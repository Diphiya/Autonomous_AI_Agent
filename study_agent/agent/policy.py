"""Deterministic policy: everything that must NOT be left to the LLM.

The router LLM *proposes* an action; :func:`apply_policy` *disposes*. It enforces
execution limits, blocks duplicate or already-failed tool calls (loop prevention),
checks preconditions (e.g. you can't grade without an active quiz), resolves
references like "week 2" or "it" against state, fills defaults, and caps the number
of consecutive clarification questions. Every override is recorded with a reason so
the decision trace explains *why* the executed action differs from the proposed one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Settings
from ..schemas import TOOL_ACTIONS, Action, RouterDecision, StudentProfile

DEFAULT_WEEKS = 4
DEFAULT_HOURS = 5.0
DEFAULT_NUM_QUESTIONS = 4


@dataclass
class Resolved:
    action: Action
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    source: str = "llm"  # llm | deterministic | policy-override | recovery
    override_reason: Optional[str] = None
    message: Optional[str] = None
    clarification_question: Optional[str] = None
    pending_action: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value, "args": self.args, "reasoning": self.reasoning, "source": self.source,
            "override_reason": self.override_reason, "message": self.message,
            "clarification_question": self.clarification_question, "pending_action": self.pending_action,
        }


def _override(base: Resolved, action: Action, reason: str, **kw) -> Resolved:
    return Resolved(action=action, reasoning=base.reasoning, source="policy-override", override_reason=reason, **kw)


def apply_policy(decision: RouterDecision, state: dict[str, Any], profile: StudentProfile, settings: Settings) -> Resolved:
    calls = state.get("tool_calls") or []
    steps = state.get("steps", 0)
    proposed = Resolved(action=decision.action, reasoning=decision.reasoning, message=decision.message,
                        clarification_question=decision.clarification_question,
                        pending_action=decision.pending_action.value if decision.pending_action else None)

    # ---- 1. hard execution limits ------------------------------------------------
    if steps > settings.max_steps_per_turn:
        return _override(proposed, Action.respond, f"step limit ({settings.max_steps_per_turn}) reached for this turn")
    if decision.action in TOOL_ACTIONS and len(calls) >= settings.max_tool_calls_per_turn:
        return _override(proposed, Action.respond, f"tool-call limit ({settings.max_tool_calls_per_turn}) reached")

    # ---- 2. clarification budget -------------------------------------------------
    if decision.action == Action.ask_clarification:
        if state.get("consecutive_clarifications", 0) >= settings.max_consecutive_clarifications:
            if (decision.pending_action == Action.create_study_plan or state.get("pending_action") == Action.create_study_plan.value) \
                    and (profile.subject or profile.goal):
                decision = decision.model_copy(update={"action": Action.create_study_plan})
                proposed.override_reason = "clarification budget used up; planning with stated defaults"
                proposed.source = "policy-override"
            else:
                return _override(proposed, Action.respond,
                                 "clarification budget used up; offering options instead of asking again",
                                 message="I don't want to keep asking questions — here's what I can do: build a study "
                                         "plan (tell me subject + weeks), explain any topic, or quiz you. Just name one!")
        else:
            return proposed

    if decision.action == Action.respond:
        return proposed

    # ---- 3. loop prevention: never retry a failed tool, never repeat a success ----
    failed = {c["action"] for c in calls if c["status"] == "failed"}
    if decision.action.value in failed:
        return _override(proposed, Action.respond, f"{decision.action.value} already failed this turn; not retrying")

    # ---- 4. per-action preconditions + argument resolution -------------------------
    if decision.action == Action.evaluate_answers:
        if any(c["action"] == Action.evaluate_answers.value and c["status"] == "ok" for c in calls):
            return _override(proposed, Action.respond,
                             "duplicate call blocked: answers were already graded this turn")
        quiz = state.get("active_quiz")
        if not quiz:
            return _override(proposed, Action.respond, "evaluate_answers requested but there is no active quiz",
                             message="There's no open quiz to grade right now. Want me to generate one? Just tell me the topic.")
        answers = decision.student_answers or state.get("parsed_answers")
        if not answers:
            return _override(proposed, Action.ask_clarification, "no answers could be read from the message",
                             clarification_question=f"Please send your answers in the form `1. B 2. A …` "
                                                    f"({len(quiz['questions'])} questions).",
                             pending_action=Action.evaluate_answers.value)
        res = _keep(proposed, decision.action)
        res.args = {"answers": answers}
        return _dup_check(res, calls)

    if decision.action in (Action.explain_topic, Action.generate_quiz):
        topic, why, err = resolve_topic(decision.topic, state, profile)
        if err:
            return _override(proposed, Action.ask_clarification, err_reason(err), clarification_question=err_question(err, state),
                             pending_action=decision.action.value)
        res = _keep(proposed, decision.action)
        if why:
            res.override_reason = why if res.source == "llm" else res.override_reason
        level = (decision.difficulty or profile.level).value if (decision.difficulty or profile.level) else "beginner"
        focus = _focus_for(topic, state)
        if ":" in topic:  # "gradient descent: learning rate" from adaptive path
            topic, sub = [p.strip() for p in topic.split(":", 1)]
            focus = list(dict.fromkeys([sub] + focus))
        res.args = {"topic": topic, "level": level, "focus_on": focus}
        if decision.action == Action.generate_quiz:
            res.args["num_questions"] = max(1, min(10, decision.num_questions or DEFAULT_NUM_QUESTIONS))
        return _dup_check(res, calls)

    if decision.action == Action.create_study_plan:
        res = _keep(proposed, decision.action)
        subject = profile.subject or profile.goal
        weeks, hours = profile.duration_weeks, profile.hours_per_week
        missing = [n for n, v in (("subject", subject), ("duration_weeks", weeks), ("hours_per_week", hours)) if not v]
        if missing:
            budget_left = state.get("consecutive_clarifications", 0) < settings.max_consecutive_clarifications
            if "subject" in missing or budget_left:
                return _override(proposed, Action.ask_clarification, f"plan needs: {', '.join(missing)}",
                                 clarification_question=plan_question(missing, profile),
                                 pending_action=Action.create_study_plan.value)
            weeks, hours = weeks or DEFAULT_WEEKS, hours or DEFAULT_HOURS
            dflt = f"defaults used for {', '.join(missing)} ({weeks} weeks, {hours:g} h/week)"
            res.override_reason = "; ".join(x for x in (res.override_reason, dflt) if x)
            res.source = "policy-override"
            res.args["assumptions"] = missing
        res.args.update({
            "goal": profile.goal or f"Learn {subject}", "subject": subject, "topics": profile.topics,
            "level": profile.level.value, "duration_weeks": int(weeks), "hours_per_week": float(hours),
            "learning_style": profile.learning_style, "weak_concepts": state.get("weak_concepts") or [],
        })
        return _dup_check(res, calls)

    return proposed


def _keep(proposed: Resolved, action: Action) -> Resolved:
    return Resolved(action=action, reasoning=proposed.reasoning, source=proposed.source,
                    override_reason=proposed.override_reason)


def _dup_check(res: Resolved, calls: list[dict]) -> Resolved:
    key = _call_key(res.action.value, res.args)
    for c in calls:
        if c["status"] == "ok" and _call_key(c["action"], c["args"]) == key:
            return _override(res, Action.respond, f"duplicate call blocked: {res.action.value} already succeeded with the same arguments this turn")
    return res


def _call_key(action: str, args: dict) -> tuple:
    topic = (args.get("topic") or args.get("subject") or "").strip().lower()
    return (action, topic)


# --------------------------------------------------------------------------- #
# Reference resolution
# --------------------------------------------------------------------------- #
def resolve_topic(raw: Optional[str], state: dict[str, Any], profile: StudentProfile) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (topic, how_it_was_resolved, error)."""
    plan = state.get("study_plan")
    if raw:
        m = re.fullmatch(r"__week_(\d+)", raw.strip()) or re.fullmatch(r"(?i)week\s*(\d+)", raw.strip())
        if m:
            n = int(m.group(1))
            if not plan:
                return None, None, "no_plan_for_week"
            if not 1 <= n <= len(plan["weeks"]):
                return None, None, f"week_out_of_range:{len(plan['weeks'])}"
            return ", ".join(plan["weeks"][n - 1]["topics"]), f"resolved 'week {n}' from the study plan", None
        return raw.strip(), None, None
    if state.get("last_topic"):
        return state["last_topic"], "no topic given; using the last topic discussed", None
    if profile.topics:
        return profile.topics[0], "no topic given; using the first topic from the profile", None
    if plan:
        return ", ".join(plan["weeks"][0]["topics"]), "no topic given; using week 1 of the plan", None
    return None, None, "no_topic"


def _focus_for(topic: str, state: dict[str, Any]) -> list[str]:
    ev = state.get("last_evaluation") or (state.get("quiz_history") or [None])[-1]
    if ev and ev.get("topic", "").lower() in topic.lower():
        return list(ev.get("weak_concepts") or [])
    return []


def err_reason(err: str) -> str:
    if err == "no_plan_for_week":
        return "student referred to a plan week but no study plan exists"
    if err.startswith("week_out_of_range"):
        return f"requested week is outside the plan (plan has {err.split(':')[1]} weeks)"
    return "no topic given and none can be inferred from state"


def err_question(err: str, state: dict[str, Any]) -> str:
    if err == "no_plan_for_week":
        return "You don't have a study plan yet, so I don't know what 'week N' covers. Which topic should I use?"
    if err.startswith("week_out_of_range"):
        n = err.split(":")[1]
        return f"Your plan only has {n} weeks — which week (1–{n}) or topic did you mean?"
    return "Which topic would you like? (e.g. 'gradient descent', 'recursion')"


def plan_question(missing: list[str], profile: StudentProfile) -> str:
    asks = {"subject": "which subject or exam it's for", "duration_weeks": "how many weeks you have",
            "hours_per_week": "about how many hours per week you can study"}
    parts = [asks[m] for m in missing]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"To build your plan I need to know {joined}."


# --------------------------------------------------------------------------- #
# Deterministic answer parsing (used for the fast path)
# --------------------------------------------------------------------------- #
_NUMBERED = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[,;]))(\d{1,2})\s*[\.\):\-]\s+", re.M)


def parse_answers(text: str, n_questions: int) -> Optional[list[str]]:
    """Parse '1. B 2. A 3) some text' or 'B, A, C, D'. Returns a list of length n or None."""
    if not text or n_questions <= 0:
        return None
    matches = list(_NUMBERED.finditer(text))
    if matches:
        answers = [""] * n_questions
        found = 0
        for i, m in enumerate(matches):
            qid = int(m.group(1))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            ans = text[m.end():end].strip().rstrip(",;").strip()
            if 1 <= qid <= n_questions and ans:
                answers[qid - 1] = ans
                found += 1
        return answers if found else None
    tokens = re.findall(r"\b([A-Da-d])\b", text)
    stripped = re.sub(r"[\sA-Da-d,;/]", "", text)
    if len(tokens) == n_questions and not stripped:
        return [t.upper() for t in tokens]
    return None
