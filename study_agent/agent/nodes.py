"""LangGraph node implementations.

Flow of one user turn (see docs/ARCHITECTURE.md for the diagram):

    ingest ──(answers to active quiz?)──► evaluate_answers   [deterministic fast path]
       │
       └──► router ──► policy ──► {create_study_plan | explain_topic | generate_quiz | evaluate_answers}
               ▲                             │ ok                         │ failed
               └─────────────────────────────┘                            ▼
               ▲                                                       recover ──(degraded retry)──► generate_quiz
               └──────────────────────────────────────────────────────────┘
            router ──► ask_clarification ──► END
            router ──► respond ──► END
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .. import prompts
from ..config import Settings
from ..llm import FaultInjector, LLMClient, LLMError, structured_call
from ..schemas import (
    Action,
    ExplainInput,
    Quiz,
    QuizInput,
    RouterDecision,
    StudentProfile,
    StudyPlanInput,
)
from ..state import AgentState
from ..tools import ToolContext, ToolFailure, run_tool
from ..tools import create_study_plan as t_plan
from ..tools import evaluate_answers as t_eval
from ..tools import explain_topic as t_explain
from ..tools import generate_quiz as t_quiz
from . import render
from .policy import Resolved, apply_policy, parse_answers

log = logging.getLogger(__name__)


@dataclass
class Deps:
    llm: LLMClient
    settings: Settings
    faults: FaultInjector


def _session_id(config: RunnableConfig) -> Optional[str]:
    return (config or {}).get("configurable", {}).get("thread_id")


def _trace(state: AgentState, node: str, decision: str, reasoning: str, source: str, detail: str | None = None) -> list:
    return list(state.get("trace") or []) + [{
        "step": state.get("steps", 0), "node": node, "decision": decision,
        "reasoning": reasoning, "source": source, "detail": detail,
    }]


class Nodes:
    def __init__(self, deps: Deps):
        self.d = deps

    # ------------------------------------------------------------------ #
    # ingest: reset per-turn scratch, record the message, deterministic parsing
    # ------------------------------------------------------------------ #
    def ingest(self, state: AgentState, config: RunnableConfig) -> dict:
        msg = state["user_input"].strip()
        quiz = state.get("active_quiz")
        parsed = parse_answers(msg, len(quiz["questions"])) if quiz else None
        update: dict[str, Any] = {
            "messages": [HumanMessage(content=msg)],
            "profile": state.get("profile") or StudentProfile().model_dump(mode="json"),
            "quiz_history": state.get("quiz_history") or [],
            "weak_concepts": state.get("weak_concepts") or [],
            "consecutive_clarifications": state.get("consecutive_clarifications", 0),
            "steps": 0, "tool_calls": [], "outputs": [], "notes": [], "trace": [],
            "decision": None, "final_reply": None, "pending_retry": None, "parsed_answers": parsed,
        }
        # Deterministic fast path: the message is *only* answers to every question of the active quiz.
        if quiz and parsed and all(a.strip() for a in parsed) and _looks_like_answers_only(msg):
            res = Resolved(action=Action.evaluate_answers, args={"answers": parsed}, source="deterministic",
                           reasoning=f"Message is a complete set of {len(parsed)} answers to the active quiz; "
                                     "graded without an LLM routing call.")
            update["decision"] = res.to_dict()
            update["steps"] = 1
            update["trace"] = [{"step": 1, "node": "ingest", "decision": "evaluate_answers",
                                "reasoning": res.reasoning, "source": "deterministic", "detail": None}]
        return update

    @staticmethod
    def after_ingest(state: AgentState) -> str:
        d = state.get("decision")
        return "evaluate_answers" if d and d["action"] == Action.evaluate_answers.value else "router"

    # ------------------------------------------------------------------ #
    # router: LLM proposes, policy disposes
    # ------------------------------------------------------------------ #
    def router(self, state: AgentState, config: RunnableConfig) -> dict:
        steps = state.get("steps", 0) + 1
        st = {**state, "steps": steps}
        profile = StudentProfile(**state["profile"])

        if steps > self.d.settings.max_steps_per_turn:
            res = Resolved(action=Action.respond, source="policy-override", reasoning="",
                           override_reason=f"step limit ({self.d.settings.max_steps_per_turn}) reached")
            return self._router_update(st, res, profile, proposed=None)

        summary = self._state_summary(st, profile)
        history = _history(state, self.d.settings.history_window)
        messages = [
            {"role": "system", "content": prompts.ROUTER_SYSTEM},
            {"role": "user", "content": prompts.router_user_prompt(summary, history, state["user_input"])},
        ]
        context = {**summary, "message": state["user_input"], "history": history,
                   "parsed_answers": state.get("parsed_answers")}
        attempt_log: list[str] = []
        try:
            decision = structured_call(
                self.d.llm, "route", messages, RouterDecision, context,
                max_retries=self.d.settings.llm_max_retries, faults=self.d.faults,
                session_id=_session_id(config), attempt_log=attempt_log,
                backoff_s=0.05 if self.d.llm.name == "mock" else 0.5,
            )
        except LLMError as e:
            # Router itself is down/invalid after retries: degrade deterministically.
            log.error("router failed: %s", e)
            msg = ("Sorry — I had trouble processing that just now. Could you try again or rephrase? "
                   "I can build a study plan, explain a topic, or quiz you.")
            if state.get("tool_calls"):
                msg = "That's what I managed so far — I hit a problem deciding the next step, so let me know what you'd like next."
            res = Resolved(action=Action.respond, source="recovery", reasoning="Router LLM failed after retries.",
                           override_reason=str(e), message=msg)
            return self._router_update(st, res, profile, proposed=None, attempt_log=attempt_log)

        profile = profile.merge(decision.profile_updates)
        res = apply_policy(decision, st, profile, self.d.settings)
        return self._router_update(st, res, profile, proposed=decision, attempt_log=attempt_log)

    def _router_update(self, st, res: Resolved, profile: StudentProfile, proposed: Optional[RouterDecision],
                       attempt_log: Optional[list[str]] = None) -> dict:
        detail_parts = []
        if proposed is not None and proposed.action != res.action:
            detail_parts.append(f"LLM proposed {proposed.action.value} → executed {res.action.value}")
        if res.override_reason:
            detail_parts.append(res.override_reason)
        if attempt_log and len(attempt_log) > 1:
            detail_parts.append(" | ".join(attempt_log))
        if res.args:
            shown = {k: v for k, v in res.args.items() if k not in ("weak_concepts",) and v not in (None, [], "")}
            detail_parts.append(f"args={shown}")
        return {
            "steps": st["steps"], "profile": profile.model_dump(mode="json"), "decision": res.to_dict(),
            "trace": _trace(st, "router", res.action.value, res.reasoning, res.source, "; ".join(detail_parts) or None),
        }

    @staticmethod
    def after_router(state: AgentState) -> str:
        return state["decision"]["action"]

    def _state_summary(self, st: AgentState, profile: StudentProfile) -> dict[str, Any]:
        plan = st.get("study_plan")
        quiz = st.get("active_quiz")
        last_eval = (st.get("quiz_history") or [None])[-1]
        return {
            "profile": profile.model_dump(mode="json"),
            "has_plan": bool(plan),
            "plan_week_topics": [w["topics"] for w in plan["weeks"]] if plan else [],
            "active_quiz": {"topic": quiz["topic"], "num_questions": len(quiz["questions"])} if quiz else None,
            "last_topic": st.get("last_topic"),
            "weak_concepts": st.get("weak_concepts") or [],
            "last_evaluation": last_eval,
            "pending_action": st.get("pending_action"),
            "consecutive_clarifications": st.get("consecutive_clarifications", 0),
            "tool_calls_this_turn": [{k: c.get(k) for k in ("action", "args", "status", "error", "summary")}
                                     for c in st.get("tool_calls") or []],
            "turn_progress": _turn_progress(st),
            "step": st.get("steps", 0),
            "step_limit": self.d.settings.max_steps_per_turn,
        }

    # ------------------------------------------------------------------ #
    # tool nodes
    # ------------------------------------------------------------------ #
    def _ctx(self, config: RunnableConfig) -> ToolContext:
        return ToolContext(self.d.llm, self.d.settings, self.d.faults, _session_id(config))

    def _run(self, state: AgentState, config: RunnableConfig, action: Action,
             fn: Callable[[ToolContext], Any], on_ok: Callable[[Any], dict]) -> dict:
        ctx = self._ctx(config)
        args = state["decision"]["args"]
        record = {"action": action.value, "args": {k: v for k, v in args.items() if k != "answers"}}
        try:
            result = run_tool(action.value, lambda: fn(ctx), self.d.settings.tool_timeout_s)
        except ToolFailure as e:
            record.update(status="failed", error=e.reason, attempts=len(ctx.attempt_log) or e.attempts)
            return {
                "tool_calls": list(state.get("tool_calls") or []) + [record],
                "trace": _trace(state, action.value, "failed", f"Tool failed: {e.reason}", "tool",
                                " | ".join(ctx.attempt_log) or None),
            }
        update = on_ok(result)
        record.update(status="ok", error=None, attempts=len(ctx.attempt_log), summary=update.pop("_summary", None))
        retried = len(ctx.attempt_log) > 1 or any("failed" in a for a in ctx.attempt_log)
        update["tool_calls"] = list(state.get("tool_calls") or []) + [record]
        update["consecutive_clarifications"] = 0
        if state.get("pending_action") == action.value:
            update["pending_action"] = None
        update["trace"] = _trace(state, action.value, "ok", record["summary"] or "done", "tool",
                                 " | ".join(ctx.attempt_log) if retried else None)
        return update

    def create_study_plan(self, state: AgentState, config: RunnableConfig) -> dict:
        args = dict(state["decision"]["args"])
        assumptions = args.pop("assumptions", None)
        inp = StudyPlanInput(**args)

        def ok(plan):
            p = plan.model_dump(mode="json")
            notes = list(state.get("notes") or [])
            if assumptions:
                said = {"duration_weeks": f"{inp.duration_weeks} weeks", "hours_per_week": f"{inp.hours_per_week:g} h/week"}
                assumed = " and ".join(said[a] for a in assumptions if a in said)
                notes.append(f"I assumed {assumed} since that wasn't specified — tell me if you'd like to change it.")
            return {"study_plan": p, "outputs": list(state.get("outputs") or []) + [{"kind": "plan", "data": p}],
                    "notes": notes, "_summary": f"{plan.duration_weeks}-week plan for '{plan.goal}'"}

        return self._run(state, config, Action.create_study_plan, lambda c: t_plan(inp, c), ok)

    def explain_topic(self, state: AgentState, config: RunnableConfig) -> dict:
        a = state["decision"]["args"]
        profile = StudentProfile(**state["profile"])
        inp = ExplainInput(topic=a["topic"], level=a.get("level", "beginner"), subject=profile.subject,
                           learning_style=profile.learning_style, focus_on=a.get("focus_on") or [])

        def ok(expl):
            e = expl.model_dump(mode="json")
            return {"last_topic": a["topic"], "outputs": list(state.get("outputs") or []) + [{"kind": "explanation", "data": e}],
                    "_summary": f"explained '{a['topic']}'" + (f" (focus: {', '.join(inp.focus_on)})" if inp.focus_on else "")}

        return self._run(state, config, Action.explain_topic, lambda c: t_explain(inp, c), ok)

    def generate_quiz(self, state: AgentState, config: RunnableConfig) -> dict:
        a = state["decision"]["args"]
        inp = QuizInput(topic=a["topic"], level=a.get("level", "beginner"), num_questions=a.get("num_questions", 4),
                        question_types=a.get("question_types") or ["mcq", "short"], focus_on=a.get("focus_on") or [])

        def ok(quiz):
            q = quiz.model_dump(mode="json")
            notes = list(state.get("notes") or [])
            if a.get("degraded"):
                notes.append("The full quiz couldn't be generated, so this is a shorter multiple-choice version.")
            return {"active_quiz": q, "last_topic": a["topic"], "notes": notes,
                    "outputs": list(state.get("outputs") or []) + [{"kind": "quiz", "data": q}],
                    "_summary": f"{len(quiz.questions)}-question quiz on '{quiz.topic}'"}

        return self._run(state, config, Action.generate_quiz, lambda c: t_quiz(inp, c), ok)

    def evaluate_answers(self, state: AgentState, config: RunnableConfig) -> dict:
        quiz = Quiz(**state["active_quiz"])
        answers = state["decision"]["args"]["answers"]

        def ok(ev):
            e = ev.model_dump(mode="json")
            weak = [w for w in (state.get("weak_concepts") or [])
                    if w not in {r.concept for r in ev.results if r.correct}]
            weak = list(dict.fromkeys(weak + ev.weak_concepts))
            hist = list(state.get("quiz_history") or []) + [
                {"topic": ev.topic, "total_score": ev.total_score, "weak_concepts": ev.weak_concepts}]
            return {"active_quiz": None, "quiz_history": hist, "weak_concepts": weak, "last_topic": ev.topic,
                    "outputs": list(state.get("outputs") or []) + [{"kind": "evaluation", "data": e}],
                    "_summary": f"score {ev.total_score:.0%}; weak: {', '.join(ev.weak_concepts) or 'none'}"}

        return self._run(state, config, Action.evaluate_answers, lambda c: t_eval(quiz, answers, c), ok)

    @staticmethod
    def after_tool(state: AgentState) -> str:
        calls = state.get("tool_calls") or []
        return "recover" if calls and calls[-1]["status"] == "failed" else "router"

    # ------------------------------------------------------------------ #
    # recovery: deterministic fallback strategies after a tool failure
    # ------------------------------------------------------------------ #
    def recover(self, state: AgentState, config: RunnableConfig) -> dict:
        failed = state["tool_calls"][-1]
        action = failed["action"]
        notes = list(state.get("notes") or [])
        prior_degraded = any(c["action"] == action and c["args"].get("degraded") for c in state["tool_calls"][:-1])

        if action == Action.generate_quiz.value and not failed["args"].get("degraded") and not prior_degraded:
            # Strategy: retry once with a simpler request (fewer, MCQ-only questions) — simpler
            # outputs are much less likely to fail validation or time out.
            args = dict(state["decision"]["args"])
            args.update(num_questions=min(3, args.get("num_questions", 4)), question_types=["mcq"], degraded=True)
            decision = {**state["decision"], "args": args, "source": "recovery"}
            return {"decision": decision, "notes": notes, "pending_retry": {"action": action},
                    "trace": _trace(state, "recover", "retry generate_quiz (degraded)",
                                    "Quiz generation failed; retrying once with 3 MCQ-only questions.", "recovery",
                                    f"original error: {failed['error']}")}

        friendly = {
            "create_study_plan": "build your study plan",
            "explain_topic": "generate the explanation",
            "generate_quiz": "generate the quiz",
            "evaluate_answers": "grade your answers",
        }[action]
        notes.append(f"⚠️ I couldn't {friendly} right now (the AI service didn't respond properly after retries). "
                     "Nothing was lost — just ask again in a moment.")
        return {"notes": notes, "pending_retry": None,
                "trace": _trace(state, "recover", "report failure, return to router",
                                f"{action} failed permanently this turn; marked as blocked so it won't be retried.",
                                "recovery", failed["error"])}

    @staticmethod
    def after_recover(state: AgentState) -> str:
        return "generate_quiz" if state.get("pending_retry") else "router"

    # ------------------------------------------------------------------ #
    # terminal nodes
    # ------------------------------------------------------------------ #
    def ask_clarification(self, state: AgentState, config: RunnableConfig) -> dict:
        d = state["decision"]
        parts = [render.render_block(b) for b in state.get("outputs") or []] + list(state.get("notes") or [])
        q = d.get("clarification_question") or "Could you tell me a bit more about what you need?"
        reply = "\n\n".join(parts + [q])
        return {"final_reply": reply, "messages": [AIMessage(content=reply)],
                "consecutive_clarifications": state.get("consecutive_clarifications", 0) + 1,
                "pending_action": d.get("pending_action") or state.get("pending_action")}

    def respond(self, state: AgentState, config: RunnableConfig) -> dict:
        d = state.get("decision") or {}
        parts = [render.render_block(b) for b in state.get("outputs") or []]
        parts += list(state.get("notes") or [])
        if d.get("override_reason") and "limit" in (d.get("override_reason") or ""):
            parts.append("(I stopped here to avoid doing too much in one go — tell me what to do next.)")
        if d.get("message"):
            parts.append(d["message"])
        if not parts:
            parts.append("What would you like to do next? I can plan, explain, or quiz.")
        reply = "\n\n".join(parts)
        return {"final_reply": reply, "messages": [AIMessage(content=reply)]}


def _turn_progress(st: AgentState) -> str:
    """Plain-language summary of what has ALREADY been done for the latest message."""
    calls = st.get("tool_calls") or []
    if not calls:
        return "Nothing done yet for the latest message."
    parts = [f"{c['action']} {c['status']}" + (f" ({c.get('summary')})" if c.get("summary") else "") for c in calls]
    msg = "Already done for the latest message: " + "; ".join(parts) + ". Do NOT redo these."
    ev = (st.get("quiz_history") or [None])[-1]
    if any(c["action"] == "evaluate_answers" and c["status"] == "ok" for c in calls) and ev:
        msg += " The student's answers HAVE BEEN GRADED; the quiz is closed."
        if ev["total_score"] < 0.6 and not any(c["action"] == "explain_topic" for c in calls):
            msg += f" Score {ev['total_score']:.0%} is below 60%: rule 3 applies (explain the weakest concept)."
    return msg


def _history(state: AgentState, window: int) -> list[dict[str, str]]:
    msgs = list(state.get("messages") or [])[:-1]  # exclude the current message (passed separately)
    out = []
    for m in msgs[-window:]:
        role = "student" if isinstance(m, HumanMessage) else "assistant"
        out.append({"role": role, "content": str(m.content)})
    return out


def _looks_like_answers_only(msg: str) -> bool:
    """True if nothing before the first numbered answer looks like a new request."""
    import re
    head = re.split(r"(?:^|\s)1\s*[\.\):\-]", msg, maxsplit=1)[0]
    if re.search(r"\b(explain|plan|quiz|instead|don't|dont)\b", head.lower()):
        return False
    return len(head.strip()) < 40
