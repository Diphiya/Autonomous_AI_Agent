"""Deterministic, offline stand-in for an LLM.

Purpose: run the *whole* agent (graph, routing, policy, tools, retries, persistence,
API) in tests, CI and demos without network access or API keys, with reproducible
output. It is rule-based on purpose and makes no claim to be smart — decision quality
must be measured with a real model (see eval/ and docs/EVALUATION.md).

It honours the same contract as a real client: returns validated Pydantic objects and
reads only the structured ``context`` the agent passes alongside the prompt.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from ..schemas import (
    Action,
    Explanation,
    Level,
    ProfileUpdate,
    Quiz,
    QuizQuestion,
    RouterDecision,
    ShortAnswerGrade,
    StudyPlan,
    StudyWeek,
)
from . import mock_kb

T = TypeVar("T", bound=BaseModel)

INTENT_PATTERNS: list[tuple[Action, str]] = [
    (Action.create_study_plan, r"\b(study plan|plan|schedule|roadmap|timetable|revision plan)\b"),
    (Action.explain_topic, r"\b(explain|what is|what's|what are|how does|how do|teach me|don'?t understand|don'?t get|confused about|re-?explain)\b"),
    (Action.generate_quiz, r"\b(quiz|test me|practice questions?|practice problems?|questions on|questions about|mcqs?|exercises?)\b"),
    (Action.evaluate_answers, r"\b(check my answers|grade my|here are my answers|my answers)\b"),
]
GREETING = r"^\s*(hi|hello|hey|thanks|thank you|thx|ok|okay|cool|great|bye)\b"
VAGUE_STUDY = r"\b(help me (study|learn|prepare)|i have an exam|exam (is )?coming|want to learn|need to study|get better at)\b"

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2}


class MockLLM:
    name = "mock"

    def generate(self, task: str, messages: list[dict[str, str]], schema: Type[T], context: dict[str, Any]) -> T:
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ValueError(f"MockLLM has no handler for task {task!r}")
        result = handler(context)
        return schema.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)

    # ------------------------------------------------------------------ #
    # Router
    # ------------------------------------------------------------------ #
    def _task_route(self, ctx: dict[str, Any]) -> RouterDecision:
        msg: str = ctx["message"]
        low = msg.lower()
        profile = ctx.get("profile") or {}
        upd = extract_profile(msg, ctx.get("pending_action"), profile)
        merged = {**profile, **upd.model_dump(exclude_none=True)}
        calls = ctx.get("tool_calls_this_turn") or []
        done_ok = {c["action"] for c in calls if c["status"] == "ok"}
        attempted = {c["action"] for c in calls}
        intents = detect_intents(low)
        topic = _ordinal_topic(low, ctx.get("plan_week_topics")) or extract_topic(msg) \
            or _pronoun_topic(low, ctx.get("last_topic"))
        num_q = extract_num_questions(low)

        def decide(action: Action, reasoning: str, **kw) -> RouterDecision:
            return RouterDecision(reasoning=reasoning, action=action, profile_updates=upd if _nonempty(upd) else None, **kw)

        # ---------- follow-up step inside the same turn ----------
        if calls:
            last = calls[-1]
            ev = ctx.get("last_evaluation")
            if (
                last["action"] == Action.evaluate_answers.value and last["status"] == "ok" and ev
                and ev["total_score"] < 0.6 and ev.get("weak_concepts") and Action.explain_topic.value not in attempted
            ):
                weak = ev["weak_concepts"][0]
                return decide(Action.explain_topic,
                              f"Score was {ev['total_score']:.0%} (<60%); adapting by re-explaining the weakest concept "
                              f"'{weak}' before moving on.", topic=f"{ev['topic']}: {weak}")
            for a in intents:
                if a.value not in attempted and a != Action.evaluate_answers:
                    return decide(a, f"The student also asked for {a.value.replace('_', ' ')} in the same message; "
                                     f"{', '.join(sorted(done_ok)) or 'previous step'} is done.",
                                  topic=topic, num_questions=num_q if a == Action.generate_quiz else None)
            return decide(Action.respond, "All requested work for this message is done; summarise and suggest a next step.",
                          message=_next_step_message(done_ok, ctx))

        # ---------- first step of the turn ----------
        if ctx.get("parsed_answers") and ctx.get("active_quiz") and (not intents or intents[0] == Action.evaluate_answers):
            return decide(Action.evaluate_answers, "The message contains answers to the active quiz.",
                          student_answers=ctx["parsed_answers"])

        if not intents:
            if ctx.get("pending_action") == Action.create_study_plan.value:
                intents = [Action.create_study_plan]
            elif ctx.get("pending_action") in (Action.explain_topic.value, Action.generate_quiz.value) and msg.strip():
                intents = [Action(ctx["pending_action"])]
                topic = topic or msg.strip().rstrip(".!?")
            elif re.search(GREETING, low):
                return decide(Action.respond, "Greeting/acknowledgement; no tool needed.",
                              message="Happy to help! I can build a study plan, explain a topic, or quiz you. What are you working on?")
            elif re.search(VAGUE_STUDY, low) or len(low.split()) <= 4:
                why = (f"Subject is now known ({merged['subject']}) but not what kind of help is wanted."
                       if merged.get("subject") else "The request is too vague to pick a tool: no subject, goal or timeframe given.")
                return decide(Action.ask_clarification, why,
                              clarification_question=_clarify_vague(merged))
            else:
                return decide(Action.respond, "No study action requested; answer conversationally.",
                              message="I can make a study plan, explain a concept, or generate a quiz. Which would help most?")

        first = intents[0]
        if first == Action.create_study_plan:
            missing = [f for f, label in (("subject", "subject"), ("duration_weeks", "weeks"), ("hours_per_week", "hours"))
                       if not merged.get(f)]
            if missing:
                return decide(Action.ask_clarification,
                              f"Planning needs subject, timeframe and weekly hours; missing: {', '.join(missing)}.",
                              clarification_question=_clarify_plan(missing, merged),
                              pending_action=Action.create_study_plan)
            return decide(Action.create_study_plan,
                          "Student asked for a plan and subject, timeframe and weekly hours are all known.")
        if first == Action.evaluate_answers:
            return decide(Action.evaluate_answers, "The student wants their answers checked.",
                          student_answers=ctx.get("parsed_answers"))
        if first in (Action.explain_topic, Action.generate_quiz) and not topic and not ctx.get("last_topic") \
                and not merged.get("topics") and not ctx.get("has_plan"):
            what = "explain" if first == Action.explain_topic else "quiz you on"
            return decide(Action.ask_clarification, f"Asked to {what} something but no topic is known yet.",
                          clarification_question=f"Sure — which topic should I {what}?", pending_action=first)
        label = "an explanation" if first == Action.explain_topic else "practice questions"
        shown = f"week {topic[7:]} of the plan" if topic and topic.startswith("__week_") else f"'{topic}'"
        return decide(first, f"Student asked for {label}" + (f" on {shown}." if topic else "; topic resolved from context."),
                      topic=topic, num_questions=num_q if first == Action.generate_quiz else None)

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def _task_create_study_plan(self, ctx: dict[str, Any]) -> StudyPlan:
        inp = ctx["input"]
        weeks_n, hours = inp["duration_weeks"], float(inp["hours_per_week"])
        pool = list(dict.fromkeys(inp.get("weak_concepts", []) + inp.get("topics", []) + mock_kb.subject_topics(inp["subject"])))
        if not pool:
            pool = [f"{inp['subject']} fundamentals", "core concepts", "worked problems", "applications", "advanced topics", "review"]
        per_week = max(1, -(-len(pool) // weeks_n)) if len(pool) <= weeks_n * 3 else 3
        style = (inp.get("learning_style") or "").lower()
        weeks = []
        for i in range(weeks_n):
            chunk = pool[i * per_week:(i + 1) * per_week] or [f"review & practice: {pool[i % len(pool)]}"]
            last = i == weeks_n - 1
            acts = [f"Study {t} ({'worked examples first' if 'example' in style else 'read + summarise notes'})" for t in chunk]
            acts.append("Mixed practice quiz on this week's topics")
            if last:
                acts.append("Full revision session and a timed mock test")
            weeks.append(StudyWeek(week=i + 1, focus=chunk[0].title() if not last or len(chunk) > 0 else "Revision",
                                   topics=chunk, activities=acts, estimated_hours=round(hours, 1),
                                   checkpoint=f"Score ≥70% on a quiz covering: {', '.join(chunk)}"))
        tips = [f"Split your {hours:g} h/week into 3–4 shorter sessions; spacing beats cramming.",
                "Start each session by recalling last session's key points without notes."]
        if inp.get("weak_concepts"):
            tips.insert(0, f"Weak spots from your quizzes ({', '.join(inp['weak_concepts'])}) are scheduled first.")
        return StudyPlan(goal=inp["goal"], duration_weeks=weeks_n, hours_per_week=hours, weeks=weeks, tips=tips)

    def _task_explain_topic(self, ctx: dict[str, Any]) -> Explanation:
        inp = ctx["input"]
        name, kb = mock_kb.lookup_topic(inp["topic"])
        level = inp.get("level", "beginner")
        if kb is None:
            t = inp["topic"]
            return Explanation(
                topic=t, summary=f"{t} is a core idea in {inp.get('subject') or 'this subject'}.",
                explanation=(f"[offline mock content] A real LLM would explain {t} here at {level} level. "
                             "Configure LLM_MODEL to get real explanations."),
                example=f"A worked example of {t}.", key_points=[f"Definition of {t}", f"When {t} is used", "A common pitfall"],
                common_mistakes=[], check_question=f"In one sentence, what is {t}?")
        focus = inp.get("focus_on") or []
        expl = kb["explanation"]
        if focus:
            expl = f"Let's focus on {', '.join(focus)}, which tripped you up. " + expl
        if level == "beginner":
            expl += " Don't worry about the maths yet; focus on the intuition."
        return Explanation(topic=name, summary=kb["summary"], explanation=expl, analogy=kb["analogy"],
                           example=kb["example"], key_points=kb["key_points"],
                           common_mistakes=kb["common_mistakes"], check_question=kb["check_question"])

    def _task_generate_quiz(self, ctx: dict[str, Any]) -> Quiz:
        inp = ctx["input"]
        n = inp.get("num_questions", 4)
        name, kb = mock_kb.lookup_topic(inp["topic"])
        types = inp.get("question_types") or ["mcq", "short"]
        if kb is not None:
            pool = [q for q in kb["questions"] if q["type"] in types]
            focus = [f.lower() for f in inp.get("focus_on") or []]
            pool.sort(key=lambda q: 0 if any(f in q["concept"] for f in focus) else 1)
            chosen = pool[:n]
            qs = [QuizQuestion(id=i + 1, **{k: v for k, v in q.items() if k != "keywords"}) for i, q in enumerate(chosen)]
            return Quiz(topic=name, level=inp.get("level", "beginner"), questions=qs)
        t = inp["topic"]
        qs = []
        for i in range(n):
            qs.append(QuizQuestion(
                id=i + 1, type="mcq", question=f"[offline mock] Which statement about {t} is correct? (#{i + 1})",
                options=[f"Correct statement about {t}", f"Common misconception about {t}",
                         f"Statement about an unrelated topic", f"Over-generalisation of {t}"],
                correct_option="A", reference_answer=f"Option A is the accurate description of {t}.", concept=t))
        return Quiz(topic=t, level=inp.get("level", "beginner"), questions=qs)

    def _task_grade_short_answer(self, ctx: dict[str, Any]) -> ShortAnswerGrade:
        answer = (ctx.get("answer") or "").lower()
        keywords: list[str] = ctx.get("keywords") or _kb_keywords(ctx.get("question", "")) \
            or _keywords_from(ctx.get("reference_answer", ""))
        if not answer.strip():
            return ShortAnswerGrade(score=0.0, feedback="No answer given.", missing_points=keywords[:3])
        hits = [k for k in keywords if any(alt in answer for alt in k.split("|"))]
        score = round(len(hits) / max(1, len(keywords)), 2)
        missing = [k.split("|")[0] for k in keywords if k not in hits]
        fb = ("Good — you covered the key ideas." if score >= 0.75 else
              "Partly right." if score >= 0.4 else "This misses the main idea.")
        if missing:
            fb += f" Consider mentioning: {', '.join(missing)}."
        return ShortAnswerGrade(score=score, feedback=fb, missing_points=missing)


# ---------------------------------------------------------------------- #
# helpers (rule-based "understanding")
# ---------------------------------------------------------------------- #
def detect_intents(low: str) -> list[Action]:
    found = []
    for action, pat in INTENT_PATTERNS:
        m = re.search(pat, low)
        if m:
            found.append((m.start(), action))
    # "explain" inside "give me a plan and explain" etc. ordering by position
    return [a for _, a in sorted(found)]


_TOPIC_PATTERNS = [
    r"(?:quiz me on|quiz on|test me on|questions (?:on|about)|practice (?:questions|problems) (?:on|about|for))\s+(.+)",
    r"(?:explain|teach me|what is|what's|what are|how does|how do|confused about|don'?t understand|don'?t get)\s+(.+)",
    r"\b(?:quiz|questions)\s+(?:for|about)\s+(.+)",
]
_STOP = re.compile(r"\s*(?:\band\b|\bthen\b|\bplease\b|[,.?!;]|\bwith\b|\bat (?:a )?\w+ level\b|\bfor (?:a )?beginners?\b|\bin simple terms\b|\bwork\b)", re.I)


def extract_topic(msg: str) -> Optional[str]:
    low = msg.lower()
    wk = re.search(r"\bweek\s*(\d+)\b", low)
    if wk and re.search(r"\b(quiz|questions|test|explain)\b", low):
        return f"__week_{wk.group(1)}"
    for pat in _TOPIC_PATTERNS:
        m = re.search(pat, msg, re.I)
        if m:
            cand = _STOP.split(m.group(1), maxsplit=1)[0].strip()
            cand = re.sub(r"^(?:me |to me |the |a |an |about |on |some |\d+ |questions? |practice questions? )+", "", cand, flags=re.I).strip()
            cand = re.sub(r"^(?:\d+|one|two|three|four|five)\s+(?:practice\s+)?questions?\s+(?:on|about)\s+", "", cand, flags=re.I)
            if cand and cand.lower() not in {"it", "this", "that", "me", "them", "again", "a quiz", "quiz"} and len(cand) < 60:
                return cand
    return None


_ORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}


def _ordinal_topic(low: str, plan_week_topics: Optional[list]) -> Optional[str]:
    m = re.search(r"\b(first|1st|second|2nd|third|3rd) topic\b", low)
    if m and plan_week_topics:
        flat = [t for wk in plan_week_topics for t in wk]
        i = _ORD[m.group(1)]
        return flat[i] if i < len(flat) else None
    return None


def _pronoun_topic(low: str, last_topic: Optional[str]) -> Optional[str]:
    if last_topic and re.search(r"\b(it|this|that|again|same topic)\b", low):
        return last_topic
    return None


def extract_num_questions(low: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|a couple of)\s+(?:\w+\s+)?(?:questions|mcqs|problems)", low)
    if not m:
        return None
    v = m.group(1)
    n = int(v) if v.isdigit() else NUM_WORDS[v]
    return max(1, min(10, n))


def extract_profile(msg: str, pending_action: Optional[str], profile: dict) -> ProfileUpdate:
    low = msg.lower()
    data: dict[str, Any] = {}
    if m := re.search(r"(\d{1,2}|one|two|three|four|five|six|eight|ten)\s*(?:-|\s)?weeks?", low):
        v = m.group(1)
        data["duration_weeks"] = int(v) if v.isdigit() else NUM_WORDS[v]
    elif m := re.search(r"(\d{1,2})\s*months?", low):
        data["duration_weeks"] = min(52, int(m.group(1)) * 4)
    if m := re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:hours?|hrs?|h)\s*(?:a|per|/|each|every)\s*week", low):
        data["hours_per_week"] = float(m.group(1))
    elif m := re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:hours?|hrs?|h)\s*(?:a|per|/|each|every)\s*day", low):
        data["hours_per_week"] = float(m.group(1)) * 7
    elif m := re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:hours?|hrs?|h)\b", low):
        if pending_action == Action.create_study_plan.value:
            data["hours_per_week"] = float(m.group(1))
    for lvl in Level:
        if re.search(rf"\b{lvl.value}\b", low) or (lvl == Level.beginner and re.search(r"\b(new to|never (studied|learned)|from scratch)\b", low)):
            data["level"] = lvl
            break
    if "example" in low and re.search(r"\b(learn|prefer|like)\b", low):
        data["learning_style"] = "examples first"
    subj = None
    for name in list(mock_kb.SUBJECT_TOPICS) + list(mock_kb.SUBJECT_ALIASES):
        if re.search(rf"\b{re.escape(name)}\b", low):
            subj = mock_kb.SUBJECT_ALIASES.get(name, name)
            break
    if not subj and (m := re.search(r"(?:learn|study|studying|prepare for|preparing for|exam in|course in|plan for)\s+(?:my |the |an? )?([a-z][a-z \-]{2,30}?)(?:\s+exam|\s+course|\s+in\b|\s+over\b|\s+for\b|[,.!?]|$)", low)):
        cand = m.group(1).strip()
        if cand not in {"it", "this", "something", "stuff", "exam", "my exam", "a plan", "plan"} and not re.search(r"\d|week|hour", cand):
            subj = cand
    if not subj and pending_action == Action.create_study_plan.value and not profile.get("subject"):
        bare = re.sub(r"[^a-z \-]", " ", low).strip()
        bare = re.sub(r"\b(i|im|i'm|am|a|an|the|and|it's|its|about|is|for|beginner|intermediate|advanced|weeks?|hours?|per|week|day|maybe|around)\b", " ", bare)
        bare = " ".join(bare.split())
        if 2 <= len(bare) <= 30 and len(bare.split()) <= 3:
            subj = bare
    if subj:
        data["subject"] = subj
        if not profile.get("goal"):
            data["goal"] = f"Learn {subj}" + (" for an upcoming exam" if "exam" in low else "")
    if "exam" in low and subj:
        data["goal"] = f"Prepare for {subj} exam"
    return ProfileUpdate(**data)


def _nonempty(upd: ProfileUpdate) -> bool:
    return bool(upd.model_dump(exclude_none=True))


def _clarify_vague(profile: dict) -> str:
    if profile.get("subject"):
        return (f"Happy to help with {profile['subject']}! Would you like a study plan, an explanation of a "
                "specific topic, or a practice quiz?")
    return ("Happy to help! What subject are you studying, and what's the goal — e.g. an exam date, a course, or "
            "a specific topic you're stuck on?")


def _clarify_plan(missing: list[str], profile: dict) -> str:
    asks = {"subject": "which subject or exam it's for",
            "duration_weeks": "how many weeks you have",
            "hours_per_week": "roughly how many hours per week you can study"}
    parts = [asks[m] for m in missing]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    lead = f"I can build a {profile['subject']} plan" if profile.get("subject") else "I can build that plan"
    lvl = "" if profile.get("level_confirmed") else " Also, would you call yourself a beginner, intermediate or advanced?"
    return f"{lead} — could you tell me {joined}?{lvl}"


def _next_step_message(done: set[str], ctx: dict) -> str:
    plan_topics = ctx.get("plan_week_topics") or []
    if "evaluate_answers" in done:
        ev = ctx.get("last_evaluation") or {}
        if ev.get("total_score", 0) >= 0.8:
            return "Great work! Want to move on to the next topic in your plan?"
        return "Review the explanation above, then say 'quiz me again' to try a fresh set of questions."
    if "generate_quiz" in done:
        return "Reply with your answers like `1. B 2. A 3. ...` and I'll grade them."
    if "create_study_plan" in done and plan_topics:
        return f"Want a quick quiz on week 1 ({plan_topics[0][0]}) to check your starting point?"
    if "explain_topic" in done:
        return "Want a few practice questions on this to lock it in?"
    return "What would you like to do next?"


def _kb_keywords(question: str) -> list[str]:
    for t in mock_kb.TOPICS.values():
        for q in t["questions"]:
            if q["question"] == question and q.get("keywords"):
                return q["keywords"]
    return []


def _keywords_from(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z]{6,}", text.lower())]
    return list(dict.fromkeys(words))[:5]
