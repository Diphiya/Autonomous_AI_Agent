"""The four study tools. Each has a typed input model and a typed output model.

LLM vs deterministic split inside tools:
* planner / explainer / quiz generator: content generation -> LLM (with schema validation).
* evaluator: MCQ grading is an exact comparison -> deterministic code (no LLM, no cost,
  no randomness). Only free-text short answers are graded by the LLM, one call each;
  if that call fails the question is marked "ungraded" instead of failing the whole tool.
"""
from __future__ import annotations

import re
from typing import Optional

from .. import prompts
from ..schemas import (
    Evaluation,
    Explanation,
    ExplainInput,
    QuestionResult,
    Quiz,
    QuizInput,
    ShortAnswerGrade,
    StudyPlan,
    StudyPlanInput,
)
from ..llm import LLMError
from .base import ToolContext, ToolFailure


def create_study_plan(inp: StudyPlanInput, ctx: ToolContext) -> StudyPlan:
    payload = inp.model_dump(mode="json")
    plan = ctx.call("create_study_plan", prompts.PLANNER_SYSTEM, prompts.tool_user_prompt(payload),
                    StudyPlan, {"input": payload})
    # Business rule that the schema can't know: plan must match what was requested.
    if plan.duration_weeks != inp.duration_weeks:
        raise ToolFailure("create_study_plan", f"plan length {plan.duration_weeks} != requested {inp.duration_weeks}")
    return plan


def explain_topic(inp: ExplainInput, ctx: ToolContext) -> Explanation:
    payload = inp.model_dump(mode="json")
    return ctx.call("explain_topic", prompts.EXPLAINER_SYSTEM, prompts.tool_user_prompt(payload),
                    Explanation, {"input": payload})


def generate_quiz(inp: QuizInput, ctx: ToolContext) -> Quiz:
    payload = inp.model_dump(mode="json")
    quiz = ctx.call("generate_quiz", prompts.QUIZ_SYSTEM, prompts.tool_user_prompt(payload), Quiz, {"input": payload})
    if len(quiz.questions) > inp.num_questions:  # trim rather than fail; fewer is accepted
        quiz = Quiz(topic=quiz.topic, level=quiz.level, questions=quiz.questions[: inp.num_questions])
    return quiz


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
LETTERS = "ABCD"


def normalise_mcq_answer(answer: str, options: list[str]) -> Optional[str]:
    """Map 'b', 'B)', 'option b', '(b)', or the option text itself to 'B'. None if unclear."""
    a = (answer or "").strip()
    if not a:
        return None
    m = re.fullmatch(r"\(?\s*(?:option\s*)?([a-dA-D])\s*[\)\.:]?\s*(?:[-–:].*)?", a)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\(?([a-dA-D])[\)\.:]\s+\S", a)
    if m:
        return m.group(1).upper()
    low = a.lower().rstrip(".")
    for i, opt in enumerate(options):
        if low == opt.lower().rstrip("."):
            return LETTERS[i]
    return None


def evaluate_answers(quiz: Quiz, answers: list[str], ctx: ToolContext) -> Evaluation:
    if not answers:
        raise ToolFailure("evaluate_answers", "no answers provided")
    answers = list(answers) + [""] * (len(quiz.questions) - len(answers))
    results: list[QuestionResult] = []
    for q, ans in zip(quiz.questions, answers):
        if q.type == "mcq":
            letter = normalise_mcq_answer(ans, q.options or [])
            correct = letter == q.correct_option
            if not ans.strip():
                fb = f"Not answered. Correct: {q.correct_option}) {q.options[LETTERS.index(q.correct_option)]}."
            elif letter is None:
                fb = f"Couldn't read '{ans}' as A–D. Correct: {q.correct_option}. {q.reference_answer}"
            elif correct:
                fb = f"Correct. {q.reference_answer}"
            else:
                fb = f"You chose {letter}; correct is {q.correct_option}. {q.reference_answer}"
            results.append(QuestionResult(id=q.id, correct=correct, score=1.0 if correct else 0.0, feedback=fb, concept=q.concept))
        else:
            if not ans.strip():
                results.append(QuestionResult(id=q.id, correct=False, score=0.0, concept=q.concept,
                                              feedback=f"Not answered. A good answer: {q.reference_answer}"))
                continue
            payload = {"question": q.question, "reference_answer": q.reference_answer, "answer": ans}
            try:
                grade = ctx.call("grade_short_answer", prompts.GRADER_SYSTEM, prompts.tool_user_prompt(payload),
                                 ShortAnswerGrade, payload)
                results.append(QuestionResult(id=q.id, correct=grade.score >= 0.6, score=grade.score,
                                              feedback=grade.feedback, concept=q.concept))
            except LLMError:
                # Graceful partial failure: don't lose the MCQ grades because one LLM call failed.
                results.append(QuestionResult(id=q.id, correct=False, score=0.0, concept=q.concept,
                                              feedback=f"(Could not be auto-graded right now.) Compare with: {q.reference_answer}"))
                ctx.attempt_log.append(f"grade_short_answer q{q.id}: gave up, marked ungraded")

    total = round(sum(r.score for r in results) / len(results), 2)
    weak = list(dict.fromkeys(r.concept for r in results if r.score < 0.6))
    if total >= 0.8:
        summary = "Strong result — you're ready to move on."
    elif total >= 0.6:
        summary = "Solid, with a couple of gaps worth reviewing."
    else:
        summary = "Some core ideas need another pass — let's fix those before moving on."
    return Evaluation(topic=quiz.topic, results=results, total_score=total, weak_concepts=weak, summary_feedback=summary)
