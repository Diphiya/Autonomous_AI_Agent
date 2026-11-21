"""Deterministic policy guards, answer parsing and the evaluator."""
import pytest

from study_agent.agent.policy import apply_policy, parse_answers
from study_agent.config import Settings
from study_agent.llm import LLMCallFailed
from study_agent.schemas import Action, Quiz, RouterDecision, StudentProfile
from study_agent.tools import ToolContext, evaluate_answers, normalise_mcq_answer

S = Settings(llm_model="mock", max_steps_per_turn=6, max_tool_calls_per_turn=4, max_consecutive_clarifications=2)
PLAN = {"weeks": [{"topics": ["linear regression", "gradient descent"]}, {"topics": ["overfitting"]}]}
QUIZ = {"topic": "t", "level": "beginner", "questions": [
    {"id": 1, "type": "mcq", "question": "q1", "options": ["a", "b", "c", "d"], "correct_option": "B", "reference_answer": "r", "concept": "c1"},
    {"id": 2, "type": "short", "question": "q2", "options": None, "correct_option": None, "reference_answer": "penalty on weights", "concept": "c2"},
]}


def dec(action, **kw):
    if action == Action.ask_clarification:
        kw.setdefault("clarification_question", "?")
    return RouterDecision(reasoning="r", action=action, **kw)


def run(d, state=None, profile=None):
    return apply_policy(d, {"steps": 1, "tool_calls": [], **(state or {})}, profile or StudentProfile(), S)


class TestPolicy:
    def test_step_limit_forces_respond(self):
        r = run(dec(Action.explain_topic, topic="x"), {"steps": 7})
        assert r.action == Action.respond and "step limit" in r.override_reason

    def test_tool_call_limit(self):
        calls = [{"action": "explain_topic", "args": {"topic": f"t{i}"}, "status": "ok"} for i in range(4)]
        assert run(dec(Action.explain_topic, topic="new"), {"tool_calls": calls}).action == Action.respond

    def test_duplicate_successful_call_blocked(self):
        calls = [{"action": "explain_topic", "args": {"topic": "Recursion"}, "status": "ok"}]
        r = run(dec(Action.explain_topic, topic="recursion"), {"tool_calls": calls})
        assert r.action == Action.respond and "duplicate" in r.override_reason

    def test_same_tool_different_topic_allowed(self):
        calls = [{"action": "explain_topic", "args": {"topic": "recursion"}, "status": "ok"}]
        assert run(dec(Action.explain_topic, topic="binary search"), {"tool_calls": calls}).action == Action.explain_topic

    def test_failed_tool_not_retried_in_same_turn(self):
        calls = [{"action": "generate_quiz", "args": {"topic": "x"}, "status": "failed"}]
        assert run(dec(Action.generate_quiz, topic="y"), {"tool_calls": calls}).action == Action.respond

    def test_evaluate_without_active_quiz_is_overridden(self):
        r = run(dec(Action.evaluate_answers, student_answers=["A"]))
        assert r.action == Action.respond and "no active quiz" in r.override_reason

    def test_evaluate_without_readable_answers_asks(self):
        assert run(dec(Action.evaluate_answers), {"active_quiz": QUIZ}).action == Action.ask_clarification

    def test_week_reference_resolved_from_plan(self):
        r = run(dec(Action.generate_quiz, topic="week 1"), {"study_plan": PLAN})
        assert r.args["topic"] == "linear regression, gradient descent" and r.args["num_questions"] == 4

    def test_week_out_of_range_asks(self):
        r = run(dec(Action.generate_quiz, topic="__week_9"), {"study_plan": PLAN})
        assert r.action == Action.ask_clarification and "only has 2 weeks" in r.clarification_question

    def test_missing_topic_falls_back_to_last_topic_then_clarifies(self):
        assert run(dec(Action.explain_topic), {"last_topic": "recursion"}).args["topic"] == "recursion"
        assert run(dec(Action.explain_topic)).action == Action.ask_clarification

    def test_plan_missing_fields_clarifies(self):
        r = run(dec(Action.create_study_plan), profile=StudentProfile(subject="ml"))
        assert r.action == Action.ask_clarification and r.pending_action == "create_study_plan"

    def test_plan_uses_defaults_when_clarification_budget_spent(self):
        r = run(dec(Action.create_study_plan), {"consecutive_clarifications": 2},
                StudentProfile(subject="ml", duration_weeks=3))
        assert r.action == Action.create_study_plan and r.args["hours_per_week"] == 5.0
        assert r.args["assumptions"] == ["hours_per_week"]

    def test_plan_without_subject_never_defaults(self):
        r = run(dec(Action.create_study_plan), {"consecutive_clarifications": 5}, StudentProfile())
        assert r.action == Action.ask_clarification

    def test_clarification_budget_converts_to_respond(self):
        r = run(dec(Action.ask_clarification), {"consecutive_clarifications": 2})
        assert r.action == Action.respond


class TestAnswerParsing:
    @pytest.mark.parametrize("text,expected", [
        ("1. B 2. A 3. C", ["B", "A", "C"]),
        ("1) b\n2) a\n3) regularisation adds a penalty", ["b", "a", "regularisation adds a penalty"]),
        ("B, A, C", ["B", "A", "C"]),
        ("1. B 3. C", ["B", "", "C"]),
    ])
    def test_parses(self, text, expected):
        assert parse_answers(text, 3) == expected

    @pytest.mark.parametrize("text", ["explain gradient descent", "B, A", "the value was 0.5 here"])
    def test_rejects_non_answers(self, text):
        assert parse_answers(text, 3) is None


class TestEvaluator:
    @pytest.mark.parametrize("ans,expected", [("b", "B"), ("B)", "B"), ("option b", "B"), ("(b)", "B"), ("b) because", "B"), ("bee", None), ("Banana", "B")])
    def test_normalise_mcq(self, ans, expected):
        assert normalise_mcq_answer(ans, ["Apple", "Banana", "Cherry", "Date"]) == expected

    def test_mcq_graded_deterministically_and_short_answer_failure_is_partial(self):
        class FailingLLM:
            name = "mock"

            def generate(self, *a, **k):
                raise AssertionError("must not be called for MCQ")

        ctx = ToolContext(FailingLLM(), S)
        # Short answer grading call explodes -> but we should still get the MCQ result.
        ctx.call = lambda *a, **k: (_ for _ in ()).throw(LLMCallFailed("grade_short_answer", 3, Exception("down")))
        ev = evaluate_answers(Quiz(**QUIZ), ["B", "some answer"], ctx)
        assert ev.results[0].correct and ev.results[0].score == 1.0
        assert "could not be auto-graded" in ev.results[1].feedback.lower()
        assert ev.total_score == 0.5


def test_regrading_in_same_turn_is_a_duplicate_not_no_quiz():
    """Real-model bug: after grading, the LLM proposed grading again and the student was told 'no open quiz'."""
    calls = [{"action": "evaluate_answers", "args": {}, "status": "ok"}]
    r = run(dec(Action.evaluate_answers, student_answers=["A"]), {"tool_calls": calls, "active_quiz": None})
    assert r.action == Action.respond and "already graded" in r.override_reason and r.message is None
