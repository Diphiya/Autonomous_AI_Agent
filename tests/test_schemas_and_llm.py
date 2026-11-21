"""Structured-output validation and the retry/repair layer."""
import pytest
from pydantic import ValidationError

from study_agent.llm import FaultInjector, InvalidLLMOutput, LLMCallFailed, LLMTransientError, structured_call
from study_agent.schemas import (
    Action, ProfileUpdate, QuizQuestion, RouterDecision, StudentProfile, StudyPlan, StudyWeek,
)


def _week(n, hours=5):
    return StudyWeek(week=n, focus="x", topics=["t"], activities=["a"], estimated_hours=hours, checkpoint="c")


class TestSchemas:
    def test_plan_rejects_wrong_number_of_weeks(self):
        with pytest.raises(ValidationError, match="3 weeks but duration_weeks=4"):
            StudyPlan(goal="g", duration_weeks=4, hours_per_week=5, weeks=[_week(1), _week(2), _week(3)])

    def test_plan_rejects_week_over_budget(self):
        with pytest.raises(ValidationError, match="exceeds"):
            StudyPlan(goal="g", duration_weeks=1, hours_per_week=5, weeks=[_week(1, hours=12)])

    def test_plan_rejects_misnumbered_weeks(self):
        with pytest.raises(ValidationError, match="numbered"):
            StudyPlan(goal="g", duration_weeks=2, hours_per_week=5, weeks=[_week(1), _week(3)])

    def test_mcq_needs_four_distinct_options_and_answer(self):
        with pytest.raises(ValidationError):
            QuizQuestion(id=1, type="mcq", question="q", options=["a", "b", "c"], correct_option="A", reference_answer="r", concept="c")
        with pytest.raises(ValidationError):
            QuizQuestion(id=1, type="mcq", question="q", options=["a", "a", "b", "c"], correct_option="A", reference_answer="r", concept="c")
        with pytest.raises(ValidationError):
            QuizQuestion(id=1, type="mcq", question="q", options=["a", "b", "c", "d"], correct_option="E", reference_answer="r", concept="c")

    def test_clarification_requires_question(self):
        with pytest.raises(ValidationError):
            RouterDecision(reasoning="r", action=Action.ask_clarification)

    def test_profile_merge_unions_topics_and_overwrites_scalars(self):
        p = StudentProfile(subject="ml", topics=["a"], duration_weeks=2)
        p2 = p.merge(ProfileUpdate(topics=["A", "b"], duration_weeks=4, level="advanced"))
        assert p2.topics == ["a", "b"] and p2.duration_weeks == 4 and p2.level.value == "advanced" and p2.level_confirmed


class _ScriptedLLM:
    """Returns/raises items from a script, one per call."""
    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate(self, task, messages, schema, context):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return schema.model_validate(item)


GOOD = {"reasoning": "r", "action": "respond", "message": "hi"}


class TestStructuredCall:
    def test_repairs_after_invalid_output_and_feeds_error_back(self):
        llm = _ScriptedLLM([InvalidLLMOutput("bad json"), GOOD])
        out = structured_call(llm, "route", [{"role": "user", "content": "x"}], RouterDecision, backoff_s=0)
        assert out.action == Action.respond
        assert "failed validation" in llm.calls[1][-1]["content"]  # repair prompt was sent

    def test_semantically_invalid_dict_is_retried(self):
        llm = _ScriptedLLM([{"reasoning": "r", "action": "ask_clarification"}, GOOD])  # missing question
        assert structured_call(llm, "route", [], RouterDecision, backoff_s=0).action == Action.respond

    def test_transient_errors_retried_then_give_up(self):
        llm = _ScriptedLLM([LLMTransientError("timeout")] * 3)
        with pytest.raises(LLMCallFailed) as ei:
            structured_call(llm, "route", [], RouterDecision, max_retries=2, backoff_s=0)
        assert ei.value.attempts == 3

    def test_unexpected_errors_are_not_retried(self):
        llm = _ScriptedLLM([KeyError("bug"), GOOD])
        with pytest.raises(KeyError):
            structured_call(llm, "route", [], RouterDecision, backoff_s=0)

    def test_fault_injector_consumes_per_attempt_and_scopes_by_session(self):
        fi = FaultInjector()
        fi.add("s1", "route", ["invalid_output", "ok"])
        llm = _ScriptedLLM([GOOD])
        log = []
        structured_call(llm, "route", [], RouterDecision, faults=fi, session_id="s1", attempt_log=log, backoff_s=0)
        assert len(log) == 2 and "failed" in log[0]
        assert fi.pop("s2", "route") is None
        with pytest.raises(ValueError):
            fi.add("s1", "route", ["explode"])


def test_real_provider_permanent_error_degrades_gracefully(settings):
    """A real LangChain client with a bad key must not crash the agent."""
    from study_agent.agent import StudyAssistant
    from study_agent.llm.langchain_llm import LangChainLLM

    class Boom:
        def invoke(self, *_):
            raise RuntimeError("AuthenticationError: invalid api key")

    llm = LangChainLLM.__new__(LangChainLLM)
    llm.name, llm._structured = "fake:model", {}
    llm._runnable = lambda schema: Boom()
    r = StudyAssistant(settings, llm=llm).chat("explain recursion", "x")
    assert r.trace[0]["source"] == "recovery" and "trouble" in r.reply
