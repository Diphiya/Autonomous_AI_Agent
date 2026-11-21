"""End-to-end multi-turn flows through the compiled LangGraph (offline mock LLM)."""
from conftest import actions, tools_ok

from study_agent.agent import StudyAssistant
from study_agent.config import Settings
from study_agent.llm.mock_llm import MockLLM
from study_agent.schemas import Action, RouterDecision

PLAN_MSG = "Make me a study plan for machine learning, 4 weeks, 6 hours a week, beginner"


def test_plan_then_quiz_then_grade_then_adapt(bot):
    r1 = bot.chat(PLAN_MSG, "s")
    assert tools_ok(r1) == ["create_study_plan"] and actions(r1)[-1] == "respond"
    r2 = bot.chat("Quiz me on week 1", "s")
    assert tools_ok(r2) == ["generate_quiz"]
    assert r2.tool_calls[0]["args"]["topic"] == "linear regression, gradient descent"
    r3 = bot.chat("1. A 2. A 3. A 4. no idea", "s")
    assert r3.trace[0]["source"] == "deterministic"  # fast path: no routing LLM call
    assert tools_ok(r3) == ["evaluate_answers", "explain_topic"]  # adapted to the low score
    st = bot.public_state("s")
    assert st["active_quiz"] is None and st["weak_concepts"] and st["quiz_history"][0]["total_score"] < 0.6


def test_answer_key_never_exposed(bot):
    bot.chat("Explain recursion", "k")
    r = bot.chat("quiz me on it", "k")
    quiz = r.state["active_quiz"]
    assert quiz and all("correct_option" not in q and "reference_answer" not in q for q in quiz["questions"])
    assert "base case stops the recursion" not in r.reply  # reference answer text not rendered


def test_ambiguous_request_asks_and_remembers_context(bot):
    r1 = bot.chat("Help me study", "c")
    assert actions(r1) == ["ask_clarification"]
    r2 = bot.chat("I want a study plan for machine learning", "c")
    assert actions(r2) == ["ask_clarification"]  # missing weeks + hours
    r3 = bot.chat("4 weeks, 5 hours per week", "c")
    assert tools_ok(r3) == ["create_study_plan"]  # resumed the pending plan action
    assert bot.public_state("c")["pending_action"] is None


def test_compound_request_chains_tools_and_stops(bot):
    r = bot.chat("Explain gradient descent and then give me 3 practice questions", "m")
    assert tools_ok(r) == ["explain_topic", "generate_quiz"]
    assert r.tool_calls[1]["args"]["num_questions"] == 3
    assert actions(r)[-1] == "respond"


def test_invalid_output_is_repaired(bot):
    bot.inject_faults("f", "generate_quiz", ["invalid_output"])
    r = bot.chat("quiz me on overfitting", "f")
    assert tools_ok(r) == ["generate_quiz"]
    assert "attempt 1 failed" in (r.trace[1]["detail"] or "")


def test_persistent_failure_degrades_then_reports_and_session_survives(bot):
    bot.inject_faults("f2", "generate_quiz", ["invalid_output"] * 3)  # full quiz fails, degraded retry succeeds
    r = bot.chat("quiz me on overfitting", "f2")
    assert [c["status"] for c in r.tool_calls] == ["failed", "ok"]
    assert r.tool_calls[1]["args"]["degraded"] and "shorter multiple-choice" in r.reply
    bot.inject_faults("f2", "explain_topic", ["timeout"] * 3)
    r2 = bot.chat("explain recursion", "f2")
    assert r2.tool_calls[0]["status"] == "failed" and "couldn't generate the explanation" in r2.reply
    assert bot.public_state("f2")["active_quiz"] is not None  # earlier state intact
    r3 = bot.chat("explain recursion", "f2")
    assert tools_ok(r3) == ["explain_topic"]


def test_router_outage_gives_graceful_reply(bot):
    bot.inject_faults("r", "route", ["error"] * 3)
    r = bot.chat("explain recursion", "r")
    assert r.trace[0]["source"] == "recovery" and "trouble" in r.reply


def test_sessions_are_isolated(bot):
    bot.chat(PLAN_MSG, "a")
    assert bot.public_state("a")["study_plan"] and not bot.public_state("b")["study_plan"]


class LoopingLLM(MockLLM):
    """Always proposes a *new* tool call, so duplicate detection alone can't stop it."""
    n = 0

    def _task_route(self, ctx):
        LoopingLLM.n += 1
        return RouterDecision(reasoning="more!", action=Action.explain_topic, topic=f"topic {LoopingLLM.n}")


def test_execution_limits_stop_a_runaway_agent(settings):
    bot = StudyAssistant(settings, llm=LoopingLLM())
    r = bot.chat("explain everything", "loop")
    assert len(r.tool_calls) == settings.max_tool_calls_per_turn
    assert r.trace[-1]["source"] == "policy-override" and "limit" in r.trace[-1]["detail"]


def test_sqlite_checkpointer_persists_across_restarts(tmp_path):
    s = Settings(llm_model="mock", checkpoint_db=str(tmp_path / "cp.db"))
    StudyAssistant(s, llm=MockLLM()).chat(PLAN_MSG, "p")
    again = StudyAssistant(s, llm=MockLLM())  # new process, same DB
    assert again.public_state("p")["study_plan"]["duration_weeks"] == 4
    r = again.chat("quiz me on week 2", "p")
    assert tools_ok(r) == ["generate_quiz"]


class RegradingLLM(MockLLM):
    """Reproduces the real-model bug: proposes evaluate_answers again after grading."""

    def _task_route(self, ctx):
        if any(c["action"] == "evaluate_answers" for c in ctx.get("tool_calls_this_turn") or []):
            assert "HAVE BEEN GRADED" in ctx["turn_progress"]  # the router is told explicitly
            return RouterDecision(reasoning="grade", action=Action.evaluate_answers, student_answers=["A"])
        return super()._task_route(ctx)


def test_partial_answers_go_through_router_and_regrade_is_blocked(settings):
    bot = StudyAssistant(settings, llm=RegradingLLM())
    bot.chat("quiz me on overfitting", "g")
    r = bot.chat("1. A 2. A 3. A", "g")  # 4-question quiz, only 3 answered -> no fast path
    assert tools_ok(r) == ["evaluate_answers"]
    assert "no open quiz" not in r.reply.lower()


def test_temperature_not_sent_when_unset(monkeypatch):
    import langchain.chat_models as cm
    from study_agent.llm.langchain_llm import LangChainLLM
    seen = {}
    monkeypatch.setattr(cm, "init_chat_model", lambda model, **kw: seen.update(kw) or object())
    LangChainLLM("anthropic:claude-x")
    assert "temperature" not in seen
    LangChainLLM("openai:gpt-x", temperature=0.2)
    assert seen["temperature"] == 0.2
