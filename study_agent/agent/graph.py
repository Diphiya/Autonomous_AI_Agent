"""Builds the LangGraph StateGraph with conditional routing."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..schemas import Action
from ..state import AgentState
from .nodes import Deps, Nodes

TOOL_NODES = [a.value for a in (Action.create_study_plan, Action.explain_topic, Action.generate_quiz, Action.evaluate_answers)]


def build_graph(deps: Deps, checkpointer=None):
    n = Nodes(deps)
    g = StateGraph(AgentState)

    g.add_node("ingest", n.ingest)
    g.add_node("router", n.router)
    g.add_node(Action.create_study_plan.value, n.create_study_plan)
    g.add_node(Action.explain_topic.value, n.explain_topic)
    g.add_node(Action.generate_quiz.value, n.generate_quiz)
    g.add_node(Action.evaluate_answers.value, n.evaluate_answers)
    g.add_node("recover", n.recover)
    g.add_node(Action.ask_clarification.value, n.ask_clarification)
    g.add_node(Action.respond.value, n.respond)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", n.after_ingest, {"router": "router", "evaluate_answers": Action.evaluate_answers.value})
    g.add_conditional_edges("router", n.after_router, {a.value: a.value for a in Action})
    for tool in TOOL_NODES:
        g.add_conditional_edges(tool, n.after_tool, {"router": "router", "recover": "recover"})
    g.add_conditional_edges("recover", n.after_recover, {"router": "router", "generate_quiz": Action.generate_quiz.value})
    g.add_edge(Action.ask_clarification.value, END)
    g.add_edge(Action.respond.value, END)

    return g.compile(checkpointer=checkpointer)
