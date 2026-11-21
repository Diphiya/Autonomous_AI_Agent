"""Graph state.

Two kinds of state live here:

* **Session state** (persisted across turns by the LangGraph checkpointer, keyed by
  session/thread id): messages, profile, study plan, active quiz, quiz history,
  weak concepts, last topic, clarification counter.
* **Turn state** (reset by the `ingest` node at the start of each user message):
  step counter, tool-call log, pending decision, rendered output blocks, decision trace.

Values are plain JSON-friendly dicts (Pydantic models are dumped) so the state
serialises cleanly to any checkpointer backend and is easy to inspect over the API.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ToolCallRecord(TypedDict, total=False):
    action: str
    args: dict[str, Any]
    status: str  # "ok" | "failed" | "blocked"
    attempts: int
    error: Optional[str]
    summary: Optional[str]


class TraceEntry(TypedDict, total=False):
    step: int
    node: str
    decision: str
    reasoning: str
    source: str  # "llm" | "deterministic" | "policy-override" | "recovery"
    detail: Optional[str]


class AgentState(TypedDict, total=False):
    # ---- session (persisted) ----
    messages: Annotated[list[AnyMessage], add_messages]
    profile: dict[str, Any]
    study_plan: Optional[dict[str, Any]]
    active_quiz: Optional[dict[str, Any]]
    quiz_history: list[dict[str, Any]]
    weak_concepts: list[str]
    last_topic: Optional[str]
    consecutive_clarifications: int
    pending_action: Optional[str]  # action waiting on a clarification answer

    # ---- turn scratch (reset every user message) ----
    user_input: str
    steps: int
    tool_calls: list[ToolCallRecord]
    decision: Optional[dict[str, Any]]  # validated RouterDecision (+ resolved args)
    parsed_answers: Optional[list[str]]
    outputs: list[dict[str, Any]]  # render blocks: {"kind": ..., "data": ...}
    notes: list[str]  # user-visible notes (assumptions, degraded results, failures)
    trace: list[TraceEntry]
    final_reply: Optional[str]
    pending_retry: Optional[dict[str, Any]]  # set by recovery node for a degraded retry
