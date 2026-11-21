"""Facade used by the API, CLI, scenarios and tests."""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from ..config import Settings, get_settings
from ..llm import FaultInjector, LLMClient, build_llm
from .graph import build_graph
from .nodes import Deps

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    session_id: str
    reply: str
    trace: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    latency_ms: int
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_checkpointer(settings: Settings):
    if settings.checkpoint_db:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        return SqliteSaver(conn)
    return MemorySaver()


class StudyAssistant:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLMClient] = None, checkpointer=None):
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings.llm_model, self.settings.llm_temperature, self.settings.llm_timeout_s)
        self.faults = FaultInjector()
        self.graph = build_graph(Deps(self.llm, self.settings, self.faults), checkpointer or _make_checkpointer(self.settings))

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": self.settings.recursion_limit}

    def chat(self, message: str, session_id: Optional[str] = None) -> TurnResult:
        session_id = session_id or uuid.uuid4().hex[:12]
        if not message or not message.strip():
            raise ValueError("message must not be empty")
        if len(message) > 4000:
            raise ValueError("message too long (max 4000 characters)")
        t0 = time.perf_counter()
        try:
            out = self.graph.invoke({"user_input": message}, self._config(session_id))
            reply = out.get("final_reply") or "…"
        except GraphRecursionError:
            # Last-resort backstop; the step/tool limits should make this unreachable.
            log.exception("recursion limit hit")
            out = self.graph.get_state(self._config(session_id)).values
            reply = "I got stuck working on that, so I stopped. Could you rephrase or break it into smaller requests?"
        return TurnResult(
            session_id=session_id, reply=reply, trace=list(out.get("trace") or []),
            tool_calls=list(out.get("tool_calls") or []), latency_ms=int((time.perf_counter() - t0) * 1000),
            state=self.public_state(session_id),
        )

    def public_state(self, session_id: str) -> dict[str, Any]:
        """Session state safe to expose (quiz answer key stripped)."""
        values = self.graph.get_state(self._config(session_id)).values or {}
        quiz = values.get("active_quiz")
        if quiz:
            quiz = {**quiz, "questions": [{k: v for k, v in q.items() if k not in ("correct_option", "reference_answer")}
                                          for q in quiz["questions"]]}
        return {
            "profile": values.get("profile"),
            "study_plan": values.get("study_plan"),
            "active_quiz": quiz,
            "quiz_history": values.get("quiz_history") or [],
            "weak_concepts": values.get("weak_concepts") or [],
            "last_topic": values.get("last_topic"),
            "pending_action": values.get("pending_action"),
            "turns": sum(1 for m in values.get("messages") or [] if m.type == "human"),
        }

    def inject_faults(self, session_id: str, task: str, faults: list[str]) -> dict[str, list[str]]:
        if not self.settings.allow_fault_injection:
            raise PermissionError("fault injection disabled (ALLOW_FAULT_INJECTION=false)")
        self.faults.add(session_id, task, faults)
        return self.faults.pending(session_id)

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
