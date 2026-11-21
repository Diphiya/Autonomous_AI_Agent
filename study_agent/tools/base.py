"""Shared tool plumbing: execution context, typed failure, timeout."""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from ..config import Settings
from ..llm import FaultInjector, LLMClient, LLMError, structured_call

R = TypeVar("R")

# One shared pool; a timed-out call keeps running in the background but its result is discarded.
_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


class ToolFailure(Exception):
    def __init__(self, tool: str, reason: str, attempts: int = 1):
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason
        self.attempts = attempts


@dataclass
class ToolContext:
    llm: LLMClient
    settings: Settings
    faults: Optional[FaultInjector] = None
    session_id: Optional[str] = None
    attempt_log: list[str] = field(default_factory=list)

    def call(self, task: str, system: str, user: str, schema, context: dict[str, Any]):
        """Structured LLM call with this context's retry policy and fault injection."""
        return structured_call(
            self.llm, task,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema, context,
            max_retries=self.settings.llm_max_retries,
            faults=self.faults, session_id=self.session_id, attempt_log=self.attempt_log,
            backoff_s=0.05 if self.llm.name == "mock" else 0.5,
        )


def run_tool(name: str, fn: Callable[[], R], timeout_s: float) -> R:
    """Run a tool with a wall-clock timeout, converting every failure into ToolFailure."""
    future = _POOL.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except cf.TimeoutError:
        raise ToolFailure(name, f"timed out after {timeout_s:.0f}s")
    except LLMError as e:
        raise ToolFailure(name, str(e), attempts=getattr(e, "attempts", 1))
    except ToolFailure:
        raise
    except Exception as e:  # unexpected bug inside a tool: still don't crash the agent
        raise ToolFailure(name, f"unexpected error: {type(e).__name__}: {e}")
