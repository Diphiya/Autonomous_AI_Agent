"""Provider-agnostic structured LLM calls with validation, repair-retries and backoff.

Every LLM call in the project goes through :func:`structured_call`. It guarantees that
callers either get a *validated* Pydantic object or a typed exception — never raw text.

Retry policy
------------
* ``InvalidLLMOutput`` (unparseable JSON / schema or semantic validation failure):
  retried up to ``max_retries`` times. The validation error is fed back to the model
  ("repair prompt") so the retry is informed rather than a blind re-roll.
* ``LLMTransientError`` (timeout, rate limit, connection error): retried with
  exponential backoff.
* Anything else is a bug/permanent failure and propagates immediately.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class for LLM failures."""


class InvalidLLMOutput(LLMError):
    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


class LLMTransientError(LLMError):
    """Timeouts, rate limits, network errors: worth retrying with backoff."""


class LLMPermanentError(LLMError):
    """Auth/config/bad-request errors from the provider: retrying won't help."""


class LLMCallFailed(LLMError):
    """Raised after retries are exhausted."""

    def __init__(self, task: str, attempts: int, last_error: Exception):
        super().__init__(f"{task} failed after {attempts} attempt(s): {last_error}")
        self.task = task
        self.attempts = attempts
        self.last_error = last_error


# --------------------------------------------------------------------------- #
# Client protocol
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    name: str

    def generate(
        self,
        task: str,
        messages: list[dict[str, str]],
        schema: Type[T],
        context: dict[str, Any],
    ) -> T:
        """Return a validated instance of ``schema``.

        ``messages`` is the full prompt (used by real LLMs). ``context`` is the same
        information as structured data; real clients ignore it, the offline mock uses
        it so that it doesn't have to parse prompts. Implementations raise
        InvalidLLMOutput / LLMTransientError for retryable problems.
        """
        ...


# --------------------------------------------------------------------------- #
# Fault injection (for demos and tests; works with real and mock LLMs)
# --------------------------------------------------------------------------- #
class FaultInjector:
    """Queue of faults per (session_id, task). Each LLM *attempt* for that task pops one.

    Fault kinds: ``"invalid_output"`` (model returns malformed/invalid data),
    ``"timeout"`` (call times out), ``"error"`` (upstream 5xx), ``"ok"`` (no fault).
    """

    KINDS = {"invalid_output", "timeout", "error", "ok"}

    def __init__(self) -> None:
        self._faults: dict[tuple[str, str], deque[str]] = defaultdict(deque)
        self._lock = threading.Lock()

    def add(self, session_id: str, task: str, faults: list[str]) -> None:
        bad = set(faults) - self.KINDS
        if bad:
            raise ValueError(f"unknown fault kinds: {bad}")
        with self._lock:
            self._faults[(session_id, task)].extend(faults)

    def pop(self, session_id: Optional[str], task: str) -> Optional[str]:
        if session_id is None:
            return None
        with self._lock:
            q = self._faults.get((session_id, task))
            if q:
                kind = q.popleft()
                return None if kind == "ok" else kind
        return None

    def pending(self, session_id: str) -> dict[str, list[str]]:
        with self._lock:
            return {t: list(q) for (s, t), q in self._faults.items() if s == session_id and q}

    def clear(self, session_id: str) -> None:
        with self._lock:
            for key in [k for k in self._faults if k[0] == session_id]:
                del self._faults[key]


# --------------------------------------------------------------------------- #
# The one entry point
# --------------------------------------------------------------------------- #
REPAIR_TEMPLATE = (
    "Your previous response could not be used because it failed validation:\n"
    "{error}\n\nReturn a corrected response that strictly matches the required schema. "
    "Do not add commentary."
)


def structured_call(
    llm: LLMClient,
    task: str,
    messages: list[dict[str, str]],
    schema: Type[T],
    context: Optional[dict[str, Any]] = None,
    *,
    max_retries: int = 2,
    backoff_s: float = 0.5,
    faults: Optional[FaultInjector] = None,
    session_id: Optional[str] = None,
    attempt_log: Optional[list[str]] = None,
) -> T:
    context = dict(context or {})
    msgs = list(messages)
    last_error: Exception | None = None
    attempts = 0

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            fault = faults.pop(session_id, task) if faults else None
            if fault == "invalid_output":
                # Simulate what a real model sometimes does: wrong shape.
                try:
                    schema.model_validate({"unexpected": "free text instead of the schema"})
                except ValidationError as ve:
                    raise InvalidLLMOutput(_short(ve), raw="<injected invalid output>") from ve
            elif fault == "timeout":
                raise LLMTransientError("injected timeout")
            elif fault == "error":
                raise LLMTransientError("injected upstream error (HTTP 503)")

            result = llm.generate(task, msgs, schema, context)
            if not isinstance(result, schema):  # defensive: client bug
                result = schema.model_validate(result)
            if attempt_log is not None:
                attempt_log.append(f"{task}: attempt {attempts} ok")
            return result

        except ValidationError as ve:  # client returned a dict that fails validation
            last_error = InvalidLLMOutput(_short(ve))
        except InvalidLLMOutput as e:
            last_error = e
        except LLMTransientError as e:
            last_error = e

        if attempt_log is not None:
            attempt_log.append(f"{task}: attempt {attempts} failed ({type(last_error).__name__}: {last_error})")
        log.warning("LLM task %s attempt %d failed: %s", task, attempts, last_error)

        if attempt < max_retries:
            if isinstance(last_error, InvalidLLMOutput):
                msgs = messages + [
                    {"role": "user", "content": REPAIR_TEMPLATE.format(error=str(last_error)[:800])}
                ]
                context["repair_attempt"] = attempt + 1
            else:
                time.sleep(backoff_s * (2**attempt))

    raise LLMCallFailed(task, attempts, last_error)  # type: ignore[arg-type]


def _short(ve: ValidationError) -> str:
    parts = []
    for err in ve.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {err.get('msg')}")
    return "; ".join(parts)
