"""Real LLM client built on LangChain's provider-agnostic ``init_chat_model``.

Works with any provider LangChain supports, e.g. ``openai:gpt-4o-mini``,
``anthropic:claude-haiku-5-5``, ``groq:llama-3.1-8b-instant``, ``ollama:llama3.1``
(install the matching ``langchain-<provider>`` package and set its API key).
"""
from __future__ import annotations

from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .base import InvalidLLMOutput, LLMPermanentError, LLMTransientError

T = TypeVar("T", bound=BaseModel)

_TRANSIENT_HINTS = ("timeout", "timed out", "rate limit", "ratelimit", "429", "503", "502", "connection", "overloaded")


class LangChainLLM:
    def __init__(self, model: str, temperature: Optional[float] = None, timeout_s: float = 45):
        from langchain.chat_models import init_chat_model

        self.name = model
        provider, _, model_name = model.partition(":")
        if not model_name:  # allow "gpt-4o-mini" without provider prefix
            provider, model_name = None, model

        kwargs: dict[str, Any] = {"timeout": timeout_s, "max_retries": 0}  # retries are handled by structured_call
        if temperature is not None:  # only send it if configured; some newer models reject it
            kwargs["temperature"] = temperature

        self._chat = init_chat_model(model_name, model_provider=provider, **kwargs)
        self._structured: dict[type, Any] = {}

    def _runnable(self, schema: Type[T]):
        if schema not in self._structured:
            self._structured[schema] = self._chat.with_structured_output(schema, include_raw=True)
        return self._structured[schema]

    def generate(self, task: str, messages: list[dict[str, str]], schema: Type[T], context: dict[str, Any]) -> T:
        try:
            out = self._runnable(schema).invoke([(m["role"], m["content"]) for m in messages])
        except ValidationError as ve:
            raise InvalidLLMOutput(str(ve)) from ve
        except Exception as e:  # provider SDK errors differ; classify by message
            text = f"{type(e).__name__}: {e}".lower()
            if any(h in text for h in _TRANSIENT_HINTS):
                raise LLMTransientError(str(e)) from e
            raise LLMPermanentError(f"{type(e).__name__}: {e}") from e

        parsed, error = out.get("parsed"), out.get("parsing_error")
        if error is not None or parsed is None:
            raw = out.get("raw")
            raise InvalidLLMOutput(f"could not parse structured output: {error}", raw=getattr(raw, "content", raw))
        return parsed
