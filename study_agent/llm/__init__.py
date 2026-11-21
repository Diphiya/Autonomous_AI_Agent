from .base import (
    FaultInjector,
    InvalidLLMOutput,
    LLMCallFailed,
    LLMClient,
    LLMError,
    LLMPermanentError,
    LLMTransientError,
    structured_call,
)


def build_llm(model: str, temperature: "float | None" = None, timeout_s: float = 45) -> "LLMClient":
    """Factory: 'mock' -> offline deterministic client, anything else -> LangChain provider."""
    if model.strip().lower() == "mock":
        from .mock_llm import MockLLM

        return MockLLM()
    from .langchain_llm import LangChainLLM

    return LangChainLLM(model, temperature=temperature, timeout_s=timeout_s)


__all__ = [
    "FaultInjector", "InvalidLLMOutput", "LLMCallFailed", "LLMClient", "LLMError", "LLMPermanentError",
    "LLMTransientError", "structured_call", "build_llm",
]
