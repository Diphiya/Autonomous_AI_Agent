"""Central configuration, loaded from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM -------------------------------------------------------------
    # "mock" = deterministic offline LLM (no API key needed, used by tests/demos).
    # Anything else is passed to LangChain's init_chat_model, e.g.
    #   "openai:gpt-4o-mini", "anthropic:claude-haiku-4-5", "ollama:llama3.1"
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "mock"))
    # Leave LLM_TEMPERATURE empty to not send it (some newer models reject the parameter).
    llm_temperature: Optional[float] = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE")) if os.getenv("LLM_TEMPERATURE", "").strip() else None
    )
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "45")))

    # --- Execution limits ------------------------------------------------
    max_steps_per_turn: int = field(default_factory=lambda: int(os.getenv("MAX_STEPS_PER_TURN", "6")))
    max_tool_calls_per_turn: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "4")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))
    tool_timeout_s: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_S", "60")))
    max_consecutive_clarifications: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONSECUTIVE_CLARIFICATIONS", "2"))
    )
    recursion_limit: int = field(default_factory=lambda: int(os.getenv("GRAPH_RECURSION_LIMIT", "40")))
    history_window: int = field(default_factory=lambda: int(os.getenv("HISTORY_WINDOW", "10")))

    # --- Persistence -----------------------------------------------------
    # Empty -> in-memory checkpointer. A path -> SQLite checkpointer (survives restarts).
    checkpoint_db: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DB", ""))

    # --- Demo / testing --------------------------------------------------
    allow_fault_injection: bool = field(default_factory=lambda: _bool("ALLOW_FAULT_INJECTION", True))


def get_settings() -> Settings:
    return Settings()
