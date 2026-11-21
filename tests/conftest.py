import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_agent.agent import StudyAssistant  # noqa: E402
from study_agent.config import Settings  # noqa: E402
from study_agent.llm.mock_llm import MockLLM  # noqa: E402


@pytest.fixture
def settings():
    return Settings(llm_model="mock", checkpoint_db="", allow_fault_injection=True)


@pytest.fixture
def bot(settings):
    return StudyAssistant(settings, llm=MockLLM())


def actions(result):
    """Executed actions (router/ingest decisions) in order for a TurnResult."""
    return [t["decision"] for t in result.trace if t["node"] in ("router", "ingest")]


def tools_ok(result):
    return [c["action"] for c in result.tool_calls if c["status"] == "ok"]
