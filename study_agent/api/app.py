"""FastAPI app: JSON API + a single-page chat UI at '/'.

Run:  uvicorn study_agent.api.app:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..agent import StudyAssistant
from ..config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="AI Study Assistant",
    version="1.0.0",
    description="Agentic study assistant: plans, explains, quizzes and grades, with an inspectable decision trace.",
)
_assistant: Optional[StudyAssistant] = None


def assistant() -> StudyAssistant:
    global _assistant
    if _assistant is None:
        _assistant = StudyAssistant(get_settings())
    return _assistant


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, examples=["Make me a 4-week ML study plan, 6 h/week"])
    session_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_\-]{1,64}$",
                                      description="Omit to start a new session")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    trace: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    latency_ms: int
    state: dict[str, Any]


class FaultRequest(BaseModel):
    task: Literal["route", "create_study_plan", "explain_topic", "generate_quiz", "grade_short_answer"]
    faults: list[Literal["invalid_output", "timeout", "error", "ok"]] = Field(..., min_length=1, max_length=10)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    a = assistant()
    return {"status": "ok", "llm": a.llm.name, "fault_injection": a.settings.allow_fault_injection}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return assistant().chat(req.message, req.session_id).to_dict()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # never leak a stack trace to the client
        logging.exception("chat failed")
        raise HTTPException(status_code=500, detail=f"internal error: {type(e).__name__}")


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    return assistant().public_state(session_id)


@app.post("/api/sessions/{session_id}/faults")
def inject_faults(session_id: str, req: FaultRequest):
    """Queue faults for the next LLM attempts of a task in this session (demo of failure handling)."""
    try:
        return {"pending": assistant().inject_faults(session_id, req.task, req.faults)}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/api/graph", response_class=PlainTextResponse)
def graph_mermaid():
    return assistant().mermaid()
