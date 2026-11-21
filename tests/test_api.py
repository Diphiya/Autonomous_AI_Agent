from fastapi.testclient import TestClient

import study_agent.api.app as api
from study_agent.agent import StudyAssistant
from study_agent.config import Settings
from study_agent.llm.mock_llm import MockLLM


def client(allow_faults=True):
    api._assistant = StudyAssistant(Settings(llm_model="mock", allow_fault_injection=allow_faults), llm=MockLLM())
    return TestClient(api.app)


def test_chat_roundtrip_and_session_state():
    c = client()
    r = c.post("/api/chat", json={"message": "Explain recursion"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] and "Recursion" in body["reply"] and body["trace"]
    s = c.get(f"/api/sessions/{body['session_id']}").json()
    assert s["last_topic"] == "recursion" and s["turns"] == 1


def test_validation_errors():
    c = client()
    assert c.post("/api/chat", json={"message": ""}).status_code == 422
    assert c.post("/api/chat", json={"message": "hi", "session_id": "bad id!"}).status_code == 422


def test_fault_endpoint():
    c = client()
    sid = c.post("/api/chat", json={"message": "hi"}).json()["session_id"]
    r = c.post(f"/api/sessions/{sid}/faults", json={"task": "generate_quiz", "faults": ["invalid_output"]})
    assert r.status_code == 200 and r.json()["pending"] == {"generate_quiz": ["invalid_output"]}
    assert c.post(f"/api/sessions/{sid}/faults", json={"task": "nope", "faults": ["timeout"]}).status_code == 422


def test_fault_endpoint_can_be_disabled():
    c = client(allow_faults=False)
    assert c.post("/api/sessions/x/faults", json={"task": "route", "faults": ["error"]}).status_code == 403


def test_index_and_graph():
    c = client()
    assert "AI Study Assistant" in c.get("/").text
    assert "router" in c.get("/api/graph").text
