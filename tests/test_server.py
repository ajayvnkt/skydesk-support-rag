"""FastAPI site server with a fake pipeline: no index build, no API."""
import pytest
from fastapi.testclient import TestClient

import server
from rag.llm import BudgetExceeded

CHUNK = {"chunk_id": "policy-pets#0", "doc_id": "policy-pets", "title": "Pets", "section": "Cabin",
         "updated": "2026-04-22", "text": "Cats up to 8 kg with carrier fly in cabin for $95.",
         "rerank": 3, "dense_rank": 1, "bm25_rank": 2}


def fake_ask(query, mode="hybrid_rerank", log_path=None, **kw):
    base = {"query": query, "timings": {"total": 5}, "cost_usd": 0.0001, "chunks": [CHUNK]}
    if "funeral" in query:
        return {**base, "route": "escalate", "answer": "I'm so sorry.", "citations": [],
                "confidence": None, "reason": "bereavement",
                "ticket": {"reason": "bereavement", "priority": "high", "customer_message": query}}
    if "wifi" in query:
        return {**base, "route": "escalate", "answer": "Not covered.", "citations": [],
                "confidence": 0.1, "reason": "not_in_kb", "ticket": {"priority": "normal"}}
    if "budget" in query:
        raise BudgetExceeded("cap")
    if "boom" in query:
        raise RuntimeError("secret traceback")
    return {**base, "route": "answer", "answer": "Yes, up to 8 kg for $95 [policy-pets].",
            "citations": ["policy-pets"], "confidence": 0.9, "reason": None, "ticket": None}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server.pipeline, "ask", fake_ask)
    monkeypatch.setattr(server, "UI_LOGS", tmp_path)
    return TestClient(server.app)


def test_answer_shape(client):
    r = client.post("/api/ask", json={"message": "cat in cabin?", "session_id": "s1"}).json()
    assert r["route"] == "answer" and r["handoff"] is None and r["latency_ms"] == 5
    c = r["citations"][0]
    assert c["doc_id"] == "policy-pets" and c["title"] == "Pets" and c["rerank"] == 3 and "$95" in c["snippet"]


def test_escalation_has_handoff(client):
    r = client.post("/api/ask", json={"message": "dad died, funeral tomorrow"}).json()
    assert r["route"] == "escalate"
    assert r["handoff"]["team"] == "Care Team" and r["handoff"]["priority"] == "high"


def test_refusal_has_no_handoff(client):
    r = client.post("/api/ask", json={"message": "wifi?"}).json()
    assert r["route"] == "escalate" and r["handoff"] is None


def test_errors_are_friendly(client):
    assert client.post("/api/ask", json={"message": "over budget"}).status_code == 503
    r = client.post("/api/ask", json={"message": "boom"})
    assert r.status_code == 500 and "secret" not in r.text
    assert client.post("/api/ask", json={"message": "   "}).status_code == 400


def test_session_id_is_sanitized():
    assert server.session_log("../../etc/x").name == "session-web-etcx.arl.jsonl"


def test_health_and_site(client):
    assert client.get("/api/health").json() == {"ok": True}
    assert "SkyDesk Airways" in client.get("/").text
