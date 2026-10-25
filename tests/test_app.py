"""Streamlit AppTest with a fake pipeline: no index build, no API."""
import pytest
from streamlit.testing.v1 import AppTest

from rag import pipeline

CHUNK = {"chunk_id": "policy-pets#0", "doc_id": "policy-pets", "title": "Pets", "section": "Cabin",
         "doc_type": "policy", "updated": "2026-04-22", "superseded": False,
         "text": "Cats up to 8 kg with carrier fly in cabin for $95.", "fused": 0.03,
         "rerank": 3, "dense_rank": 1, "bm25_rank": 2}


def fake_ask(query, mode="hybrid_rerank", log_path=None, **kw):
    base = {"query": query, "mode": mode, "timings": {"total": 5}, "cost_usd": 0.0001,
            "run_id": "sdr-test", "candidates": [CHUNK], "chunks": [CHUNK], "scores": [3]}
    if "lawyer" in query:
        return {**base, "route": "escalate", "answer": "Handing you to Customer Relations.",
                "citations": [], "confidence": None, "reason": "fraud+legal", "chunks": [],
                "candidates": [], "ticket": {"reason": "fraud+legal", "priority": "high",
                                             "customer_message": query, "suggested_docs": []}}
    if "budget" in query:
        from rag.llm import BudgetExceeded
        raise BudgetExceeded("cap")
    return {**base, "route": "answer", "answer": "Yes, up to 8 kg for $95 [policy-pets].",
            "citations": ["policy-pets"], "confidence": 0.9, "reason": None, "ticket": None}


@pytest.fixture
def at(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline, "ask", lambda q, **kw: calls.append(kw) or fake_ask(q, **kw))
    a = AppTest.from_file("../app.py", default_timeout=20)
    a.calls = calls
    return a.run()


def test_renders_chat_page(at):
    assert not at.exception
    assert at.title[0].value == "SkyDesk Support Assistant"
    assert at.sidebar.radio[0].value == "Hybrid + rerank"


def test_answer_shows_route_and_citation(at):
    at.chat_input[0].set_value("can my cat fly in cabin?").run()
    assert not at.exception
    assert any("Answered" in m.value for m in at.markdown)
    assert any("[policy-pets]" in m.value for m in at.markdown)
    assert at.calls[0]["mode"] == "hybrid_rerank"


def test_escalation_card_and_mode(at):
    at.sidebar.radio[0].set_value("Dense only").run()
    at.chat_input[0].set_value("chargeback and my lawyer").run()
    assert not at.exception
    assert at.calls[0]["mode"] == "dense"
    assert any("Routed to:** Payments Integrity, Customer Relations" in m.value for m in at.markdown)


def test_whitespace_input_is_ignored(at):
    at.chat_input[0].set_value("   ").run()
    assert not at.exception and at.calls == []


def test_budget_error_is_friendly(at):
    at.chat_input[0].set_value("budget please").run()
    assert not at.exception
    assert any("1.00 API budget" in w.value for w in at.warning)


def _eval_page():
    import sys
    sys.path.insert(0, ".")
    import app
    app.eval_page()


def _how_page():
    import sys
    sys.path.insert(0, ".")
    import app
    app.how_page()


@pytest.mark.parametrize("fn,title", [(_eval_page, "Evaluation"), (_how_page, "How it works")])
def test_other_pages_render(monkeypatch, fn, title):
    import pathlib
    monkeypatch.chdir(pathlib.Path(__file__).resolve().parent.parent)
    at = AppTest.from_function(fn, default_timeout=20).run()
    assert not at.exception
    assert at.title[0].value == title
