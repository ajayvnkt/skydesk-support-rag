import json
from types import SimpleNamespace

import pytest

from conftest import FakeLLM
from rag import arl
from rag.llm import LLM, BudgetExceeded, cost_usd
from rag.pipeline import ask


class FakeClient:
    """Mimics the openai client response shapes."""
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)
        self.embed_calls = 0

    def _chat(self, **kw):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    def _embed(self, model, input):
        self.embed_calls += 1
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10 * len(input)),
                               data=[SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])


def test_cost_math():
    assert cost_usd("gpt-4.1-nano", 1_000_000, 1_000_000) == pytest.approx(0.50)
    assert cost_usd("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)


def test_ledger_records_and_guard_raises(tmp_path):
    llm = LLM(tmp_path, budget=0.0002, client=FakeClient())
    text, meta = llm.chat([{"role": "user", "content": "hi"}], purpose="t", max_tokens=10)
    assert text == "ok" and meta["tokens_in"] == 1000 and meta["cost_usd"] == pytest.approx(0.00014)
    rows = json.loads((tmp_path / "cost_ledger.json").read_text())
    assert rows[0]["purpose"] == "t" and rows[0]["model"] == "gpt-4.1-nano"
    assert {"ts", "tokens_in", "tokens_out", "usd", "latency_ms"} <= set(rows[0])
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "x" * 4000}], purpose="t2", max_tokens=200)
    assert len(json.loads((tmp_path / "cost_ledger.json").read_text())) == 1  # refused call not billed


def test_guard_uses_persisted_spend(tmp_path):
    (tmp_path / "cost_ledger.json").write_text(json.dumps([{"purpose": "old", "usd": 1.0}]))
    with pytest.raises(BudgetExceeded):
        LLM(tmp_path, client=FakeClient()).embed(["hello world"])


def test_embedding_cache_is_free_on_reuse(tmp_path):
    fc = FakeClient()
    v1 = LLM(tmp_path, client=fc).embed(["a", "bb", "a"])
    v2 = LLM(tmp_path, client=fc).embed(["bb", "a"])  # fresh instance -> disk cache
    assert fc.embed_calls == 1 and v1[1] == v2[0]
    assert len(json.loads((tmp_path / "cost_ledger.json").read_text())) == 1


def test_real_client_blocked_in_tests(tmp_path):
    with pytest.raises(AssertionError):
        LLM(tmp_path).chat([{"role": "user", "content": "hi"}], purpose="x")


def test_arl_answer_shape(index, tmp_path):
    log = tmp_path / "c.arl.jsonl"
    r = ask("How many checked bags can I bring on Economy Saver?", index, FakeLLM(), log_path=log)
    ev = [json.loads(l) for l in log.read_text().splitlines()]
    assert [e["seq"] for e in ev] == list(range(len(ev)))
    assert all(e["run_id"] == r["run_id"] and e["agent"] == "skydesk-support-rag" for e in ev)
    types = [e["type"] for e in ev]
    assert types[0] == "run_start" and types[1] == "user_input" and types[-1] == "run_end"
    tools = [e["tool"] for e in ev if e["type"] == "tool_call"]
    assert tools == ["search_kb", "rerank_kb"]
    llm_ev = [e for e in ev if e["type"] == "llm_response"]
    assert len(llm_ev) == 2 and all(e["tokens_in"] and "cost_usd" in e for e in llm_ev)
    assert "cost_usd" not in ev[-1] and ev[-1]["outcome"]["total_cost_usd"] > 0  # no double count
    fa = next(e for e in ev if e["type"] == "final_answer")
    assert fa["claimed_success"] is True and "[faq-baggage]" in fa["content"]
    assert ev[-1]["outcome"]["resolved"] is True
    assert ev[0]["model"] == "gpt-4.1-nano" and ev[0]["task_type"] == "support_qa"


def test_arl_escalation_shape(index, tmp_path):
    log = tmp_path / "c.arl.jsonl"
    ask("I will sue SkyDesk", index, FakeLLM(), log_path=log)
    ev = [json.loads(l) for l in log.read_text().splitlines()]
    types = [e["type"] for e in ev]
    assert "escalation" in types
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["tool"] == "escalate_to_human" and call["args"]["reason"] == "legal"
    assert next(e for e in ev if e["type"] == "final_answer")["claimed_success"] is False
    assert ev[-1]["outcome"]["escalated"] is True


def test_check_model_is_priced_and_ledgered(tmp_path):
    from rag.llm import CHECK_MODEL
    llm = LLM(tmp_path, client=FakeClient())
    _, meta = llm.chat([{"role": "user", "content": "hi"}], purpose="price_check", model=CHECK_MODEL)
    assert meta["model"] == CHECK_MODEL and meta["cost_usd"] == pytest.approx(cost_usd(CHECK_MODEL, 1000, 100))
    assert json.loads((tmp_path / "cost_ledger.json").read_text())[0]["model"] == CHECK_MODEL
