import pytest

from rag.index import RRF_K, rrf, tokenize


def test_tokenizer_keeps_codes_intact():
    toks = tokenize("Why do I get SD-ERR-112 on booking AB12CD?")
    assert "sd-err-112" in toks and "ab12cd" in toks
    assert "112" in toks  # parts indexed too
    assert "do" not in toks and "i" not in toks


def test_rrf_math():
    s = rrf([["a", "b", "c"], ["b", "a"]], k=60)
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["c"] == pytest.approx(1 / 63)
    assert RRF_K == 60
    # doc found by both retrievers beats one found by only one at the same rank
    s2 = rrf([["x", "y"], ["y"]])
    assert s2["y"] > s2["x"]


def test_bm25_finds_error_code(index):
    assert index.keyword("SD-ERR-112")[0].startswith("ticket-4821")


def test_hybrid_search_shape_and_topk(index):
    res = index.search("checked bag allowance economy saver", top_k=3)
    assert len(res) == 3
    assert res[0]["doc_id"] == "faq-baggage"
    assert {"fused", "dense_rank", "bm25_rank", "chunk_id"} <= set(res[0])


def test_superseded_downweighted(index):
    q = "refunds processed business days original payment method"
    full = {r["chunk_id"]: r["fused"] for r in index.search(q, top_k=50, superseded_weight=1.0)}
    half = {r["chunk_id"]: r["fused"] for r in index.search(q, top_k=50)}
    cid = "policy-refunds-2024#0"
    assert half[cid] == pytest.approx(full[cid] * 0.5, rel=1e-3)
