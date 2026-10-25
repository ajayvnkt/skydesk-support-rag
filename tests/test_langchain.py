"""Track 2: LangChain primitives (Document, BaseRetriever, ChatPromptTemplate) without changing
retrieval results. Golden ranks were recorded from HybridIndex.search before the refactor."""
import pytest
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from conftest import FIXTURES, fake_embed
from rag.index import HybridIndex, HybridRetriever
from rag.ingest import chunk_corpus, chunk_documents, from_document, load_corpus, to_document

GOLDEN = {  # (mode, query) -> chunk ids in order, recorded before the LangChain refactor
    ("hybrid", "checked bag allowance economy saver"):
        ["faq-baggage#0", "faq-baggage#1", "policy-refunds#0", "policy-refunds#1", "policy-refunds#2",
         "policy-refunds#3", "ticket-4821#0", "manual-checkin#0", "manual-checkin#1", "policy-refunds-2024#0"],
    ("hybrid", "SD-ERR-112 at checkout"):
        ["ticket-4821#0", "faq-baggage#1", "policy-refunds#0", "faq-baggage#0", "policy-refunds#3",
         "policy-refunds#1", "policy-refunds#2", "manual-checkin#0", "manual-checkin#1", "policy-refunds-2024#0"],
    ("hybrid", "refund policy for cancelled flights"):
        ["policy-refunds#3", "policy-refunds#0", "policy-refunds#1", "policy-refunds#2", "faq-baggage#1",
         "faq-baggage#0", "ticket-4821#0", "manual-checkin#0", "manual-checkin#1", "policy-refunds-2024#0"],
    ("dense", "SD-ERR-112 at checkout"):
        ["ticket-4821#0", "faq-baggage#1", "policy-refunds-2024#0", "policy-refunds#0", "faq-baggage#0",
         "policy-refunds#3", "policy-refunds#1", "policy-refunds#2", "manual-checkin#0", "manual-checkin#1"],
    ("bm25", "How many checked bags can I bring on Economy Saver?"): ["faq-baggage#0"],
}


def test_chunks_are_documents_and_round_trip():
    dicts = chunk_corpus(load_corpus(FIXTURES))
    docs = chunk_documents(load_corpus(FIXTURES))
    assert all(isinstance(d, Document) for d in docs)
    assert [from_document(d) for d in docs] == dicts  # same text and metadata, same order
    assert docs[0].page_content == dicts[0]["text"] and docs[0].id == dicts[0]["chunk_id"]
    assert from_document(to_document(dicts[0])) == dicts[0]


def test_index_from_documents_equals_index_from_dicts():
    a = HybridIndex(chunk_documents(load_corpus(FIXTURES)), fake_embed)
    b = HybridIndex(chunk_corpus(load_corpus(FIXTURES)), fake_embed)
    q = "refund policy for cancelled flights"
    assert a.search(q) == b.search(q)


def test_retriever_is_a_langchain_base_retriever(index):
    r = HybridRetriever(index=index, mode="hybrid", k=3)
    assert isinstance(r, BaseRetriever)
    docs = r.invoke("checked bag allowance economy saver")
    assert len(docs) == 3 and all(isinstance(d, Document) for d in docs)
    assert {"fused", "dense_rank", "bm25_rank", "chunk_id", "doc_id"} <= set(docs[0].metadata)


@pytest.mark.parametrize("mode,query", list(GOLDEN))
def test_retriever_golden_ranks_unchanged(index, mode, query):
    got = [d.metadata["chunk_id"] for d in HybridRetriever(index=index, mode=mode, k=12).invoke(query)]
    assert got == GOLDEN[(mode, query)]
    assert got == [c["chunk_id"] for c in index.search(query, top_k=12, mode=mode)]


def test_prompts_are_chat_prompt_templates_rendered_for_the_ledgered_llm():
    from rag.prompts import GEN_PROMPT, JUDGE_PROMPT, RERANK_PROMPT, REPAIR_PROMPT, to_messages
    for p in (GEN_PROMPT, REPAIR_PROMPT, RERANK_PROMPT, JUDGE_PROMPT):
        assert isinstance(p, ChatPromptTemplate)
    m = to_messages(RERANK_PROMPT, question="bags {x}?", listing="(0) [a] b")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"] == "<question>bags {x}?</question>\n\nPassages:\n(0) [a] b"
    assert '{"scores": {"<n>": <0-3>, ...}, "ambiguous": <true|false>}' in m[0]["content"]
    r = to_messages(REPAIR_PROMPT, context="C", question="Q", draft="D", doc_ids="a, b", example_id="a")
    assert [x["role"] for x in r] == ["system", "user", "assistant", "user"] and r[2]["content"] == "D"
