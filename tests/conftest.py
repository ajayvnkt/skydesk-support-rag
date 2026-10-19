import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
FARES = Path(__file__).parent / "fixtures" / "fares"  # unseen fare/tier names (not SkyDesk's)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to build a real OpenAI client fails the test: tests must never call the API."""
    import openai

    def boom(*a, **k):
        raise AssertionError("real OpenAI client constructed in a test")
    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def fake_embed(texts):
    """Deterministic hashed bag-of-words vectors: similar words -> similar vectors."""
    out = []
    for t in texts:
        v = np.zeros(256)
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        out.append(v.tolist())
    return out


class FakeLLM:
    """Scripted stand-in for rag.llm.LLM.chat. rerank: keyword overlap -> 0-3 scores."""
    def __init__(self, answer="Economy Saver includes one checked bag up to 23 kg [faq-baggage].",
                 rerank_json=None, ambiguous=False, repair_answer=None):
        self.answer, self.rerank_json, self.ambiguous = answer, rerank_json, ambiguous
        self.repair_answer = repair_answer  # reply to the citation-repair turn (default: same answer)
        self.calls, self.messages = [], []

    def chat(self, messages, purpose, **kw):
        self.calls.append(purpose)
        self.messages.append(messages)
        meta = {"model": "gpt-4.1-nano", "tokens_in": 100, "tokens_out": 20,
                "cost_usd": 0.000018, "latency_ms": 5.0}
        if purpose == "rerank":
            if self.rerank_json is not None:
                return self.rerank_json, meta
            user = messages[-1]["content"]
            q = set(re.findall(r"[a-z0-9]+", re.search(r"<question>(.*?)</question>", user, re.S)
                               .group(1).lower())) - {"the", "a", "i", "my", "can", "how", "what", "on", "is"}
            scores = {}
            for m in re.finditer(r"^\((\d+)\) (.*)$", user, re.M):
                hits = len(q & set(re.findall(r"[a-z0-9]+", m.group(2).lower())))
                scores[m.group(1)] = min(3, hits)
            return json.dumps({"scores": scores, "ambiguous": self.ambiguous}), meta
        if purpose == "generate_repair" and self.repair_answer is not None:
            return self.repair_answer, meta
        return self.answer, meta


@pytest.fixture
def index():
    from rag.index import HybridIndex
    from rag.ingest import chunk_corpus, load_corpus
    return HybridIndex(chunk_corpus(load_corpus(FIXTURES)), fake_embed)


@pytest.fixture
def fares_index():
    from rag.index import HybridIndex
    from rag.ingest import chunk_documents, load_corpus
    return HybridIndex(chunk_documents(load_corpus(FARES)), fake_embed)
