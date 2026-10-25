"""Regression tests for the validation-tool failures, on paraphrases and on an unseen fixture
corpus (tests/fixtures/fares: Saver/Standard/Plus fares, Bronze/Silver/Gold tiers), so the rules
are tested on names and wording the eval set never used."""
import json

import pytest

from conftest import FakeLLM
from rag.graph import (build_graph, option_ambiguity, option_tables, pinned_option_chunk,
                       route_confidence, stated_options, support)
from rag.pipeline import ask

BAGS_Q = "what's the checked luggage limit for an Economy Saver ticket?"


# ---- 1. one citation repair before escalating ----
def test_uncited_draft_is_repaired_once_then_answered(index, tmp_path):
    llm = FakeLLM(answer="Economy Saver includes one checked bag up to 23 kg.",
                  repair_answer="Economy Saver includes one checked bag up to 23 kg [faq-baggage].")
    log = tmp_path / "l.jsonl"
    r = ask(BAGS_Q, index, llm, log_path=log)
    assert r["route"] == "answer" and r["citations"] == ["faq-baggage"], r
    assert r["reason"] == "citations_repaired"
    assert llm.calls == ["rerank", "generate", "generate_repair"]
    repair_msgs = llm.messages[-1]
    assert repair_msgs[-2] == {"role": "assistant", "content": llm.answer}
    assert "faq-baggage" in repair_msgs[-1]["content"]
    ev = [json.loads(x) for x in log.read_text().splitlines()]
    rep = [e for e in ev if e["type"] == "llm_response" and e["meta"]["purpose"] == "generate_repair"]
    assert rep and rep[0]["meta"]["repair_reason"] == "no_citations"
    assert not [e for e in ev if e["type"] == "error"]  # a repair is not an error


def test_repair_happens_at_most_once(index, tmp_path):
    llm = FakeLLM(answer="One checked bag up to 23 kg.")  # repair also comes back uncited
    r = ask(BAGS_Q, index, llm, log_path=tmp_path / "l.jsonl")
    assert r["route"] == "escalate" and r["reason"] == "no_citations"
    assert llm.calls == ["rerank", "generate", "generate_repair"]


def test_hallucinated_only_citation_gets_repair_then_escalates(index, tmp_path):
    llm = FakeLLM(answer="You get 3 bags [made-up-doc].")
    r = ask(BAGS_Q, index, llm, log_path=tmp_path / "l.jsonl")
    assert r["route"] == "escalate" and r["reason"].startswith("citation_not_in_context")
    assert llm.calls.count("generate_repair") == 1


def test_refusal_is_not_repaired(index, tmp_path):
    llm = FakeLLM(answer="INSUFFICIENT_CONTEXT")
    r = ask(BAGS_Q, index, llm, log_path=tmp_path / "l.jsonl")
    assert r["reason"] == "not_in_kb" and "generate_repair" not in llm.calls


# ---- 2. included allowances before fees ----
def test_generation_rules_subtract_included_allowance():
    from rag.prompts import GEN_SYSTEM
    assert "does NOT already include" in GEN_SYSTEM and "1 included = 1 extra" in GEN_SYSTEM


def test_named_fare_pins_its_allowance_row_into_context(fares_index):
    """3 bags on a Standard fare: the fee table alone can't be totalled correctly, so the
    passage saying what Standard includes is pinned in even when rerank cut it (top_n=1)."""
    llm = FakeLLM(answer="3 bags - 1 included = 2 extra: $30 + $50 [policy-extra-bags] [policy-allowance].")
    q = "Travelling on Standard with 3 suitcases, what do the extra bags cost online?"
    s = build_graph(fares_index, llm, top_n=1).invoke(
        {"query": q, "raw_query": q, "timings": {}, "cost": 0.0, "trace": []})
    ids = [c["chunk_id"] for c in s["chunks"]]
    assert ids[0] == "policy-extra-bags#0" and "policy-allowance#0" in ids, ids
    assert s["route"] == "answer"
    assert "Standard | 1 checked bag" in llm.messages[-1][1]["content"]


def test_no_pin_when_no_option_named():
    ranked = [{"chunk_id": "a", "rerank": 3, "text": "x"},
              {"chunk_id": "b", "rerank": 2, "text": "| Fare type | Bags |\n|---|---|\n| Saver | 0 |\n| Plus | 2 |"}]
    assert pinned_option_chunk("how much is an extra bag", ranked, ranked[:1]) is None
    assert pinned_option_chunk("extra bag on plus?", ranked, ranked[:1])["chunk_id"] == "b"
    ranked[1]["rerank"] = 0  # irrelevant chunks are never pinned
    assert pinned_option_chunk("extra bag on plus?", ranked, ranked[:1]) is None


# ---- 3. clarify when the answer differs per fare / tier and the customer didn't say ----
@pytest.mark.parametrize("q,family,options", [
    ("how many checked bags are included with my ticket?", "fare type", ["Saver", "Standard", "Plus"]),
    ("what is the seat selection price?", "membership tier", ["Bronze", "Silver", "Gold"]),
])
def test_option_dependent_question_gets_clarify_with_options(fares_index, tmp_path, q, family, options):
    llm = FakeLLM()
    r = ask(q, fares_index, llm, log_path=tmp_path / "l.jsonl")
    assert r["route"] == "clarify" and r["reason"] == "option_dependent", r
    assert family in r["answer"] and all(o in r["answer"] for o in options)
    assert r["citations"] and "generate" not in llm.calls  # deterministic, no generation cost


@pytest.mark.parametrize("q", ["I'm on a Plus fare, how many checked bags are included?",
                               "Gold member here: what is the seat selection price?",
                               "what is the online price of an extra bag?"])
def test_stated_option_or_non_customer_table_answers(fares_index, q):
    llm = FakeLLM(answer="See [policy-allowance] [policy-seat-prices] [policy-extra-bags].")
    r = ask(q, fares_index, llm, log_path=None)
    assert r["route"] == "answer", r


def test_option_table_parsing_and_stated_names():
    text = ("| Fare class | Fee |\n|---|---|\n| SkyLite | Not permitted |\n| SkyClassic | $75 |\n\n"
            "| Delay | Pay |\n|---|---|\n| 3 hours | $100 |\n| 5 hours | $250 |\n\n"
            "| Tier | Lounge |\n|---|---|\n| Silver | No |\n| Gold | No |")
    t = option_tables(text)
    assert len(t) == 1 and list(t[0]["rows"]) == ["SkyLite", "SkyClassic"]  # delay: not a customer attribute; tier: same answer
    opts = ["SkyLite", "SkyClassic", "SkyFlex"]
    assert stated_options("i fly sky classic", opts) == ["SkyClassic"]
    assert stated_options("on a classic fare", opts) == ["SkyClassic"]
    assert stated_options("is it flexible?", opts) == []
    chunks = [{"doc_id": "d", "rerank": 3, "text": text}, {"doc_id": "e", "rerank": 3, "text": "x"}]
    assert option_ambiguity("change fee?", chunks)["doc_ids"] == ["d"]
    assert option_ambiguity("change fee on skylite?", chunks) is None
    chunks[0]["rerank"] = 2  # not among the best-supported chunks
    assert option_ambiguity("change fee?", chunks) is None


# ---- 4. reported confidence = P(route is right) ----
def test_route_confidence_points_the_right_way():
    assert route_confidence("escalate", "low_confidence", 0.04) > 0.9  # clear "not in KB"
    assert route_confidence("escalate", "low_confidence", 0.45) < 0.6  # borderline
    assert route_confidence("answer", None, 0.96) > route_confidence("answer", None, 0.96, repaired=True)
    assert route_confidence("answer", None, 0.96, flagged_vague=True) < 0.96
    assert 0.5 <= route_confidence("escalate", "not_in_kb", 0.96) < route_confidence("escalate", "not_in_kb", 0.6)


def test_low_support_escalation_reports_high_confidence(index, tmp_path):
    llm = FakeLLM(rerank_json=json.dumps({"scores": {"0": 0, "1": 1}, "ambiguous": False}))
    r = ask("Can I bring my parrot in the cabin on a long-haul flight?", index, llm,
            log_path=tmp_path / "l.jsonl")
    assert r["route"] == "escalate" and r["reason"] == "low_confidence"
    assert r["confidence"] > 0.5 and r["ticket"]["confidence"] == r["confidence"]


def test_named_fare_row_is_pinned_even_when_retrieval_missed_it(fares_index):
    """top_k=1: retrieval returns only the fee table. The per-fare table naming the customer's
    fare joins the rerank pool as a pin-only candidate and reaches the generator."""
    llm = FakeLLM(answer="3 bags - 1 included = 2 extra: $30 + $50 [policy-extra-bags] [policy-allowance].")
    q = "Travelling on Standard with 3 suitcases, what do the extra bags cost online?"
    s = build_graph(fares_index, llm, top_k=1, top_n=1).invoke(
        {"query": q, "raw_query": q, "timings": {}, "cost": 0.0, "trace": []})
    assert [c["chunk_id"] for c in s["candidates"]] == ["policy-extra-bags#0"]
    assert [c["chunk_id"] for c in s["chunks"]] == ["policy-extra-bags#0", "policy-allowance#0"]
    assert s["chunks"][1]["pinned"]
    assert s["support"] == support(s["chunks"][:1])  # a pinned chunk never inflates support


class PriceCheckLLM(FakeLLM):
    def __init__(self, check_reply, **kw):
        super().__init__(**kw)
        self.check_reply = check_reply

    def chat(self, messages, purpose, **kw):
        if purpose == "price_check":
            self.calls.append(purpose)
            self.messages.append(messages)
            return json.dumps(self.check_reply), {"model": "gpt-4.1-nano", "tokens_in": 1,
                                                  "tokens_out": 1, "cost_usd": 1e-6, "latency_ms": 1.0}
        return super().chat(messages, purpose, **kw)


PLUS_Q = "I'm on a Plus fare and bringing 3 checked bags. What's the online price for the extra ones?"


def test_price_check_replaces_a_wrong_total(fares_index):
    wrong = "Extra bags cost $30 + $50 = $80 [policy-extra-bags]."
    right = "Plus includes 2 checked bags, so 3 - 2 = 1 extra bag: $30 [policy-allowance] [policy-extra-bags]."
    llm = PriceCheckLLM({"answer_correct": False, "corrected_answer": right}, answer=wrong)
    r = ask(PLUS_Q, fares_index, llm, log_path=None)
    assert r["route"] == "answer" and r["answer"] == right and "price_check" in llm.calls


def test_price_check_keeps_a_correct_answer(fares_index):
    ok = "Plus includes 2 checked bags, so 1 extra bag: $30 [policy-allowance] [policy-extra-bags]."
    llm = PriceCheckLLM({"answer_correct": True, "corrected_answer": ""}, answer=ok)
    assert ask(PLUS_Q, fares_index, llm, log_path=None)["answer"] == ok


def test_price_check_skipped_without_prices_or_fare_tables(index, fares_index):
    llm = PriceCheckLLM({"answer_correct": False, "corrected_answer": "x"},
                        answer="Snacks are sold on every flight [faq-onboard].")
    ask("is food free on board?", fares_index, llm, log_path=None)
    llm2 = PriceCheckLLM({"answer_correct": False, "corrected_answer": "x"},
                         answer="Economy Saver includes one checked bag; extras are $30 [faq-baggage].")
    ask(BAGS_Q, index, llm2, log_path=None)  # a price, but no per-fare table in context
    assert "price_check" not in llm.calls + llm2.calls


def test_no_pin_when_context_already_has_the_named_fare_row():
    table = "| Fare type | Bags |\n|---|---|\n| Saver | 0 |\n| Plus | 2 |"
    chosen = [{"chunk_id": "a", "rerank": 3, "text": table}]
    ranked = chosen + [{"chunk_id": "b", "rerank": 3, "text": table.replace("Bags", "Seats")}]
    assert pinned_option_chunk("bags on plus?", ranked, chosen) is None


def test_same_family_tables_in_best_chunks_are_merged_into_one_question():
    a = "| Fare type | Change fee |\n|---|---|\n| Saver | Not allowed |\n| Plus | Free |"
    b = "| Fare type | Same-day change |\n|---|---|\n| Saver | No |\n| Plus | Free |"
    opt = option_ambiguity("change my flight?", [{"doc_id": "x", "rerank": 3, "text": a},
                                                 {"doc_id": "y", "rerank": 3, "text": b}])
    assert opt["columns"] == ["Change fee", "Same-day change"] and opt["doc_ids"] == ["x", "y"]
    assert opt["rows"]["Saver"] == ["Not allowed", "No"]
