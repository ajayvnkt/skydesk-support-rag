"""Evaluate the SkyDesk RAG bot on eval/queries.yaml (real API, within the $1 ledger budget).

    python eval/run_eval.py                    # ablation + end-to-end at the default threshold
    python eval/run_eval.py --threshold 0.45 --tag tuned --skip-ablation

Writes eval/results.json (runs keyed by tag) and the ARL log of the run to
logs/conversations.arl.jsonl (fresh file per run; the last run is the one the Week 1 grader reads).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import arl  # noqa: E402
from rag.graph import CONFIDENCE_THRESHOLD  # noqa: E402
from rag.llm import LLM  # noqa: E402
from rag.pipeline import ask, build_index  # noqa: E402
from rag.prompts import JUDGE_PROMPT, to_messages  # noqa: E402
from rag.rerank import rerank  # noqa: E402

RESULTS = ROOT / "eval" / "results.json"
RETRIEVAL_CATS = {"answerable", "multi_doc", "exact_match"}
FCR_CATS = RETRIEVAL_CATS | {"ambiguous"}
K = 4


def norm(s: str) -> str:
    s = s.lower().replace(" ", " ").replace("›", ">").replace("→", ">")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-").replace("**", "")
    return re.sub(r"\s+", " ", s)


def fact_hits(answer: str, facts: list[str]) -> list[bool]:
    a = norm(answer)
    return [norm(f) in a for f in facts]


def retrieval_metrics(ranked_docs: list[str], expected: list[str], k: int = K) -> dict:
    """ranked_docs: doc_id per retrieved chunk, best first (what the generator would see)."""
    top = ranked_docs[:k]
    first = next((i + 1 for i, d in enumerate(ranked_docs) if d in expected), None)
    return {"hit": any(d in expected for d in top),
            "recall": len(set(top) & set(expected)) / len(expected),
            "rr": 1 / first if first else 0.0, "first_rank": first}


def ablation(queries, index, llm) -> dict:
    by = lambda ids: [index.by_id[c]["doc_id"] for c in ids]  # noqa: E731
    per, methods = [], ("dense", "bm25", "hybrid", "hybrid_rerank")
    for q in queries:
        if q["category"] not in RETRIEVAL_CATS:
            continue
        hyb = index.search(q["query"], top_k=12)
        top, info = rerank(q["query"], hyb, llm, top_n=12)
        ranks = {"dense": by(index.dense(q["query"], 20)), "bm25": by(index.keyword(q["query"], 20)),
                 "hybrid": [c["doc_id"] for c in hyb], "hybrid_rerank": [c["doc_id"] for c in top]}
        row = {"id": q["id"], "query": q["query"], "category": q["category"],
               "expected": q["expected_doc_ids"], "rerank_ok": info["ok"]}
        for m in methods:
            row[m] = {**retrieval_metrics(ranks[m], q["expected_doc_ids"]), "top4": ranks[m][:K]}
        per.append(row)
    summary = {m: {"hit@4": round(statistics.mean(r[m]["hit"] for r in per), 3),
                   "recall@4": round(statistics.mean(r[m]["recall"] for r in per), 3),
                   "mrr": round(statistics.mean(r[m]["rr"] for r in per), 3)} for m in methods}
    return {"n": len(per), "summary": summary, "per_query": per}




def judge(q: dict, r: dict, llm) -> dict:
    """Claim-level faithfulness: yes = every claim has a supporting quote, partial = most, no = half or fewer."""
    cited = [c for c in r["chunks"] if c["doc_id"] in r["citations"]] or r["chunks"]
    src = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in cited)
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(q.get("key_facts") or [], 1)) or "(none)"
    msgs = to_messages(JUDGE_PROMPT, question=q['query'], sources=src, answer=r['answer'], facts=facts)
    try:
        text, meta = llm.chat(msgs, purpose="judge", max_tokens=450, json_mode=True)
        out = json.loads(text)
        claims = out.get("claims") or []
        ok = sum(bool(c.get("quote")) for c in claims)
        share = ok / len(claims) if claims else 1.0
        verdict = "yes" if share == 1 else "partial" if share > 0.5 else "no"
        return {"faithful": verdict, "supported_claims": f"{ok}/{len(claims)}",
                "relevance": int(out.get("relevance", 0)),
                "unsupported": [c.get("claim") for c in claims if not c.get("quote")],
                "key_facts_semantic": [bool(x) for x in (out.get("key_facts") or [])][:len(q.get("key_facts") or [])],
                "cost_usd": meta["cost_usd"]}
    except Exception as e:  # judge failure is recorded, not fatal
        return {"faithful": "error", "relevance": None, "unsupported": [], "error": str(e), "cost_usd": 0.0}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))]


def end_to_end(queries, index, llm, threshold: float, log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # this run's log only
    rows = []
    for q in queries:
        r = ask(q["query"], index, llm, threshold=threshold, log_path=log_path, run_id=f"eval-{q['id']}")
        exp = q["expected_behavior"]
        facts = q.get("key_facts") or []
        hits = fact_hits(r["answer"], facts)
        ctx = {c["doc_id"] for c in r["chunks"]}
        row = {"id": q["id"], "query": q["query"], "category": q["category"], "expected": exp,
               "route": r["route"], "reason": r["reason"], "route_correct": r["route"] == exp,
               "confidence": r["confidence"], "support": r.get("support"), "answer": r["answer"], "citations": r["citations"],
               "context_docs": sorted(ctx), "expected_docs": q["expected_doc_ids"],
               "key_facts": facts, "facts_hit": hits,
               "latency_ms": r["timings"]["total"], "cost_usd": r["cost_usd"]}
        if r["citations"]:  # answers and grounded clarifications
            row["citations_valid"] = all(c in ctx for c in r["citations"])
            row["cites_expected_doc"] = bool(set(r["citations"]) & set(q["expected_doc_ids"]))
            row["judge"] = judge(q, r, llm)
        # Resolved on first contact without a human: correct answer, or the right clarifying
        # question for an ambiguous query (an answer that covers every key fact also counts there).
        if q["category"] in FCR_CATS:
            answered_ok = r["route"] == "answer" and all(hits)
            row["fcr"] = answered_ok or (exp == "clarify" and r["route"] == "clarify")
        rows.append(row)
        print(f"{q['id']} {q['category']:<13} exp={exp:<8} got={r['route']:<8} "
              f"conf={r['confidence']} facts={sum(hits)}/{len(facts)} {r['reason'] or ''}", flush=True)
    return {"threshold": threshold, "per_query": rows, "metrics": metrics(rows)}


def metrics(rows) -> dict:
    fcr = [r for r in rows if "fcr" in r]
    has_sem = any("key_facts_semantic" in (r.get("judge") or {}) for r in rows)  # v0 judge lacked it
    exp_esc = [r for r in rows if r["expected"] == "escalate"]
    got_esc = [r for r in rows if r["route"] == "escalate"]
    tp = sum(r["expected"] == "escalate" for r in got_esc)
    answered = [r for r in rows if r["route"] == "answer"]
    cited = [r for r in rows if "judge" in r]
    judged = [r["judge"] for r in cited if r["judge"]["faithful"] in ("yes", "partial", "no")]
    sem = [x for r in rows if r["expected"] == "answer" for x in
           (r["judge"].get("key_facts_semantic") or [False] * len(r["key_facts"]) if "judge" in r
            else [False] * len(r["key_facts"]))]
    exp_ans = [r for r in rows if r["expected"] == "answer"]
    lat = [r["latency_ms"] for r in rows]
    lat_llm = [r["latency_ms"] for r in rows if not _is_guard(r)]
    return {
        "n": len(rows),
        "route_accuracy": round(sum(r["route_correct"] for r in rows) / len(rows), 3),
        "first_contact_resolution": round(sum(r["fcr"] for r in fcr) / len(fcr), 3),
        "fcr_n": f"{sum(r['fcr'] for r in fcr)}/{len(fcr)}",
        "first_contact_resolution_semantic": round(sum(_fcr_semantic(r) for r in fcr) / len(fcr), 3) if has_sem else None,
        "escalation_precision": round(tp / len(got_esc), 3) if got_esc else None,
        "escalation_recall": round(tp / len(exp_esc), 3) if exp_esc else None,
        "escalations": f"{tp} correct of {len(got_esc)} escalated; {len(exp_esc)} expected",
        "unanswerable_refused": f"{sum(r['route'] == 'escalate' for r in rows if r['category'] == 'unanswerable')}/"
                                f"{sum(r['category'] == 'unanswerable' for r in rows)}",
        "must_escalate_caught": f"{sum(r['route'] == 'escalate' for r in rows if r['category'] == 'must_escalate')}/"
                                f"{sum(r['category'] == 'must_escalate' for r in rows)}",
        "key_fact_coverage_expected_answer": round(
            sum(sum(r["facts_hit"]) for r in exp_ans) / max(1, sum(len(r["key_facts"]) for r in exp_ans)), 3),
        "key_fact_coverage_expected_answer_semantic": round(sum(sem) / len(sem), 3) if has_sem else None,
        "key_fact_coverage_when_answered": round(
            sum(sum(r["facts_hit"]) for r in answered) / max(1, sum(len(r["key_facts"]) for r in answered)), 3),
        "answered": len(answered),
        "judged_responses": len(judged),
        "faithfulness_yes_rate": round(sum(j["faithful"] == "yes" for j in judged) / len(judged), 3) if judged else None,
        "faithfulness_score": round(statistics.mean({"yes": 1, "partial": .5, "no": 0}[j["faithful"]]
                                                    for j in judged), 3) if judged else None,
        "judge_verdicts": {v: sum(j["faithful"] == v for j in judged) for v in ("yes", "partial", "no")},
        "relevance_mean": round(statistics.mean(j["relevance"] for j in judged), 2) if judged else None,
        "citation_validity": round(sum(r["citations_valid"] for r in cited) / len(cited), 3) if cited else None,
        "cites_expected_doc": round(sum(r["cites_expected_doc"] for r in cited) / len(cited), 3) if cited else None,
        "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95),
        "latency_ms_p50_llm_path": pct(lat_llm, 50) if lat_llm else None,
        "latency_ms_p95_llm_path": pct(lat_llm, 95) if lat_llm else None,
        "bot_cost_usd_total": round(sum(r["cost_usd"] for r in rows), 6),
        "bot_cost_usd_per_query": round(sum(r["cost_usd"] for r in rows) / len(rows), 6),
        "judge_cost_usd_total": round(sum(r["judge"]["cost_usd"] for r in cited), 6),
    }


def _fcr_semantic(r) -> bool:
    """FCR with the judge's meaning-level key-fact check instead of literal substrings."""
    if r["expected"] == "clarify" and r["route"] == "clarify":
        return True
    sem = (r.get("judge") or {}).get("key_facts_semantic") or []
    return r["route"] == "answer" and len(sem) == len(r["key_facts"]) and all(sem)


def threshold_sweep(rows, grid=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)) -> list[dict]:
    """Offline counterfactual (no API): re-route each retrieved query by its recorded gate score
    (`support` from v5 on, where `confidence` became P(route right); `confidence` before).
    A query that passed the gate keeps its observed outcome; one that would now pass but was
    gated is counted as an answer of unknown quality (shown separately)."""
    out = []
    for t in grid:
        ok = unknown = 0
        for r in rows:
            # guard escalation, or (v0) reranker-ambiguous clarify that ran before the gate
            gate = r.get("support", r["confidence"])
            if gate is None or r["reason"] in ("ambiguous", "vague_query"):
                ok += r["route_correct"]
                continue
            passes = gate >= t
            if not passes:
                ok += r["expected"] == "escalate"
            elif r["reason"] == "low_confidence":
                unknown += 1
            else:
                ok += r["route_correct"]
        out.append({"threshold": t, "route_correct": ok, "unknown": unknown, "n": len(rows)})
    return out


GUARD_NAMES = {"bereavement", "fraud", "legal", "prompt_injection", "human_requested"}


def _is_guard(r) -> bool:
    """Guard escalations never reach retrieval/LLM; kept out of the LLM-path latency numbers."""
    return bool(r["reason"]) and set(r["reason"].split("+")) <= GUARD_NAMES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute metrics from results.json, no API")
    args = ap.parse_args(argv)

    if args.rescore:
        out = json.loads(RESULTS.read_text())
        for run in out["runs"].values():
            run["metrics"] = metrics(run["per_query"])
            run["threshold_sweep"] = threshold_sweep(run["per_query"])
        RESULTS.write_text(json.dumps(out, indent=2))
        print(json.dumps({k: v["metrics"] for k, v in out["runs"].items()}, indent=2))
        return
    queries = yaml.safe_load((ROOT / "eval" / "queries.yaml").read_text())
    llm = LLM()
    spend0 = llm.spent()
    index = build_index(llm=llm)
    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    if not args.skip_ablation:
        out["ablation"] = ablation(queries, index, llm)
        print(json.dumps(out["ablation"]["summary"], indent=2))
    out["runs"][args.tag] = end_to_end(queries, index, llm, args.threshold, arl.LOG_PATH)
    out["runs"][args.tag]["threshold_sweep"] = threshold_sweep(out["runs"][args.tag]["per_query"])
    out["runs"][args.tag]["spend_this_run_usd"] = round(llm.spent() - spend0, 6)
    out["index"] = {"docs": len({c["doc_id"] for c in index.chunks}), "chunks": len(index.chunks)}
    out["ledger_total_usd"] = round(llm.spent(), 6)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["runs"][args.tag]["metrics"], indent=2))
    print(f"spend this run ${llm.spent() - spend0:.6f}; ledger total ${llm.spent():.6f}")


if __name__ == "__main__":
    main()
