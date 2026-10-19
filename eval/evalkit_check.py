"""Statistical check of the final eval run with EvalKit (my own stats library).

Run:  PYTHONPATH=/Users/vnktajay/Projects/evalkit/src .venv/bin/python eval/evalkit_check.py
Reads eval/results.json, writes eval/evalkit_results.json. No API calls.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from evalkit.calibration import auroc, brier, ece_result
from evalkit.intervals import wilson
from evalkit.paired import exact_mcnemar, newcombe_paired_interval

HERE = Path(__file__).resolve().parent
BASE, PREV, FINAL = "v0_baseline", "v4_final", "v5_final"


def frac(s: str) -> tuple[int, int]:
    k, n = map(int, re.fullmatch(r"(\d+)/(\d+)", s).groups())
    return k, n


def ci(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


def paired(a_run: dict, b_run: dict) -> dict:
    """Exact McNemar + Newcombe interval on route_correct, a (before) vs b (after)."""
    a = {q["id"]: q["route_correct"] for q in a_run["per_query"]}
    b = {q["id"]: q["route_correct"] for q in b_run["per_query"]}
    assert a.keys() == b.keys(), "runs must cover the same queries"
    gains = sum(b[i] and not a[i] for i in b)
    losses = sum(a[i] and not b[i] for i in b)
    both_ok = sum(b[i] and a[i] for i in b)
    both_bad = sum(not b[i] and not a[i] for i in b)
    newc = newcombe_paired_interval(gains, losses, both_ok, both_bad)
    return {"before_correct": sum(a.values()), "after_correct": sum(b.values()),
            "gains": [i for i in b if b[i] and not a[i]], "losses": [i for i in b if a[i] and not b[i]],
            "both_correct": both_ok, "both_wrong": both_bad,
            "exact_mcnemar_p": exact_mcnemar(gains, losses),
            "newcombe": {k: v for k, v in newc.items() if not isinstance(v, (dict, list)) or k in ("interval", "ci")}}


def calibration(rows: list[dict], key: str) -> dict:
    """Does `key` (confidence or support) predict route_correct? Guard-path runs have no score."""
    scored = [q for q in rows if q.get(key) is not None]
    conf = [float(q[key]) for q in scored]
    ok = [bool(q["route_correct"]) for q in scored]
    out = {"score": key, "n_with_score": len(scored), "n_route_wrong": ok.count(False),
           "wrong_ids": [q["id"] for q in scored if not q["route_correct"]],
           "ece_5bins": ece_result(conf, ok, bins=5), "brier": brier(conf, ok)}
    out["auroc"] = auroc(conf, ok) if 0 < ok.count(False) < len(ok) else None
    return out


def main() -> dict:
    runs = json.loads((HERE / "results.json").read_text())["runs"]
    fin = runs[FINAL]
    pq = fin["per_query"]
    route_k = sum(q["route_correct"] for q in pq)
    out = {
        "run": FINAL, "baseline": BASE, "previous": PREV,
        # v5 confidence = P(route right); support = the old retrieval-evidence score the gate uses.
        "calibration_confidence": calibration(pq, "confidence"),
        "calibration_support": calibration(pq, "support"),
        "calibration_v4_confidence": calibration(runs[PREV]["per_query"], "confidence"),
        "route_accuracy": ci(route_k, len(pq)),
        "fcr_strict_literal": ci(*frac(fin["metrics"]["fcr_n"])),
        "fcr_manual": ci(*frac(fin["manual_review"]["fcr"])),
        "faithful_manual": ci(*frac(fin["manual_review"]["fully_faithful"])),
        "paired_route_v0_vs_v5": paired(runs[BASE], fin),
        "paired_route_v4_vs_v5": paired(runs[PREV], fin),
    }
    (HERE / "evalkit_results.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
