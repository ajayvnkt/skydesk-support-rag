"""ARL log -> Agent VCR traces, plus golden expected traces from eval/queries.yaml.

Run:  .venv/bin/python eval/arl_to_vcr.py [log.arl.jsonl] [actual_dir]
Writes vcr/<actual_dir>/qNN.json (default vcr/actual/) (from the log's tool_call events) and vcr/expected/qNN.json.
Args are reduced to {query} on both sides so tuning top_k or the ticket fields never breaks a trace.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
GOLDEN = {  # expected_behavior -> tool sequence
    "answer": ["search_kb", "rerank_kb"],
    "clarify": ["search_kb", "rerank_kb"],  # grade -> clarify happens after retrieval, with no tool of its own
    "escalate_guard": ["escalate_to_human"],  # must_escalate: guard fires before retrieval
    "escalate_low_conf": ["search_kb", "rerank_kb", "escalate_to_human"],
}


def actual_traces(log: Path, ids: dict[str, str]) -> dict[str, dict]:
    """Runs are matched to eval ids by their query text, so any ARL log works (the last run of a query wins)."""
    calls, query = defaultdict(list), {}
    for line in log.read_text().splitlines():
        e = json.loads(line)
        rid = e["run_id"]
        if e["type"] == "user_input":
            query[rid] = e["content"]
        elif e["type"] == "tool_call":
            calls[rid].append(e["tool"])
    out = {}
    for rid in query:  # dicts keep log order
        if query[rid] in ids:
            qid = ids[query[rid]]
            out[qid] = {"version": 1, "scenario": qid,
                        "calls": [{"name": n, "args": {"query": query[rid]}} for n in calls[rid]]}
    return out


def golden(q: dict) -> dict:
    b = q["expected_behavior"]
    key = b if b != "escalate" else ("escalate_guard" if q["category"] == "must_escalate" else "escalate_low_conf")
    return {"version": 1, "scenario": q["id"], "calls": [{"name": n, "args": {"query": q["query"]}} for n in GOLDEN[key]]}


def main(log: Path, actual_dir: str = "actual") -> None:
    qs = yaml.safe_load((PROJECT / "eval" / "queries.yaml").read_text())
    acts = actual_traces(log, {q["query"]: q["id"] for q in qs})
    for q in qs:
        for kind, trace in (("expected", golden(q)), (actual_dir, acts.get(q["id"]))):
            if trace is None:
                continue
            p = PROJECT / "vcr" / kind / f"{q['id']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(acts)} actual traces from {log.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "logs" / "v4_final.arl.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "actual")
