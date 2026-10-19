"""SkyDesk Airways demo site: static airline homepage (web/) + support chat API over rag.pipeline.

uvicorn server:app --port 9006 --host 127.0.0.1
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import llm as llm_mod
from rag import pipeline

PROJECT = Path(__file__).resolve().parent
UI_LOGS = PROJECT / "logs" / "ui"
MODES = {"hybrid_rerank", "hybrid", "dense", "bm25"}
TEAMS = {"bereavement": "Care Team", "fraud": "Payments Integrity", "legal": "Customer Relations",
         "prompt_injection": "SkyDesk agent", "human_requested": "SkyDesk agent"}
REASONS = {"bereavement": "Bereavement travel is always handled by a person",
           "fraud": "Possible fraud or disputed charge",
           "legal": "Legal or regulatory matter",
           "prompt_injection": "Message asked the assistant to change its rules",
           "human_requested": "You asked for a person"}
BUDGET_MSG = ("The demo has reached its $1.00 API budget, so I can't look this up right now. "
              "A SkyDesk agent can still help you.")

app = FastAPI(title="SkyDesk Airways demo")


class Ask(BaseModel):
    message: str = Field(max_length=4000)
    mode: str = "hybrid_rerank"
    session_id: str = "anon"


def session_log(sid: str) -> Path:
    # sid comes from the browser: keep only safe chars so it cannot escape logs/ui/
    return UI_LOGS / f"session-web-{re.sub(r'[^A-Za-z0-9_-]', '', sid)[:32] or 'anon'}.arl.jsonl"


def shape(r: dict) -> dict:
    """pipeline.ask result -> the small JSON the chat widget renders (no debug panels)."""
    by_doc: dict[str, dict] = {}
    for c in r.get("chunks", []):
        by_doc.setdefault(c["doc_id"], c)
    cites = [{"doc_id": d, "title": (c := by_doc.get(d, {})).get("title") or d,
              "section": c.get("section"), "updated": c.get("updated"),
              "snippet": (c.get("text") or "")[:600], "dense_rank": c.get("dense_rank"),
              "bm25_rank": c.get("bm25_rank"), "rerank": c.get("rerank")}
             for d in r.get("citations") or []]
    out = {"answer": r.get("answer", ""), "route": r.get("route"), "reason": r.get("reason"),
           "confidence": r.get("confidence"), "citations": cites,
           "latency_ms": (r.get("timings") or {}).get("total"), "cost_usd": r.get("cost_usd"),
           "handoff": None}
    if r.get("reason") == "budget_exceeded":
        out["answer"] = BUDGET_MSG
    topics = [t for t in (r.get("reason") or "").split("+") if t in TEAMS]
    if r.get("route") == "escalate" and topics:  # policy hand-off; other escalations are polite refusals
        t = r.get("ticket") or {}
        out["handoff"] = {"team": ", ".join(dict.fromkeys(TEAMS[x] for x in topics)),
                          "priority": t.get("priority", "normal"),
                          "reason": "; ".join(REASONS[x] for x in topics),
                          "summary": t.get("customer_message", r.get("query", ""))[:300]}
    return out


@app.post("/api/ask")
def ask(body: Ask):
    if not body.message.strip():
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    try:
        r = pipeline.ask(body.message, mode=body.mode if body.mode in MODES else "hybrid_rerank",
                         log_path=session_log(body.session_id))
    except llm_mod.BudgetExceeded:
        return JSONResponse({"error": BUDGET_MSG}, status_code=503)
    except Exception:  # noqa: BLE001 -- never leak a traceback to the customer
        return JSONResponse({"error": "Sorry, something went wrong on our side. Please try again "
                                      "or ask for a human agent."}, status_code=500)
    return shape(r)


@app.get("/api/cost")
def cost():
    return llm_mod.summary()


@app.get("/api/log")
def log(session_id: str = "anon"):
    p = session_log(session_id)
    if not p.exists():
        return JSONResponse({"error": "No conversation logged for this session yet."}, status_code=404)
    return FileResponse(p, media_type="application/x-ndjson", filename=p.name)


@app.get("/api/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=PROJECT / "web", html=True), name="web")
