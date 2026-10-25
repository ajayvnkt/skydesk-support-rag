"""SkyDesk Support Assistant: Streamlit chat UI over rag.pipeline (ops console, port 9007).

streamlit run app.py --server.port 9007
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from rag import llm as llm_mod
from rag import pipeline

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "eval" / "results.json"
UI_LOGS = PROJECT / "logs" / "ui"
ONE_LINER = ("My RAG app helps SkyDesk support agents and customers answer booking, refund, baggage "
             "and policy questions from the SkyDesk help center (55 docs: policies, FAQs, app manual, "
             "resolved tickets) in a chat widget on the airline's website with 94% faithfulness against a 90% target (15/16 "
             "answers, my manual review) and 4.06/5 relevance (LLM judge), under a 3-second latency "
             "ceiling (measured p95 1.78 s), and escalates to a human when it is not sure.")

MODES = {"Hybrid + rerank": "hybrid_rerank", "Hybrid (RRF, no rerank)": "hybrid",
         "Dense only": "dense", "BM25 only": "bm25"}
SAMPLES = [
    ("Exact code", "app shows BAG-301 when I add my bag"),
    ("Escalation", "There's a booking on my card I never made. I already filed a chargeback with "
                   "my bank and my lawyer says you owe me, so just refund it now."),
    ("Not in the KB", "is there wifi on the plane and how much does it cost?"),
    ("Exact term", "SkyFlex"),
    ("Paraphrase", "my suitcase didnt arrive and I need to buy clothes, how much will you "
                   "reimburse while i wait?"),
    ("Pets", "can my cat fly in the cabin with me? what's the weight limit and fee"),
    ("Fare-dependent", "how much does it cost to change my flight"),
]
ROUTE_BADGE = {"answer": ("Answered", "green", ":material/check_circle:"),
               "clarify": ("Asked to clarify", "orange", ":material/help:"),
               "escalate": ("Escalated to a human", "red", ":material/support_agent:")}
TEAMS = {"bereavement": "Care Team", "fraud": "Payments Integrity",
         "legal": "Customer Relations", "prompt_injection": "SkyDesk agent",
         "human_requested": "SkyDesk agent"}
REASONS = {"bereavement": "Bereavement (policy: always a human)",
           "fraud": "Possible fraud or disputed charge (policy: always a human)",
           "legal": "Legal or regulatory matter (policy: always a human)",
           "prompt_injection": "Message tried to change the assistant's instructions",
           "human_requested": "Customer asked for a person",
           "low_confidence": "Retrieved evidence below the confidence threshold",
           "not_in_kb": "The help center does not cover this",
           "no_citations": "Draft answer had no citations",
           "budget_exceeded": "API budget cap reached",
           "claimed_action": "Draft claimed a refund/voucher/change was done (the bot cannot act)",
           "no_answer": "No answer was generated",
           "option_dependent": "Answer differs per fare class / tier and the customer did not say theirs",
           "citations_repaired": "First draft had no valid citation; one repair turn fixed it",
           "vague_query": "Question too vague to search"}
BUDGET_MSG = ("The demo has reached its $1.00 API budget, so I can't look this up right now. "
              "A SkyDesk agent can still help you.")

st.set_page_config(page_title="SkyDesk Support Assistant", page_icon=":material/flight:",
                   layout="wide")


def md(text: str) -> str:
    """Escape $ so prices like "$40 + $60" are not rendered as LaTeX."""
    return (text or "").replace("$", r"\$")


# ---------------- sidebar ----------------
def cost_meter():
    s = llm_mod.summary()
    st.sidebar.markdown("**API spend**")
    st.sidebar.progress(min(1.0, s["total_usd"] / s["budget_usd"]),
                        text=md(f"${s['total_usd']:.4f} of ${s['budget_usd']:.2f} cap "
                                    f"({s['calls']} calls)"))


def session_log() -> Path:
    if "sid" not in st.session_state:
        st.session_state.sid = uuid.uuid4().hex[:10]
    return UI_LOGS / f"session-{st.session_state.sid}.arl.jsonl"


# ---------------- chat rendering ----------------
def reason_label(reason: str | None) -> str:
    if not reason:
        return "n/a"
    if reason.startswith("citation_not_in_context"):
        return "Draft cited a document that was not retrieved"
    return " + ".join(REASONS.get(r, r) for r in reason.split("+"))


def chunk_card(c: dict):
    st.markdown(f"**{c['title']}**" + (f" > {c['section']}" if c.get("section") else ""))
    st.caption(f"`{c['doc_id']}` | {c.get('doc_type') or 'doc'} | updated "
               f"{c.get('updated') or 'n/a'}" + (" | SUPERSEDED" if c.get("superseded") else ""))
    rr = "n/a" if c.get("rerank") is None else f"{c['rerank']}/3"
    st.markdown(f"Dense rank **{c.get('dense_rank') or '-'}** | BM25 rank "
                f"**{c.get('bm25_rank') or '-'}** | Rerank **{rr}** | RRF {c.get('fused')}")
    st.markdown(md(c["text"][:1500]))


def render_answer(r: dict, details: bool):
    label, color, icon = ROUTE_BADGE.get(r["route"], (r["route"], "gray", None))
    conf = r.get("confidence")
    sup = r.get("support")
    conf_txt = "Confidence n/a (policy guard)" if conf is None else f"Confidence {conf:.2f}" + (
        f" (evidence {sup:.2f})" if sup is not None else "")
    with st.container(horizontal=True):
        st.badge(label, color=color, icon=icon)
        st.badge(conf_txt, color="blue" if conf is not None and conf >= pipeline.CONFIDENCE_THRESHOLD else "gray")
        st.badge(next(k for k, v in MODES.items() if v == r.get("mode", "hybrid_rerank")),
                 color="violet")
    st.markdown(md(r["answer"]))

    if r["route"] == "answer" and r["citations"]:
        st.caption("Sources (click to open the passage)")
        with st.container(horizontal=True):
            for doc in r["citations"]:
                with st.popover(doc, icon=":material/description:"):
                    for c in [c for c in r["chunks"] if c["doc_id"] == doc]:
                        chunk_card(c)

    if r["route"] == "escalate" and r.get("ticket"):
        t = r["ticket"]
        topics = (t.get("reason") or "").split("+")
        teams = list(dict.fromkeys(TEAMS[x] for x in topics if x in TEAMS)) or ["SkyDesk agent"]
        with st.container(border=True):
            st.markdown("**:material/support_agent: Hand-off ticket**")
            st.markdown(f"**Routed to:** {', '.join(teams)}  \n**Priority:** "
                        f"{t.get('priority', 'normal').title()}")
            st.markdown(f"**Reason:** {reason_label(t.get('reason'))}")
            st.markdown(f"**Customer message:** {md(t.get('customer_message', ''))}")
            if t.get("suggested_docs"):
                st.markdown("**Docs for the agent:** " + ", ".join(f"`{d}`" for d in t["suggested_docs"]))
            if t.get("draft_answer"):
                st.markdown(f"**Unverified draft (not shown to customer):** {md(t['draft_answer'])}")

    if details:
        with st.expander("Retrieval details"):
            tm = r.get("timings", {})
            st.caption(f"Total {tm.get('total', '-')} ms | retrieve {tm.get('retrieve', '-')} ms | "
                       f"rerank {tm.get('rerank', '-')} ms | generate {tm.get('generate', '-')} ms | "
                       f"cost \\${r.get('cost_usd', 0):.6f} | run `{r.get('run_id')}`")
            if r.get("candidates"):
                top = {c["chunk_id"]: c.get("rerank") for c in r["chunks"]}
                st.dataframe(pd.DataFrame([{
                    "doc": c["doc_id"], "chunk": c["chunk_id"], "dense rank": c.get("dense_rank"),
                    "BM25 rank": c.get("bm25_rank"), "RRF": c.get("fused"),
                    "sent to LLM": c["chunk_id"] in top, "rerank": top.get(c["chunk_id"])}
                    for c in r["candidates"]]), hide_index=True, width="stretch")
            else:
                st.caption("No retrieval: a policy guard or clarify rule handled this turn.")


def chat_page():
    st.title("SkyDesk Support Assistant")
    st.caption(ONE_LINER)
    sb = st.sidebar
    sb.markdown("**Try these**")
    for tag, q in SAMPLES:
        if sb.button(f"{tag}: {q[:48]}{'...' if len(q) > 48 else ''}", key=f"s-{tag}",
                     width="stretch"):
            st.session_state.pending = q
    sb.divider()
    mode_label = sb.radio("Retrieval mode", list(MODES), key="mode",
                          help="Non-rerank modes take the top 4 retrieved chunks as-is; evidence "
                               "then comes only from dense/BM25 agreement, so they escalate more.")
    details = sb.toggle("Show retrieval details", key="details")
    log = session_log()
    sb.download_button("Download conversation log (ARL)",
                       data=log.read_bytes() if log.exists() else b"",
                       file_name=log.name, mime="application/x-ndjson",
                       disabled=not log.exists(), width="stretch",
                       help="Agent Run Log v1, for the Week 1 Agent Report Card: "
                            "python -m arc analyze <file>")
    if sb.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    msgs = st.session_state.setdefault("messages", [])
    if not msgs:
        st.info("Ask about baggage, refunds, delays, pets, the SkyDesk app or error codes. "
                "Answers cite the help-center page they come from.", icon=":material/info:")
    for i, m in enumerate(msgs):
        with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user"
                             else ":material/flight:"):
            if m["role"] == "user":
                st.markdown(md(m["content"]))
            elif "error" in m:
                st.warning(md(m["error"]), icon=":material/savings:")
            else:
                render_answer(m["result"], details)

    prompt = (st.chat_input("Ask SkyDesk support...") or st.session_state.pop("pending", None)
              or "").strip()
    if not prompt:  # whitespace-only input would make pipeline.ask raise
        return
    msgs.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Searching the help center..."):
            r = pipeline.ask(prompt, mode=MODES[mode_label], log_path=log)
        if r.get("reason") == "budget_exceeded":
            r["answer"] = BUDGET_MSG
        msgs.append({"role": "assistant", "result": r})
    except llm_mod.BudgetExceeded:
        msgs.append({"role": "assistant", "error": BUDGET_MSG})
    st.rerun()


# ---------------- evaluation page ----------------
def eval_page():
    st.title("Evaluation")
    st.caption("Read from eval/results.json (no re-running). Full write-up: docs/EVAL_REPORT.md.")
    if not RESULTS.exists():
        st.warning("eval/results.json not found. Run: python eval/run_eval.py")
        return
    d = json.loads(RESULTS.read_text())
    final = next(k for k in ("v5_final", "v4_final", "v3") if k in d["runs"])
    v0, v3 = d["runs"]["v0_baseline"], d["runs"][final]
    m, mr = v3["metrics"], v3.get("manual_review", {})

    st.subheader(f"Headline (final config {final}, threshold 0.5)")
    a, b, c = st.columns(3)
    a.metric("Route accuracy", f"{round(m['route_accuracy'] * m['n'])}/{m['n']}",
             f"{(m['route_accuracy'] - v0['metrics']['route_accuracy']) * 100:+.0f} pts vs baseline")
    b.metric("First-contact resolution", mr.get("fcr", m["fcr_n"]),
             help=f"Manual review. Judge semantic {m['first_contact_resolution_semantic']:.0%}, "
                  f"strict literal {m['fcr_n']}.")
    c.metric("Escalation precision / recall",
             f"{m['escalation_precision']:.2f} / {m['escalation_recall']:.2f}")
    a, b, c = st.columns(3)
    a.metric("Cost per query", f"${m['bot_cost_usd_per_query']:.5f}")
    b.metric("Faithfulness (manual)", mr.get("fully_faithful", "?"),
             f"LLM judge {m['judge_verdicts']['yes']}/{m['judged_responses']}", delta_color="off",
             help="Manual review of every response against the full cited docs; the judge is a smoke test.")
    c.metric("Relevance (judge, 1-5)", f"{m['relevance_mean']:.2f}")
    a, b, c = st.columns(3)
    a.metric("Citation validity", f"{m['citation_validity']:.0%}",
             f"{m['cites_expected_doc']:.0%} cite an expected doc", delta_color="off")
    b.metric("Latency p50 / p95", f"{m['latency_ms_p50'] / 1000:.2f} / {m['latency_ms_p95'] / 1000:.2f} s")
    c.metric("Refused / must-escalate", f"{m['unanswerable_refused']} / {m['must_escalate_caught']}",
             help="Unanswerable queries refused / policy escalations caught.")
    st.warning(f"Faithfulness in the one-liner is my manual review; LLM judge "
               f"{m['judge_verdicts']['yes']}/{m['judged_responses']}, manual review {mr.get('fully_faithful', '?')} "
               "fully faithful. All fixes were tuned on these same 20 queries; there is no "
               "held-out set.", icon=":material/warning:")

    st.subheader("Retrieval ablation")
    abl = d["ablation"]
    names = {"dense": "Dense only", "bm25": "BM25 only", "hybrid": "Hybrid (RRF)",
             "hybrid_rerank": "Hybrid + rerank"}
    st.dataframe(pd.DataFrame([{"Retriever": names[k], **v} for k, v in abl["summary"].items()]),
                 hide_index=True)
    st.caption(f"{abl['n']} answerable, multi-doc and exact-match queries; top 4 chunks.")

    st.subheader("Baseline vs final")
    keys = [("route_accuracy", "Route accuracy"), ("escalation_precision", "Escalation precision"),
            ("escalation_recall", "Escalation recall"), ("first_contact_resolution", "FCR (literal)"),
            ("faithfulness_yes_rate", "Faithfulness (judge)"), ("relevance_mean", "Relevance")]
    st.dataframe(pd.DataFrame([{"Metric": n, "Baseline (v0)": v0["metrics"][k], f"Final ({final})": m[k]}
                               for k, n in keys]), hide_index=True)

    st.subheader("Per query (final run)")
    st.dataframe(pd.DataFrame([{
        "id": q["id"], "category": q["category"], "query": q["query"], "expected": q["expected"],
        "route": q["route"], "correct": q["route_correct"], "confidence": q["confidence"],
        "citations": ", ".join(q.get("citations") or []),
        "judge": (q.get("judge") or {}).get("faithful"), "latency ms": q["latency_ms"]}
        for q in v3["per_query"]]), hide_index=True, width="stretch")

    st.subheader("Confidence threshold sweep")
    sweep = pd.DataFrame(v3["threshold_sweep"]).set_index("threshold")[["route_correct"]]
    st.line_chart(sweep, y_label="routes correct (of 20)")
    if mr.get("failures"):
        st.subheader("Known failures (manual review)")
        for qid, why in mr["failures"].items():
            st.markdown(md(f"- **{qid}**: {why}"))


# ---------------- how it works ----------------
FLOW = """digraph {
  rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  q [label="Question", shape=oval, fillcolor="#ffffff"];
  classify [label="classify\\n(regex guards, no LLM)"];
  retrieve [label="retrieve\\ndense + BM25, RRF\\ndrop superseded"];
  rerank [label="rerank\\ngpt-4.1-nano 0-3"];
  grade [label="grade\\nevidence >= 0.5?\\nper-fare table?"];
  generate [label="generate\\ncited answer"];
  verify [label="verify\\ncitations in context?\\nno claimed actions"];
  escalate [label="escalate\\nhand-off ticket", fillcolor="#fde2e1"];
  clarify [label="clarify", fillcolor="#fff1d6"];
  done [label="Answer", shape=oval, fillcolor="#dff3e4"];
  q -> classify; classify -> escalate [label="bereavement / fraud /\\nlegal / injection"];
  classify -> clarify [label="too vague"]; classify -> retrieve -> rerank -> grade;
  grade -> escalate [label="low"]; grade -> clarify [label="fare not stated"];
  grade -> generate -> verify; verify -> generate [label="no valid cite\\n(1 repair)"];
  verify -> done [label="ok"]; verify -> escalate [label="refusal / bad cite /\\nclaims an action"];
}"""


def how_page():
    st.title("How it works")
    st.caption("Track 2: LangChain + LangGraph. Use case 4: customer support knowledge base with "
               "hybrid search.")
    st.markdown("""
| Step | What SkyDesk does |
|---|---|
| **Use case** | Answer booking, refund, baggage and policy questions for agents and customers; escalate to a human when unsure or when policy requires it. |
| **Corpus** | 55 docs, about 14,800 words: 24 policies, 9 FAQs, 9 app-manual pages, 13 resolved tickets. 5 of them are raw HTML help-center pages; 1 is a superseded 2024 policy kept for the freshness test. |
| **Ingestion + cleaning** | Markdown with YAML front-matter (id, doc_type, updated, tags). HTML is parsed with the standard-library parser: scripts, nav, footer and cookie banner dropped, entities decoded, table cells kept apart. |
| **Freshness** | Each doc carries an `updated` date; superseded pages stay searchable but are dropped before reranking when a current page exists. Embeddings are cached by content hash, so re-indexing only pays for changed chunks. |
| **Chunking + embedding** | Split on headings first (the heading becomes the citation section), then ~350 tokens with 50 overlap. FAQ answers and tickets stay whole. 206 chunks embedded with text-embedding-3-small. |
| **Retrieve** | Dense cosine (top 20) + BM25 with code-aware tokens like SD-ERR-112 (top 20), fused with Reciprocal Rank Fusion; gpt-4.1-nano reranks the top 12 and keeps 4. |
| **Generate + guard** | gpt-4.1-nano answers only from the 4 passages with [doc-id] citations; every cited id must be in the context, otherwise the turn escalates. |
| **Cost control** | Every OpenAI call goes through one module with a ledger and a hard \\$1.00 cap. About \\$0.00026 per question (v5_final). |
""")
    st.subheader("LangGraph flow for one turn")
    st.graphviz_chart(FLOW)


if __name__ == "__main__":  # streamlit runs the script as __main__; tests import pages directly
    cost_meter()
    st.navigation([st.Page(chat_page, title="Chat", icon=":material/chat:", default=True),
                   st.Page(eval_page, title="Evaluation", icon=":material/analytics:"),
                   st.Page(how_page, title="How it works", icon=":material/account_tree:")]).run()
