# Week 2 handout compliance check

An independent check of this repo against `Week 2 Project Handout ( Oct 2026 ).md`, made on 2026-10-18. Each row gives a requirement from the handout, where the evidence is, and a verdict.

## Checks run for this pass

- `pytest -q`: 101 passed, before and after the doc edits.
- `python -m rag ask "What does SD-ERR-112 mean?"`: route `answer`, confidence 0.925, cites `[manual-error-codes]`, 2.2 s, $0.000334.
- `python -m rag ask "my grandmother passed away, can I get a refund on my flight?"`: route `escalate` (bereavement guard), $0, 14 ms.
- Customer site on port 9006: `GET /` returns 200 and `/api/health` returns `{"ok":true}`. Ops console on port 9007: 200, also after its restart.
- Five numbers in `PROJECT_DOC.md` checked against `eval/results.json` → `runs.v5_final`. All five match:

  | Number | Value |
  |---|---|
  | Routes | 0.95 = 19/20 |
  | Manual first-contact resolution | 12/16; judge 10/16 |
  | Faithfulness | manual 15/16; judge 13/16, 0.812, score 0.844 |
  | Relevance | 4.06 |
  | Latency p50 / p95 | 1500.5 / 1781.3 ms |
  | Cost per query | $0.000265 |
  | Hybrid + rerank MRR | 0.839 |
  | Index | 55 docs / 206 chunks |

- Ledger: $0.108002 over 757 calls. That is $0.000334 more than the $0.107668 (before the demo video; $0.1139 in total with the video recording) / 755 calls the docs quote, and the difference is the two verification queries above.

## Compliance table

| # | Requirement (handout) | Evidence | Verdict |
|---|---|---|---|
| **One-liner** | | | |
| 1 | Uses the template: user, question type, corpus, surface, % faithfulness and/or % relevance | `docs/PROJECT_DOC.md:17`, `README.md:17`, `docs/EVAL_REPORT.md:5`, `app.py:20` | PASS |
| 2 | Rule 1: names the corpus specifically | "SkyDesk help center (55 docs: policies, FAQs, app manual, resolved tickets)"; full breakdown at `PROJECT_DOC.md:28` | PASS |
| 3 | Rule 2: measures faithfulness, not just relevance, **and sets a target %** | Measured 94% (15/16, manual review) and judge 81%. **Fixed in this pass:** before it, no target was stated anywhere, only measured values. The one-liner now says "against a 90% target" in all four places and in the .docx | PASS (after fix; see note 1) |
| 4 | Rule 3: picks a latency ceiling | "under a 3-second latency ceiling (measured p95 1.78 s)" | PASS |
| **Framework fields (1 to 2 sentences each)** | | | |
| 5 | Use case | `PROJECT_DOC.md:27` | PASS |
| 6 | Corpus: sources, count, formats, language, owner | `PROJECT_DOC.md:28` | PASS |
| 7 | Ingestion + cleaning: markup, boilerplate, entities | `PROJECT_DOC.md:29`; code in `rag/ingest.py` (HTMLParser, `html` unescape) | PASS |
| 8 | Ingestion + freshness: how docs get in, **refresh cadence**, freshness SLA | `PROJECT_DOC.md:30`. **Fixed:** this row had no cadence and ran to 5 sentences. It now says "re-index on demand whenever the policy team publishes (manual today)" and gives a same-day SLA | PASS (after fix) |
| 9 | Chunking + embedding: strategy, size, model, why | `PROJECT_DOC.md:31`; `rag/ingest.py:204` (`RecursiveCharacterTextSplitter`), ~350 tokens / 50 overlap, `text-embedding-3-small` | PASS |
| 10 | Retrieve: store, dense/sparse/hybrid, top-k | `PROJECT_DOC.md:32`; `rag/index.py:41` (`rrf`), `:77` (`search`), top 20 + 20 → RRF 12 → rerank 4 | PASS |
| 11 | "1 to 2 sentences max" per field | **Fixed:** 4 of the 6 rows ran to 3 to 5 sentences; all 6 are now 1 to 2 sentences, in the .md and the .docx | PASS (after fix) |
| 12 | Tip: design the "I don't know" path first | `PROJECT_DOC.md` §8; `rag/graph.py:349` classify guards, `:402` grade threshold, `:440` verify | PASS |
| **Track 2: LangChain + LangGraph** | | | |
| 13 | LangGraph state machine | `rag/graph.py:20` `from langgraph.graph import END, StateGraph`; `:481-496` builds 8 nodes with `add_node` and 3 `add_conditional_edges`; invoked in `rag/pipeline.py:53` | PASS |
| 14 | LangChain primitives | `langchain_core.documents.Document` (`rag/ingest.py:25`, `rag/index.py:11`); `RecursiveCharacterTextSplitter` (`rag/ingest.py:203-204`); `BaseRetriever` subclass `HybridRetriever` (`rag/index.py:12`, `:97`); `ChatPromptTemplate.from_messages` for all 5 prompts (`rag/prompts.py:11, 30, 33, 50, 62, 89`) | PASS |
| 15 | Handout's other building blocks: Embeddings/VectorStore classes, LCEL chains, LangSmith | Not used. `ChatOpenAI`/`OpenAIEmbeddings` are skipped on purpose so that every call goes through the cost-capped `rag/llm.py` (reason given at `PROJECT_DOC.md:61`). There is no LangSmith; tracing uses the ARL run log instead. The handout lists these as building blocks, not requirements | PASS (with documented deviation) |
| **Use case 4: description** | | | |
| 16 | Hybrid (keyword + semantic) retrieval | `rag/index.py:1-2, 13, 63` (BM25Okapi) plus numpy cosine, fused by RRF; ablation at `PROJECT_DOC.md:157-166` | PASS |
| 17 | Over support tickets, FAQs **and** product manuals | Corpus: 24 policies, 9 FAQs, 9 app-manual pages, 13 resolved tickets (`results.json` index: 55 docs / 206 chunks) | PASS |
| 18 | Confidence-based fallback that escalates to a human instead of hallucinating | `rag/graph.py:28` `CONFIDENCE_THRESHOLD = 0.5`, `:408` low support leads to escalate, plus a hand-off ticket. Escalation precision and recall are 1.00; unanswerable 2/2 and must-escalate 2/2 | PASS |
| 19 | Tested with 20 real-world-style support queries | `eval/queries.yaml` (20 queries, typos and casual phrasing); `eval/validate_dataset.py` | PASS |
| 20 | Measures first-contact resolution | `eval/run_eval.py:130-157`; reported in `PROJECT_DOC.md:189-200` (12/16 manual, 10/16 judge) | PASS |
| 21 | Overview table: "Hybrid + **multilingual**" retrieval pattern | English only. The corpus, the eval set and the regex guards are all English. Disclosed as a limitation (`PROJECT_DOC.md` §13, `README.md` Limitations). The handout's detailed description of use case 4 does not mention multilingual | PARTIAL (disclosed) |
| **Use case 4: deliverable and submission** | | | |
| 22 | Working support bot with hybrid search + escalation logic | Live: two `rag ask` runs above, site on 9006, console on 9007 | PASS |
| 23 | Tested against 20 queries with resolution metrics | `docs/EVAL_REPORT.md`, `eval/results.json` v5_final | PASS |
| 24 | Submission: demo recording | Not made yet. `README.md:12` says "to be added" | FAIL (open: video) |
| 25 | Submission: GitHub link or zip | See row 30 | FAIL (open: push) |
| 26 | Submission: evaluation metrics document | `docs/EVAL_REPORT.md`; also `PROJECT_DOC.md` §7 | PASS |
| **Week 2 deliverables** | | | |
| 27 | Project documentation as a **Google Doc** | Content is ready in `docs/PROJECT_DOC.md` and `submission/SkyDesk_Support_RAG_Week2.docx` (7 images; patched in this pass with the target and the tightened framework table). It has not been uploaded to Google Docs yet | PARTIAL (open: upload the .docx to Google Drive and open it as a Google Doc) |
| 28 | Doc includes overview, datasets, prompts or agent instructions, iterations, learnings | `PROJECT_DOC.md` §1, §4, §5 (5.1 Claude Code prompts, 5.2-5.5 system prompts), §6, §12 | PASS (note 2) |
| 29 | Video of 5 minutes or less: walkthrough, what you built, how you used AI coding tools, live demo | Not started (out of scope for this pass). `video/narration.json` still has v4 numbers | FAIL (open: video) |
| 30 | Code uploaded to GitHub, link shared in the form | The README links `github.com/ajayvnkt/skydesk-support-rag`, but the GitHub API returns 404, there is no `git remote`, and `main` has no commits. `.gitignore` covers `.env`, `.venv/`, `store/`, `submission/` and local logs | FAIL (open: create the repo, commit, push; nothing was committed in this pass) |
| **Bonus** | | | |
| 31 | Chatbot UI connected to the RAG system | `web/` + `server.py:71` `POST /api/ask` (FastAPI, port 9006); Streamlit console `app.py` (port 9007). Both return 200 | PASS |

## Notes

1. **The 90% faithfulness target was added during this check.** No earlier doc stated one. The handout asks you to set a target, and 90% is below the measured 94% from manual review but above the judge's 81%. If you would rather state a different target, or say plainly that it was set after the fact, change the one-liner in `README.md:17`, `docs/PROJECT_DOC.md:17`, `docs/EVAL_REPORT.md:5`, `app.py:22` and the .docx.
2. `PROJECT_DOC.md` §5.1 lists the prompts given to Claude Code. They are reconstructed and condensed, and only item 9 is verbatim, so check them before you submit.
3. The ledger figure in the docs ($0.107668 (before the demo video; $0.1139 in total with the video recording), 755 calls) is a snapshot taken before this check, which added $0.000334. Any later UI testing will move the ledger again.

## What remains (all need you; none can be done without a commit, a push or a recording)

1. **Video demo** (5 minutes or less). Refresh `video/narration.json` from v4 to v5_final numbers first.
2. **GitHub.** Create `ajayvnkt/skydesk-support-rag`, commit and push, then confirm the README link resolves.
3. **Google Doc.** Upload `submission/SkyDesk_Support_RAG_Week2.docx` to Drive, open it as a Google Doc, and share the link.
4. Optional: multilingual support (row 21) is disclosed but not implemented.
