# SkyDesk Evaluation Set

`queries.yaml` holds **20** real-world-style support queries (casual phrasing, typos like "bereavment", "didnt", "u guys") used to measure retrieval quality, faithfulness, escalation behaviour and first-contact resolution (FCR).

## Schema

| Field | Meaning |
|---|---|
| `id` | q01..q20 |
| `query` | The customer's message, verbatim |
| `category` | `answerable` (single doc), `multi_doc` (needs 2 docs), `exact_match` (error code or fare-class name, where keyword search should shine), `ambiguous` (should ask to clarify or answer per case with a stated assumption), `unanswerable` (not in corpus: say not found and offer a human), `must_escalate` (bereavement, chargeback/fraud/legal: hand to a human, no answer attempt) |
| `expected_doc_ids` | Doc ids that contain the answer (empty for unanswerable). Use for recall@k / MRR. |
| `expected_behavior` | `answer`, `clarify` or `escalate` |
| `key_facts` | Short phrases a correct answer must contain. Each appears literally (case-insensitive, whitespace-normalised) in at least one expected doc. For `clarify` items they are the facts a good per-case answer would mention. |

## Mix

| Category | Count | Ids |
|---|---|---|
| answerable | 8 | q01-q08 |
| multi_doc | 3 | q09-q11 |
| exact_match | 3 | q12-q14 |
| ambiguous | 2 | q15-q16 |
| unanswerable | 2 | q17-q18 |
| must_escalate | 2 | q19-q20 |

q07 also tests freshness: the superseded 2024 policy gives $150 for a 5-hour delay; the correct answer is the current $250 travel credit.

## Suggested scoring

- **Retrieval:** hit@k and recall@k against `expected_doc_ids` (skip unanswerable).
- **Behaviour accuracy:** predicted answer/clarify/escalate vs `expected_behavior`.
- **Key-fact coverage:** fraction of `key_facts` present in the answer.
- **Faithfulness:** LLM judge, every claim supported by retrieved context.
- **First-contact resolution:** a query counts as resolved when the behaviour is correct and, for `answer` items, all key facts are present and the answer is faithful; a correct escalation on `must_escalate` / `unanswerable` also counts as correctly handled.

## Validate

`.venv/bin/python eval/validate_dataset.py` checks front-matter, unique ids, that every expected doc exists and that every key fact is literally supported. No API calls.
