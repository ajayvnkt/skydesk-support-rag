# Agent Report Card

**File format:** ARL JSONL (canonical) (confidence 0.98) -- 20 runs, 195 events.
**Policy:** default (generic) (built-in generic default).  **Success score:** trained model.

## Headline

- Runs: **20**; agent said "done" in **65%**.
- Mean predicted success: **89%** (no ground-truth outcomes in this file).
- Runs with a critical finding: **0**; said done although something failed / never happened: **0**.
- Cost: **$0.0054** total, $0.0003 per run, $0.0003 per expected success. (estimated)

## Autonomy Ladder

| Task | Runs | Success | Rung | Why |
|---|---|---|---|---|
| support_qa | 20 | 89% (predicted) | Human approves | success 89% (predicted) is below 90% |

## Runs most likely to have failed

| Run | Task | P(success) | Outcome | Findings |
|---|---|---|---|---|
| eval-q17 | support_qa | 1% | - | - |
| eval-q18 | support_qa | 1% | - | - |
| eval-q05 | support_qa | 98% | - | - |
| eval-q15 | support_qa | 98% | - | - |
| eval-q16 | support_qa | 98% | - | - |
| eval-q19 | support_qa | 99% | - | - |
| eval-q20 | support_qa | 99% | - | - |
| eval-q01 | support_qa | 99% | - | - |
| eval-q02 | support_qa | 99% | - | - |
| eval-q03 | support_qa | 99% | - | - |
| eval-q04 | support_qa | 99% | - | - |
| eval-q06 | support_qa | 99% | - | - |
| eval-q07 | support_qa | 99% | - | - |
| eval-q08 | support_qa | 99% | - | - |
| eval-q09 | support_qa | 99% | - | - |
