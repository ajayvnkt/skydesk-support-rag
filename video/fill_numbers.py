"""Fill the {{PLACEHOLDERS}} in video/narration.json with spoken-word numbers -> video/narration.filled.json.

Sources: eval/results.json (run v5_final) and store/cost_ledger.json (every model call's usd).

    python video/fill_numbers.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n: int) -> str:
    """Integer 0..999999 -> words, no hyphens (TTS-friendly)."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))
    return words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + words(n % 1000))


def frac(s: str) -> str:
    """'19/20' -> 'nineteen of twenty'."""
    k, n = (int(x) for x in s.split("/"))
    return f"{words(k)} of {words(n)}"


def pct(x: float) -> str:
    return words(round(x * 100)) + " percent"


def seconds(ms: float) -> str:
    a, b = divmod(round(ms / 100), 10)   # tenths of a second
    return words(a) + ("" if b == 0 else " point " + ONES[b]) + (" second" if (a, b) == (1, 0) else " seconds")


def cents(usd: float) -> str:
    c = round(usd * 100)
    return "under one cent" if c < 1 else f"about {words(c)} cent" + ("" if c == 1 else "s")


def numbers() -> dict:
    res = json.loads((ROOT / "eval/results.json").read_text())
    run = res["runs"]["v5_final"]
    m, mr = run["metrics"], run["manual_review"]

    n_faith = int(mr["fully_faithful"].split("/")[1])
    p, r = m["escalation_precision"], m["escalation_recall"]
    esc = "perfect precision and recall" if p == r == 1 else f"{pct(p)} precision and {pct(r)} recall"

    ledger = json.loads((ROOT / "store/cost_ledger.json").read_text())
    total = sum(x["usd"] for x in ledger)
    assert total < 1.0, f"ledger total ${total:.4f} is not under the one dollar cap"

    return {
        "ROUTE_ACC": frac(mr["route"]),
        "ESC_PR": esc,
        "FAITH_MANUAL": frac(mr["fully_faithful"]),
        "FAITH_JUDGE": frac(f"{round(m['faithfulness_yes_rate'] * n_faith)}/{n_faith}"),  # the console shows "LLM judge 13/16"
        "LATENCY_P50": seconds(m["latency_ms_p50"]),
        "COST_PER_Q": cents(m["bot_cost_usd_per_query"] * 1000) + " per thousand questions",
        "TOTAL_SPEND": cents(total),
    }


def main() -> int:
    assert words(56) == "fifty six" and words(1500) == "one thousand five hundred" and seconds(1500.5) == "one point five seconds"
    nums = numbers()
    for k, v in nums.items():
        print(f"{k:<13} {v}")
    text = (HERE / "narration.json").read_text(encoding="utf-8")
    filled = re.sub(r"\{\{(\w+)\}\}", lambda mt: nums[mt[1]], text)
    left = re.findall(r"\{\{\w+\}\}", filled)
    if left:
        print("unfilled placeholders:", left)
        return 1
    scenes = json.loads(filled)
    long = [(b["key"], len(b["say"].split())) for s in scenes for b in s["beats"] if len(b["say"].split()) > 26]
    total = sum(len(b["say"].split()) for s in scenes for b in s["beats"])
    (HERE / "narration.filled.json").write_text(filled, encoding="utf-8")
    print(f"wrote video/narration.filled.json: {total} words" + (f"; over 26 words: {long}" if long else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
