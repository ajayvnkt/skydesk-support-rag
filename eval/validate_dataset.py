"""Offline sanity check for corpus/ and eval/queries.yaml (no API calls).

Checks: front-matter / HTML meta parse, required fields, unique ids, every
expected_doc_id exists, every key_fact appears literally (case-insensitive,
whitespace-normalised) in at least one cited doc. Prints an inventory table.
Usage: python eval/validate_dataset.py [--inventory]
"""
import html
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
DOC_TYPES = {"faq", "policy", "manual", "ticket"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("‑", "-")).strip().lower()


def load_md(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    assert m, f"{path.name}: missing front-matter"
    meta = yaml.safe_load(m.group(1))
    return {**meta, "updated": str(meta["updated"]), "body": m.group(2), "file": path.name}


def load_html(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    meta = dict(re.findall(r'<meta name="([\w-]+)" content="([^"]*)"', raw))
    title = re.search(r"<title>(.*?)</title>", raw, re.S).group(1).split("|")[0].strip()
    main = re.search(r"<main.*?</main>", raw, re.S).group(0)
    body = html.unescape(re.sub(r"<[^>]+>", " ", main))
    return {"id": meta["doc-id"], "title": html.unescape(title), "doc_type": meta["doc-type"],
            "updated": meta["updated"], "tags": [t.strip() for t in meta["tags"].split(",")],
            "body": body, "file": f"html/{path.name}"}


def main() -> int:
    docs = [load_md(p) for p in sorted(CORPUS.glob("*.md")) if p.name != "README.md"]
    docs += [load_html(p) for p in sorted((CORPUS / "html").glob("*.html"))]
    errors = []
    for d in docs:
        for k in ("id", "title", "doc_type", "updated", "tags"):
            if not d.get(k):
                errors.append(f"{d['file']}: missing {k}")
        if d["doc_type"] not in DOC_TYPES:
            errors.append(f"{d['file']}: bad doc_type {d['doc_type']}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["updated"]):
            errors.append(f"{d['file']}: bad updated {d['updated']}")
    ids = Counter(d["id"] for d in docs)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    by_id = {d["id"]: d for d in docs}

    queries = yaml.safe_load((ROOT / "eval" / "queries.yaml").read_text())
    if len(queries) != 20:
        errors.append(f"expected 20 queries, got {len(queries)}")
    for q in queries:
        for did in q["expected_doc_ids"]:
            if did not in by_id:
                errors.append(f"{q['id']}: unknown doc {did}")
        cited = " ".join(norm(by_id[d]["body"]) for d in q["expected_doc_ids"] if d in by_id)
        for fact in q["key_facts"]:
            if norm(fact) not in cited:
                errors.append(f"{q['id']}: key fact not in cited docs: {fact!r}")

    words = {d["id"]: len(re.findall(r"\b\w[\w'$.,-]*", d["body"])) for d in docs}
    if "--inventory" in sys.argv:
        print("| id | type | updated | words | file |\n|---|---|---|---|---|")
        for d in sorted(docs, key=lambda d: (d["doc_type"], d["id"])):
            flag = " (superseded)" if d.get("superseded") else ""
            print(f"| {d['id']}{flag} | {d['doc_type']} | {d['updated']} | {words[d['id']]} | {d['file']} |")
    print(f"\ndocs={len(docs)} by_type={dict(Counter(d['doc_type'] for d in docs))} "
          f"total_words={sum(words.values())} queries={len(queries)} "
          f"categories={dict(Counter(q['category'] for q in queries))}")
    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
