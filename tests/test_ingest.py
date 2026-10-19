from rag.ingest import (CHUNK_TOKENS, OVERLAP_TOKENS, chunk_corpus, chunk_doc, clean_html,
                        count_tokens, load_corpus, split_sections)
from conftest import FIXTURES


def test_load_corpus_metadata():
    docs = {d["doc_id"]: d for d in load_corpus(FIXTURES)}
    assert set(docs) == {"faq-baggage", "policy-refunds", "policy-refunds-2024", "ticket-4821",
                         "manual-checkin"}
    assert docs["faq-baggage"]["doc_type"] == "faq"
    assert docs["faq-baggage"]["updated"] == "2026-08-01"
    assert docs["policy-refunds-2024"]["superseded"] is True
    assert docs["policy-refunds"]["superseded"] is False
    assert not docs["faq-baggage"]["text"].startswith("---")


def test_html_cleaning():
    d = {x["doc_id"]: x for x in load_corpus(FIXTURES)}["manual-checkin"]
    t = d["text"]
    assert d["title"] == "Online Check-in" and d["doc_type"] == "manual"
    assert "SHOULD_NOT_APPEAR" not in t and "color:red" not in t  # script/style dropped
    assert "Sign in" not in t and "Refunds" not in t               # header/nav dropped
    assert "All rights reserved" not in t and "helpful" not in t   # footer/boilerplate dropped
    assert "24 hours before departure & closes" in t               # entities decoded
    assert "it’s also emailed" in t and "—" in t
    assert "# Online check-in" in t and "## Boarding pass" in t    # headings preserved
    assert "  " not in t                                           # whitespace collapsed


def test_clean_html_entities_and_whitespace():
    title, text, meta = clean_html("<title>X &amp; Y</title><p>a&nbsp;&nbsp; b\n\n\n\n c</p>")
    assert title == "X & Y" and "a b" in text and "\n\n\n" not in text


def test_split_sections_paths():
    secs = split_sections("intro\n# A\none\n## B\ntwo\n# C\nthree")
    assert secs == [("", "intro"), ("A", "one"), ("A > B", "two"), ("C", "three")]


def test_chunking_sizes_and_whole_units():
    chunks = chunk_corpus(load_corpus(FIXTURES))
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
        assert set(c) >= {"chunk_id", "doc_id", "title", "section", "doc_type", "updated",
                          "superseded", "text"}
    # FAQ: one chunk per Q&A, question kept as section
    assert len(by_doc["faq-baggage"]) == 2
    assert by_doc["faq-baggage"][1]["section"].startswith("What happens if my bag")
    assert "100 USD per day" in by_doc["faq-baggage"][1]["text"]
    # ticket: whole, problem and resolution together
    assert len(by_doc["ticket-4821"]) == 1
    assert "Problem" in by_doc["ticket-4821"][0]["text"] and "3-D Secure" in by_doc["ticket-4821"][0]["text"]
    # long policy section split under the cap with overlap
    long = [c for c in by_doc["policy-refunds"] if c["section"].endswith("Long section")]
    assert len(long) >= 2
    assert all(count_tokens(c["text"]) <= CHUNK_TOKENS for c in long)
    assert by_doc["policy-refunds-2024"][0]["superseded"] is True
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_faq_section_kept_whole_even_if_long():
    long_answer = "word " * 1000
    doc = {"doc_id": "f", "title": "F", "doc_type": "faq", "updated": "", "superseded": False,
           "tags": [], "text": f"## Q?\n{long_answer}"}
    assert len(chunk_doc(doc)) == 1
    doc["doc_type"] = "manual"
    assert len(chunk_doc(doc)) > 1


def test_html_hyphen_meta_cookie_banner_and_table_cells():
    raw = ('<html><head><title>Pets | SkyDesk</title><meta name="doc-id" content="pets">'
           '<meta name="doc-type" content="policy"></head><body>'
           '<div id="cookie-banner" class="banner"><div>We use cookies here</div><button>OK</button></div>'
           '<h1>Pets</h1><table><tr><th>Service</th><th>Fee</th></tr>'
           '<tr><td>Pet in cabin</td><td>$95&nbsp;each way</td></tr></table></body></html>')
    title, text, meta = clean_html(raw)
    assert meta["doc_id"] == "pets" and meta["doc_type"] == "policy"
    assert "cookies" not in text
    assert "Pet in cabin | $95 each way" in text
