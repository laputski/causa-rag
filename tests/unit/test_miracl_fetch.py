"""eval/miracl/fetch.py — the identity chain and the retrieval pool.

Two claims carry the Configuration Report. The first is that a MIRACL
docid survives the trip through the filesystem and comes back as the same
string the golden set names, which is checked here against the real
ingestion reader rather than a restatement of it. The second is that the
pool keeps the passages a human judged and rejected: without them every
document in the corpus answers every question, and configurations that
differ in the field would come out equal on paper.

Nothing here touches the network. The two readers that do are replaced.
"""
from __future__ import annotations

import json

import pytest

from eval.miracl import fetch
from services.ingestion.cli import _read_file

# Two questions. q1 has two relevant passages and two rejected ones,
# q2 has one of each, and q3 was judged but nothing was relevant.
_TOPICS = {"q1": "first question", "q2": "second question", "q3": "unanswerable"}
_QRELS = {
    "q1": [("151236#0", 1), ("151236#1", 1), ("77#3", 0), ("904#0", 0)],
    "q2": [("42#7", 1), ("151236#1", 0)],
    "q3": [("500#0", 0)],
}


@pytest.fixture
def slice_(monkeypatch):
    monkeypatch.setattr(fetch, "read_topics", lambda lang: dict(_TOPICS))
    monkeypatch.setattr(fetch, "read_qrels", lambda lang: (
        {q: [d for d, r in rows if r > 0] for q, rows in _QRELS.items() if any(r > 0 for _, r in rows)},
        {q: [d for d, _ in rows] for q, rows in _QRELS.items()},
    ))
    return fetch.build_slice("ar")


def test_docid_becomes_the_ref_id_ingestion_will_produce(tmp_path):
    """The whole reason for this layout. Checked through the real reader in
    services/ingestion/cli.py, so a change to how source_code or article_no
    is derived breaks this test instead of the report's numbers."""
    fetch.write_passage(tmp_path, "151236#1", "A title", "Some passage text.")

    doc = _read_file(tmp_path / "151236" / "1.txt")

    assert doc.metadata["source_code"] == "151236"
    assert doc.metadata["article_no"] == "1"
    ref = f"{doc.metadata['source_code']}/{doc.metadata['article_no']}"
    assert ref == "/".join(fetch.split_docid("151236#1"))


def test_passage_keeps_its_title(tmp_path):
    """MIRACL stores the article title apart from the passage. A passage
    that arrives without it loses the term most likely to match a query
    that names the subject."""
    fetch.write_passage(tmp_path, "7#0", "Damascus", "It is the capital.")
    text = (tmp_path / "7" / "0.txt").read_text(encoding="utf-8")
    assert text.startswith("Damascus\n\n")
    assert "It is the capital." in text


def test_passage_without_a_title_has_no_leading_blank_line(tmp_path):
    fetch.write_passage(tmp_path, "7#0", "", "Body only.")
    assert (tmp_path / "7" / "0.txt").read_text(encoding="utf-8") == "Body only.\n"


def test_pool_keeps_the_rejected_passages(slice_):
    """The distractors. Their absence would not fail anything — it would
    quietly raise every recall number and flatten the differences the
    report exists to show."""
    assert slice_.judged == {"151236#0", "151236#1", "77#3", "904#0", "42#7", "500#0"} - {"500#0"}
    assert slice_.n_relevant == 3


def test_question_with_no_relevant_passage_is_dropped(slice_):
    """q3 was judged and nothing was relevant. Scoring it would count a
    question no configuration can answer as every configuration's failure."""
    assert "q3" not in slice_.questions
    assert set(slice_.questions) == {"q1", "q2"}


def test_limit_narrows_the_pool_with_the_questions(monkeypatch):
    monkeypatch.setattr(fetch, "read_topics", lambda lang: dict(_TOPICS))
    monkeypatch.setattr(fetch, "read_qrels", lambda lang: (
        {q: [d for d, r in rows if r > 0] for q, rows in _QRELS.items() if any(r > 0 for _, r in rows)},
        {q: [d for d, _ in rows] for q, rows in _QRELS.items()},
    ))
    sl = fetch.build_slice("ar", limit=1)

    assert set(sl.questions) == {"q1"}
    # Still carries q1's own rejected passages, which is what makes a small
    # dry run measure anything at all.
    assert sl.judged == {"151236#0", "151236#1", "77#3", "904#0"}


def test_unlayoutable_docids_are_named_not_skipped():
    bad = fetch.unlayoutable({"151236#1", "abc#1", "7#", "#3", "12#4"})
    assert bad == ["#3", "7#", "abc#1"]


def test_no_unlayoutable_docids_in_a_clean_set():
    assert fetch.unlayoutable({"151236#1", "7#0", "42#12"}) == []


def test_golden_set_has_no_ground_truth(slice_, tmp_path):
    """MIRACL ships no reference answers. Writing an empty string instead
    would let answer-similarity report a confident zero on every question."""
    path = tmp_path / "miracl-ar.v1.fast.jsonl"
    fetch.write_golden(slice_, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all("ground_truth" not in r for r in rows)
    assert all("reference_answer" not in r for r in rows)


def test_golden_refs_are_the_positives_in_layout_form(slice_, tmp_path):
    path = tmp_path / "g.jsonl"
    fetch.write_golden(slice_, path)
    rows = {json.loads(x)["id"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}

    assert rows["q1"]["article_refs"] == ["151236/0", "151236/1"]
    assert rows["q2"]["article_refs"] == ["42/7"]
    assert rows["q1"]["question"] == "first question"


def test_golden_set_is_written_atomically(slice_, tmp_path):
    """A half-written golden set read by a run would score against a
    truncated answer key and look like a retrieval regression."""
    path = tmp_path / "g.jsonl"
    fetch.write_golden(slice_, path)
    assert not list(tmp_path.glob("*.tmp"))


def test_only_the_dev_split_is_used():
    """test-a and test-b withhold their judgements, so a run against them
    scores every configuration at zero and reads as a broken pipeline."""
    assert fetch.SPLIT == "dev"
