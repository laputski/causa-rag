"""core/chunking/sentence.py — SentenceChunkingStrategy with razdel.sentenize().

No prior test coverage existed for this strategy; added alongside the
razdel swap (replaced a hand-rolled regex boundary that mis-split bare
list/clause numerals off as their own sentence — see
core/chunking/structure_aware.py for the production bug this caused there).
"""
from __future__ import annotations

from core.chunking.sentence import SentenceChunkingStrategy
from core.interfaces import ChunkingStrategy
from core.models import Document


def _doc(content: str) -> Document:
    return Document(source="test.txt", content=content)


def test_satisfies_protocol():
    assert isinstance(SentenceChunkingStrategy(), ChunkingStrategy)


def test_empty_content_returns_no_chunks():
    assert SentenceChunkingStrategy().chunk(_doc("")) == []


def test_groups_sentences_up_to_max_chars():
    sentences = [f"This is sentence number {i} with some text in it." for i in range(10)]
    text = " ".join(sentences)
    chunks = SentenceChunkingStrategy(max_chars=120, min_chars=10).chunk(_doc(text))
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_does_not_split_off_a_bare_numeral_as_its_own_sentence():
    """Same class of bug fixed in structure_aware.py: a numbered clause
    marker like "1." must stay attached to its content, not become its own
    near-empty chunk."""
    text = "1. The first provision with enough text in it for the check. 2. The second provision, also with text."
    chunks = SentenceChunkingStrategy(max_chars=1000, min_chars=1).chunk(_doc(text))
    assert all(c.text.strip() not in ("1.", "2.") for c in chunks)
    assert any(c.text.startswith("1. The first") for c in chunks)


def test_short_trailing_group_dropped_below_min_chars():
    chunks = SentenceChunkingStrategy(max_chars=1000, min_chars=50).chunk(_doc("Short."))
    assert chunks == []
