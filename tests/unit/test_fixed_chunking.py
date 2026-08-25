import pytest

from core.chunking.fixed import FixedChunkingStrategy
from core.interfaces import ChunkingStrategy
from core.models import Document


def _doc(text: str) -> Document:
    return Document(source="test.txt", content=text)


def test_satisfies_protocol():
    assert isinstance(FixedChunkingStrategy(), ChunkingStrategy)


def test_single_chunk_short_text():
    s = FixedChunkingStrategy(chunk_size=100, overlap=10)
    chunks = s.chunk(_doc("hello"))
    assert len(chunks) == 1
    assert chunks[0].text == "hello"


def test_multiple_chunks():
    s = FixedChunkingStrategy(chunk_size=10, overlap=2)
    text = "a" * 25
    chunks = s.chunk(_doc(text))
    assert len(chunks) > 1
    assert all(c.strategy_id == "fixed" for c in chunks)


def test_overlap_preserved():
    s = FixedChunkingStrategy(chunk_size=10, overlap=5)
    text = "0123456789ABCDEFGHIJ"
    chunks = s.chunk(_doc(text))
    # second chunk starts at offset 5, so it starts with '56789'
    assert chunks[1].text[:5] == "56789"


def test_covers_full_text():
    s = FixedChunkingStrategy(chunk_size=10, overlap=0)
    text = "x" * 30
    chunks = s.chunk(_doc(text))
    assert sum(len(c.text) for c in chunks) == 30


def test_empty_text():
    s = FixedChunkingStrategy()
    chunks = s.chunk(_doc(""))
    assert chunks == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        FixedChunkingStrategy(chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        FixedChunkingStrategy(chunk_size=10, overlap=10)


def test_chunk_doc_id_propagated():
    doc = _doc("some text here")
    s = FixedChunkingStrategy(chunk_size=5, overlap=0)
    for c in s.chunk(doc):
        assert c.doc_id == doc.doc_id
