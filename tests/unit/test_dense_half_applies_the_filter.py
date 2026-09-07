"""A filter both halves of a hybrid were given, and one of them used.

The dense adapter's `retrieve` took `filters`, the protocol declared it, and
the body called the store with no filter at all. The lexical adapter beside it
built a term clause for every key. So a question asked with a filter that
matches nothing came back with an empty page from one half and a full page
from the other, and a caller who asked to narrow the search was answered from
fragments the filter excluded.

Found while trying to stage the catalogue's entry for a filter that excludes
everything, which could not be staged on this platform because on this half a
filter excluded nothing.

The stub is tested here and the real adapter on the proving ground, and both
are tested, because a stub that ignores a filter the adapter applies makes
every unit test above it agree with itself.
"""
from __future__ import annotations

import ast
from pathlib import Path

from adapters.qdrant import QdrantRetrieverStub
from core.models import Chunk

ROOT = Path(__file__).resolve().parents[2]


def _store() -> QdrantRetrieverStub:
    store = QdrantRetrieverStub()
    store.upsert(
        [Chunk(chunk_id="a", doc_id="01.md", text="Плановое обслуживание узла.",
               metadata={"source_code": "handbook"}),
         Chunk(chunk_id="b", doc_id="02.md", text="Перенос срока обслуживания.",
               metadata={"source_code": "other"})],
        [[1.0, 0.0], [0.9, 0.1]],
    )
    return store


def test_a_filter_naming_a_field_of_the_fragment_narrows_the_result() -> None:
    hits = _store().retrieve(query="обслуживание", k=10, query_vector=[1.0, 0.0],
                             filters={"doc_id": "02.md"})
    assert [h.chunk.chunk_id for h in hits] == ["b"]


def test_a_filter_naming_a_field_of_the_metadata_narrows_it_too() -> None:
    hits = _store().retrieve(query="обслуживание", k=10, query_vector=[1.0, 0.0],
                             filters={"source_code": "handbook"})
    assert [h.chunk.chunk_id for h in hits] == ["a"]


def test_a_filter_matching_nothing_returns_nothing() -> None:
    """The half of this that the catalogue is about: a filter that excludes
    everything has to be able to, or the entry for it cannot be staged."""
    hits = _store().retrieve(query="обслуживание", k=10, query_vector=[1.0, 0.0],
                             filters={"doc_id": "a-document-nobody-loaded"})
    assert hits == []


def test_no_filter_still_returns_everything() -> None:
    """The silent half. A filter applied where none was asked for narrows a
    search nobody meant to narrow, and reads as a retrieval failure."""
    hits = _store().retrieve(query="обслуживание", k=10, query_vector=[1.0, 0.0])
    assert len(hits) == 2
    assert _store().retrieve(query="x", k=10, query_vector=[1.0, 0.0], filters={}) != []


def test_the_real_adapter_hands_the_filter_to_the_store() -> None:
    """Read off the source, because the adapter needs a server and this is
    exactly the line whose absence nobody noticed: the call was there, the
    parameter was in the signature, and the two were not connected."""
    tree = ast.parse((ROOT / "adapters" / "qdrant.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "query_points"]
    assert calls, "the adapter no longer queries the store by that name"
    for call in calls:
        assert "query_filter" in {kw.arg for kw in call.keywords}, (
            f"the query at line {call.lineno} passes no filter, so the parameter is decorative"
        )
