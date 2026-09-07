"""A registered system says where it sits, and the atlas reads the declaration.

Where a system sits in the space of architectures decides which catalogue
entries can occur in it at all, and a probe cannot see it: there is no request
whose answer tells you whether the thing on the other end fuses two sources.
So it is declared, for the same reason `supported_params` beside it is.

Two halves, and the second is what keeps the first from being decorative. The
declaration is checked against the schema when it arrives, so a typo is an
error at registration and never a system the atlas quietly says nothing
about; and the atlas counts a declared point beside the ones this platform
runs itself, so the declaration decides something.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from services.api_gateway.routers.external_rags import ExternalRagCreateRequest, create_external_rag


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Store:
    def __init__(self, docs: list[dict[str, Any]] | None = None) -> None:
        self.docs = docs or []

    async def insert_one(self, collection: str, doc: dict[str, Any]) -> str:
        self.docs.append(dict(doc))
        return "id"

    async def find_many(self, collection: str, query: dict[str, Any] | None = None,
                        **kw: Any) -> list[dict[str, Any]]:
        return list(self.docs)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    import adapters.mongodb as mdb

    fake = _Store()
    for name in ("insert_one", "find_many"):
        monkeypatch.setattr(mdb, name, getattr(fake, name))
    return fake


def _registration(**overrides: Any) -> ExternalRagCreateRequest:
    return ExternalRagCreateRequest(**{
        "name": "Someone else's hybrid", "url": "http://example.test/rag", **overrides,
    })


def test_a_registration_keeps_the_coordinates_it_declared(store: _Store) -> None:
    doc = _run(create_external_rag(_registration(
        coordinates={"C3": "rrf", "D1": "cross_encoder"})))
    assert doc["coordinates"] == {"C3": "rrf", "D1": "cross_encoder"}


def test_a_dimension_the_schema_does_not_hold_is_refused(store: _Store) -> None:
    """Stored unchecked, an unresolvable coordinate is a coordinate no entry
    will ever match, and the system would read as one no failure can happen
    in. Refusing at registration is the difference between an error and a
    silence."""
    with pytest.raises(HTTPException) as raised:
        _run(create_external_rag(_registration(coordinates={"Z9": "whatever"})))
    assert raised.value.status_code == 400
    assert "Z9" in str(raised.value.detail)


def test_a_value_the_dimension_does_not_take_is_refused(store: _Store) -> None:
    with pytest.raises(HTTPException) as raised:
        _run(create_external_rag(_registration(coordinates={"C3": "telepathy"})))
    assert raised.value.status_code == 400
    assert "telepathy" in str(raised.value.detail)
    assert "rrf" in str(raised.value.detail), "the refusal does not say what would resolve"


def test_declaring_nothing_is_allowed_and_says_nothing(store: _Store) -> None:
    """Empty is an answer: the atlas can then say only what it says about
    every system, which is honest for a system nobody has described."""
    assert _run(create_external_rag(_registration()))["coordinates"] == {}


def test_the_atlas_counts_a_declared_point_beside_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half that keeps the declaration from being decorative."""
    import adapters.mongodb as mdb
    from core.eval import rag_space
    from services.api_gateway.routers.atlas import read_atlas

    fake = _Store([{
        "id": "abc", "name": "Someone else's dense RAG",
        "coordinates": {"A5": "dense_single", "C3": "none", "D1": "none"},
    }])
    monkeypatch.setattr(mdb, "find_many", fake.find_many)

    payload = _run(read_atlas())
    assert "Someone else's dense RAG" in payload["points"]
    declared = payload["points"]["Someone else's dense RAG"]
    assert declared["declared"] is True
    assert declared["applicable"] > 0, "no entry can occur in it, which cannot be right"
    for name in rag_space.POINTS:
        assert payload["points"][name]["declared"] is False

    narrowed = _run(read_atlas(point="Someone else's dense RAG"))
    assert narrowed["point"] == "Someone else's dense RAG"
    assert any(e["applies_here"] is True for e in narrowed["entries"])


def test_a_system_that_declared_nothing_is_not_offered_as_a_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A point with no coordinates would answer every applicability question
    with "nothing is ruled out", which reads as knowledge and is its absence."""
    import adapters.mongodb as mdb
    from core.eval import rag_space
    from services.api_gateway.routers.atlas import read_atlas

    fake = _Store([{"id": "abc", "name": "Undescribed", "coordinates": {}}])
    monkeypatch.setattr(mdb, "find_many", fake.find_many)

    payload = _run(read_atlas())
    assert set(payload["points"]) == set(rag_space.POINTS)
