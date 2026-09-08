"""A read that names no limit gets every document, and used to get a thousand.

The adapter asked the driver for `limit or 1000`, so a collection with more in
it came back short and looked complete: the right shape, the right documents,
fewer of them, and nothing anywhere saying so.

Found by counting, where reading the adapter had not found it. The windows
that moved out
of the run documents are 3795 documents on the store this was written against,
so one query for every run's windows answered for 24 runs of 129 and the other
105 were served as runs that recorded no window at all, which is exactly what a
run genuinely missing one looks like.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

import adapters.mongodb as mdb


class _Cursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.asked_for: Any = "never asked"
        self.limited_to: int | None = None

    def sort(self, sort: Any) -> _Cursor:
        return self

    def limit(self, n: int) -> _Cursor:
        self.limited_to = n
        return self

    async def to_list(self, length: Any) -> list[dict[str, Any]]:
        """The driver's own contract: `None` is every document, a number is at
        most that many."""
        self.asked_for = length
        documents = self.documents if self.limited_to is None else self.documents[:self.limited_to]
        return documents if length is None else documents[:length]


class _Collection:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.cursor = _Cursor(documents)

    def find(self, query: Any, projection: Any = None) -> _Cursor:
        return self.cursor


@pytest.fixture
def collection(monkeypatch: pytest.MonkeyPatch) -> _Collection:
    """More documents than the number that used to be the ceiling."""
    fake = _Collection([{"n": i} for i in range(2500)])
    monkeypatch.setattr(mdb, "get_collection", lambda name: fake)
    return fake


def test_every_document_comes_back(collection: _Collection) -> None:
    found = asyncio.run(mdb.find_many("anything"))
    assert len(found) == 2500, (
        f"a read with no limit returned {len(found)} of 2500, so a caller reading a large "
        "collection is served a subset that looks like the whole of it"
    )
    assert collection.cursor.asked_for is None, "the driver was asked for a bounded page"


def test_a_limit_that_was_asked_for_is_still_a_limit(collection: _Collection) -> None:
    """The half that keeps the fix from turning paging off. A caller that names
    a limit means it, and one query for a page must not read a collection."""
    found = asyncio.run(mdb.find_many("anything", limit=10))
    assert len(found) == 10
    assert collection.cursor.limited_to == 10
    assert collection.cursor.asked_for == 10
