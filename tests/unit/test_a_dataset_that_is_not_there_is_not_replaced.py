"""A dataset that is named and not found is reported, never replaced.

Found live, re-running a stored run of a realm whose datasets live in the
store: the run document keeps the dataset's short name, the lookup by name
cut it at the first dot, and the loader returned five invented questions
instead of saying so. The re-run finished green with no retrieval metric at
all, and the comparison and resampling paths, which load a stored run's
dataset by that short name, had been measuring the stub the same way.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from services.api_gateway.routers.experiments import DatasetNotFound, _load_dataset

_STORED = {"filename": "notes.v0.fast.jsonl", "name": "notes.v0.fast", "version": "v0",
           "speed": "fast", "questions": [{"id": "q1", "question": "?", "article_refs": ["A/1"]}]}


async def _find_one(collection, query):
    if collection != "datasets":
        return None
    if query.get("filename") == _STORED["filename"] or query.get("name") == _STORED["name"]:
        return dict(_STORED)
    return None


def test_a_stored_run_s_short_name_finds_its_dataset() -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one)):
        dataset = asyncio.run(_load_dataset("notes.v0.fast"))
    assert [q["id"] for q in dataset.questions] == ["q1"]


def test_a_name_that_matches_nothing_is_reported() -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one)), \
         pytest.raises(DatasetNotFound):
        asyncio.run(_load_dataset("notes.v9.missing"))


def test_the_stub_is_still_there_for_whoever_asks_for_it() -> None:
    assert asyncio.run(_load_dataset("stub")).name == "stub"
