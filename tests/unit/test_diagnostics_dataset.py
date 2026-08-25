"""The deep-diagnostics dataset comes from the database, and not only from a file.

`eval/golden/` holds one demonstration dataset. A Realm's own datasets live in
Mongo, so while diagnostics looked for a file alone, RAGAS and TruLens failed on
every Realm except the demo one with "Control dataset not found". The default
behaved worse: it pointed at that same demonstration file, failed at nothing,
and checked another Realm's corpus with questions about the demo handbook,
honestly returning context_recall = 0.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.api_gateway.routers.corpus import _load_diagnostics_dataset

DOC = {
    "filename": "Cosmos 1.v0.full.jsonl", "name": "Cosmos 1", "version": "v0",
    "speed": "full", "realm_id": "acme",
    "questions": [{"id": "q1", "question": "Was the section changed?", "ground_truth": "Yes"}],
}


@pytest.mark.asyncio
async def test_a_realm_s_own_dataset_is_found_in_mongo() -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=DOC)):
        ds = await _load_diagnostics_dataset("Cosmos 1.v0.full.jsonl", "acme")
    assert ds.name == "Cosmos 1"
    assert len(ds.questions) == 1


@pytest.mark.asyncio
async def test_the_realm_is_part_of_the_lookup() -> None:
    seen: list[dict] = []

    async def find_one(collection: str, query: dict):
        seen.append(query)
        return DOC

    with patch("adapters.mongodb.find_one", find_one):
        await _load_diagnostics_dataset("Cosmos 1.v0.full.jsonl", "acme")
    # This realm's dataset is asked for first: two realms can hold datasets
    # under the same name, and taking whichever comes first means returning
    # another realm's questions.
    assert seen[0] == {"filename": "Cosmos 1.v0.full.jsonl", "realm_id": "acme"}


@pytest.mark.asyncio
async def test_a_dataset_in_neither_place_is_a_404_and_not_a_silent_substitution() -> None:
    from fastapi import HTTPException

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await _load_diagnostics_dataset("nothing-like-this.jsonl", "acme")
    assert exc.value.status_code == 404
