"""A field added to a connector type reaches an installation that already
has the catalogue.

Found by reading, before any field was added: the seed wrote the catalogue
only into an empty collection, so an installation seeded once kept the
schema of its first start for good. The first field to be added (a remote
store's address or key) would have existed in the code and in no form on
any machine that had run the platform before. The catalogue has no endpoint
that writes it, so the seed in the code is its only source of truth.
"""
from __future__ import annotations

import pytest

import adapters.mongodb as mdb
from services.api_gateway.routers import realms as R


class _Store:
    def __init__(self, docs):
        self.docs = [dict(d) for d in docs]

    async def count(self, collection, query=None):
        return len(self.docs)

    async def upsert_one(self, collection, query, doc):
        self.docs = [d for d in self.docs if d.get("type") != query["type"]]
        self.docs.append(dict(doc))


@pytest.mark.asyncio
async def test_a_seeded_type_takes_the_schema_the_code_declares(monkeypatch):
    old = {"type": "qdrant", "params_schema": {"host": {"type": "string"}}}
    store = _Store([old])
    monkeypatch.setattr(mdb, "count", store.count)
    monkeypatch.setattr(mdb, "upsert_one", store.upsert_one)

    await R.seed_connector_types()

    declared = next(ct for ct in R._CONNECTOR_TYPES_SEED if ct["type"] == "qdrant")
    stored = next(d for d in store.docs if d["type"] == "qdrant")
    assert stored == declared
    assert {d["type"] for d in store.docs} >= {ct["type"] for ct in R._CONNECTOR_TYPES_SEED}
