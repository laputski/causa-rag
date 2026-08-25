"""Moving a realm between installations as one file.

What is checked is what an edit easily loses: secrets do not travel by default,
`dry_run` genuinely writes nothing, and a clash of ids does not merge two run
histories into one.
"""
import pytest

import adapters.mongodb as mdb
from services.api_gateway.routers import realms as R


class FakeMongo:
    """A small Mongo stand-in with real matching semantics: an absent field never
    equals the value being asked for."""

    def __init__(self, data: dict[str, list[dict]]):
        self.data = {k: [dict(d) for d in v] for k, v in data.items()}

    def _match(self, doc, query):
        return all(doc.get(k) == v for k, v in query.items())

    async def find_one(self, collection, query=None):
        for d in self.data.get(collection, []):
            if self._match(d, query or {}):
                return dict(d)
        return None

    async def find_many(self, collection, query=None):
        return [dict(d) for d in self.data.get(collection, []) if self._match(d, query or {})]

    async def insert_one(self, collection, doc):
        self.data.setdefault(collection, []).append(dict(doc))

    async def update_one(self, collection, query, update):
        for d in self.data.get(collection, []):
            if self._match(d, query):
                d.update(update.get("$set", {}))


@pytest.fixture
def fake(monkeypatch):
    store = FakeMongo({
        "realms": [{
            "id": "demo", "name": "Demo", "description": "",
            "resources": [
                {"type": "qdrant", "host": "localhost", "port": 6333},
                {"type": "neo4j", "uri": "bolt://localhost:7687", "password": "ragplatform"},
            ],
            "created_at": "2026-01-01T00:00:00Z",
        }],
        "external_rags": [{"id": "r1", "name": "demo-rag", "realm_id": "demo"}],
        "prompts": [{"id": "demo_prompt_v1", "realm_id": "demo"}],
        "generation_presets": [{"id": "p1", "realm_id": "demo"}],
        "datasets": [{"filename": "handbook.v1.fast.jsonl", "realm_id": "demo",
                      "questions": [{"id": "q-1", "question": "?"}]}],
        "corpora": [{"corpus_id": "handbook", "realm_id": "demo", "description": "The staff handbook"}],
        "settings": [{"realm_id": "demo", "model": "qwen3:8b"}],
    })
    for name in ("find_one", "find_many", "insert_one", "update_one"):
        monkeypatch.setattr(mdb, name, getattr(store, name))
    return store


# @lat: [[realm#Realm#Выгрузка и загрузка реалма#Секреты маскируются по умолчанию]]
@pytest.mark.asyncio
async def test_export_masks_secrets_by_default(fake):
    """An export file is something people email. A password inside it is a
    password sent by email, whatever the sender intended."""
    bundle = await R.export_realm("demo")
    neo4j = next(r for r in bundle["realm"]["resources"] if r["type"] == "neo4j")
    assert neo4j["password"] == R._MASK
    # The list is required: a connection with dots instead of a password looks
    # configured and does not work, and the receiving side must learn that before
    # the first run rather than from its failure.
    assert bundle["masked_fields"] == ["neo4j.password"]


# @lat: [[realm#Realm#Выгрузка и загрузка реалма#Корпус переносится ссылкой]]
@pytest.mark.asyncio
async def test_export_carries_corpora_by_reference_only(fake):
    """A corpus's contents live in Qdrant and OpenSearch and do not fit in a file
    somebody emails. What travels is a list of what has to be ingested."""
    bundle = await R.export_realm("demo")
    assert bundle["corpora"] == [{"corpus_id": "handbook", "description": "The staff handbook"}]
    assert "chunks" not in str(bundle["corpora"])


# @lat: [[realm#Realm#Выгрузка и загрузка реалма#Круговой перенос]]
@pytest.mark.asyncio
async def test_round_trip_creates_a_separate_realm(fake):
    """Export and import back onto the same installation is the commonest way to
    check the transfer. A clash of ids must not merge two run histories: keeping
    them separate is the whole point of a realm."""
    bundle = await R.export_realm("demo", include_secrets=True)
    report = await R.import_realm(bundle, on_conflict="rename")

    assert report["realm_id"] == "demo-2"
    assert await mdb.find_one("realms", {"id": "demo"}) is not None
    imported = await mdb.find_one("realms", {"id": "demo-2"})
    assert imported["name"] == "Demo (2)"
    assert len(await mdb.find_many("prompts", {"realm_id": "demo-2"})) == 1
    assert len(await mdb.find_many("datasets", {"realm_id": "demo-2"})) == 1


# @lat: [[realm#Realm#Выгрузка и загрузка реалма#Сверка ничего не пишет]]
@pytest.mark.asyncio
async def test_dry_run_writes_nothing(fake):
    """A preview that leaves something written is not a preview."""
    bundle = await R.export_realm("demo")
    before = {k: len(v) for k, v in fake.data.items()}
    report = await R.import_realm(bundle, on_conflict="rename", dry_run=True)
    assert report["dry_run"] is True
    assert report["entries"]
    assert {k: len(v) for k, v in fake.data.items()} == before


# @lat: [[realm#Realm#Выгрузка и загрузка реалма#Совпадение без указания отказывает]]
@pytest.mark.asyncio
async def test_conflict_fails_loudly_by_default(fake):
    """By default the import refuses rather than renaming silently: a quiet
    rename hides the fact that somebody is loading a file into the wrong
    place."""
    from fastapi import HTTPException
    bundle = await R.export_realm("demo")
    with pytest.raises(HTTPException) as exc:
        await R.import_realm(bundle)
    assert exc.value.status_code == 409


# @lat: [[realm#Realm#Выгрузка и загрузка реалма#Чужой формат отклоняется]]
@pytest.mark.asyncio
async def test_unknown_format_is_rejected(fake):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await R.import_realm({"format": "keycloak/v1", "realm": {"id": "x"}})
    assert exc.value.status_code == 400
