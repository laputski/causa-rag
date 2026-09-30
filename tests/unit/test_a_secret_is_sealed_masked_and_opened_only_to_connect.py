"""A realm's secret is sealed where it is stored, masked on the way out of the
API, absent from any export, and opened only where a connection is made.

Before this, a Neo4j password stood in the realm document in the clear, came
back in every read of the realm, and travelled in an export whenever
`include_secrets` was asked for.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import adapters.mongodb as mdb
from adapters.fernet_secrets import FernetSecretBox, SecretBoxStub
from core.models import ResourceConfig
from core.secrets import MASK, SEALED, SecretUnavailable, for_storage
from services.api_gateway.routers import corpus as C
from services.api_gateway.routers import realms as R


class _Store:
    def __init__(self, realm: dict):
        self.realms = [dict(realm)]

    async def find_one(self, collection, query=None):
        if collection == "realms":
            return next((dict(r) for r in self.realms if r["id"] == (query or {}).get("id")), None)
        return None

    async def find_many(self, collection, query=None, sort=None):
        return [dict(r) for r in self.realms] if collection == "realms" else []

    async def update_one(self, collection, query, update):
        for r in self.realms:
            if r["id"] == query["id"]:
                r.update(update.get("$set", {}))

    async def count(self, collection, query=None):
        return 0


@pytest.fixture
def store(monkeypatch):
    s = _Store({"id": "acme", "name": "Acme", "resources": [
        {"type": "neo4j", "uri": "bolt://g:7687", "user": "neo4j", "password": "legacy-plain"}]})
    for name in ("find_one", "find_many", "update_one", "count"):
        monkeypatch.setattr(mdb, name, getattr(s, name))
    monkeypatch.setattr(R, "secret_box", lambda: SecretBoxStub())

    async def known_types():
        return {"qdrant", "neo4j", "opensearch", "ollama"}
    monkeypatch.setattr(R, "_valid_connector_types", known_types)
    return s


def _stored_password(store: _Store):
    return next(r for r in store.realms[0]["resources"] if r["type"] == "neo4j").get("password")


def _put(password):
    fields = {"type": "neo4j", "uri": "bolt://g:7687", "user": "neo4j"}
    if password is not ...:
        fields["password"] = password
    return asyncio.run(R.set_realm_resources("acme", [ResourceConfig(**fields)]))


# ── Storing ───────────────────────────────────────────────────────────────────

def test_a_new_secret_is_stored_sealed_and_shown_masked(store) -> None:
    shown = _put("new-password")
    stored = _stored_password(store)
    assert isinstance(stored, dict) and SEALED in stored
    assert "new-password" not in str(store.realms)
    assert next(r for r in shown["resources"] if r["type"] == "neo4j")["password"] == MASK


def test_the_mask_sent_back_keeps_the_secret_and_absence_keeps_it_too(store) -> None:
    """The interface edits what it read, and what it read is the mask."""
    _put("new-password")
    sealed = _stored_password(store)
    _put(MASK)
    assert _stored_password(store) == sealed
    _put(...)
    assert _stored_password(store) == sealed


def test_an_empty_string_clears_the_secret(store) -> None:
    _put("new-password")
    _put("")
    assert _stored_password(store) is None


def test_without_a_key_a_new_secret_is_refused_and_never_stored_in_the_clear(store, monkeypatch) -> None:
    monkeypatch.setattr(R, "secret_box", lambda: FernetSecretBox(key=""))
    with pytest.raises(HTTPException) as exc:
        _put("new-password")
    assert exc.value.status_code == 400
    assert "new-password" not in str(store.realms)


# ── Reading ───────────────────────────────────────────────────────────────────

def test_every_read_of_a_realm_masks_its_secrets_sealed_or_not(store) -> None:
    """The stored value here predates sealing, and is masked all the same."""
    for realm in (asyncio.run(R.get_realm("acme")), asyncio.run(R.list_realms())[0]):
        assert next(r for r in realm["resources"] if r["type"] == "neo4j")["password"] == MASK
    assert "legacy-plain" not in str(asyncio.run(R.list_realms()))


def test_a_connection_gets_the_secret_opened(store) -> None:
    _put("new-password")
    resource = asyncio.run(C._get_realm_resource("acme", "neo4j"))
    assert resource["password"] == "new-password"


def test_a_secret_stored_before_sealing_still_connects(store) -> None:
    assert asyncio.run(C._get_realm_resource("acme", "neo4j"))["password"] == "legacy-plain"


def test_a_secret_that_cannot_be_opened_stops_the_connection_and_says_why(store, monkeypatch) -> None:
    """Handed on empty, the graph adapter would connect with its own default."""
    _put("new-password")
    monkeypatch.setattr(R, "secret_box", lambda: FernetSecretBox(key=""))
    with pytest.raises(SecretUnavailable):
        asyncio.run(C._get_realm_resource("acme", "neo4j"))
    reply = asyncio.run(R.test_realm_resource("acme", R.ResourceTestRequest(type="neo4j")))
    assert reply["status"] == "error" and "CAUSA_SECRET_KEY" in reply["detail"]


# ── Leaving ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("include_secrets", [False, True])
def test_an_export_carries_no_secret_whatever_it_is_asked(store, monkeypatch, include_secrets) -> None:
    async def nothing(*a, **k):
        return []

    async def no_settings(*a, **k):
        return None
    monkeypatch.setattr(mdb, "find_many", nothing)
    monkeypatch.setattr(mdb, "find_one", lambda c, q=None: store.find_one(c, q) if c == "realms" else no_settings())
    _put("new-password")
    bundle = asyncio.run(R.export_realm("acme", include_secrets=include_secrets))
    assert "new-password" not in str(bundle) and SEALED not in str(bundle)
    assert "neo4j.password" in bundle["masked_fields"]


# ── The box ───────────────────────────────────────────────────────────────────

def test_the_box_opens_what_it_sealed_and_nothing_sealed_under_another_key() -> None:
    from cryptography.fernet import Fernet

    box = FernetSecretBox(key=Fernet.generate_key().decode())
    token = box.seal("s3cret")
    assert "s3cret" not in token and box.open(token) == "s3cret"
    with pytest.raises(SecretUnavailable):
        FernetSecretBox(key=Fernet.generate_key().decode()).open(token)
    assert FernetSecretBox(key="not a key").available is False


def test_the_rules_hold_without_the_router() -> None:
    box = SecretBoxStub()
    stored = {"password": {SEALED: box.seal("old")}}
    assert for_storage({"password": MASK}, stored, {"password"}, box) == stored
    assert for_storage({}, stored, {"password"}, box) == stored
    assert "password" not in for_storage({"password": ""}, stored, {"password"}, box)
    assert box.open(for_storage({"password": "new"}, stored, {"password"}, box)["password"][SEALED]) == "new"
