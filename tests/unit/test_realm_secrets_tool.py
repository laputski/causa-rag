"""Sealing what was stored in the clear, and resealing under a new key."""
from __future__ import annotations

import asyncio

import pytest
from cryptography.fernet import Fernet

import adapters.mongodb as mdb
from adapters.fernet_secrets import FernetSecretBox
from core.secrets import SEALED, SecretUnavailable
from services.api_gateway.routers.realms import _secret_fields
from tools import realm_secrets as T


class _Store:
    def __init__(self):
        self.data = {
            "realms": [{"id": "acme", "resources": [
                {"type": "neo4j", "uri": "bolt://g", "password": "plain-pass"},
                {"type": "qdrant", "host": "q", "port": 6333}]}],
            "external_rags": [{"id": "r1", "headers": {"Authorization": "Bearer plain"}}],
        }

    async def find_many(self, collection, query=None, sort=None):
        return [dict(d) for d in self.data.get(collection, [])]

    async def find_one(self, collection, query):
        return next((dict(d) for d in self.data[collection] if d["id"] == query["id"]), None)

    async def update_one(self, collection, query, update):
        for d in self.data[collection]:
            if d["id"] == query["id"]:
                d.update(update["$set"])


@pytest.fixture
def store(monkeypatch):
    s = _Store()
    for name in ("find_many", "find_one", "update_one"):
        monkeypatch.setattr(mdb, name, getattr(s, name))
    return s


def test_a_dry_run_changes_nothing(store, monkeypatch) -> None:
    monkeypatch.setenv("CAUSA_SECRET_KEY", Fernet.generate_key().decode())
    before = repr(store.data)
    assert asyncio.run(T.run(write=False, old_key=None)) == 0
    assert repr(store.data) == before


def test_sealing_leaves_nothing_in_the_clear_and_reads_back(store, monkeypatch, capsys) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("CAUSA_SECRET_KEY", key)
    assert asyncio.run(T.run(write=True, old_key=None)) == 0
    assert "plain-pass" not in repr(store.data) and "Bearer plain" not in repr(store.data)
    box = FernetSecretBox(key=key)
    neo4j = store.data["realms"][0]["resources"][0]
    assert box.open(neo4j["password"][SEALED]) == "plain-pass"
    assert "read-back failures: 0" in capsys.readouterr().out


def test_rotation_reseals_under_the_new_key_and_the_old_one_opens_nothing(store, monkeypatch) -> None:
    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    monkeypatch.setenv("CAUSA_SECRET_KEY", old)
    asyncio.run(T.run(write=True, old_key=None))
    monkeypatch.setenv("CAUSA_SECRET_KEY", new)
    assert asyncio.run(T.run(write=True, old_key=old)) == 0
    token = store.data["external_rags"][0]["headers"]["Authorization"][SEALED]
    assert FernetSecretBox(key=new).open(token) == "Bearer plain"
    with pytest.raises(SecretUnavailable):
        FernetSecretBox(key=old).open(token)


def test_without_a_key_nothing_is_attempted(store, monkeypatch) -> None:
    monkeypatch.delenv("CAUSA_SECRET_KEY", raising=False)
    before = repr(store.data)
    assert asyncio.run(T.run(write=True, old_key=None)) == 2
    assert repr(store.data) == before


def test_a_resource_without_secrets_is_left_alone() -> None:
    realm = {"resources": [{"type": "qdrant", "host": "q", "port": 6333}]}
    box = FernetSecretBox(key=Fernet.generate_key().decode())
    updated, changed = T.resealed_realm(realm, box, None, _secret_fields)
    assert changed == [] and updated == realm
