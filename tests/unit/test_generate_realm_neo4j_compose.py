"""tools/generate_realm_neo4j_compose.py — pure-logic checks against a mocked
Mongo, no live docker-compose or database mutation. The script itself is a
one-shot ops tool (like tools/migrate_corpus_aliases.py) meant to be run by
an operator, not imported at runtime — these tests just lock in the port-
assignment/idempotency rules so a future edit doesn't silently reassign an
already-migrated Realm's ports out from under it.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tools.generate_realm_neo4j_compose import main


def _realm(id_: str, created_at: str, neo4j_uri: str | None = "bolt://localhost:7687") -> dict:
    resources = [{"type": "qdrant", "host": "localhost", "port": 6333}]
    if neo4j_uri:
        resources.append({"type": "neo4j", "uri": neo4j_uri, "user": "neo4j", "password": ""})
    return {"id": id_, "created_at": created_at, "resources": resources}


@pytest.mark.asyncio
async def test_first_realm_by_created_at_is_untouched(tmp_path, monkeypatch) -> None:
    import tools.generate_realm_neo4j_compose as mod
    monkeypatch.setattr(mod, "_OUTPUT_PATH", tmp_path / "out.yml")
    monkeypatch.setattr(mod, "_port_is_free", lambda port: True)

    realms = [_realm("acme", "2026-01-01"), _realm("acme", "2026-01-02")]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=realms)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        await main()

    # Only acme should get a Mongo write — demo (first by created_at) is the default.
    mock_update.assert_called_once()
    assert mock_update.call_args[0][1] == {"id": "acme"}


@pytest.mark.asyncio
async def test_assigns_sequential_ports_starting_at_base(tmp_path, monkeypatch) -> None:
    import tools.generate_realm_neo4j_compose as mod
    monkeypatch.setattr(mod, "_OUTPUT_PATH", tmp_path / "out.yml")
    monkeypatch.setattr(mod, "_port_is_free", lambda port: True)

    realms = [_realm("acme", "2026-01-01"), _realm("acme", "2026-01-02")]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=realms)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        await main()

    new_resources = mock_update.call_args[0][2]["$set"]["resources"]
    neo4j_res = next(r for r in new_resources if r["type"] == "neo4j")
    assert neo4j_res["uri"] == f"bolt://localhost:{mod._BASE_BOLT_PORT}"

    content = (tmp_path / "out.yml").read_text()
    assert "neo4j-acme:" in content
    assert f'"{mod._BASE_HTTP_PORT}:7474"' in content
    assert f'"{mod._BASE_BOLT_PORT}:7687"' in content


@pytest.mark.asyncio
async def test_already_migrated_realm_is_left_untouched(tmp_path, monkeypatch) -> None:
    """Idempotency — a Realm whose neo4j.uri already differs from the shared
    default is treated as already provisioned, not reassigned new ports on
    every re-run."""
    import tools.generate_realm_neo4j_compose as mod
    monkeypatch.setattr(mod, "_OUTPUT_PATH", tmp_path / "out.yml")
    monkeypatch.setattr(mod, "_port_is_free", lambda port: True)

    realms = [
        _realm("demo", "2026-01-01"),
        _realm("acme", "2026-01-02", neo4j_uri="bolt://localhost:7476"),
    ]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=realms)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        await main()

    mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_third_realm_gets_next_port_pair_after_an_already_migrated_one(tmp_path, monkeypatch) -> None:
    import tools.generate_realm_neo4j_compose as mod
    monkeypatch.setattr(mod, "_OUTPUT_PATH", tmp_path / "out.yml")
    monkeypatch.setattr(mod, "_port_is_free", lambda port: True)

    realms = [
        _realm("demo", "2026-01-01"),
        _realm("acme", "2026-01-02", neo4j_uri="bolt://localhost:7476"),
        _realm("acme", "2026-01-03"),
    ]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=realms)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        await main()

    mock_update.assert_called_once()
    new_resources = mock_update.call_args[0][2]["$set"]["resources"]
    neo4j_res = next(r for r in new_resources if r["type"] == "neo4j")
    # Ports for acme must NOT collide with acme's already-assigned 7476.
    assert neo4j_res["uri"] == f"bolt://localhost:{mod._BASE_BOLT_PORT + 2}"
    assert neo4j_res["uri"] != "bolt://localhost:7476"


@pytest.mark.asyncio
async def test_skips_a_port_pair_already_bound_by_an_unrelated_process(tmp_path, monkeypatch) -> None:
    """A port can be unclaimed in Mongo (no other Realm's resources[] mentions
    it) yet already bound by some unrelated local process — the Mongo-only
    claimed_bolt_ports set can't see that, only a real socket-level check
    can. Simulates the base pair being taken by mocking _port_is_free."""
    import tools.generate_realm_neo4j_compose as mod
    monkeypatch.setattr(mod, "_OUTPUT_PATH", tmp_path / "out.yml")
    monkeypatch.setattr(
        mod, "_port_is_free",
        lambda port: port not in (mod._BASE_HTTP_PORT, mod._BASE_BOLT_PORT),
    )

    realms = [_realm("acme", "2026-01-01"), _realm("acme", "2026-01-02")]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=realms)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        await main()

    new_resources = mock_update.call_args[0][2]["$set"]["resources"]
    neo4j_res = next(r for r in new_resources if r["type"] == "neo4j")
    assert neo4j_res["uri"] == f"bolt://localhost:{mod._BASE_BOLT_PORT + 2}"
