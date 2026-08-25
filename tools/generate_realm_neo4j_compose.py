"""One Neo4j container per Realm.

Neo4j Community has no multi-database support (unlike Enterprise), and the
graph itself has no corpus_id/realm_id partitioning at all (see
adapters/neo4j_graph.py's module docstring, and the design notes) —
unlike Qdrant/OpenSearch, this can't be fixed by namespacing collection
names. The only real isolation is a separate container (and volume) per
Realm.

This script formalizes the manual copy-paste convention already sketched as
a commented-out example in deploy/compose/docker-compose.yml into something
regenerable for however many Realms are actually registered, since not every
Realm needs its container running at once. Each gets its own
`profiles: ["graph", "graph-<realm_id>"]`, so `docker compose --profile
graph-acme up -d` starts that one Realm's container while `--profile graph`
starts all of them.

The FIRST-registered Realm (by created_at) keeps the base `neo4j` service in
docker-compose.yml itself (bolt://localhost:7687) — no container churn for
whichever Realm was already using it before per-Realm containers existed. Every other Realm gets
its own `neo4j-<realm_id>` service, written to a generated override file,
with sequential host ports assigned once and left stable on re-run (a Realm
that already has a non-default neo4j URI registered is treated as already
migrated and left untouched).

Usage:
    python3 -m tools.generate_realm_neo4j_compose
    docker compose -f deploy/compose/docker-compose.yml \\
                    -f deploy/compose/neo4j-realms.generated.yml \\
                    --profile graph-<realm_id> up -d neo4j-<realm_id>

`generate_neo4j_overlay()` below is also imported directly by
services/api_gateway/routers/realms.py's `POST /realms/{id}/neo4j/provision` — the UI-triggered self-service path that additionally runs
`docker compose up -d` itself instead of leaving that to the operator. This
module stays docker-subprocess-free on purpose (mirrors
tools/migrate_corpus_aliases.py's pure-Mongo-plus-file shape); the actual
`docker`/subprocess orchestration lives in the router, not here.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

_COMPOSE_DIR = Path(__file__).parent.parent / "deploy" / "compose"
_COMPOSE_BASE = _COMPOSE_DIR / "docker-compose.yml"
_OUTPUT_PATH = _COMPOSE_DIR / "neo4j-realms.generated.yml"
_DEFAULT_BOLT_URI = "bolt://localhost:7687"
_BASE_HTTP_PORT = 7475
_BASE_BOLT_PORT = 7476


def _service_block(realm_id: str, http_port: int, bolt_port: int) -> str:
    safe_name = realm_id.replace("_", "-")
    return f"""  neo4j-{safe_name}:
    image: neo4j:5-community
    profiles: ["graph", "graph-{safe_name}"]
    ports:
      - "{http_port}:7474"
      - "{bolt_port}:7687"
    volumes:
      - neo4j_{safe_name.replace('-', '_')}_data:/data
    environment:
      NEO4J_AUTH: neo4j/ragplatform
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_server_memory_heap_initial__size: "2G"
      NEO4J_server_memory_heap_max__size: "2G"
      NEO4J_server_memory_pagecache_size: "1G"
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s
"""


def _port_is_free(port: int) -> bool:
    """Real OS-level check — a port can be unclaimed in Mongo bookkeeping (no
    other Realm's `resources[]` mentions it) yet already bound by some
    unrelated local process; only a socket bind test catches that, the
    Mongo-claimed-ports set below can't."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


async def generate_neo4j_overlay() -> dict[str, Any]:
    """Regenerates deploy/compose/neo4j-realms.generated.yml for every Realm
    that doesn't already have its own dedicated neo4j resource, and returns a
    per-Realm summary services/api_gateway/routers/realms.py's
    `POST /realms/{id}/neo4j/provision` uses to then actually start
    the container. Pure Mongo-plus-file logic — no docker subprocess here
    (see module docstring).

    Returns ``{realm_id: {"status": "default"|"already_migrated"|"provisioned",
    "uri": ..., "service_name": ..., "http_port"?: ..., "bolt_port"?: ...}}``.
    """
    import adapters.mongodb as mdb

    realms = await mdb.find_many("realms", query={"deleted_at": None}, sort=[("created_at", 1)])
    result: dict[str, Any] = {}
    if not realms:
        return result

    default_realm = realms[0]
    result[default_realm["id"]] = {"status": "default", "uri": _DEFAULT_BOLT_URI}

    # First pass: find every bolt port already claimed by a previously-
    # migrated Realm, so a Realm being assigned fresh ports in THIS run can't
    # collide with one assigned in an earlier run (idempotency across runs,
    # not just "same realm, same result").
    claimed_bolt_ports: set[int] = set()
    for realm in realms[1:]:
        neo4j_res = next((r for r in realm.get("resources", []) if r["type"] == "neo4j"), None)
        uri = (neo4j_res or {}).get("uri") or ""
        if uri and uri != _DEFAULT_BOLT_URI and ":" in uri:
            try:
                claimed_bolt_ports.add(int(uri.rsplit(":", 1)[-1]))
            except ValueError:
                pass

    services: list[str] = []
    volumes: list[str] = []
    next_http, next_bolt = _BASE_HTTP_PORT, _BASE_BOLT_PORT

    for realm in realms[1:]:
        realm_id = realm["id"]
        safe_name = realm_id.replace("_", "-")
        neo4j_res = next((r for r in realm.get("resources", []) if r["type"] == "neo4j"), None)
        already_migrated = neo4j_res and neo4j_res.get("uri") not in (None, "", _DEFAULT_BOLT_URI)
        if already_migrated:
            result[realm_id] = {
                "status": "already_migrated", "uri": neo4j_res["uri"],
                "service_name": f"neo4j-{safe_name}",
            }
            continue

        while next_bolt in claimed_bolt_ports or not (_port_is_free(next_http) and _port_is_free(next_bolt)):
            next_http += 2
            next_bolt += 2
        http_port, bolt_port = next_http, next_bolt
        claimed_bolt_ports.add(bolt_port)
        next_http += 2
        next_bolt += 2

        services.append(_service_block(realm_id, http_port, bolt_port))
        volumes.append(f"  neo4j_{safe_name.replace('-', '_')}_data:")

        new_uri = f"bolt://localhost:{bolt_port}"
        resources: list[dict[str, Any]] = [r for r in realm.get("resources", []) if r["type"] != "neo4j"]
        resources.append({"type": "neo4j", "uri": new_uri, "user": "neo4j", "password": "ragplatform"})
        await mdb.update_one("realms", {"id": realm_id}, {"$set": {"resources": resources}})
        result[realm_id] = {
            "status": "provisioned", "uri": new_uri, "service_name": f"neo4j-{safe_name}",
            "http_port": http_port, "bolt_port": bolt_port,
        }

    if services:
        header = (
            "# GENERATED by tools/generate_realm_neo4j_compose.py — do not hand-edit.\n"
            "# Re-run the script after registering a new Realm to add its service here.\n"
            "# Usage: docker compose -f deploy/compose/docker-compose.yml "
            "-f deploy/compose/neo4j-realms.generated.yml up -d neo4j-<realm_id>\n\n"
            "services:\n"
        )
        footer = "\nvolumes:\n" + "\n".join(volumes) + "\n"
        _OUTPUT_PATH.write_text(header + "\n".join(services) + footer, encoding="utf-8")

    return result


async def main() -> None:
    result = await generate_neo4j_overlay()
    if not result:
        print("No realms registered — nothing to generate.")
        return

    provisioned = 0
    for realm_id, info in result.items():
        if info["status"] == "default":
            print(f"Default Realm (keeps base `neo4j` service, {info['uri']}): {realm_id}")
        elif info["status"] == "already_migrated":
            print(f"[{realm_id}] already has its own neo4j resource ({info['uri']}) — leaving untouched")
        else:
            print(f"[{realm_id}] assigned {info['uri']} (http {info['http_port']}) — registered in realms.resources")
            provisioned += 1

    if not provisioned:
        print("Every non-default Realm already has its own neo4j resource — nothing to generate.")
    else:
        print(f"\nWrote {_OUTPUT_PATH} ({provisioned} Realm-specific Neo4j service(s))")


if __name__ == "__main__":
    asyncio.run(main())
