"""Realm registry — isolated namespaces each containing shared
infrastructure (Qdrant/Neo4j/OpenSearch/Ollama) and multiple RAG endpoint
implementations. Resources are typed against the connector_types catalog seeded
at gateway startup; unknown types are rejected at write time.
"""
from __future__ import annotations

import asyncio
import re
import shutil
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import adapters.mongodb as mdb
from core.models import ResourceConfig
from tools.generate_realm_neo4j_compose import _COMPOSE_BASE, _OUTPUT_PATH, generate_neo4j_overlay

router = APIRouter(prefix="/realms", tags=["realms"])
connector_types_router = APIRouter(tags=["realms"])

_REALMS_COLLECTION = "realms"
_CONNECTOR_TYPES_COLLECTION = "connector_types"

_CONNECTOR_TYPES_SEED: list[dict[str, Any]] = [
    {
        "type": "qdrant",
        "label": "Qdrant",
        "description": "Dense vector store — ANN cosine search",
        "params_schema": {
            "host": {"type": "string", "default": "localhost"},
            "port": {"type": "integer", "default": 6333},
        },
    },
    {
        "type": "neo4j",
        "label": "Neo4j",
        "description": "Knowledge graph — GDS community detection",
        "params_schema": {
            "uri": {"type": "string", "default": "bolt://localhost:7687"},
            "user": {"type": "string", "default": "neo4j"},
            "password": {"type": "string"},
        },
    },
    {
        "type": "opensearch",
        "label": "OpenSearch",
        "description": "BM25 sparse index",
        "params_schema": {
            "host": {"type": "string", "default": "localhost"},
            "port": {"type": "integer", "default": 9200},
        },
    },
    {
        "type": "ollama",
        "label": "Ollama",
        "description": "LLM — generation + LLM-judge deep diagnostics",
        "params_schema": {
            "host": {"type": "string", "default": "localhost"},
            "port": {"type": "integer", "default": 11434},
            "model": {"type": "string", "default": "qwen3:8b"},
        },
    },
]


async def seed_connector_types() -> None:
    """Idempotent seed — only inserts if the collection is empty."""
    n = await mdb.count(_CONNECTOR_TYPES_COLLECTION)
    if n == 0:
        for ct in _CONNECTOR_TYPES_SEED:
            await mdb.upsert_one(_CONNECTOR_TYPES_COLLECTION, {"type": ct["type"]}, ct)


def _slug(name: str) -> str:
    """Derive a URL-safe id from a human name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


async def _valid_connector_types() -> set[str]:
    docs = await mdb.find_many(_CONNECTOR_TYPES_COLLECTION)
    return {d["type"] for d in docs}


# ── A realm's key metrics ───────────────────────────────────────────────────
#
# One list serving two places: the run table's columns and the number band on
# the run screen. The evaluator computes ten metrics, and showing all ten in
# both places means showing none: there is nothing to read.
#
# Stored on the realm rather than in the browser's localStorage, for two
# reasons: the choice belongs to the project rather than to the machine looking
# at it, and it travels in the realm export along with everything else.
DEFAULT_KEY_METRICS = [
    "retrieval_recall_at_k",
    "answer_similarity",
    "context_support",
    "correct_refusal",
    "retrieval_precision_at_k",
]



def _strip_id(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    # A realm created before key metrics existed returns the default list rather
    # than an empty one: on the interface side an empty list is indistinguishable
    # from "the reader removed every column", and an older realm's run table
    # would end up with no metric at all.
    out.setdefault("key_metrics", list(DEFAULT_KEY_METRICS))
    return out


# ── Connector types catalog ──────────────────────────────────────────────────

@connector_types_router.get("/connector-types")
async def list_connector_types() -> list[dict[str, Any]]:
    docs = await mdb.find_many(_CONNECTOR_TYPES_COLLECTION)
    return [_strip_id(d) for d in docs]


# ── Realm CRUD ───────────────────────────────────────────────────────────────

# What a realm is for. Empty is the ordinary case and stays empty on every realm
# that already exists, so nothing has to be rewritten to gain the field.
#
# `proving_ground` marks a realm whose data is broken on purpose. Without such a
# mark its diagnostics read as a broken installation, and a green one would mean
# the guards had stopped firing on the defects put there for them. It also keeps
# the installation check off it: asking whether the platform works is the wrong
# question to put to a realm built to fail.
PURPOSES = ("", "proving_ground")


class RealmCreateRequest(BaseModel):
    name: str
    id: str | None = None
    description: str = ""
    purpose: str = ""


@router.get("")
async def list_realms(include_deleted: bool = False) -> list[dict[str, Any]]:
    docs = await mdb.find_many(_REALMS_COLLECTION, sort=[("created_at", -1)])
    if not include_deleted:
        docs = [d for d in docs if not d.get("deleted_at")]
    return [_strip_id(d) for d in docs]


@router.post("", status_code=201)
async def create_realm(body: RealmCreateRequest) -> dict[str, Any]:
    realm_id = body.id or _slug(body.name)
    existing = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if existing and not existing.get("deleted_at"):
        raise HTTPException(status_code=409, detail=f"Realm '{realm_id}' already exists")
    doc: dict[str, Any] = {
        "id": realm_id,
        "name": body.name,
        "description": body.description.strip(),
        "purpose": body.purpose,
        "resources": existing.get("resources", []) if existing else [],
        "created_at": existing.get("created_at", datetime.now(UTC).isoformat()) if existing else datetime.now(UTC).isoformat(),
        "deleted_at": None,
    }
    await mdb.upsert_one(_REALMS_COLLECTION, {"id": realm_id}, doc)
    return _strip_id(doc)



# ── Realm export and import ─────────────────────────────────────────────────
#
# Moving a project between installations as one file, the way Keycloak does it.
#
# What travels: the realm itself with its resources and key metrics, the RAG
# endpoints, the prompts, the generation presets, the control-question datasets
# together with their questions, and the settings (active model, active domain
# packs).
#
# What does not, and why: **the corpus contents**. They live in Qdrant and
# OpenSearch and run to tens of thousands of fragments, which does not fit in a
# file somebody emails. Corpora travel as references, so on the receiving side
# this is a list of what has to be ingested again. Run history stays behind for
# the same reason plus one of its own: a run carried over without the corpus it
# was made against is numbers with nothing to explain them.

_EXPORT_FORMAT = "causa-realm/v1"

# Resource fields whose values must not go into a file somebody emails. The same
# rule already applies when they are displayed on the resources page
# (ui/src/pages/RealmResourcesPage.tsx#SENSITIVE_KEY_RE); writing it a second
# time here is deliberate, because a secrets rule that holds only in the
# interface protects against somebody reading over your shoulder and nothing
# else.
_SENSITIVE_KEY = re.compile(r"password|secret|token|api[_-]?key", re.IGNORECASE)
_MASK = "••••••••"


def _mask_resources(resources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Returns the resources with their secrets masked, plus a list of what was
    masked. The receiving side needs that list: a connection whose password has
    been replaced by dots looks configured and does not work."""
    masked: list[dict[str, Any]] = []
    fields: list[str] = []
    for resource in resources:
        copy = dict(resource)
        for key in copy:
            if _SENSITIVE_KEY.search(key) and copy[key]:
                copy[key] = _MASK
                fields.append(f"{resource.get('type', '?')}.{key}")
        masked.append(copy)
    return masked, fields


@router.get("/{realm_id}/export")
async def export_realm(realm_id: str, include_secrets: bool = False) -> dict[str, Any]:
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc or doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")

    realm = _strip_id(doc)
    masked_fields: list[str] = []
    if not include_secrets:
        realm["resources"], masked_fields = _mask_resources(realm.get("resources", []))

    async def scoped(collection: str) -> list[dict[str, Any]]:
        return [_strip_id(d) for d in await mdb.find_many(collection, {"realm_id": realm_id})]

    settings_doc = await mdb.find_one("settings", {"realm_id": realm_id}) or {}

    return {
        "format": _EXPORT_FORMAT,
        "exported_at": datetime.now(UTC).isoformat(),
        "realm": realm,
        "external_rags": await scoped("external_rags"),
        "prompts": await scoped("prompts"),
        "generation_presets": await scoped("generation_presets"),
        "datasets": await scoped("datasets"),
        "settings": _strip_id(settings_doc),
        # References only: `corpus_id` and a description, not one fragment.
        "corpora": [
            {"corpus_id": c.get("corpus_id"), "description": c.get("description", "")}
            for c in await mdb.find_many("corpora", {"realm_id": realm_id})
            if not c.get("deleted_at")
        ],
        "masked_fields": masked_fields,
    }


class ImportEntry(BaseModel):
    kind: str
    created: int = 0
    conflicted: int = 0
    skipped: int = 0


@router.post("/import")
async def import_realm(
    bundle: dict[str, Any],
    on_conflict: str = "fail",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Imports a realm from an export file.

    `dry_run=true` returns the same reconciliation and writes nothing: the
    preview dialog shows it before anything is confirmed, because importing to
    see what happens is not a preview.
    """
    if bundle.get("format") != _EXPORT_FORMAT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {bundle.get('format')!r}, expected {_EXPORT_FORMAT!r}",
        )
    if on_conflict not in {"fail", "rename"}:
        raise HTTPException(status_code=400, detail="on_conflict must be 'fail' or 'rename'")

    realm = dict(bundle.get("realm") or {})
    realm_id = realm.get("id")
    if not realm_id:
        raise HTTPException(status_code=400, detail="bundle has no realm.id")

    existing = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if existing:
        if on_conflict == "fail":
            raise HTTPException(status_code=409, detail=f"Realm '{realm_id}' already exists")
        # A rename rather than a merge: merging two realms mixes two run
        # histories, and keeping those separate is what a realm is for.
        suffix = 2
        while await mdb.find_one(_REALMS_COLLECTION, {"id": f"{realm_id}-{suffix}"}):
            suffix += 1
        realm_id = f"{realm_id}-{suffix}"
        realm["id"] = realm_id
        realm["name"] = f"{realm.get('name', realm_id)} ({suffix})"

    warnings: list[str] = []
    entries: list[dict[str, Any]] = []

    collections = [
        ("external_rags", "external_rags"),
        ("prompts", "prompts"),
        ("generation_presets", "generation_presets"),
        ("datasets", "datasets"),
    ]
    for key, collection in collections:
        docs = bundle.get(key) or []
        entries.append({"kind": key, "created": len(docs), "conflicted": 0, "skipped": 0})
        if dry_run:
            continue
        for d in docs:
            await mdb.insert_one(collection, {**d, "realm_id": realm_id})
        if key == "prompts" and docs:
            # The pipeline reads prompts from files, synchronously, with no
            # database access. Import wrote to Mongo alone, so an imported realm
            # answered through whichever prompt happened to have a file
            # beside it.
            from core.prompt_store import prompt_store
            prompt_store.sync_from([{**d, "realm_id": realm_id} for d in docs])

    corpora = bundle.get("corpora") or []
    entries.append({"kind": "corpora", "created": 0, "conflicted": 0, "skipped": len(corpora)})
    if corpora:
        warnings.append(
            "Corpus contents do not travel: "
            + ", ".join(str(c.get("corpus_id")) for c in corpora)
        )

    masked = list(bundle.get("masked_fields") or [])
    if masked:
        warnings.append("These need entering again: " + ", ".join(masked))

    if not dry_run:
        realm.pop("deleted_at", None)
        realm["created_at"] = realm.get("created_at") or datetime.now(UTC).isoformat()
        await mdb.insert_one(_REALMS_COLLECTION, realm)
        settings = bundle.get("settings") or {}
        if settings:
            await mdb.insert_one("settings", {**settings, "realm_id": realm_id})

    return {
        "dry_run": dry_run,
        "realm_id": realm_id,
        "entries": entries,
        "masked_fields": masked,
        "warnings": warnings,
    }


@router.get("/{realm_id}")
async def get_realm(realm_id: str) -> dict[str, Any]:
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc or doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")
    return _strip_id(doc)


class RealmUpdateRequest(BaseModel):
    name: str
    description: str = ""
    purpose: str | None = None


@router.put("/{realm_id}")
async def update_realm(realm_id: str, body: RealmUpdateRequest) -> dict[str, Any]:
    """Updates a Realm's display name and description. The `id` (slug) is
    immutable — it's the join key used everywhere else (external_rags.realm_id,
    experiment runs, settings), so changing it would silently orphan all of that."""
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc or doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Name must not be empty")
    update = {"name": body.name.strip(), "description": body.description.strip()}
    # Omitted leaves it as it was: a form that does not know about the field
    # must not be able to clear it by staying silent.
    if body.purpose is not None:
        if body.purpose not in PURPOSES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown purpose {body.purpose!r}. Known: {list(PURPOSES)}",
            )
        update["purpose"] = body.purpose
    await mdb.update_one(_REALMS_COLLECTION, {"id": realm_id}, {"$set": update})
    doc.update(update)
    return _strip_id(doc)


@router.delete("/{realm_id}", status_code=204)
async def delete_realm(realm_id: str, hard: bool = False) -> None:
    """Soft-delete by default (marks `deleted_at`, hides from listings) — a
    Realm carries enough attached config (resources, RAG endpoints, runs,
    settings) that recovering from an accidental delete should not require
    re-entering all of it. Pass `?hard=true` to actually remove the document,
    which still requires zero linked RAG endpoints (same guard as before)."""
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc or (doc.get("deleted_at") and not hard):
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")

    if not hard:
        await mdb.update_one(
            _REALMS_COLLECTION, {"id": realm_id},
            {"$set": {"deleted_at": datetime.now(UTC).isoformat()}},
        )
        return

    linked = await mdb.count("external_rags", {"realm_id": realm_id})
    if linked:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot permanently delete Realm '{realm_id}': {linked} RAG endpoint(s) still linked",
        )
    deleted = await mdb.delete_one(_REALMS_COLLECTION, {"id": realm_id})
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")


@router.post("/{realm_id}/restore")
async def restore_realm(realm_id: str) -> dict[str, Any]:
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc or not doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"No soft-deleted Realm '{realm_id}' found")
    await mdb.update_one(_REALMS_COLLECTION, {"id": realm_id}, {"$set": {"deleted_at": None}})
    doc["deleted_at"] = None
    return _strip_id(doc)


# ── Resource management ──────────────────────────────────────────────────────

class KeyMetricsRequest(BaseModel):
    key_metrics: list[str]


@router.put("/{realm_id}/key-metrics")
async def set_key_metrics(realm_id: str, body: KeyMetricsRequest) -> dict[str, Any]:
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")
    await mdb.update_one(
        _REALMS_COLLECTION, {"id": realm_id}, {"$set": {"key_metrics": body.key_metrics}},
    )
    doc["key_metrics"] = body.key_metrics
    return _strip_id(doc)



@router.put("/{realm_id}/resources")
async def set_realm_resources(realm_id: str, resources: list[ResourceConfig]) -> dict[str, Any]:
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")

    valid_types = await _valid_connector_types()
    for r in resources:
        if r.type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown connector type '{r.type}'. Supported: {sorted(valid_types)}",
            )

    serialized = [r.model_dump() for r in resources]
    await mdb.update_one(
        _REALMS_COLLECTION,
        {"id": realm_id},
        {"$set": {"resources": serialized}},
    )
    doc["resources"] = serialized
    return _strip_id(doc)


class ResourceTestRequest(BaseModel):
    type: str


@router.post("/{realm_id}/resources/test")
async def test_realm_resource(realm_id: str, body: ResourceTestRequest) -> dict[str, Any]:
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")

    cfg = next((r for r in doc.get("resources", []) if r["type"] == body.type), None)
    if cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"No '{body.type}' resource registered for Realm '{realm_id}'",
        )

    try:
        result = await _ping_resource(body.type, cfg)
    except Exception as e:
        return {"status": "error", "type": body.type, "detail": str(e)}
    return {"status": "ok", "type": body.type, **result}


async def _ping_resource(resource_type: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if resource_type == "qdrant":
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 6333)
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"http://{host}:{port}/collections")
            resp.raise_for_status()
            data = resp.json()
            collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
            return {"collections": collections}

    if resource_type == "neo4j":
        # `verify()`, not `is_available()`. The latter only asks whether the
        # `neo4j` Python package imports, so this endpoint used to answer
        # `{"status": "ok"}` for a Neo4j nobody had started — found live with
        # port 7687 closed while the reply said the resource was fine. The
        # other three types here have always made a real request; this one
        # only looked like it did.
        from adapters.neo4j_graph import Neo4jGraphRetriever
        uri = cfg.get("uri", "bolt://localhost:7687")
        user = cfg.get("user", "neo4j")
        password = cfg.get("password", "ragplatform")
        graph = Neo4jGraphRetriever(
            uri=uri, user=user, password=password,
            # Same 5s bound the other three resource pings use. The driver's
            # own default is 30s, which turns "check everything" into half a
            # minute of nothing on a stand with one service down.
            connection_timeout=5.0,
        )
        if not graph.is_available():
            raise RuntimeError("neo4j driver not installed ('.[graph]')")
        try:
            if not graph.verify():
                raise RuntimeError(f"Neo4j unreachable at {uri}")
        finally:
            graph.close()
        return {"neo4j_uri": uri}

    if resource_type == "opensearch":
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 9200)
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"http://{host}:{port}/")
            resp.raise_for_status()
            data = resp.json()
            return {"cluster_name": data.get("cluster_name", "")}

    if resource_type == "ollama":
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 11434)
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"http://{host}:{port}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models}

    raise ValueError(f"No ping logic for type '{resource_type}'")


# ── Neo4j per-Realm provisioning ─────────────────────────────────
#
# Self-service front-end for the two-step manual process documented in
# README.md (per-realm Neo4j): generate the compose overlay +
# register the new bolt URI (tools/generate_realm_neo4j_compose.py, pure
# Mongo/file logic, no docker dependency), then actually run
# `docker compose up -d <service>` and poll the container's healthcheck.
# In-memory only (not persisted to Mongo) — this is a rare admin action, not
# something that needs to survive a gateway restart; losing track of an
# in-flight job just means re-POSTing, which is itself idempotent since
# generate_neo4j_overlay() and `docker compose up -d` both are.
_neo4j_provision_status: dict[str, dict[str, Any]] = {}


@router.post("/{realm_id}/neo4j/provision")
async def provision_realm_neo4j(realm_id: str) -> dict[str, Any]:
    """Provisions a dedicated Neo4j container for this Realm. Backgrounds the
    actual `docker compose up -d` + healthcheck wait (image pull alone can
    exceed an HTTP request timeout on a cold cache) — poll
    GET /realms/{realm_id}/neo4j/provision for status."""
    doc = await mdb.find_one(_REALMS_COLLECTION, {"id": realm_id})
    if not doc or doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' not found")

    current = _neo4j_provision_status.get(realm_id)
    if current and current["status"] == "running":
        return current

    if shutil.which("docker") is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "docker CLI not found on the API Gateway host — provision manually "
                "(see the per-realm Neo4j notes)."
            ),
        )

    overlay = await generate_neo4j_overlay()
    info = overlay.get(realm_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Realm '{realm_id}' has no created_at / is not in the registry scan")

    if info["status"] == "default":
        result = {
            "status": "not_needed",
            "detail": f"This Realm already uses the shared default Neo4j instance ({info['uri']}).",
        }
        _neo4j_provision_status[realm_id] = result
        return result

    service_name = info["service_name"]

    if info["status"] == "already_migrated":
        health = await _container_health(service_name)
        if health == "healthy":
            result = {"status": "done", "service_name": service_name, "uri": info["uri"], "detail": "already provisioned and healthy"}
            _neo4j_provision_status[realm_id] = result
            return result
        # Resource is registered but the container isn't confirmed healthy
        # (e.g. after a fresh `docker compose down`/volume prune) — fall
        # through and (re)start it below, same as a first-time provision.

    _neo4j_provision_status[realm_id] = {
        "status": "running", "service_name": service_name, "detail": "starting container…",
    }
    asyncio.create_task(_run_neo4j_provision(realm_id, service_name, info.get("uri")))
    return _neo4j_provision_status[realm_id]


@router.get("/{realm_id}/neo4j/provision")
async def get_neo4j_provision_status(realm_id: str) -> dict[str, Any]:
    status = _neo4j_provision_status.get(realm_id)
    if status is None:
        raise HTTPException(status_code=404, detail="No provisioning job found for this Realm — POST first")
    return status


async def _run_neo4j_provision(realm_id: str, service_name: str, uri: str | None) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "compose",
            "-f", str(_COMPOSE_BASE), "-f", str(_OUTPUT_PATH),
            "up", "-d", service_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            _neo4j_provision_status[realm_id] = {
                "status": "error", "service_name": service_name,
                "detail": stderr.decode(errors="replace")[-2000:],
            }
            return

        for _ in range(30):  # ~60s at 2s polling interval
            health = await _container_health(service_name)
            if health == "healthy":
                _neo4j_provision_status[realm_id] = {
                    "status": "done", "service_name": service_name, "uri": uri, "detail": "container healthy",
                }
                return
            if health == "unhealthy":
                _neo4j_provision_status[realm_id] = {
                    "status": "error", "service_name": service_name,
                    "detail": "container reported unhealthy — check `docker logs`",
                }
                return
            await asyncio.sleep(2)

        _neo4j_provision_status[realm_id] = {
            "status": "error", "service_name": service_name,
            "detail": "container did not become healthy within 60s — check `docker logs`",
        }
    except Exception as e:
        _neo4j_provision_status[realm_id] = {"status": "error", "service_name": service_name, "detail": str(e)}


async def _container_health(service_name: str) -> str:
    id_proc = await asyncio.create_subprocess_exec(
        "docker", "compose",
        "-f", str(_COMPOSE_BASE), "-f", str(_OUTPUT_PATH),
        "ps", "-q", service_name,
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await id_proc.communicate()
    container_id = out.decode().strip()
    if not container_id:
        return "unknown"

    inspect = await asyncio.create_subprocess_exec(
        "docker", "inspect", "--format", "{{.State.Health.Status}}", container_id,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out2, _ = await inspect.communicate()
    return out2.decode().strip()
