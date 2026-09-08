"""Corpus management router — upload files, trigger ingestion, view history."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

log = structlog.get_logger()

router = APIRouter(prefix="/corpus", tags=["corpus"])

_progress: dict[str, list[dict[str, Any]]] = {}

# Single source of truth for "what corpora exist, per Realm,
# backed by what physical storage" (see the design notes). Deliberately
# NOT derived from a raw `GET /collections` scan of Qdrant/OpenSearch: that
# would surface every fitness-test/integration-test collection alongside real
# corpora with no way to tell them apart, and Neo4j has no collection-like
# enumeration primitive at all. A corpus is "real" exactly when something
# registered it here — via /corpus/ingest, the migration script for
# pre-existing collections, or an external RAG's registration (Phase 5).
_CORPORA_COLLECTION = "corpora"


# ── History helpers ────────────────────────────────────────────────────────────

async def _save_ingest(doc: dict[str, Any]) -> None:
    try:
        import adapters.mongodb as mdb
        await mdb.upsert_one("corpus_ingests", {"job_id": doc["job_id"]}, doc)
    except Exception:
        pass


async def _load_history(realm_id: str | None = None) -> list[dict[str, Any]]:
    try:
        import adapters.mongodb as mdb
        query = {"realm_id": realm_id} if realm_id else None
        return await mdb.find_many("corpus_ingests", query=query, sort=[("started_at", -1)], limit=50)
    except Exception:
        return []


# ── Corpora registry ─────────────────────────────────────────────────

async def _register_corpus(
    realm_id: str | None,
    corpus_id: str,
    storage_type: str,
    backends: dict[str, dict[str, Any]],
    owner: str = "platform",
    description: str = "",
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upsert one (realm_id, corpus_id) registry record — idempotent, safe to
    call on every successful ingest (re-ingesting the same corpus just
    refreshes `backends`/`updated_at`, doesn't create a duplicate record).

    Always clears `deleted_at` — re-ingesting (or registering a new dump
    into) a corpus that was previously soft-deleted is a deliberate "this is
    active again" signal, same as `realms.py#create_realm` reviving a
    soft-deleted Realm id on re-create."""
    import adapters.mongodb as mdb
    doc = {
        "realm_id": realm_id,
        "corpus_id": corpus_id,
        "storage_type": storage_type,
        "backends": backends,
        "owner": owner,
        "description": description,
        "updated_at": datetime.now(UTC).isoformat(),
        "deleted_at": None,
    }
    # What the index was built from and built by. Omitted, never written
    # empty, when a caller has none: an empty manifest on a record
    # that had a real one would replace knowledge with a claim of ignorance,
    # and re-registering a corpus is the commonest thing that happens to one.
    if manifest:
        doc["manifest"] = manifest
    existing = await mdb.find_one(_CORPORA_COLLECTION, {"realm_id": realm_id, "corpus_id": corpus_id})
    if existing:
        await mdb.update_one(_CORPORA_COLLECTION, {"realm_id": realm_id, "corpus_id": corpus_id}, {"$set": doc})
        return {**existing, **doc}
    doc["id"] = str(uuid.uuid4())[:8]
    doc["created_at"] = doc["updated_at"]
    await mdb.insert_one(_CORPORA_COLLECTION, dict(doc))
    return doc


async def _list_corpora(realm_id: str | None = None, include_deleted: bool = False) -> list[dict[str, Any]]:
    import adapters.mongodb as mdb
    query = {"realm_id": realm_id} if realm_id else {}
    docs = await mdb.find_many(_CORPORA_COLLECTION, query=query, sort=[("corpus_id", 1)])
    if not include_deleted:
        docs = [d for d in docs if not d.get("deleted_at")]
    return [{k: v for k, v in d.items() if k != "_id"} for d in docs]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("")
async def list_ingests(realm_id: str | None = None) -> list[dict[str, Any]]:
    # corpus_id (which dataset to browse/ingest) is itself
    # realm-scoped now: without this filter, switching Realm left the
    # Upload/Content/Health/Graph tabs showing every other Realm's
    # corpus_ids too, since the ingest history collection wasn't filtered.
    return await _load_history(realm_id)


@router.get("/collections")
async def list_collections(realm_id: str | None = None, include_deleted: bool = False) -> list[dict[str, Any]]:
    """The actual, discoverable list of corpora for a Realm, read
    from the `corpora` registry (see `_register_corpus`) rather than the
    ingest-job history `list_ingests` above returns. History is empty for
    anything ingested outside the upload form (CLI runs, migrated older
    collections) — this endpoint is what NewExperimentPage/ChatPage/CorpusPage
    should use to populate a corpus_id picker going forward.

    Soft-deleted corpora (see DELETE /collections/{id}) are hidden by default,
    same `?include_deleted=true` convention as GET /realms.
    """
    return await _list_corpora(realm_id, include_deleted)


class CorpusRegisterRequest(BaseModel):
    realm_id: str
    corpus_id: str
    storage_type: str
    # Per-backend info, e.g. {"qdrant": {"collection": "...", "embedder_id": "bge_m3"}}.
    # embedder_id is per-backend, not a single corpus-level field — a corpus
    # can legitimately have more than one (e.g. GraphRAG-type: BGE-M3 for
    # Qdrant chunk vectors, a separate embedder for Neo4j community/graph
    # embeddings — see the design notes).
    backends: dict[str, dict[str, Any]]
    owner: str = "external_ingestor"
    description: str = ""


@router.post("/collections", status_code=201)
async def register_collection(body: CorpusRegisterRequest) -> dict[str, Any]:
    """Public counterpart to GET /collections: lets a corpus
    prepared OUTSIDE the platform's own /corpus/ingest (an external RAG's own
    ingestor, writing directly to Qdrant/OpenSearch/Neo4j) become known to
    the registry — same idempotent upsert `_register_corpus` already uses
    internally, exposed for external callers. Mirrors POST /datasets: any owner, one registry, no second-class citizens.

    Registering here does NOT by itself grant Content/Health/Graph access
    to the corpus's contents — that still requires the Realm's `resources[]`
    to actually point at the same physical storage (Case A, `uses_realm_resources`).
    This endpoint only makes the corpus *known*: listed by GET /collections,
    selectable as corpus_id in a new run, and its embedder(s) discoverable.
    """
    return await _register_corpus(
        realm_id=body.realm_id,
        corpus_id=body.corpus_id,
        storage_type=body.storage_type,
        backends=body.backends,
        owner=body.owner,
        description=body.description,
    )


class CorpusUpdateRequest(BaseModel):
    description: str = ""


@router.put("/collections/{corpus_registry_id}")
async def update_collection(corpus_registry_id: str, body: CorpusUpdateRequest) -> dict[str, Any]:
    """Edits a corpus registry record's description. `corpus_id`/`realm_id`
    are immutable — they're the join key used everywhere else a corpus_id is
    referenced (experiment configs, datasets, chat), same reasoning as
    realms.py#update_realm keeping a Realm's `id` immutable."""
    import adapters.mongodb as mdb
    doc = await mdb.find_one(_CORPORA_COLLECTION, {"id": corpus_registry_id})
    if not doc or doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"Corpus {corpus_registry_id!r} not found")
    update = {"description": body.description.strip(), "updated_at": datetime.now(UTC).isoformat()}
    await mdb.update_one(_CORPORA_COLLECTION, {"id": corpus_registry_id}, {"$set": update})
    doc.update(update)
    return {k: v for k, v in doc.items() if k != "_id"}


@router.delete("/collections/{corpus_registry_id}", status_code=204)
async def delete_collection(corpus_registry_id: str, hard: bool = False) -> None:
    """Soft-delete by default (marks `deleted_at`, hides from every corpus_id
    picker) — mirrors realms.py#delete_realm's reasoning: a corpus registry
    record is cheap to recreate, but recovering from an accidental delete of
    the *pointer* to real ingested data should not require re-ingesting.

    `?hard=true` additionally attempts to physically drop the underlying
    Qdrant collection / OpenSearch index (best-effort — a backend that's
    unreachable or already gone doesn't block removing the registry record,
    it's logged and skipped). Neo4j is deliberately never touched here:
    it has no corpus_id partitioning at all — there is no "this corpus's slice
    of the graph" to drop, only the whole Realm's shared graph, which a
    corpus-level delete must not reach into."""
    import adapters.mongodb as mdb
    doc = await mdb.find_one(_CORPORA_COLLECTION, {"id": corpus_registry_id})
    if not doc or (doc.get("deleted_at") and not hard):
        raise HTTPException(status_code=404, detail=f"Corpus {corpus_registry_id!r} not found")

    if not hard:
        await mdb.update_one(
            _CORPORA_COLLECTION, {"id": corpus_registry_id},
            {"$set": {"deleted_at": datetime.now(UTC).isoformat()}},
        )
        return

    backends = doc.get("backends") or {}
    realm_id = doc.get("realm_id")
    if "qdrant" in backends and backends["qdrant"].get("collection"):
        try:
            from qdrant_client import QdrantClient
            cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
            host = cfg.get("host", "localhost") if cfg else os.getenv("QDRANT_HOST", "localhost")
            port = int(cfg.get("port", 6333)) if cfg else int(os.getenv("QDRANT_PORT", "6333"))
            QdrantClient(host=host, port=port, timeout=30, check_compatibility=False).delete_collection(
                backends["qdrant"]["collection"],
            )
        except Exception as e:
            log.warning("delete_collection.qdrant_drop_failed", corpus_registry_id=corpus_registry_id, error=str(e))
    if "opensearch" in backends and backends["opensearch"].get("index"):
        try:
            from opensearchpy import OpenSearch
            cfg = await _get_realm_resource(realm_id, "opensearch") if realm_id else None
            host = cfg.get("host", "localhost") if cfg else os.getenv("OPENSEARCH_HOST", "localhost")
            port = int(cfg.get("port", 9200)) if cfg else int(os.getenv("OPENSEARCH_PORT", "9200"))
            OpenSearch(hosts=[{"host": host, "port": port}], use_ssl=False).indices.delete(
                index=backends["opensearch"]["index"], ignore=[404],
            )
        except Exception as e:
            log.warning("delete_collection.opensearch_drop_failed", corpus_registry_id=corpus_registry_id, error=str(e))

    deleted = await mdb.delete_one(_CORPORA_COLLECTION, {"id": corpus_registry_id})
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Corpus {corpus_registry_id!r} not found")


@router.post("/collections/{corpus_registry_id}/restore")
async def restore_collection(corpus_registry_id: str) -> dict[str, Any]:
    import adapters.mongodb as mdb
    doc = await mdb.find_one(_CORPORA_COLLECTION, {"id": corpus_registry_id})
    if not doc or not doc.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"No soft-deleted corpus {corpus_registry_id!r} found")
    await mdb.update_one(_CORPORA_COLLECTION, {"id": corpus_registry_id}, {"$set": {"deleted_at": None}})
    doc["deleted_at"] = None
    return {k: v for k, v in doc.items() if k != "_id"}


@router.post("/ingest")
async def ingest_corpus(
    files: list[UploadFile] = File(...),
    strategy: str = Form("structure_aware"),
    chunk_size: int = Form(512),
    overlap: int = Form(64),
    exclude: str = Form("full.txt"),
    corpus_id: str = Form("default"),
    realm_id: str = Form(""),
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())[:8]
    _progress[job_id] = []

    # Save uploaded files to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_corpus_"))
    filenames = []
    for f in files:
        dest = tmp_dir / (f.filename or "unknown.txt")
        content = await f.read()
        dest.write_bytes(content)
        filenames.append(f.filename or "unknown.txt")

    exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]

    doc: dict[str, Any] = {
        "job_id": job_id,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "files": filenames,
        "n_files": len(filenames),
        "n_chunks": 0,
        "hit_ratio": 0.0,
        "status": "running",
        "error": None,
        "corpus_id": corpus_id,
        "realm_id": realm_id or None,
    }

    # Resolve Realm connection params if realm_id provided
    realm_resources: dict[str, Any] = {}
    if realm_id:
        import adapters.mongodb as mdb
        realm_doc = await mdb.find_one("realms", {"id": realm_id})
        if realm_doc:
            for r in realm_doc.get("resources", []):
                realm_resources[r["type"]] = r

    # Run ingestion in background
    asyncio.create_task(
        _run_ingestion(job_id, tmp_dir, strategy, chunk_size, overlap, exclude_list, doc, corpus_id, realm_resources)
    )

    return {"job_id": job_id, "status": "started", "n_files": len(filenames)}


async def _run_ingestion(
    job_id: str,
    tmp_dir: Path,
    strategy: str,
    chunk_size: int,
    overlap: int,
    exclude: list[str],
    doc: dict[str, Any],
    corpus_id: str = "default",
    realm_resources: dict[str, Any] | None = None,
) -> None:
    def _emit(event: dict[str, Any]) -> None:
        _progress[job_id].append(event)

    try:
        _emit({"type": "start", "message": f"Indexing {doc['n_files']} files..."})

        use_real = os.getenv("USE_REAL_BGE_M3", "true").lower() in ("1", "true", "yes")
        env = {**os.environ, "USE_REAL_BGE_M3": "true" if use_real else "false"}

        cmd = [
            "python3", "-m", "services.ingestion.cli", "ingest", str(tmp_dir),
            "--strategy", strategy,
            "--chunk-size", str(chunk_size),
            "--overlap", str(overlap),
            "--corpus-id", corpus_id,
        ]
        if doc.get("realm_id"):
            cmd += ["--realm-id", doc["realm_id"]]
        for ex in exclude:
            cmd += ["--exclude", ex]
        if realm_resources:
            if "qdrant" in realm_resources:
                q = realm_resources["qdrant"]
                cmd += ["--qdrant-host", str(q.get("host", "localhost")),
                        "--qdrant-port", str(q.get("port", 6333))]
            if "opensearch" in realm_resources:
                o = realm_resources["opensearch"]
                cmd += ["--opensearch-host", str(o.get("host", "localhost")),
                        "--opensearch-port", str(o.get("port", 9200))]
            if "neo4j" in realm_resources:
                n = realm_resources["neo4j"]
                if n.get("uri"):
                    cmd += ["--neo4j-uri", n["uri"]]
                if n.get("user"):
                    cmd += ["--neo4j-user", n["user"]]
                if n.get("password"):
                    cmd += ["--neo4j-password", n["password"]]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        n_chunks = 0
        hit_ratio = 0.0
        assert proc.stdout
        async for line in proc.stdout:
            text = line.decode(errors="replace").strip()
            if "ingest.file_done" in text:
                try:
                    import re
                    m = re.search(r"chunks=(\d+)", text)
                    if m:
                        n_chunks += int(m.group(1))
                except Exception:
                    pass
                _emit({"type": "progress", "message": text, "n_chunks": n_chunks})
            elif "ingest.done" in text:
                try:
                    import re
                    m = re.search(r"hit_ratio=([\d.]+)", text)
                    if m:
                        hit_ratio = float(m.group(1))
                    mc = re.search(r"size=(\d+)", text)
                    if mc:
                        n_chunks = int(mc.group(1))
                except Exception:
                    pass

        await proc.wait()
        doc.update({
            "finished_at": datetime.now(UTC).isoformat(),
            "n_chunks": n_chunks,
            "hit_ratio": hit_ratio,
            "status": "done" if proc.returncode == 0 else "error",
            "error": None if proc.returncode == 0 else f"exit code {proc.returncode}",
        })
        if proc.returncode == 0:
            # Same naming functions the CLI subprocess itself used (adapters.
            # qdrant/opensearch), called here with the identical inputs rather
            # than parsed out of subprocess stdout — one function, two callers,
            # not two independent implementations of the naming rule.
            from adapters.opensearch import _index_name
            from adapters.qdrant import _collection_name
            realm_id = doc.get("realm_id")
            backends: dict[str, Any] = {
                # embedder_id lives per-backend — a corpus can have
                # more than one (e.g. a GraphRAG-type corpus also has a
                # separate Neo4j community-embedding model); this ingest path
                # only ever produces a dense Qdrant one, BGE-M3.
                "qdrant": {"collection": _collection_name(strategy, "bge_m3", corpus_id, realm_id), "embedder_id": "bge_m3"},
                # /corpus/ingest never passes --no-opensearch, so the sparse
                # index always gets created alongside the dense one (see
                # ingest()'s use_opensearch default in services/ingestion/cli.py).
                "opensearch": {"index": _index_name(strategy, corpus_id, realm_id)},
            }
            await _register_corpus(
                realm_id=realm_id,
                corpus_id=corpus_id,
                storage_type="dense_sparse",
                backends=backends,
                owner="platform",
            )
        _emit({"type": "done", "job_id": job_id, "n_chunks": n_chunks, "hit_ratio": hit_ratio})

    except Exception as e:
        doc.update({"finished_at": datetime.now(UTC).isoformat(), "status": "error", "error": str(e)})
        _emit({"type": "error", "message": str(e)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        await _save_ingest(doc)


# ── Corpus observability ─────────────────────────────────────────────

async def _get_realm_resource(realm_id: str, resource_type: str) -> dict[str, Any] | None:
    """Look up a typed resource from the Realm's resources list."""
    import adapters.mongodb as mdb
    doc = await mdb.find_one("realms", {"id": realm_id})
    if not doc:
        return None
    return next((r for r in doc.get("resources", []) if r["type"] == resource_type), None)


def _resolve_qdrant(
    corpus_id: str, strategy: str, embedder: str, cfg: dict[str, Any] | None = None,
    realm_id: str | None = None,
) -> Any:
    from adapters.qdrant import QdrantRetriever
    host = cfg.get("host", "localhost") if cfg else os.getenv("QDRANT_HOST", "localhost")
    port = int(cfg.get("port", 6333)) if cfg else int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantRetriever(
        host=host,
        port=port,
        strategy_id=strategy,
        embedder_id=embedder,
        corpus_id=corpus_id,
        realm_id=realm_id,
    )


def _resolve_neo4j(cfg: dict[str, Any] | None = None) -> Any:
    from adapters.neo4j_graph import Neo4jGraphRetriever
    if cfg:
        return Neo4jGraphRetriever(
            uri=cfg.get("uri", "bolt://localhost:7687"),
            user=cfg.get("user", "neo4j"),
            password=cfg.get("password", "ragplatform"),
        )
    return Neo4jGraphRetriever()


# ── Alternate dump ingestion — one physical backend at a time ──────────────────
#
# /corpus/ingest above is the only ingestion path that chunks raw text itself;
# these three accept a dump already prepared in each backend's own native (or
# native-adjacent) shape and load it directly, skipping the chunking pipeline
# entirely. Each ends by calling _register_corpus so the result shows up in
# every corpus_id picker exactly like a text-chunked corpus would — the
# frontend and every other consumer of GET /collections don't need to know
# which path produced a given corpus.
#
# Mechanics differ enough per backend that "upload a file" isn't one uniform
# interaction — see each endpoint's docstring for what it actually accepts and
# why (found live, working through this: naively assuming one shared upload
# UX across all three would have been wrong for at least two of them).

@router.post("/ingest/qdrant-snapshot")
async def ingest_qdrant_snapshot(
    file: UploadFile = File(...),
    realm_id: str = Form(...),
    corpus_id: str = Form(...),
    strategy: str = Form("structure_aware"),
    embedder: str = Form("bge_m3"),
) -> dict[str, Any]:
    """Restores a Qdrant collection directly from an uploaded `.snapshot` file
    (Qdrant's own binary format — produced by `POST /collections/{name}/snapshots`,
    `qdrant-client`'s `create_snapshot()`, or downloaded from another instance).

    Uses Qdrant's own upload-and-recover REST operation
    (`client.http.snapshots_api.recover_from_uploaded_snapshot`) — the file's
    bytes go straight to Qdrant over HTTP, no shared filesystem path needed,
    unlike the OpenSearch path below.

    This REPLACES the target collection wholesale; it is not a merge with
    whatever chunks (if any) already live in that realm-scoped collection —
    restoring a second, independently-produced snapshot on top of an
    existing one is not supported here, because "replace or merge" has no
    answer this endpoint could pick on the caller's behalf."""
    from adapters.qdrant import _collection_name
    collection = _collection_name(strategy, embedder, corpus_id, realm_id)

    qdrant_cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
    host = qdrant_cfg.get("host", "localhost") if qdrant_cfg else os.getenv("QDRANT_HOST", "localhost")
    port = int(qdrant_cfg.get("port", 6333)) if qdrant_cfg else int(os.getenv("QDRANT_PORT", "6333"))

    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    client = QdrantClient(host=host, port=port, timeout=120, check_compatibility=False)
    try:
        client.http.snapshots_api.recover_from_uploaded_snapshot(
            collection_name=collection, snapshot=file.file,
        )
    except Exception as e:
        # UnexpectedResponse.__str__() truncates the raw response body to a
        # short preview ("...") — found live: the actual Qdrant-side reason
        # (e.g. "failed to restore RocksDB backup: NotFound: Backup not
        # found") was cut off in this very log line, forcing a trip into the
        # Qdrant container's own logs to see the real error. .structured()
        # parses the untruncated body instead.
        detail = str(e)
        if isinstance(e, UnexpectedResponse):
            try:
                detail = e.structured().get("status", {}).get("error", detail)
            except Exception:
                pass
        log.error(
            "ingest_qdrant_snapshot.recover_failed", realm_id=realm_id, corpus_id=corpus_id,
            collection=collection, error=detail, exc_info=True,
        )
        # Found live, again: "failed to restore RocksDB backup: NotFound:
        # Backup not found" is Qdrant's own per-segment error when the
        # uploaded file untars fine but isn't a snapshot ITS OWN snapshot API
        # produced (see the design notes) — a real user just saw this
        # raw Qdrant-internals string with no idea what to actually do about
        # it. The wording is stable across which segment failed, so it's a
        # reliable signature to match on; the hint is only added when it
        # actually applies, not guessed at for every restore failure (a
        # genuinely different cause, e.g. a network drop mid-upload, gets no
        # hint and shouldn't be told to regenerate a snapshot that may have
        # been fine). The hint itself is a translation KEY, not baked-in
        # English prose — `detail`/`message` stay Qdrant's own untranslated
        # system message (there's nothing to localize about that), but the
        # plain-language explanation is user-facing and must follow the
        # UI's selected language, so [[ui/src/pages/CorpusPage.tsx]] looks
        # `hint` up via i18n (`corpusPage.upload.dumpErrorHints`) instead of
        # the backend hardcoding one language into the response.
        message = f"Qdrant snapshot restore failed: {detail}"
        if "failed to restore rocksdb backup" in detail.lower():
            raise HTTPException(
                status_code=502,
                detail={"message": message, "hint": "qdrantSnapshotNotNative"},
            ) from e
        raise HTTPException(status_code=502, detail=message) from e

    doc = await _register_corpus(
        realm_id=realm_id, corpus_id=corpus_id, storage_type="dense_only",
        backends={"qdrant": {"collection": collection, "embedder_id": embedder}},
        owner="platform", description=f"Restored from Qdrant snapshot {file.filename!r}",
    )
    return {"corpus": doc, "collection": collection}


# Found live: apoc.export.cypher.all replays the source graph's own
# CREATE INDEX/CONSTRAINT statements verbatim — on a re-upload (or after
# `replace=true`, which only clears nodes/relationships, never schema, see
# Neo4jGraphRetriever.clear()) the target graph can already have an
# equivalent index/constraint. Each code is Neo4j's own signal that the
# schema object the statement wanted already exists in equivalent form —
# not a real failure, so ingest_neo4j_cypher treats these as no-ops rather
# than aborting the whole script.
_BENIGN_SCHEMA_ALREADY_EXISTS_CODES = frozenset({
    "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
    "Neo.ClientError.Schema.IndexAlreadyExists",
    "Neo.ClientError.Schema.ConstraintAlreadyExists",
})


@router.post("/ingest/neo4j-cypher")
async def ingest_neo4j_cypher(
    file: UploadFile = File(...),
    realm_id: str = Form(...),
    corpus_id: str = Form(...),
    confirm_arbitrary_cypher: bool = Form(False),
    # Asked directly: should this endpoint have a "replace" option like the
    # OpenSearch one below, instead of relying entirely on the uploaded
    # script's own content (a CREATE-based dump duplicates on re-upload; only
    # a MERGE-based one is idempotent — see the docstring below). When true,
    # the ENTIRE graph is wiped before the script runs — there's no
    # narrower "replace just this corpus_id's slice" option, since Neo4j
    # has no corpus_id partitioning to scope a narrower wipe to.
    replace: bool = Form(False),
) -> dict[str, Any]:
    """Executes an uploaded Cypher script (`;`-separated statements) verbatim
    against the Realm's Neo4j graph.

    `confirm_arbitrary_cypher` must be explicitly set — there is no
    sandboxing or clause allowlist here. A statement can `DETACH DELETE`
    anything, and since Neo4j has no corpus_id partitioning at all, it acts on
    the Realm's *entire* shared graph, not some isolated slice belonging to
    `corpus_id` — the registry record this creates is a label for "this
    script populated the Realm's graph", not a claim of graph-level isolation
    the way the Qdrant/OpenSearch entries below actually get.

    `replace=true` wipes the whole graph first via
    [[adapters/neo4j_graph.py#Neo4jGraphRetriever#clear]] — the same
    apoc.periodic.iterate-batched delete the graph-retrieval build path
    already uses (a naive single-transaction `MATCH (n) DETACH DELETE n`
    previously failed silently on a graph of this size, see that method's
    own docstring), reused here rather than re-risking that exact bug with
    a second, hand-rolled copy of the same query.

    Lines starting with `:` (`:begin`/`:commit`/`:rollback`/`:param`/...) are
    dropped before splitting on `;` — cypher-shell REPL meta-commands, not
    Cypher, that apoc.export.cypher.all's default output wraps statement
    batches in. Harmless to drop: each statement here already runs via its
    own auto-committed session.run() call, not inside an explicit
    transaction these markers would have delimited.

    Runs statements sequentially in one session and stops at the first
    failure rather than continuing past it — a partially-applied script
    left silently half-done would be worse than an explicit error naming
    which statement broke and how many ran before it."""
    if not confirm_arbitrary_cypher:
        raise HTTPException(
            status_code=400,
            detail=(
                "Set confirm_arbitrary_cypher=true to proceed — this executes the "
                "uploaded file's Cypher statements verbatim against the Realm's Neo4j "
                "graph (CREATE/DELETE/anything), with no sandboxing."
            ),
        )

    raw = (await file.read()).decode("utf-8")
    # Found live: apoc.export.cypher.all's default output ("cypher-shell"
    # format) wraps batches of statements in `:begin`/`:commit` lines —
    # cypher-shell REPL meta-commands, not Cypher at all. The naive `;`-split
    # below folded the leading `:begin` into the first real statement's own
    # chunk, and Neo4j's parser rejected it immediately (':' is never valid
    # at the start of a statement). Stripped before splitting — any line
    # whose first non-whitespace character is `:` is a cypher-shell
    # meta-command (`:begin`/`:commit`/`:rollback`/`:param`/...), never a
    # real Cypher statement's own first line, so this can't misfire on
    # legitimate Cypher. Dropping `:begin`/`:commit` doesn't change execution
    # semantics either — each statement already runs via its own
    # session.run() below (auto-committed individually), not inside an
    # explicit transaction these markers would have delimited.
    raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith(":"))
    statements = [s.strip() for s in raw.split(";") if s.strip()]
    if not statements:
        raise HTTPException(status_code=400, detail="No Cypher statements found in the uploaded file")

    cfg = await _get_realm_resource(realm_id, "neo4j") if realm_id else None
    uri = cfg.get("uri", os.getenv("NEO4J_URI", "bolt://localhost:7687")) if cfg else os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = cfg.get("user", os.getenv("NEO4J_USER", "neo4j")) if cfg else os.getenv("NEO4J_USER", "neo4j")
    password = cfg.get("password", os.getenv("NEO4J_PASSWORD", "ragplatform")) if cfg else os.getenv("NEO4J_PASSWORD", "ragplatform")

    if replace:
        from adapters.neo4j_graph import GraphUnavailable, Neo4jGraphRetriever
        retriever = Neo4jGraphRetriever(uri=uri, user=user, password=password)
        try:
            retriever.clear()
        except GraphUnavailable as e:
            raise HTTPException(status_code=502, detail=f"Neo4j unreachable at {uri}: {e}") from e
        except Exception as e:
            log.error("ingest_neo4j_cypher.replace_failed", realm_id=realm_id, uri=uri, error=str(e), exc_info=True)
            raise HTTPException(status_code=502, detail=f"Failed to clear the graph before loading: {e}") from e
        finally:
            retriever.close()

    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Neo4j unreachable at {uri}: {e}") from e

    executed = 0
    try:
        with driver.session() as session:
            for i, stmt in enumerate(statements):
                try:
                    session.run(stmt).consume()
                    executed += 1
                except Exception as e:
                    if isinstance(e, Neo4jError) and e.code in _BENIGN_SCHEMA_ALREADY_EXISTS_CODES:
                        # Found live, on a real 24609-statement
                        # apoc.export.cypher.all dump: statement 1 was a
                        # CREATE INDEX the export replays from the source
                        # graph's own schema, which failed with
                        # EquivalentSchemaRuleAlreadyExists because this
                        # target graph already had an equivalent index (from
                        # an earlier attempt at this exact upload —
                        # `replace=true` only clears nodes/relationships via
                        # Neo4jGraphRetriever.clear(), never schema, so a
                        # retry after a partial failure keeps whatever
                        # indexes/constraints the first attempt already
                        # created). The schema object the statement wanted
                        # already exists in equivalent form — not a real
                        # failure — so it's counted as applied and the loop
                        # continues, rather than aborting a huge load over a
                        # statement that's already a no-op.
                        log.info(
                            "ingest_neo4j_cypher.schema_already_exists", realm_id=realm_id,
                            index=i, total=len(statements), code=e.code,
                        )
                        executed += 1
                        continue
                    log.error(
                        "ingest_neo4j_cypher.statement_failed", realm_id=realm_id, index=i,
                        total=len(statements), executed=executed, error=str(e),
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            f"Cypher statement {i + 1}/{len(statements)} failed: {e}. "
                            f"{executed} statement(s) already applied before this — the graph "
                            "is left partially modified, not rolled back."
                        ),
                    ) from e
    finally:
        driver.close()

    doc = await _register_corpus(
        realm_id=realm_id, corpus_id=corpus_id, storage_type="graph",
        backends={"neo4j": {"uri": uri}}, owner="platform",
        description=f"Loaded from Cypher script {file.filename!r} ({executed} statements)",
    )
    return {"corpus": doc, "statements_executed": executed, "replaced": replace}


def _decode_ndjson_bytes(raw_bytes: bytes) -> tuple[str, str | None]:
    """Decodes an uploaded NDJSON file's bytes tolerantly instead of hard-
    failing the whole upload over one bad byte: strict UTF-8, then strict
    cp1251 (Windows-1251 — the common legacy encoding for Russian/Belarusian
    text this platform's own content skews toward, see adapters/opensearch.py's
    ru_be_analyzer), then UTF-8 with unrepresentable bytes replaced (U+FFFD)
    as a last resort. Never raises for encoding reasons — the per-line JSON
    parse in the caller already discards whatever doesn't come out as valid
    JSON, so a replaced byte degrades only the one line it's in, not the
    whole batch of otherwise-valid lines around it.

    Found live: an earlier version's cp1251-fallback error message reused
    the *original* UTF-8 exception's text even when it was actually the
    cp1251 attempt that failed — always describing the same byte/position
    regardless of what actually went wrong, making a genuine second failure
    indistinguishable from the first. Centralizing both attempts here (each
    keeping its own message) and only ever warning, never raising, sidesteps
    that class of bug entirely rather than fixing the message a third time.

    Found live, again, right after that fix: the *final* fallback tier used
    to be cp1251-with-replace over the WHOLE file, not just the bad byte.
    For a file that's genuinely, fully cp1251 (no multi-byte sequences
    anywhere), strict cp1251 (the tier above) already succeeds cleanly, so
    this tier is only ever reached when strict cp1251 ALSO raised somewhere
    — meaning the file is most likely mostly-UTF-8 with one or two stray
    bad bytes, not actually cp1251 at all. Decoding a mostly-UTF-8 byte
    stream via cp1251 (a single-byte legacy encoding with no concept of
    multi-byte sequences) reinterprets every Cyrillic character's 2+ UTF-8
    bytes as 2+ separate, wrong cp1251 characters — mojibake across the
    *entire* file, not just the one bad spot, verified directly: a 3-line
    file with valid UTF-8 Cyrillic in lines 1 and 3 and one bad byte in
    line 2 decoded via cp1251-replace turned "Привет мир" into
    "РџСЂРёРІРµС‚ РјРёСЂ" — every other line in the file, garbled by the
    same fallback meant to rescue just the one bad line. UTF-8 with
    errors="replace" instead leaves every well-formed multi-byte sequence
    completely untouched and only substitutes U+FFFD for the actual
    offending byte(s) — the correct last resort when the file is
    predominantly UTF-8.

    Returns `(decoded_text, warning_or_None)` — the caller logs the warning
    when present."""
    try:
        return raw_bytes.decode("utf-8"), None
    except UnicodeDecodeError as e:
        utf8_msg = str(e)
    try:
        return raw_bytes.decode("cp1251"), None
    except UnicodeDecodeError as e:
        cp1251_msg = str(e)
    text = raw_bytes.decode("utf-8", errors="replace")
    warning = (
        f"Not valid UTF-8 ({utf8_msg}) or valid cp1251 ({cp1251_msg}) — decoded as UTF-8 with "
        "unrepresentable bytes replaced (U+FFFD); any line containing one will likely fail JSON "
        "parsing and be skipped rather than indexed."
    )
    return text, warning


def _decompress_upload_bytes(raw_bytes: bytes) -> bytes:
    """Transparently decompresses `raw_bytes` if it's gzip (`\\x1f\\x8b`
    magic bytes) or a zip archive (`PK\\x03\\x04` magic bytes, reads the
    first `.json`/`.ndjson`/`.jsonl`-named entry, or else the archive's
    first entry) — dumps are routinely shipped compressed to save space.
    Returns `raw_bytes` unchanged if neither magic-byte prefix matches.

    Shared between [[services/api_gateway/routers/corpus.py#ingest_opensearch_dump]]
    (where this was first found live: a `.gz` upload used to hit
    `.decode("utf-8")` directly and raise an unhandled `UnicodeDecodeError`
    on the compressed bytes' own first byte) and
    [[services/api_gateway/routers/corpus.py#ingest_neo4j_chunks]], which
    accepts dumps in the same NDJSON/JSON shape."""
    if raw_bytes[:2] == b"\x1f\x8b":
        import gzip
        try:
            return gzip.decompress(raw_bytes)
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"File looks gzip-compressed but failed to decompress: {e}") from e
    if raw_bytes[:4] == b"PK\x03\x04":
        import io
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                names = zf.namelist()
                if not names:
                    raise HTTPException(status_code=400, detail="Uploaded zip archive contains no files")
                preferred = next(
                    (n for n in names if n.lower().endswith((".json", ".ndjson", ".jsonl"))), names[0],
                )
                return zf.read(preferred)
        except zipfile.BadZipFile as e:
            raise HTTPException(status_code=400, detail=f"File looks zip-compressed but failed to open: {e}") from e
    return raw_bytes


def _parse_json_or_ndjson_objects(raw: str) -> tuple[list[dict[str, Any]], int]:
    """Parses `raw` as either one JSON value (a top-level array of objects,
    or a single object) or NDJSON (one JSON object per line) — whole-
    document parsing is tried FIRST, not as a fallback after NDJSON line-by-
    line parsing comes up empty.

    Found live (in [[services/api_gateway/routers/corpus.py#ingest_opensearch_dump]],
    where this logic first lived): trying NDJSON-first on a pretty-printed
    JSON array silently kept only the array's own LAST element (the only
    one with no trailing comma before the closing `]`, so it happens to
    also be valid JSON on its own line) — every other element, each with a
    trailing comma, failed the per-line parse and was silently dropped, with
    `actions`/whatever the caller built therefore never coming up empty, so
    the whole-document fallback never even ran. Trying whole-document
    parsing first side-steps that class of bug entirely.

    Returns `(objects, n_skipped)` — `n_skipped` counts non-object JSON
    values (a line that's syntactically valid JSON but not an object, e.g. a
    bare number/string/array/null) and invalid-JSON lines, dropped rather
    than raising."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    n_skipped = 0
    objs: list[dict[str, Any]] = []
    if isinstance(parsed, (list, dict)):
        candidates = parsed if isinstance(parsed, list) else [parsed]
        for obj in candidates:
            if isinstance(obj, dict):
                objs.append(obj)
            else:
                n_skipped += 1
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if isinstance(obj, dict):
                objs.append(obj)
            else:
                n_skipped += 1
    return objs, n_skipped


# Found live: a real-world Neo4j :Chunk-graph export, generated by a custom
# script (not apoc.export.cypher.all, despite ingest_neo4j_cypher's own
# docstring assuming that), inlined free-text chunk content as literal
# Cypher string values instead of driver parameters — the exact same
# chunk_id/doc_id/text/structural_path shape
# [[adapters/neo4j_graph.py#Neo4jGraphRetriever#add_chunks]] already writes
# via $-parameters. Embedded, unescaped `"` characters in the free-form
# (English/Russian technical documentation) text broke the Cypher parser
# ("must contain an even number of non-escaped quotes") partway through a
# 24609-statement load. Not fixable after the fact — there is no reliable
# way to tell "this is the end of the string" from "this is a literal quote
# inside the text" once the escaping is already wrong, and a regex "repair"
# risks silently corrupting chunk text. This endpoint sidesteps the whole
# class of bug structurally: every value goes through `add_chunks()` as a
# driver parameter, never Cypher text, so embedded quotes/newlines/anything
# else in `text` can never break query syntax.
@router.post("/ingest/neo4j-chunks")
async def ingest_neo4j_chunks(
    file: UploadFile = File(...),
    realm_id: str = Form(...),
    corpus_id: str = Form(...),
    replace: bool = Form(False),
) -> dict[str, Any]:
    """Bulk-loads `:Chunk` nodes (+ per-document `NEXT` adjacency) into the
    Realm's Neo4j graph from an uploaded JSON/NDJSON dump of
    `{chunk_id, doc_id, text, structural_path?}` objects — the same shape
    [[adapters/neo4j_graph.py#Neo4jGraphRetriever#add_chunks]] itself
    writes. `keywords` isn't read from the dump — `add_chunks()` always
    recomputes it from `text`, the same as an in-process ingest would.

    Same NDJSON-primary / whole-JSON-fallback / gzip+zip transparent
    decompression as [[services/api_gateway/routers/corpus.py#ingest_opensearch_dump]]
    (`_decompress_upload_bytes`/`_parse_json_or_ndjson_objects`, shared with
    that endpoint) — see its docstring for the reasoning behind each of
    those choices.

    `replace=true` wipes the whole graph first via
    [[adapters/neo4j_graph.py#Neo4jGraphRetriever#clear]], same as
    [[services/api_gateway/routers/corpus.py#ingest_neo4j_cypher]]'s own
    `replace` option and the same caveat: this is the ENTIRE Realm's graph,
    not just this corpus_id's slice — Neo4j has no corpus_id partitioning to
    scope a narrower wipe to."""
    from core.models import Chunk

    cfg = await _get_realm_resource(realm_id, "neo4j") if realm_id else None
    uri = cfg.get("uri", os.getenv("NEO4J_URI", "bolt://localhost:7687")) if cfg else os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = cfg.get("user", os.getenv("NEO4J_USER", "neo4j")) if cfg else os.getenv("NEO4J_USER", "neo4j")
    password = cfg.get("password", os.getenv("NEO4J_PASSWORD", "ragplatform")) if cfg else os.getenv("NEO4J_PASSWORD", "ragplatform")

    raw_bytes = _decompress_upload_bytes(await file.read())
    raw, decode_warning = _decode_ndjson_bytes(raw_bytes)
    if decode_warning:
        log.warning(
            "ingest_neo4j_chunks.decode_fallback", realm_id=realm_id, corpus_id=corpus_id,
            warning=decode_warning,
        )

    objs, n_skipped = _parse_json_or_ndjson_objects(raw)
    chunks: list[Chunk] = []
    for obj in objs:
        if not obj.get("chunk_id") or not obj.get("doc_id"):
            n_skipped += 1
            continue
        chunks.append(Chunk(
            chunk_id=str(obj["chunk_id"]),
            doc_id=str(obj["doc_id"]),
            text=str(obj.get("text", "")),
            structural_path=str(obj.get("structural_path") or obj.get("path") or ""),
        ))
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No valid chunk objects (each needs at least chunk_id and doc_id) found in the "
            "uploaded file (tried NDJSON and a single JSON array/object — neither produced one)",
        )

    from adapters.neo4j_graph import GraphUnavailable, Neo4jGraphRetriever
    retriever = Neo4jGraphRetriever(uri=uri, user=user, password=password)
    try:
        if replace:
            retriever.clear()
        retriever.ensure_indexes()
        n_written = retriever.add_chunks(chunks)
    except GraphUnavailable as e:
        raise HTTPException(status_code=502, detail=f"Neo4j unreachable at {uri}: {e}") from e
    except Exception as e:
        log.error(
            "ingest_neo4j_chunks.write_failed", realm_id=realm_id, corpus_id=corpus_id,
            error=str(e), exc_info=True,
        )
        raise HTTPException(status_code=502, detail=f"Failed to write chunks to Neo4j: {e}") from e
    finally:
        retriever.close()

    doc = await _register_corpus(
        realm_id=realm_id, corpus_id=corpus_id, storage_type="graph",
        backends={"neo4j": {"uri": uri}}, owner="platform",
        description=f"Bulk-loaded from Neo4j chunk dump {file.filename!r} ({n_written} chunks)",
    )
    return {
        "corpus": doc, "n_written": n_written, "n_skipped": n_skipped, "replaced": replace,
        "decode_warning": decode_warning,
    }


@router.post("/ingest/opensearch-dump")
async def ingest_opensearch_dump(
    file: UploadFile = File(...),
    realm_id: str = Form(...),
    corpus_id: str = Form(...),
    strategy: str = Form("structure_aware"),
    replace: bool = Form(False),
) -> dict[str, Any]:
    """Bulk-indexes documents from an uploaded JSON dump into a realm-scoped
    OpenSearch index — the primary expected shape is newline-delimited JSON
    (NDJSON, one object per line), the same document shape
    [[adapters/opensearch.py#OpenSearchRetriever#index_chunks]] itself writes
    (`text`/`doc_id`/`chunk_id`/`strategy_id`/`structural_path`/`metadata`;
    `chunk_id` becomes the OpenSearch document `_id`). A plain JSON file (a
    top-level array of documents, or a single document) is also accepted —
    tried as a fallback once NDJSON line-by-line parsing finds zero valid
    documents (found live: a pretty-printed JSON array's own lines, like
    `"["`/`"  {"`, aren't valid JSON on their own, so every one of them
    silently failed the per-line parse and the whole upload 400'd as "no
    valid lines" even though the file's actual content was perfectly good
    JSON, just not NDJSON-shaped).

    Transparently decompresses the upload first if it's gzip or a zip
    archive (`_decompress_upload_bytes`, shared with
    [[services/api_gateway/routers/corpus.py#ingest_neo4j_chunks]]) — dumps
    are routinely shipped compressed to save space. Found live: a `.gz`
    upload previously hit `(await file.read()).decode("utf-8")` directly and
    raised an unhandled `UnicodeDecodeError` (500) on the compressed bytes'
    first byte, since nothing here expected anything but plain-text NDJSON.

    Default behavior is an upsert by `chunk_id`, NOT a full index replace
    (unlike the Qdrant snapshot endpoint above, which always replaces the
    whole collection) — found live, asked directly: re-uploading a *new*
    version of a dump with fewer/different chunk_ids left the old, no-longer-
    present chunks stranded in the index, still served in search results.
    `replace=true` drops the index first (if it exists) so the index ends up
    containing exactly this dump's documents, nothing left over from before.

    Deliberately NOT OpenSearch's own native snapshot/restore feature — that
    needs the snapshot's files to already exist on the shared filesystem path
    OpenSearch reads snapshots from (`path.repo`), and the API gateway runs as
    a host process, not a container (same constraint
    [[services/api_gateway/routers/realms.py#provision_realm_neo4j]] has for
    Neo4j) — it has no direct filesystem access to that Docker volume without
    also shelling out to `docker volume inspect` to find the real mountpoint,
    which this endpoint does not attempt. NDJSON bulk-indexing sidesteps that
    gap entirely by talking to OpenSearch's regular document API instead of
    its snapshot API — simpler, and not a byte-for-byte OpenSearch snapshot
    restore."""
    from adapters.opensearch import _INDEX_SETTINGS, _index_name
    index = _index_name(strategy, corpus_id, realm_id)

    cfg = await _get_realm_resource(realm_id, "opensearch") if realm_id else None
    host = cfg.get("host", "localhost") if cfg else os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(cfg.get("port", 9200)) if cfg else int(os.getenv("OPENSEARCH_PORT", "9200"))

    from opensearchpy import OpenSearch
    from opensearchpy.helpers import bulk
    client = OpenSearch(hosts=[{"host": host, "port": port}], use_ssl=False)

    raw_bytes = _decompress_upload_bytes(await file.read())
    raw, decode_warning = _decode_ndjson_bytes(raw_bytes)
    if decode_warning:
        log.warning(
            "ingest_opensearch_dump.decode_fallback", realm_id=realm_id, corpus_id=corpus_id,
            warning=decode_warning,
        )

    # Found live: a genuinely plain-JSON dump (a top-level array of
    # documents, or a single document — pretty-printed or not, not
    # one-object-per-line) must be tried as ONE JSON value FIRST, not as a
    # fallback after NDJSON line-by-line parsing comes up empty — see
    # _parse_json_or_ndjson_objects's own docstring for why (trying NDJSON
    # first on a pretty-printed array silently kept only its last element).
    objs, n_skipped = _parse_json_or_ndjson_objects(raw)
    actions = [{"_index": index, "_id": obj.get("chunk_id") or str(uuid.uuid4()), "_source": obj} for obj in objs]

    if not actions:
        raise HTTPException(
            status_code=400,
            detail="No valid NDJSON document lines found in the uploaded file (also tried parsing it as a "
            "single JSON array/object — neither produced a valid document)",
        )

    # Index setup (exists/delete/create) is just as capable of failing as the
    # bulk call itself — found live: a cluster-wide `cluster.blocks.create_index`
    # block (OpenSearch refusing new indices, e.g. under disk pressure) raised
    # from client.indices.create() below, which sat outside this try/except
    # and surfaced as a raw unhandled 500 with a full stack trace instead of
    # a clean 502, unlike every other failure mode here.
    try:
        index_existed = client.indices.exists(index=index)
        if replace and index_existed:
            client.indices.delete(index=index)
            index_existed = False
        if not index_existed:
            client.indices.create(index=index, body=_INDEX_SETTINGS)
        n_indexed, errors = bulk(client, actions, raise_on_error=False)
    except Exception as e:
        log.error("ingest_opensearch_dump.bulk_failed", realm_id=realm_id, corpus_id=corpus_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=502, detail=f"OpenSearch bulk index failed: {e}") from e

    doc = await _register_corpus(
        realm_id=realm_id, corpus_id=corpus_id, storage_type="sparse_only",
        backends={"opensearch": {"index": index}}, owner="platform",
        description=f"Bulk-loaded from OpenSearch dump {file.filename!r}",
    )
    return {
        "corpus": doc, "index": index, "n_indexed": n_indexed, "n_skipped": n_skipped,
        "n_errors": len(errors) if isinstance(errors, list) else errors, "replaced": replace,
        "decode_warning": decode_warning,
    }


@router.get("/{corpus_id}/chunks")
async def browse_chunks(
    corpus_id: str,
    offset: str | None = None,
    limit: int = 50,
    q: str = "",
    strategy: str = "structure_aware",
    embedder: str = "bge_m3",
    realm_id: str | None = None,
) -> dict[str, Any]:
    """Paginate real indexed chunks for a corpus — the corpus content browser."""
    qdrant_cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
    try:
        qdrant = _resolve_qdrant(corpus_id, strategy, embedder, qdrant_cfg, realm_id)
        chunks, next_offset = qdrant.scroll(offset=offset, limit=limit, text_filter=q)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Corpus index unavailable: {e}") from e

    seen_texts: set[str] = set()
    items = []
    for c in chunks:
        is_dup = c.text in seen_texts
        seen_texts.add(c.text)
        items.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "structural_path": c.structural_path,
            "text": c.text,
            "length": len(c.text),
            "header_only": bool(c.structural_path) and len(c.text.strip()) < 40,
            "duplicate": is_dup,
        })
    return {"corpus_id": corpus_id, "items": items, "next_offset": next_offset}


_NEAR_DUP_SAMPLE_SIZE = 300


@router.get("/{corpus_id}/health")
async def corpus_health(
    corpus_id: str,
    strategy: str = "structure_aware",
    embedder: str = "bge_m3",
    realm_id: str | None = None,
) -> dict[str, Any]:
    """Preventive corpus diagnostics — catches bad indexing before a run.

    Includes the two fast sampled detectors (near-duplicate via Qdrant k-NN,
    language mix) alongside the pre-existing deterministic ones — all fast
    enough to run synchronously on page load. The LLM-judge deep diagnostics
    (Ragas/TruLens/chunk coherence) are intentionally NOT here — see the
    separate /diagnostics/* endpoints below, triggered on demand.
    """
    import random

    from core.eval.corpus_health import analyze, detect_near_duplicates

    qdrant_cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
    try:
        qdrant = _resolve_qdrant(corpus_id, strategy, embedder, qdrant_cfg, realm_id)
        chunks = qdrant.scroll_all()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Corpus index unavailable: {e}") from e

    health = analyze(chunks).to_dict()

    chunk_ids = [c.chunk_id for c in chunks if c.chunk_id]
    sample_ids = (
        random.sample(chunk_ids, _NEAR_DUP_SAMPLE_SIZE)
        if len(chunk_ids) > _NEAR_DUP_SAMPLE_SIZE
        else chunk_ids
    )
    near_dup_item = detect_near_duplicates(qdrant, sample_ids) if sample_ids else None
    if near_dup_item:
        health["items"].append(near_dup_item.to_dict())

    # Each finding names the catalogue entries it is evidence for, so a reader
    # can tell a known failure from a sentence about this corpus.
    from services.api_gateway.routers.atlas import attach_failure_ids
    attach_failure_ids(health["items"], "health")

    return {"corpus_id": corpus_id, **health}


# ── Graph community diagnostics (Stage "graph viz") ──────────────────────────
#
# corpus_id here only selects which Qdrant collection to read source_code
# labels from — the Neo4j graph itself has no corpus_id partitioning (see
# adapters/neo4j_graph.py:clear() docstring), so community detection always
# runs over the whole graph, never filtered to one corpus_id.

def _doc_to_source_code_map(
    corpus_id: str, strategy: str, embedder: str, cfg: dict[str, Any] | None = None,
    realm_id: str | None = None,
) -> dict[str, str]:
    """Full doc_id -> source_code map for one corpus, paginating past
    QdrantRetriever.scroll_all()'s 5000-chunk cap (this corpus has ~35k)."""
    qdrant = _resolve_qdrant(corpus_id, strategy, embedder, cfg, realm_id)
    mapping: dict[str, str] = {}
    offset: str | None = None
    while True:
        chunks, offset = qdrant.scroll(offset=offset, limit=1000)
        if not chunks:
            break
        for c in chunks:
            source_code = c.metadata.get("source_code")
            if c.doc_id and source_code:
                mapping[c.doc_id] = source_code
        if offset is None:
            break
    return mapping


def _label_communities(communities: list[dict[str, Any]], doc_to_code: dict[str, str]) -> None:
    """Mutates each community dict in place, adding a source_code
    distribution — the diagnostic this whole feature exists for: do the
    communities Leiden/Louvain find actually line up with corpus/source
    boundaries, or are they driven by something else (shared vocabulary
    noise)? See adapters/neo4j_graph.py module comment for what was found
    on this corpus before this endpoint existed."""
    import collections

    for community in communities:
        codes = [doc_to_code.get(d) for d in community["doc_ids"] if doc_to_code.get(d)]
        counts = collections.Counter(codes)
        total = len(codes)
        top = counts.most_common(5)
        community["source_code_distribution"] = [
            {"source_code": code, "count": n} for code, n in top
        ]
        community["dominant_source_share"] = (top[0][1] / total) if top and total else 0.0
        community["distinct_source_count"] = len(counts)
        del community["doc_ids"]  # internal-only — replaced by the summary above


@router.get("/{corpus_id}/graph/communities")
async def graph_communities(
    corpus_id: str,
    algorithm: str = "leiden",
    edge_type: str = "lexical",
    strategy: str = "structure_aware",
    embedder: str = "bge_m3",
    realm_id: str | None = None,
) -> dict[str, Any]:
    """Community detection over the whole Neo4j graph (GDS Leiden/Louvain),
    labeled with each community's source_code mix for this corpus_id's
    Qdrant collection. ``edge_type`` selects which relationship signal to
    cluster on — "lexical" (RELATED, shared-keyword) or "semantic"
    (RELATED_SEMANTIC, embedding cosine) — kept as two distinct,
    independently-cacheable results rather than one replacing the other.
    See adapters/neo4j_graph.py:detect_communities()."""
    from adapters.neo4j_graph import GraphUnavailable

    neo4j_cfg = await _get_realm_resource(realm_id, "neo4j") if realm_id else None
    qdrant_cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
    # A Realm with no neo4j resource of its own reads the instance named by the
    # environment, which is the same instance every other such Realm reads.
    # Whatever is found there belongs to some other Realm, and returning it as
    # this one's communities signs another Realm's graph with this Realm's name.
    # That is what the demo Realm did: it reported the communities of a corpus
    # belonging to a different Realm entirely. The only real isolation is a
    # container of its own (see the design notes).
    if realm_id and neo4j_cfg is None:
        return {
            "corpus_id": corpus_id,
            "realm_scoped": False,
            "community_count": 0,
            "modularity": None,
            "communities": [],
            "inter_community_edges": [],
        }
    graph = _resolve_neo4j(neo4j_cfg)
    try:
        result = graph.detect_communities(algorithm=algorithm, edge_type=edge_type)
    except GraphUnavailable as e:
        raise HTTPException(status_code=503, detail=f"Graph unavailable: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        doc_to_code = _doc_to_source_code_map(corpus_id, strategy, embedder, qdrant_cfg, realm_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Corpus index unavailable: {e}") from e

    _label_communities(result["communities"], doc_to_code)
    return {"corpus_id": corpus_id, "realm_scoped": True, **result}


@router.get("/{corpus_id}/graph/communities/{community_id}")
async def graph_community_detail(
    corpus_id: str,
    community_id: int,
    algorithm: str = "leiden",
    edge_type: str = "lexical",
    realm_id: str | None = None,
) -> dict[str, Any]:
    """Drill-down: nodes + edges inside one community. Assumes
    GET /graph/communities already ran detect_communities(algorithm,
    edge_type) in this session — re-run it first if results look stale."""
    from adapters.neo4j_graph import GraphUnavailable

    neo4j_cfg = await _get_realm_resource(realm_id, "neo4j") if realm_id else None
    graph = _resolve_neo4j(neo4j_cfg)
    try:
        result = graph.community_subgraph(community_id, algorithm=algorithm, edge_type=edge_type)
    except GraphUnavailable as e:
        raise HTTPException(status_code=503, detail=f"Graph unavailable: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"corpus_id": corpus_id, **result}


# ── Deep diagnostics (LLM-judge, on demand) ──────────────────────────────────
#
# Unlike /health above (deterministic, sampled, fast enough for page load),
# these run real LLM-judge calls — slow (each makes one Ollama call per
# question/chunk) and meant to be triggered explicitly by a user action, not
# loaded automatically. Kept as separate endpoints rather than folding into
# /health for that reason: six independent chunk-quality diagnostics.

_DEFAULT_DIAGNOSTICS_DATASET = "handbook.v1.fast.jsonl"
_DEFAULT_DIAGNOSTICS_MAX_QUESTIONS = 10
_DEFAULT_CHUNK_COHERENCE_SAMPLE_SIZE = 150
_GOLDEN_DIR = Path("eval/golden")
# Hard ceiling, not just a default — Ragas/TruLens make 3+ LLM-judge calls
# PER QUESTION (chunk-coherence: 1 per chunk). Letting this run unbounded
# against a large golden set (130+ questions) from a UI button with no
# progress indicator would tie up the request for many minutes per click.
_MAX_DIAGNOSTICS_QUESTIONS = 200
_MAX_CHUNK_COHERENCE_SAMPLE_SIZE = 500


async def _build_diagnostics_pipeline(
    corpus_id: str, pipeline_id: str, top_k: int, realm_id: str | None = None,
) -> tuple[Any, Any]:
    """Builds the pipeline for ad-hoc deep diagnostics through the SAME
    registry + ExperimentConfig path a real experiment run uses
    (ExperimentRunner._build_pipeline) — reusing it here means
    diagnostics see the actually-deployed embedder/generator (whatever
    /settings has active) and the real naive/hybrid_rrf/hybrid_weighted/
    graph pipeline wiring, not a hand-rolled standalone copy that could
    silently drift from what real experiments use.

    chunking_strategy/embedder/generator ComponentRefs below are required
    by ExperimentConfig's schema but unused on this path — _build_pipeline
    takes the embedder/generator from the resolved pipeline_id's own
    registered instance, not from these refs (chunking only matters at
    ingest time).

    Returns (pipeline, embedder_instance) — the embedder is also handed to
    RagasRunner directly (see eval/ragas_runner.py) rather than letting it
    spin up a second, separate embedding model for its ResponseRelevancy
    metric: not every Ollama deployment serves embeddings on its chat
    endpoint, and BGE-M3 is the embedder this platform already relies on.

    `realm_id` — without it this diagnostics pipeline hit the
    gateway's startup-time env-var Qdrant/OpenSearch regardless of which
    Realm's corpus_id was asked for, same leak as _for_the_corpus's
    docstring describes for real experiment runs.
    """
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentRunner
    from core.registry import registry

    config = ExperimentConfig(
        name="deep-diagnostics",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
        pipeline_id=pipeline_id,
        corpus_id=corpus_id,
        top_k=top_k,
    )
    qdrant_cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
    opensearch_cfg = await _get_realm_resource(realm_id, "opensearch") if realm_id else None
    runner = ExperimentRunner(registry=registry)
    pipeline = runner._build_pipeline(config, realm_id or "", qdrant_cfg, opensearch_cfg)
    return pipeline, pipeline._embedder


async def _load_diagnostics_dataset(dataset: str, realm_id: str | None = None) -> Any:
    """The deep-diagnostics dataset: from the database first, then from a file.

    This used to look only for a file under eval/golden/, which holds one
    demonstration dataset. A Realm's own datasets live in Mongo, so RAGAS and
    TruLens failed on every Realm except the demo one with "Control dataset not
    found". The button carrying the default failed at nothing and was worse for
    it: another Realm's corpus was checked with questions about the demo
    handbook, honestly returning context_recall = 0.

    The same order a run follows (experiments.py#_load_dataset): the database is
    the source, and the file is what remains from before it existed.
    """
    import adapters.mongodb as mdb
    from eval.dataset import EvalDataset

    query: dict[str, Any] = {"filename": dataset}
    if realm_id:
        query["realm_id"] = realm_id
    try:
        doc = await mdb.find_one("datasets", query) or await mdb.find_one("datasets", {"filename": dataset})
    except Exception:
        # An unreachable database reads as "not in the database", which is what
        # the file fallback below exists for. Without this, the raw driver
        # timeout escaped through both diagnostics endpoints on any machine
        # with no Mongo running, including the fresh clone CI runs on, where
        # the shipped golden file would have answered perfectly well.
        doc = None
    if doc and doc.get("questions"):
        ds = EvalDataset.__new__(EvalDataset)
        ds.name = doc.get("name", dataset)
        ds.version = doc.get("version", "v0")
        ds.speed = doc.get("speed", "fast")
        ds.questions = doc["questions"]
        return ds

    path = _GOLDEN_DIR / dataset
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Control dataset not found: {dataset}")
    return EvalDataset.from_jsonl(path)


@router.post("/{corpus_id}/diagnostics/ragas")
async def run_ragas_diagnostics(
    corpus_id: str,
    pipeline_id: str = "naive",
    top_k: int = 5,
    dataset: str = _DEFAULT_DIAGNOSTICS_DATASET,
    max_questions: int = _DEFAULT_DIAGNOSTICS_MAX_QUESTIONS,
    realm_id: str | None = None,
) -> dict[str, Any]:
    from eval.ragas_runner import RagasRunner

    max_questions = min(max_questions, _MAX_DIAGNOSTICS_QUESTIONS)
    # Every external call sits inside the try, so a Mongo that is down surfaces
    # as the 503 that names it, not as a raw driver timeout escaping the
    # endpoint as a 500. The dataset loader itself treats an unreachable
    # database as "not in the database" and falls through to the file, which is
    # what lets the endpoint validate pipeline_id and return its 400 on a
    # machine where nothing is running at all.
    try:
        eval_dataset = await _load_diagnostics_dataset(dataset, realm_id)
        pipeline, embedder_instance = await _build_diagnostics_pipeline(corpus_id, pipeline_id, top_k, realm_id)
        runner = RagasRunner(pipeline=pipeline, embedder=embedder_instance)
        result = await asyncio.to_thread(runner.run, eval_dataset, max_questions=max_questions)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        # A missing dataset is a 404 in its own right, not "diagnostics
        # unavailable".
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Diagnostics unavailable: {e}") from e
    return {"corpus_id": corpus_id, "metrics": result.metrics, "per_question": result.per_question}


@router.post("/{corpus_id}/diagnostics/trulens")
async def run_trulens_diagnostics(
    corpus_id: str,
    pipeline_id: str = "naive",
    top_k: int = 5,
    dataset: str = _DEFAULT_DIAGNOSTICS_DATASET,
    max_questions: int = _DEFAULT_DIAGNOSTICS_MAX_QUESTIONS,
    realm_id: str | None = None,
) -> dict[str, Any]:
    from eval.trulens_runner import TruLensRunner

    max_questions = min(max_questions, _MAX_DIAGNOSTICS_QUESTIONS)
    # Same order and the same reasoning as the ragas endpoint above.
    try:
        eval_dataset = await _load_diagnostics_dataset(dataset, realm_id)
        pipeline, _ = await _build_diagnostics_pipeline(corpus_id, pipeline_id, top_k, realm_id)
        runner = TruLensRunner(pipeline=pipeline)
        result = await asyncio.to_thread(runner.run, eval_dataset, max_questions=max_questions)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        # A missing dataset is a 404 in its own right, not "diagnostics
        # unavailable".
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Diagnostics unavailable: {e}") from e
    return {"corpus_id": corpus_id, "metrics": result.metrics, "per_question": result.per_question}


@router.post("/{corpus_id}/diagnostics/chunk-coherence")
async def run_chunk_coherence_diagnostics(
    corpus_id: str,
    strategy: str = "structure_aware",
    embedder: str = "bge_m3",
    sample_size: int = _DEFAULT_CHUNK_COHERENCE_SAMPLE_SIZE,
    realm_id: str | None = None,
) -> dict[str, Any]:
    from eval.chunk_coherence_judge import ChunkCoherenceJudge

    sample_size = min(sample_size, _MAX_CHUNK_COHERENCE_SAMPLE_SIZE)
    qdrant_cfg = await _get_realm_resource(realm_id, "qdrant") if realm_id else None
    try:
        qdrant = _resolve_qdrant(corpus_id, strategy, embedder, qdrant_cfg, realm_id)
        chunks = qdrant.scroll_all()
        judge = ChunkCoherenceJudge(sample_size=sample_size)
        result = await asyncio.to_thread(judge.run, chunks, corpus_id=corpus_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Diagnostics unavailable: {e}") from e
    incoherent = sorted(
        (c for c in result.per_chunk if c["score"] <= 2),
        key=lambda c: c["score"],
    )
    return {
        "corpus_id": corpus_id,
        "metrics": result.metrics,
        "sample_size": result.sample_size,
        "incoherent_chunks": incoherent,
    }


@router.delete("/{job_id}", status_code=204)
async def delete_ingest(job_id: str) -> None:
    try:
        import adapters.mongodb as mdb
        n = await mdb.delete_one("corpus_ingests", {"job_id": job_id})
        if n == 0:
            raise HTTPException(status_code=404, detail=f"Ingest {job_id!r} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.websocket("/progress/{job_id}")
async def corpus_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        sent = 0
        while True:
            events = _progress.get(job_id, [])
            for event in events[sent:]:
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
                sent += 1
                if event.get("type") in ("done", "error"):
                    return
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
