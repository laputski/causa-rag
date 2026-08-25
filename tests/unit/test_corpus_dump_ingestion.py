"""services/api_gateway/routers/corpus.py — alternate dump ingestion endpoints.

Covers the three "load a dump directly into one backend" paths added
alongside /corpus/ingest's own text-chunking pipeline: POST
/corpus/ingest/qdrant-snapshot, /neo4j-cypher, /opensearch-dump. See
the design notes "Corpus registry CRUD" section for why each backend's
mechanics differ (replace-not-merge for Qdrant, arbitrary-code-execution
safety gate for Neo4j, NDJSON bulk-index instead of native snapshot/restore
for OpenSearch).
"""
from __future__ import annotations

import importlib.util
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.corpus import router

# `pytest` on a fresh clone with the `dev` extra alone must be green, so a test
# needing an optional extra skips instead of failing. These reach the real
# driver: the endpoint raises "neo4j driver not installed" before any mock in
# the test body can stand in for it.
_needs_neo4j = pytest.mark.skipif(
    importlib.util.find_spec("neo4j") is None,
    reason="needs the neo4j driver: pip install -e '.[graph]'",
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


# ── POST /corpus/ingest/qdrant-snapshot ─────────────────────────────────────────

def test_ingest_qdrant_snapshot_recovers_and_registers_corpus(client) -> None:
    mock_qdrant_client = MagicMock()
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)) as mock_insert:
        resp = client.post(
            "/corpus/ingest/qdrant-snapshot",
            files={"file": ("dump.snapshot", b"fake-snapshot-bytes", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus"]["realm_id"] == "acme"
    assert body["corpus"]["corpus_id"] == "manuals"
    assert body["corpus"]["backends"]["qdrant"]["collection"] == body["collection"]
    mock_qdrant_client.http.snapshots_api.recover_from_uploaded_snapshot.assert_called_once()
    call_kwargs = mock_qdrant_client.http.snapshots_api.recover_from_uploaded_snapshot.call_args.kwargs
    assert call_kwargs["collection_name"] == body["collection"]
    mock_insert.assert_called_once()


def test_ingest_qdrant_snapshot_502_when_recover_fails(client) -> None:
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.http.snapshots_api.recover_from_uploaded_snapshot.side_effect = RuntimeError("corrupt snapshot")
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
        resp = client.post(
            "/corpus/ingest/qdrant-snapshot",
            files={"file": ("dump.snapshot", b"bad-bytes", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 502
    assert "corrupt snapshot" in resp.json()["detail"]


def test_ingest_qdrant_snapshot_502_surfaces_full_error_not_truncated_preview(client) -> None:
    """UnexpectedResponse.__str__() truncates the raw response body to a
    short "..." preview — found live, real incident: the actual Qdrant-side
    reason ("failed to restore RocksDB backup: NotFound: Backup not found")
    was cut off in the gateway's own log, forcing a trip into the Qdrant
    container's logs to see it. .structured() must be used instead so the
    full error reaches both the log and the HTTP response."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    long_error = "failed to restore RocksDB backup: NotFound: Backup not found" + "x" * 300
    content = json.dumps({"status": {"error": long_error}}).encode()
    exc = UnexpectedResponse(status_code=500, reason_phrase="Internal Server Error", content=content, headers={})
    # Confirms the premise: str(exc) really does truncate, so the fix is load-bearing.
    assert long_error not in str(exc)

    mock_qdrant_client = MagicMock()
    mock_qdrant_client.http.snapshots_api.recover_from_uploaded_snapshot.side_effect = exc
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
        resp = client.post(
            "/corpus/ingest/qdrant-snapshot",
            files={"file": ("dump.snapshot", b"bad-bytes", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    # This exact wording also matches the "not a real Qdrant snapshot" hint
    # signature (see test_ingest_qdrant_snapshot_502_adds_hint_for_backup_not_found,
    # which pins that shape/behavior) — `detail` is a {message, hint} object
    # here, not a plain string; this test only cares that the full,
    # untruncated error text reached `message`.
    message = detail["message"] if isinstance(detail, dict) else detail
    assert long_error in message


def test_ingest_qdrant_snapshot_502_adds_hint_for_backup_not_found(client) -> None:
    """Found live: a real user saw the raw "failed to restore RocksDB backup:
    NotFound: Backup not found" Qdrant-internals string with no indication of
    what to actually do — this signature specifically means the uploaded
    file wasn't produced by Qdrant's own snapshot API. `detail` becomes a
    {message, hint} object when this exact signature is detected — `hint` is
    an i18n key (ui/src/pages/CorpusPage.tsx looks it up under
    corpusPage.upload.dumpErrorHints), not baked-in English prose, so the
    explanation follows the UI's selected language."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    qdrant_error = 'Service internal error: Failed to restore snapshot from "...": Service runtime error: failed to restore RocksDB backup: NotFound: Backup not found'
    content = json.dumps({"status": {"error": qdrant_error}}).encode()
    exc = UnexpectedResponse(status_code=500, reason_phrase="Internal Server Error", content=content, headers={})

    mock_qdrant_client = MagicMock()
    mock_qdrant_client.http.snapshots_api.recover_from_uploaded_snapshot.side_effect = exc
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
        resp = client.post(
            "/corpus/ingest/qdrant-snapshot",
            files={"file": ("dump.snapshot", b"bad-bytes", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert qdrant_error in detail["message"]
    assert detail["hint"] == "qdrantSnapshotNotNative"


def test_ingest_qdrant_snapshot_502_no_hint_for_unrelated_failure(client) -> None:
    """The friendly "not a real Qdrant snapshot" hint must only apply to the
    specific RocksDB-backup signature — a different failure (e.g. a dropped
    connection) shouldn't be told to regenerate a snapshot that may be fine."""
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.http.snapshots_api.recover_from_uploaded_snapshot.side_effect = RuntimeError("connection reset by peer")
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
        resp = client.post(
            "/corpus/ingest/qdrant-snapshot",
            files={"file": ("dump.snapshot", b"bad-bytes", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "connection reset by peer" in detail
    assert "wasn't produced by Qdrant's own snapshot API" not in detail


# ── POST /corpus/ingest/neo4j-cypher ─────────────────────────────────────────────

def test_ingest_neo4j_cypher_requires_confirm_flag(client) -> None:
    resp = client.post(
        "/corpus/ingest/neo4j-cypher",
        files={"file": ("dump.cypher", b"CREATE (n:Foo);", "text/plain")},
        data={"realm_id": "acme", "corpus_id": "manuals"},
    )
    assert resp.status_code == 400
    assert "confirm_arbitrary_cypher" in resp.json()["detail"]


def test_ingest_neo4j_cypher_400_when_no_statements(client) -> None:
    resp = client.post(
        "/corpus/ingest/neo4j-cypher",
        files={"file": ("dump.cypher", b"   \n  ", "text/plain")},
        data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
    )
    assert resp.status_code == 400


@_needs_neo4j
def test_ingest_neo4j_cypher_executes_statements_and_registers_corpus(client) -> None:
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE (a:Foo {id: 1});\nCREATE (b:Bar {id: 2});", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["statements_executed"] == 2
    assert body["corpus"]["storage_type"] == "graph"
    assert mock_session.run.call_count == 2
    mock_driver.close.assert_called_once()


# Found live: apoc.export.cypher.all's default ("cypher-shell") output wraps
# statement batches in `:begin`/`:commit` lines — cypher-shell REPL
# meta-commands, not Cypher. The naive `;`-split folded the leading `:begin`
# into the first real statement's own chunk, and Neo4j's parser rejected it
# immediately with "Invalid input ':'" since ':' is never valid Cypher at
# the start of a statement.
@_needs_neo4j
def test_ingest_neo4j_cypher_strips_begin_commit_markers(client) -> None:
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    dump = (
        b":begin\n"
        b"CREATE (a:Foo {id: 1});\n"
        b"CREATE (b:Bar {id: 2});\n"
        b":commit\n"
    )

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", dump, "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 200
    assert resp.json()["statements_executed"] == 2
    calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert calls == ["CREATE (a:Foo {id: 1})", "CREATE (b:Bar {id: 2})"]
    assert not any(":begin" in c or ":commit" in c for c in calls)


@_needs_neo4j
def test_ingest_neo4j_cypher_strips_other_colon_prefixed_meta_commands(client) -> None:
    """Not just :begin/:commit — any cypher-shell REPL meta-command
    (:param, :rollback, ...) is dropped the same way, since none of them
    are ever valid as the first character of a real Cypher statement."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    dump = (
        b":param rows => [1, 2]\n"
        b"CREATE (a:Foo {id: 1});\n"
    )

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", dump, "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 200
    assert resp.json()["statements_executed"] == 1
    calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert calls == ["CREATE (a:Foo {id: 1})"]


@_needs_neo4j
def test_ingest_neo4j_cypher_stops_at_first_failing_statement(client) -> None:
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    good_result = MagicMock()
    mock_session.run.side_effect = [good_result, RuntimeError("syntax error")]
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE (a:Foo);\nBAD CYPHER HERE;\nCREATE (c:Baz);", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 502
    assert "1 statement(s) already applied" in resp.json()["detail"]
    # The third statement must never run — stop at first failure, don't
    # silently continue past it leaving a half-applied script undetected.
    assert mock_session.run.call_count == 2


# Found live, on a real 24609-statement apoc.export.cypher.all dump:
# statement 1 was a CREATE INDEX the export replays from the source graph's
# own schema, which failed with EquivalentSchemaRuleAlreadyExists because
# the target graph already had an equivalent index — replace=true only
# clears nodes/relationships (Neo4jGraphRetriever.clear()), never schema,
# so a retry after a partial failure keeps whatever indexes/constraints the
# first attempt already created. This isn't a real failure and must not
# abort the rest of a huge script over a statement that's already a no-op.
@_needs_neo4j
def test_ingest_neo4j_cypher_tolerates_equivalent_schema_already_exists(client) -> None:
    from neo4j.exceptions import Neo4jError
    schema_error = Neo4jError._hydrate_neo4j(
        code="Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
        message="An equivalent index already exists",
    )
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    good_result = MagicMock()
    mock_session.run.side_effect = [schema_error, good_result]
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE INDEX FOR (n:Chunk) ON (n.keywords);\nCREATE (c:Foo);", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    # Both statements count as applied — the schema one is a no-op, not
    # skipped/discarded — and the loop must continue past it.
    assert body["statements_executed"] == 2
    assert mock_session.run.call_count == 2


@_needs_neo4j
def test_ingest_neo4j_cypher_still_fails_on_unrelated_schema_error(client) -> None:
    """The tolerance above is narrow — a different Neo4j schema error code
    (not "already exists") must still abort the script like any other
    genuine failure."""
    from neo4j.exceptions import Neo4jError
    other_error = Neo4jError._hydrate_neo4j(
        code="Neo.ClientError.Schema.ConstraintValidationFailed",
        message="Node already exists with label Chunk and property id",
    )
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.side_effect = [other_error]
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE CONSTRAINT FOR (n:Chunk) REQUIRE n.id IS UNIQUE;", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 502
    assert "0 statement(s) already applied" in resp.json()["detail"]


@_needs_neo4j
def test_ingest_neo4j_cypher_no_replace_by_default_leaves_graph_untouched(client) -> None:
    """Default (replace omitted/false) must not clear anything — matches the
    file's own existing behavior before `replace` was added."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE (a:Foo {id: 1});", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true"},
        )

    assert resp.status_code == 200
    assert resp.json()["replaced"] is False
    assert mock_session.run.call_count == 1
    assert "apoc.periodic.iterate" not in mock_session.run.call_args_list[0].args[0]


@_needs_neo4j
def test_ingest_neo4j_cypher_replace_clears_graph_before_loading_statements(client) -> None:
    """`replace=true` must wipe the whole graph (via
    Neo4jGraphRetriever.clear()'s batched apoc.periodic.iterate delete —
    reused rather than a second, hand-rolled DETACH DELETE that could
    reintroduce the silent-failure-on-large-graphs bug that method's own
    docstring documents) before the uploaded script's own statements run."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("neo4j.GraphDatabase.driver", return_value=mock_driver), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE (a:Foo {id: 1});", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true", "replace": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["replaced"] is True
    assert body["statements_executed"] == 1
    calls = mock_session.run.call_args_list
    assert len(calls) == 2
    # The clear runs BEFORE the uploaded script's own statement.
    assert "apoc.periodic.iterate" in calls[0].args[0]
    assert "DETACH DELETE" in calls[0].args[0]
    assert calls[1].args[0] == "CREATE (a:Foo {id: 1})"
    # Both the retriever's own driver and the endpoint's statement-execution
    # driver must be closed — two separate connections were opened.
    assert mock_driver.close.call_count == 2


@_needs_neo4j
def test_ingest_neo4j_cypher_replace_failure_returns_502_before_running_statements(client) -> None:
    """If clearing the graph fails, the uploaded script must never run —
    partially clearing then loading on top of it would leave an
    inconsistent, hard-to-diagnose graph state."""
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.clear", side_effect=RuntimeError("boom")), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("neo4j.GraphDatabase.driver") as mock_driver_ctor:
        resp = client.post(
            "/corpus/ingest/neo4j-cypher",
            files={"file": ("dump.cypher", b"CREATE (a:Foo {id: 1});", "text/plain")},
            data={"realm_id": "acme", "corpus_id": "manuals", "confirm_arbitrary_cypher": "true", "replace": "true"},
        )

    assert resp.status_code == 502
    assert "Failed to clear the graph" in resp.json()["detail"]
    # The endpoint's own statement-execution driver must never even be
    # constructed — the failure is caught before that point.
    mock_driver_ctor.assert_not_called()


# ── POST /corpus/ingest/neo4j-chunks ──────────────────────────────────────────────
#
# Found live: a real-world :Chunk-graph export inlined free-text chunk
# content as literal Cypher string values instead of driver parameters, and
# an unescaped embedded '"' character broke the parser partway through a
# 24609-statement load — not reliably fixable after the fact for arbitrary
# free text. This endpoint sidesteps the whole class of bug: every value
# goes through Neo4jGraphRetriever.add_chunks() as a driver parameter,
# never Cypher text.

def test_ingest_neo4j_chunks_loads_ndjson_and_registers_corpus(client) -> None:
    ndjson = (
        b'{"chunk_id": "c1", "doc_id": "d1", "text": "hello", "structural_path": "Intro"}\n'
        b'{"chunk_id": "c2", "doc_id": "d1", "text": "world"}\n'
    )
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", return_value=2) as mock_add, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_written"] == 2
    assert body["n_skipped"] == 0
    assert body["replaced"] is False
    assert body["corpus"]["storage_type"] == "graph"
    chunks_arg = mock_add.call_args.args[0]
    assert [c.chunk_id for c in chunks_arg] == ["c1", "c2"]
    assert chunks_arg[0].structural_path == "Intro"


def test_ingest_neo4j_chunks_preserves_embedded_quotes_and_newlines(client) -> None:
    """The exact bug this endpoint exists to sidestep: text containing
    unescaped quotes/newlines must reach add_chunks() completely intact,
    since it travels as a driver parameter, never as Cypher text."""
    tricky_text = 'He said "hello" and then\nwent home — "goodbye" too.'
    payload = json.dumps({"chunk_id": "c1", "doc_id": "d1", "text": tricky_text}).encode()
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", return_value=1) as mock_add, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.json", payload, "application/json")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert mock_add.call_args.args[0][0].text == tricky_text


def test_ingest_neo4j_chunks_falls_back_to_json_array(client) -> None:
    payload = json.dumps([
        {"chunk_id": "c1", "doc_id": "d1", "text": "a"},
        {"chunk_id": "c2", "doc_id": "d1", "text": "b"},
    ]).encode()
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", return_value=2) as mock_add, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.json", payload, "application/json")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert len(mock_add.call_args.args[0]) == 2


def test_ingest_neo4j_chunks_skips_objects_missing_required_fields(client) -> None:
    ndjson = (
        b'{"chunk_id": "c1", "doc_id": "d1", "text": "ok"}\n'
        b'{"chunk_id": "c2", "text": "missing doc_id"}\n'
        b'{"doc_id": "d1", "text": "missing chunk_id"}\n'
        b'not even json\n'
    )
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", return_value=1) as mock_add, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_skipped"] == 3
    assert len(mock_add.call_args.args[0]) == 1


def test_ingest_neo4j_chunks_400_when_no_valid_chunks(client) -> None:
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.ndjson", b"not json at all\n", "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400


def test_ingest_neo4j_chunks_replace_clears_graph_before_writing(client) -> None:
    ndjson = b'{"chunk_id": "c1", "doc_id": "d1", "text": "a"}\n'
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.clear") as mock_clear, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", return_value=1) as mock_add, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals", "replace": "true"},
        )

    assert resp.status_code == 200
    assert resp.json()["replaced"] is True
    mock_clear.assert_called_once()
    mock_add.assert_called_once()


def test_ingest_neo4j_chunks_write_failure_returns_502(client) -> None:
    ndjson = b'{"chunk_id": "c1", "doc_id": "d1", "text": "a"}\n'
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", side_effect=RuntimeError("boom")), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None) as mock_close:
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 502
    assert "Failed to write chunks to Neo4j" in resp.json()["detail"]
    # The retriever's connection must still be released even on failure.
    mock_close.assert_called_once()


def test_ingest_neo4j_chunks_transparently_decompresses_gzip(client) -> None:
    import gzip
    ndjson = b'{"chunk_id": "c1", "doc_id": "d1", "text": "a"}\n'
    gzipped = gzip.compress(ndjson)
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.add_chunks", return_value=1) as mock_add, \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.ensure_indexes", return_value=None), \
         patch("adapters.neo4j_graph.Neo4jGraphRetriever.close", return_value=None), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/neo4j-chunks",
            files={"file": ("dump.ndjson.gz", gzipped, "application/gzip")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert mock_add.call_args.args[0][0].chunk_id == "c1"


# ── POST /corpus/ingest/opensearch-dump ──────────────────────────────────────────

def test_ingest_opensearch_dump_bulk_indexes_and_registers_corpus(client) -> None:
    ndjson = b'{"chunk_id": "c1", "text": "hello", "doc_id": "d1"}\n{"chunk_id": "c2", "text": "world", "doc_id": "d1"}\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(2, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_indexed"] == 2
    assert body["n_skipped"] == 0
    assert body["corpus"]["storage_type"] == "sparse_only"
    mock_bulk.assert_called_once()
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1", "c2"}


def test_ingest_opensearch_dump_default_upserts_without_dropping_index(client) -> None:
    """Default behavior is upsert-by-chunk_id, not a full index replace —
    a stale chunk_id from a previous upload that's absent from this dump
    stays in the index unless replace=true (see the endpoint's own
    docstring, found live: re-uploading a shorter dump left old chunks
    stranded but still searchable)."""
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["replaced"] is False
    mock_os_client.indices.delete.assert_not_called()
    mock_os_client.indices.create.assert_not_called()


def test_ingest_opensearch_dump_replace_drops_existing_index_first(client) -> None:
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals", "replace": "true"},
        )

    assert resp.status_code == 200
    assert resp.json()["replaced"] is True
    mock_os_client.indices.delete.assert_called_once()
    mock_os_client.indices.create.assert_called_once()


def test_ingest_opensearch_dump_replace_skips_delete_when_index_absent(client) -> None:
    """replace=true against a corpus_id that has no index yet must not try
    to delete something that doesn't exist — still creates it fresh."""
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = False

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals", "replace": "true"},
        )

    assert resp.status_code == 200
    mock_os_client.indices.delete.assert_not_called()
    mock_os_client.indices.create.assert_called_once()


def test_ingest_opensearch_dump_502_when_index_creation_fails(client) -> None:
    """Found live: a cluster-wide block on OpenSearch (`cluster.blocks.
    create_index`, e.g. under disk pressure) raised from indices.create()
    used to sit outside the try/except that only wrapped bulk() — surfaced
    as a raw unhandled 500 with a full stack trace instead of a clean 502
    like every other failure mode on this endpoint."""
    from opensearchpy.exceptions import AuthorizationException
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = False
    # Shape matches what opensearchpy's own Connection._raise_error actually
    # builds from a real JSON error body (a nested root_cause dict) — not a
    # flat string, which TransportError.__str__ wouldn't render the same way.
    info = {
        "error": {
            "root_cause": [{"type": "index_create_block_exception", "reason": "blocked by: [FORBIDDEN/10/cluster create-index blocked (api)];"}],
            "type": "index_create_block_exception",
            "reason": "blocked by: [FORBIDDEN/10/cluster create-index blocked (api)];",
        },
        "status": 403,
    }
    mock_os_client.indices.create.side_effect = AuthorizationException(403, "index_create_block_exception", info)

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 502
    assert "create-index blocked" in resp.json()["detail"]


def test_ingest_opensearch_dump_skips_invalid_json_lines(client) -> None:
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\nnot valid json\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["n_skipped"] == 1


def test_ingest_opensearch_dump_skips_non_object_json_lines(client) -> None:
    """Found live: a line can be syntactically valid JSON without being a
    JSON *object* — e.g. a bare number. `obj.get("chunk_id")` crashed with
    an unhandled `AttributeError` ('int' object has no attribute 'get') on
    such a line instead of being treated like any other malformed line."""
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n1025\n{"chunk_id": "c2", "text": "world"}\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(2, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson, "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_indexed"] == 2
    assert body["n_skipped"] == 1
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1", "c2"}


def test_ingest_opensearch_dump_400_when_no_valid_lines(client) -> None:
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=MagicMock()):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", b"not json at all\n", "application/x-ndjson")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400


def test_ingest_opensearch_dump_transparently_decompresses_gzip(client) -> None:
    """NDJSON dumps are routinely shipped `.gz` — found live: an uploaded
    `.gz` file hit `.decode("utf-8")` directly on the compressed bytes and
    raised an unhandled UnicodeDecodeError (500) on the gzip magic byte."""
    import gzip
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n{"chunk_id": "c2", "text": "world"}\n'
    gzipped = gzip.compress(ndjson)
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(2, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson.gz", gzipped, "application/gzip")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["n_indexed"] == 2
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1", "c2"}


def test_ingest_opensearch_dump_400_on_corrupt_gzip(client) -> None:
    truncated_gzip = b"\x1f\x8b" + b"not actually valid gzip data"
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=MagicMock()):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson.gz", truncated_gzip, "application/gzip")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400
    assert "decompress" in resp.json()["detail"]


def test_ingest_opensearch_dump_transparently_decompresses_zip(client) -> None:
    """Only gzip was supported — a zipped dump hit the same unhandled
    UnicodeDecodeError a raw .gz upload used to hit on the archive's own
    binary header, since nothing here recognized the zip magic bytes."""
    import io
    import zipfile
    ndjson = b'{"chunk_id": "c1", "text": "hello"}\n{"chunk_id": "c2", "text": "world"}\n'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dump.ndjson", ndjson)
    zipped = buf.getvalue()
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(2, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.zip", zipped, "application/zip")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["n_indexed"] == 2
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1", "c2"}


def test_ingest_opensearch_dump_zip_prefers_json_named_entry(client) -> None:
    """A zip with multiple entries (e.g. a README alongside the actual
    dump) must pick the .json/.ndjson/.jsonl-named one, not just whatever
    namelist() happens to list first."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", "not the dump")
        zf.writestr("dump.ndjson", b'{"chunk_id": "c1", "text": "hello"}\n')
    zipped = buf.getvalue()
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.zip", zipped, "application/zip")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1"}


def test_ingest_opensearch_dump_400_on_corrupt_zip(client) -> None:
    truncated_zip = b"PK\x03\x04" + b"not actually a valid zip archive"
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=MagicMock()):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.zip", truncated_zip, "application/zip")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400
    assert "zip" in resp.json()["detail"]


def test_ingest_opensearch_dump_falls_back_to_a_plain_json_array(client) -> None:
    """A plain JSON file (a top-level array, pretty-printed or not) — not
    NDJSON at all — used to 400 as "no valid lines" even though its content
    was perfectly good JSON, since every per-line json.loads() choked on
    fragments like "[" or "  {" that aren't valid JSON on their own."""
    dump = b'[\n  {"chunk_id": "c1", "text": "hello"},\n  {"chunk_id": "c2", "text": "world"}\n]\n'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(2, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.json", dump, "application/json")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["n_indexed"] == 2
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1", "c2"}


def test_ingest_opensearch_dump_falls_back_to_a_single_json_object(client) -> None:
    """A dump of exactly one document, not wrapped in an array."""
    dump = b'{"chunk_id": "c1", "text": "hello"}'
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.json", dump, "application/json")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["n_indexed"] == 1


def test_ingest_opensearch_dump_400_when_neither_ndjson_nor_json_produces_a_document(client) -> None:
    """Concatenated pretty-printed JSON objects (no array wrapper, no
    one-object-per-line) — the shape that motivated this diagnosis: neither
    per-line parsing (each line is a fragment) nor whole-body parsing
    (multiple top-level values = "Extra data") produces anything, and the
    error message should say both were tried."""
    dump = b'{\n  "chunk_id": "c1"\n}\n{\n  "chunk_id": "c2"\n}\n'
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=MagicMock()):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.json", dump, "application/json")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400
    assert "single JSON array/object" in resp.json()["detail"]


def test_ingest_opensearch_dump_falls_back_to_cp1251_when_not_utf8(client) -> None:
    """Found live: a real dump failed strict UTF-8 decoding on a byte
    (0xD7) that's an ordinary Cyrillic letter ("Ч") in cp1251 (Windows-1251)
    — the common legacy encoding Windows tools still export Cyrillic text
    as. This platform's own content skews Russian/Belarusian (see
    adapters/opensearch.py's ru_be_analyzer), so cp1251 is a deliberate,
    narrow fallback — not a generic charset-guessing library."""
    ndjson_cp1251 = '{"chunk_id": "c1", "text": "Что делать"}\n'.encode("cp1251")
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(1, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", ndjson_cp1251, "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    assert resp.json()["n_indexed"] == 1
    assert resp.json()["decode_warning"] is None
    actions = mock_bulk.call_args[0][1]
    assert actions[0]["_source"]["text"] == "Что делать"


def test_ingest_opensearch_dump_400_when_undecodable_in_either_encoding(client) -> None:
    """Byte 0x98 is undefined in both UTF-8 (invalid start byte) and cp1251
    (unmapped in that code page) — one of the very few bytes neither strict
    decode can make sense of. `_decode_ndjson_bytes` never hard-fails on this
    though: it falls through to cp1251 with `errors="replace"`, so the 400
    here comes from the *content* being unparseable NDJSON (a single line of
    binary noise), not from a decode-stage rejection."""
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=MagicMock()):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", b"\x98binary garbage", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400
    assert "No valid NDJSON" in resp.json()["detail"]


def test_ingest_opensearch_dump_tolerates_one_bad_byte_among_valid_lines(client) -> None:
    """Found live: a decode-stage rejection used to fail the ENTIRE upload
    over a single bad byte anywhere in the file, even when every other line
    was perfectly valid NDJSON. `_decode_ndjson_bytes`'s `errors="replace"`
    last resort decodes the whole file (replacing just the offending byte
    with U+FFFD) rather than raising, so surrounding valid lines still parse
    and index — only the one field containing the bad byte is degraded, and
    the response surfaces a `decode_warning` so the caller knows to check."""
    raw = (
        b'{"chunk_id": "c1", "text": "good"}\n'
        b'{"chunk_id": "c2", "text": "bad \x98 byte"}\n'
        b'{"chunk_id": "c3", "text": "good2"}\n'
    )
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(3, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", raw, "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_indexed"] == 3
    assert body["decode_warning"] is not None
    actions = mock_bulk.call_args[0][1]
    assert {a["_id"] for a in actions} == {"c1", "c2", "c3"}


def test_ingest_opensearch_dump_last_resort_preserves_multibyte_utf8_around_bad_byte(client) -> None:
    """Found live: the last-resort fallback used to be cp1251-with-replace
    over the WHOLE file, not just the bad byte. Reached only when BOTH
    strict UTF-8 and strict cp1251 raise somewhere — meaning the file is
    most likely mostly-UTF-8 with a stray bad byte, not actually cp1251.
    Decoding it via cp1251 anyway (a single-byte encoding with no concept
    of multi-byte sequences) reinterpreted every Cyrillic character's 2+
    UTF-8 bytes as 2+ wrong cp1251 characters — mojibake across the entire
    file, not just the one bad line. UTF-8 with errors="replace" leaves
    every well-formed multi-byte sequence untouched and only substitutes
    U+FFFD for the actual bad byte(s)."""
    # The explicit "utf-8" is redundant to Python and not to a reader: this test
    # is about encodings, and naming it on the two good lines is what marks the
    # middle one as the odd byte out.
    raw = (
        '{"chunk_id": "c1", "text": "Привет мир"}\n'.encode("utf-8")  # noqa: UP012
        + b'{"chunk_id": "c2", "text": "bad \x98 byte"}\n'
        + '{"chunk_id": "c3", "text": "Спасибо"}\n'.encode("utf-8")  # noqa: UP012
    )
    mock_os_client = MagicMock()
    mock_os_client.indices.exists.return_value = True

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=mock_os_client), \
         patch("opensearchpy.helpers.bulk", return_value=(3, [])) as mock_bulk, \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", raw, "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_indexed"] == 3
    actions = mock_bulk.call_args[0][1]
    by_id = {a["_id"]: a["_source"] for a in actions}
    assert by_id["c1"]["text"] == "Привет мир"
    assert by_id["c3"]["text"] == "Спасибо"


def test_ingest_opensearch_dump_400_when_non_utf8_bytes_decode_as_cp1251_but_arent_valid_ndjson(client) -> None:
    """cp1251 maps almost every byte value to *some* character (only 0x98 is
    undefined), so most non-UTF-8 byte soup "succeeds" as cp1251 without
    raising — but still isn't valid NDJSON, so this still ends in a 400,
    just via the existing "no valid lines" path rather than an encoding
    error naming UTF-8/cp1251 specifically."""
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("opensearchpy.OpenSearch", return_value=MagicMock()):
        resp = client.post(
            "/corpus/ingest/opensearch-dump",
            files={"file": ("dump.ndjson", b"\xff\xfe\x00\x01binary garbage", "application/octet-stream")},
            data={"realm_id": "acme", "corpus_id": "manuals"},
        )

    assert resp.status_code == 400
    assert "No valid NDJSON" in resp.json()["detail"]
