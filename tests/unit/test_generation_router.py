"""services/api_gateway/routers/generation.py — question generator.

Covers: generation-presets CRUD (with per-Realm idempotent seeding),
chunk-sampling strategies (single/pair/range depending on question_type,
honest degradation when no pair/range exists), article_refs derivation from
chunk metadata (never asked of the LLM), tolerant JSON parsing, partial
failure handling, and the "nothing is written to Mongo" invariant — this
endpoint only returns drafts (see module docstring for why).

POST /generate/questions itself is an async job (see generation.py's own
docstring) — it returns a job_id immediately and the actual drafts/failed
payload arrives as the "done" event on the paired progress WebSocket. Tests
below acme that real flow end-to-end via `_run_job` rather than mocking the
asyncio scheduling away, so a regression in the job/WS wiring itself (not
just the pure helper functions) would fail the suite.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.generation import (
    _BUILTIN_TEMPLATES,
    _article_refs,
    _clean_reference_answer,
    _parse_llm_json,
    _resolve_template,
    _sample_chunk_groups,
    router,
)


def _chunk(
    chunk_id: str, text: str, source_code: str | None = None, article_no: str | None = None,
    structural_path: str = "",
):
    metadata = {}
    if source_code:
        metadata["source_code"] = source_code
    if article_no:
        metadata["article_no"] = article_no
    return SimpleNamespace(chunk_id=chunk_id, text=text, doc_id=chunk_id, structural_path=structural_path, metadata=metadata)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


# ── Sampling strategies ───────────────────────────────────────────────────────

def test_sample_single_for_unknown_question_type():
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(5)]
    groups = _sample_chunk_groups(chunks, "closed", 3)
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)


def test_sample_pairs_shares_source_code():
    chunks = [
        _chunk("a1", "art 1", "HK001", "1"),
        _chunk("a2", "art 2", "HK001", "2"),
        _chunk("b1", "art b1", "HK002", "1"),
    ]
    groups = _sample_chunk_groups(chunks, "comparative", 1)
    assert len(groups) == 1
    assert len(groups[0]) == 2
    codes = {c.metadata["source_code"] for c in groups[0]}
    assert codes == {"HK001"}


def test_sample_pairs_falls_back_to_single_when_no_pair_exists():
    """No document has 2+ articles here — comparative must not fail the
    whole batch, it degrades to single-chunk questions instead."""
    chunks = [_chunk("a1", "art 1", "HK001", "1"), _chunk("b1", "art b1", "HK002", "1")]
    groups = _sample_chunk_groups(chunks, "comparative", 2)
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)


def test_sample_ranges_are_contiguous_and_same_source():
    chunks = [_chunk(f"c{i}", f"art {i}", "HK001", str(i)) for i in range(1, 10)]
    groups = _sample_chunk_groups(chunks, "navigational", 1)
    assert len(groups) == 1
    group = groups[0]
    assert 3 <= len(group) <= 8
    article_nos = [int(c.metadata["article_no"]) for c in group]
    assert article_nos == sorted(article_nos)
    assert article_nos == list(range(article_nos[0], article_nos[0] + len(article_nos)))


# ── Template resolution — chosen preset, or a built-in fallback by shape ──────

def test_resolve_template_uses_preset_when_given():
    preset = {"template": "custom {chunk_text}"}
    group = [_chunk("a", "t")]
    assert _resolve_template(preset, group) == "custom {chunk_text}"


def test_resolve_template_falls_back_to_builtin_single_for_one_chunk():
    group = [_chunk("a", "t")]
    assert _resolve_template(None, group) == _BUILTIN_TEMPLATES["single"]


def test_resolve_template_falls_back_to_builtin_pair_for_two_chunks():
    group = [_chunk("a", "t"), _chunk("b", "t")]
    assert _resolve_template(None, group) == _BUILTIN_TEMPLATES["pair"]


def test_resolve_template_falls_back_to_builtin_range_for_three_plus_chunks():
    group = [_chunk("a", "t"), _chunk("b", "t"), _chunk("c", "t")]
    assert _resolve_template(None, group) == _BUILTIN_TEMPLATES["range"]


# ── article_refs derivation ────────────────────────────────────────────────────

def test_article_refs_built_from_chunk_metadata():
    group = [_chunk("a", "t", "SRC001", "5")]
    assert _article_refs(group) == ["SRC001/5"]


def test_article_refs_falls_back_to_bare_doc_id_when_no_metadata_at_all():
    """Found live: a general document corpus (technical manuals, no
    legal-document-code system) has no source_code metadata at all — used
    to leave article_refs completely empty for every such question,
    universally scored "out_of_scope" downstream regardless of actual
    coverage. doc_id (always present on a Chunk) is the last-resort
    identity — coarser (document-level, not section-level) but still a
    real, usable ref instead of none."""
    group = [_chunk("a", "t")]
    assert _article_refs(group) == ["a"]


def test_article_refs_multiple_for_a_group():
    group = [_chunk("a", "t", "HK001", "1"), _chunk("b", "t", "HK001", "2")]
    assert _article_refs(group) == ["HK001/1", "HK001/2"]


def test_article_refs_falls_back_to_structural_path_without_article_no():
    """Realm-agnostic: a corpus not laid out as one numbered file per
    article (e.g. a single large manual) still gets a usable ref, built from
    the chunking-derived heading breadcrumb instead of a missing article_no."""
    group = [_chunk("a", "t", source_code="install-guide", structural_path="section/subsection[Montage]")]
    assert _article_refs(group) == ["install-guide#section/subsection[Montage]"]


def test_article_refs_prefers_article_no_over_structural_path_when_both_present():
    group = [_chunk("a", "t", "SRC001", "5", structural_path="chapter/article")]
    assert _article_refs(group) == ["SRC001/5"]


def test_article_refs_falls_back_to_doc_id_and_structural_path_when_source_code_missing():
    group = [_chunk("a", "t", structural_path="section/subsection[Montage]")]
    assert _article_refs(group) == ["a#section/subsection[Montage]"]


# ── reference_answer nested-JSON defense ──────────────────────────────────────
# Found live: qwen3:8b sometimes writes reference_answer as its own nested
# JSON ({"answer": "...", "reasoning": "..."}) despite the prompt explicitly
# forbidding it.

def test_clean_reference_answer_passes_through_plain_text():
    assert _clean_reference_answer("Yes, article 5 applies.") == "Yes, article 5 applies."


def test_clean_reference_answer_unwraps_answer_reasoning_shape():
    raw = '{"answer": "Yes", "reasoning": "Per the article, the client bears the cost."}'
    assert _clean_reference_answer(raw) == "Yes. Per the article, the client bears the cost."


def test_clean_reference_answer_unwraps_answer_only():
    assert _clean_reference_answer('{"answer": "No"}') == "No"


def test_clean_reference_answer_leaves_unrecognized_json_shape_unchanged():
    raw = '{"foo": "bar"}'
    assert _clean_reference_answer(raw) == raw


# ── JSON parsing tolerance ─────────────────────────────────────────────────────

def test_parse_llm_json_plain():
    assert _parse_llm_json('{"question": "q?", "reference_answer": "a"}') == {"question": "q?", "reference_answer": "a"}


def test_parse_llm_json_tolerates_wrapping_prose():
    raw = 'Here is your JSON:\n{"question": "q?", "reference_answer": "a"}\nHope that helps!'
    assert _parse_llm_json(raw) == {"question": "q?", "reference_answer": "a"}


def test_parse_llm_json_none_on_missing_fields():
    assert _parse_llm_json('{"question": "q?"}') is None


def test_parse_llm_json_none_on_garbage():
    assert _parse_llm_json("not json at all") is None


# ── generation-presets CRUD ────────────────────────────────────────────────────

def test_list_presets_returns_empty_for_a_new_realm_without_seeding(client):
    """Presets are opt-in — an earlier version auto-seeded 3
    legal-domain defaults into every new Realm, including
    non-legal ones (found live: a Realm for device manuals still offered
    a preset phrased in terms of statutes). A Realm with zero presets now just gets
    an empty list — no insert calls, nothing seeded."""
    with patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)) as mock_insert, \
         patch("adapters.mongodb.find_many", AsyncMock(return_value=[])):
        resp = client.get("/generation-presets?realm_id=acme")

    assert resp.status_code == 200
    assert resp.json() == []
    mock_insert.assert_not_called()


def test_list_presets_returns_realms_own_presets(client):
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=[
        {"id": "p1", "name": "Custom preset", "realm_id": "demo"},
    ])):
        resp = client.get("/generation-presets?realm_id=demo")

    assert resp.status_code == 200
    assert [p["id"] for p in resp.json()] == ["p1"]


def test_create_and_delete_preset(client):
    with patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post("/generation-presets", json={
            "name": "Custom", "template": "Test {chunk_text}", "realm_id": "demo",
        })
    assert resp.status_code == 201
    preset_id = resp.json()["id"]
    assert preset_id

    with patch("adapters.mongodb.delete_one", AsyncMock(return_value=1)) as mock_delete:
        resp = client.delete(f"/generation-presets/{preset_id}?realm_id=demo")
    assert resp.status_code == 204
    mock_delete.assert_called_once_with("generation_presets", {"id": preset_id, "realm_id": "demo"})


def test_delete_preset_404_when_not_found(client):
    with patch("adapters.mongodb.delete_one", AsyncMock(return_value=0)):
        resp = client.delete("/generation-presets/nope")
    assert resp.status_code == 404


def test_update_preset(client):
    existing = {"id": "p1", "name": "Old", "description": "old desc", "template": "old {chunk_text}", "realm_id": "demo"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.put("/generation-presets/p1?realm_id=demo", json={
            "name": "New name", "description": "new desc", "template": "new {chunk_text}",
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "New name"
    assert body["template"] == "new {chunk_text}"
    assert body["id"] == "p1"
    mock_update.assert_called_once_with(
        "generation_presets", {"id": "p1", "realm_id": "demo"},
        {"$set": {"name": "New name", "description": "new desc", "template": "new {chunk_text}"}},
    )


def test_update_preset_404_when_not_found(client):
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.put("/generation-presets/nope", json={"name": "X", "template": "{chunk_text}"})
    assert resp.status_code == 404


# ── POST /generate/questions ───────────────────────────────────────────────────

_PRESET = {"id": "p1", "template": 'Q about: {chunk_text}\n{"question": "...", "reference_answer": "..."}'}


def _run_job(client, payload: dict) -> dict:
    """POST kicks off a background job and returns only a job_id (see
    generation.py's docstring on why: a batch is one blocking LLM call per
    sampled group and must not tie up the event loop for the whole request).
    Drives the real flow end-to-end — opens the paired progress WebSocket and
    waits for the "done" event, which carries the final drafts/failed payload
    — instead of mocking the asyncio scheduling away."""
    resp = client.post("/generate/questions", json=payload)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    with client.websocket_connect(f"/generate/questions/{job_id}/progress") as ws:
        while True:
            event = ws.receive_json()
            if event["type"] == "done":
                return event
            if event["type"] == "error":
                raise AssertionError(f"generation job errored: {event}")


def test_generate_questions_returns_job_id_immediately(client):
    chunks = [_chunk("c1", "text", "HK001", "1")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate",
               return_value='{"question": "q", "reference_answer": "a"}'):
        resp = client.post("/generate/questions", json={
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "preset_id": "p1", "type_counts": [{"question_type": "closed", "n_questions": 1}],
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["job_id"]
    assert body["n_groups"] == 1


def test_generate_questions_happy_path_derives_article_refs(client):
    chunks = [_chunk("c1", "Article 5 text", "SRC001", "5")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate",
               return_value='{"question": "What does article 5 say?", "reference_answer": "..."}'):
        body = _run_job(client, {
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "preset_id": "p1", "type_counts": [{"question_type": "closed", "n_questions": 1}],
        })

    assert len(body["drafts"]) == 1
    draft = body["drafts"][0]
    assert draft["article_refs"] == ["SRC001/5"]
    assert draft["provenance"]["origin"] == "generated"
    assert draft["provenance"]["model"] == "qwen3:8b"
    assert draft["provenance"]["chunk_id"] == "c1"
    assert body["failed"] == []


def test_generate_questions_reports_progress_events_before_done(client):
    """The frontend shows "N of M generated" while a batch is running — this
    asserts the "progress" events actually carry processed/total/generated,
    not just that a final "done" event eventually shows up."""
    chunks = [_chunk("c1", "text 1", "HK001", "1"), _chunk("c2", "text 2", "HK001", "2")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate",
               return_value='{"question": "q", "reference_answer": "a"}'):
        resp = client.post("/generate/questions", json={
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "preset_id": "p1", "type_counts": [{"question_type": "closed", "n_questions": 2}],
        })
        job_id = resp.json()["job_id"]

        events = []
        with client.websocket_connect(f"/generate/questions/{job_id}/progress") as ws:
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "done":
                    break

    assert events[0] == {"type": "start", "total": 2}
    progress_events = [e for e in events if e["type"] == "progress"]
    assert [e["processed"] for e in progress_events] == [1, 2]
    assert [e["generated"] for e in progress_events] == [1, 2]
    assert all(e["total"] == 2 for e in progress_events)
    assert events[-1]["type"] == "done"


def test_generate_questions_supports_mixed_type_counts(client):
    """A single batch can mix question types (e.g. 2 closed + 1 clarifying)
    instead of one uniform type for the whole request — each type_counts
    entry is sampled/generated independently, and each draft is tagged with
    its own requested type, not the first/last one in the list."""
    chunks = [_chunk(f"c{i}", f"text {i}", "HK001", str(i)) for i in range(1, 4)]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate",
               return_value='{"question": "q", "reference_answer": "a"}'):
        body = _run_job(client, {
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b", "preset_id": "p1",
            "type_counts": [
                {"question_type": "closed", "n_questions": 2},
                {"question_type": "clarifying", "n_questions": 1},
            ],
        })

    assert len(body["drafts"]) == 3
    types = sorted(d["question_type"] for d in body["drafts"])
    assert types == ["clarifying", "closed", "closed"]


def test_generate_questions_unwraps_nested_json_reference_answer(client):
    """End-to-end: the LLM ignoring the prompt's plain-text instruction
    still results in a clean reference_answer in the response."""
    chunks = [_chunk("c1", "text", "HK1", "1")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks
    nested = '{"question": "q?", "reference_answer": "{\\"answer\\": \\"Yes\\", \\"reasoning\\": \\"because\\"}"}'

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", return_value=nested):
        body = _run_job(client, {
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b", "preset_id": "p1",
            "type_counts": [{"question_type": "closed", "n_questions": 1}],
        })

    assert body["drafts"][0]["reference_answer"] == "Yes. because"


def test_generate_questions_partial_failure_does_not_drop_successful_drafts(client):
    chunks = [_chunk("c1", "text 1", "HK001", "1"), _chunk("c2", "text 2", "HK001", "2")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    responses = iter([
        '{"question": "q1", "reference_answer": "a1"}',
        "garbage, not json",
    ])

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", side_effect=lambda *a, **k: next(responses)):
        body = _run_job(client, {
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "preset_id": "p1", "type_counts": [{"question_type": "closed", "n_questions": 2}],
        })

    assert len(body["drafts"]) == 1
    assert len(body["failed"]) == 1


def test_generate_questions_never_writes_to_mongo(client):
    chunks = [_chunk("c1", "text", "HK001", "1")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate",
               return_value='{"question": "q", "reference_answer": "a"}'), \
         patch("adapters.mongodb.insert_one", AsyncMock()) as mock_insert, \
         patch("adapters.mongodb.update_one", AsyncMock()) as mock_update:
        _run_job(client, {
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "preset_id": "p1", "type_counts": [{"question_type": "closed", "n_questions": 1}],
        })

    mock_insert.assert_not_called()
    mock_update.assert_not_called()


def test_generate_questions_404_when_preset_missing(client):
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.post("/generate/questions", json={
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b", "preset_id": "nope",
        })
    assert resp.status_code == 404


def test_generate_questions_works_without_a_preset(client):
    """Presets are optional — omitting preset_id entirely must not 404
    or otherwise require a Mongo lookup; generation falls back to a
    built-in, domain-neutral template (_BUILTIN_TEMPLATES)."""
    chunks = [_chunk("c1", "text", "HK001", "1")]
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = chunks

    with patch("adapters.mongodb.find_one", AsyncMock()) as mock_find_one, \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate",
               return_value='{"question": "q", "reference_answer": "a"}'):
        body = _run_job(client, {
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "type_counts": [{"question_type": "closed", "n_questions": 1}],
        })

    mock_find_one.assert_not_called()
    assert len(body["drafts"]) == 1
    assert body["drafts"][0]["provenance"]["preset_id"] is None


def test_generate_questions_503_when_qdrant_unavailable(client):
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", side_effect=RuntimeError("down")):
        resp = client.post("/generate/questions", json={
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b", "preset_id": "p1",
        })
    assert resp.status_code == 503


def test_generate_questions_503_when_corpus_has_no_chunks(client):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = []
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=_PRESET)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant):
        resp = client.post("/generate/questions", json={
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b", "preset_id": "p1",
        })
    assert resp.status_code == 503
