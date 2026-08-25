"""The whole platform, once, over the demo material.

This walks the path a new user actually takes: import a realm, check its
resources answer, ingest the demo corpus, ask a question in chat, run the
golden set twice under different configurations, compare the two runs question
by question, read the diagnostics, leave feedback, turn that feedback into a
judgment and a golden question, and export the realm back out.

Every step is one test, and they run in file order. Each records what the next
one needs in `state` and skips when a prerequisite is missing, so a failure in
step three reports itself once instead of as eleven cascading errors.

What this is for: the demo is both the showcase and the only proof that the
platform works on a machine that has just cloned the repository. Checking that
by hand does not scale and does not run in CI, so it is checked here, with a
metrics summary printed at the end so a regression in demo quality shows up as
a number rather than only as a colour.

    make test-e2e
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "corpus" / "demo_handbook"
GOLDEN_FILE = REPO_ROOT / "eval" / "golden" / "handbook.v1.fast.jsonl"
CORPUS_ID = "handbook"


def _need(state: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in state]
    if missing:
        pytest.skip(f"earlier step did not produce: {', '.join(missing)}")


def _questions() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _base_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "name": "e2e",
        "chunking_strategy": {"kind": "chunker", "component_id": "structure_aware"},
        "embedder": {"kind": "embedder", "component_id": "bge_m3"},
        "retrievers": [{"kind": "retriever", "component_id": "qdrant_dense"}],
        "generator": {"kind": "generator", "component_id": "ollama"},
        "pipeline_id": "hybrid_rrf",
        "top_k": 5,
        "corpus_id": CORPUS_ID,
    }
    config.update(overrides)
    return config


# ── 1. realm ─────────────────────────────────────────────────────────────────

def test_import_realm_from_the_demo_bundle(client, realm_id, state):
    """The bundle the welcome screen offers has to import into a clean realm."""
    from tools.seed_demo import build_bundle, load_questions

    bundle = build_bundle(load_questions())
    bundle["realm"]["id"] = realm_id
    bundle["realm"]["name"] = f"E2E {realm_id}"

    resp = client.post("/realms/import?on_conflict=fail", json=bundle)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["realm_id"] == realm_id

    created = {e["kind"]: e["created"] for e in body["entries"]}
    assert created["prompts"] == 1, created
    assert created["generation_presets"] == 1, created
    assert created["datasets"] == 1, created

    state["realm_id"] = realm_id


def test_realm_appears_in_the_listing(client, state):
    _need(state, "realm_id")
    resp = client.get("/realms")
    assert resp.status_code == 200
    assert state["realm_id"] in {r["id"] for r in resp.json()}


def test_realm_resources_answer(client, state, services):
    """A realm pointing at a service that is down is worse than no realm."""
    _need(state, "realm_id")
    for kind in ("qdrant", "opensearch"):
        resp = client.post(f"/realms/{state['realm_id']}/resources/test", json={"type": kind})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("status") == "ok", f"{kind}: {body}"


# ── 2. corpus ────────────────────────────────────────────────────────────────

def test_ingest_the_demo_corpus(state, realm_id):
    """Ingested by the CLI from a stable directory, which is what makes the
    golden refs resolvable at all: a UI upload lands in a temp directory whose
    name changes on every run, so `source_code` would differ every time."""
    _need(state, "realm_id")
    os.environ["USE_REAL_BGE_M3"] = "true"
    from services.ingestion.cli import ingest

    stats = ingest(
        CORPUS_DIR,
        strategy_id="structure_aware",
        corpus_id=CORPUS_ID,
        realm_id=realm_id,
        use_opensearch=True,
    )
    assert stats["files"] == 8, stats
    assert stats["chunks"] > 8, f"expected more chunks than documents, got {stats}"
    assert stats["qdrant_collection"], stats
    state["ingest"] = stats


def test_ref_ids_are_derived_and_stable(state):
    """Every golden ref must resolve against what actually landed in Qdrant.

    This is the assertion the first demo bundle would have failed: its refs
    read "section 2.1", which no ingest can ever produce, so every retrieval
    metric read zero and every question classified `uncovered`.
    """
    _need(state, "ingest")
    from qdrant_client import QdrantClient

    from tests.e2e.conftest import QDRANT_HOST, QDRANT_PORT

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    points, _ = qc.scroll(collection_name=state["ingest"]["qdrant_collection"], limit=500, with_payload=True)
    indexed = {
        f"{p.payload.get('metadata', {}).get('source_code')}/{p.payload.get('metadata', {}).get('article_no')}"
        for p in points
    }
    referenced = {ref for q in _questions() for ref in q["article_refs"]}
    assert referenced, "the golden set has no refs at all"
    assert referenced <= indexed, f"refs missing from the index: {sorted(referenced - indexed)}"


def test_register_the_corpus(client, state, realm_id):
    _need(state, "ingest")
    backends: dict[str, Any] = {
        "qdrant": {"collection": state["ingest"]["qdrant_collection"], "embedder_id": "bge_m3"}
    }
    if state["ingest"].get("opensearch_index"):
        backends["opensearch"] = {"index": state["ingest"]["opensearch_index"]}

    resp = client.post("/corpus/collections", json={
        "realm_id": realm_id,
        "corpus_id": CORPUS_ID,
        "storage_type": "hybrid" if "opensearch" in backends else "dense_only",
        "backends": backends,
        "owner": "platform",
    })
    assert resp.status_code in (200, 201), resp.text

    listing = client.get(f"/corpus/collections?realm_id={realm_id}")
    assert listing.status_code == 200
    assert CORPUS_ID in {c["corpus_id"] for c in listing.json()}


def test_corpus_health_reports_a_usable_corpus(client, state, realm_id):
    """Duplicates, empty chunks and a stub embedder all read as healthy data
    to retrieval and as noise to metrics, so the demo must be clean of them."""
    _need(state, "ingest")
    resp = client.get(f"/corpus/{CORPUS_ID}/health?realm_id={realm_id}")
    assert resp.status_code == 200, resp.text
    health = resp.json()
    assert health.get("n_chunks", 0) > 0, health
    assert health.get("n_duplicates", 0) == 0, f"the demo corpus has duplicates: {health}"
    assert health.get("n_header_only", 0) == 0, f"the demo corpus has header-only chunks: {health}"
    state["health"] = health


# ── 3. chat ──────────────────────────────────────────────────────────────────

def test_chat_answers_from_the_demo_corpus(client, state, realm_id, require_ollama):
    _need(state, "ingest")
    resp = client.post("/query", json={
        "text": "Who approves a purchase of 12,000 EUR?",
        "realm_id": realm_id,
        "corpus_id": CORPUS_ID,
    })
    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer.get("text", "").strip(), "empty answer"
    refs = answer.get("source_refs") or []
    assert refs, "answer cites no sources"
    assert any("demo_handbook" in json.dumps(r, ensure_ascii=False) for r in refs), refs


# ── 4. runs ──────────────────────────────────────────────────────────────────

def _run_experiment(client, realm_id, state, key: str, **config_overrides: Any) -> str:
    resp = client.post("/experiments", json={
        "config": _base_config(**config_overrides),
        "dataset_name": "handbook.v1.fast.jsonl",
        "realm_id": realm_id,
    })
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    import time
    for _ in range(600):  # generation over 15 questions can legitimately take minutes
        detail = client.get(f"/experiments/{run_id}")
        if detail.status_code == 200 and detail.json().get("finished_at"):
            state[key] = run_id
            return run_id
        time.sleep(2)
    pytest.fail(f"run {run_id} did not finish within 20 minutes")


def test_first_run_produces_real_metrics(client, state, realm_id, require_ollama):
    _need(state, "ingest")
    run_id = _run_experiment(client, realm_id, state, "run_a", name="e2e-a", top_k=5)

    detail = client.get(f"/experiments/{run_id}").json()
    metrics = detail["aggregate_metrics"]
    state["metrics_a"] = metrics

    assert len(detail["question_results"]) == len(_questions())
    assert metrics.get("retrieval_recall_at_k", 0) > 0, (
        f"recall is zero, so the refs did not resolve against the index: {metrics}"
    )
    assert metrics.get("correct_refusal", 0) > 0, (
        f"no question was correctly refused, so the out-of-scope pair is not "
        f"being classified: {metrics}"
    )


def test_no_question_is_uncovered(client, state):
    """`uncovered` removes a question from retrieval scoring without failing,
    which is how a broken demo looks identical to a working one."""
    _need(state, "run_a")
    detail = client.get(f"/experiments/{state['run_a']}").json()
    uncovered = [
        q["question_id"] for q in detail["question_results"]
        if q.get("answerability") == "uncovered"
    ]
    assert not uncovered, f"questions classified uncovered: {uncovered}"


def test_second_run_under_a_different_configuration(client, state, realm_id, require_ollama):
    _need(state, "run_a")
    run_id = _run_experiment(client, realm_id, state, "run_b", name="e2e-b", top_k=10)

    a = client.get(f"/experiments/{state['run_a']}").json()
    b = client.get(f"/experiments/{run_id}").json()
    assert a["config"]["config_hash"] != b["config"]["config_hash"], (
        "two different configurations produced the same hash, so runs cannot be told apart"
    )
    state["metrics_b"] = b["aggregate_metrics"]


def test_compare_the_two_runs_question_by_question(client, state):
    """Aggregates hide the thing that matters: which questions moved."""
    _need(state, "run_a", "run_b")
    resp = client.post("/experiments/compare", json={"ids": [state["run_a"], state["run_b"]]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body, "comparison returned nothing"
    state["comparison"] = body


# ── 5. diagnostics ───────────────────────────────────────────────────────────

def test_diagnostics_do_not_fire_falsely_on_healthy_data(client, state):
    """A detector that fires on a working demo is worse than one that is silent:
    it teaches the reader to ignore the panel."""
    _need(state, "run_a")
    detail = client.get(f"/experiments/{state['run_a']}").json()
    diagnostics = detail.get("diagnostics") or []
    stub_alerts = [d for d in diagnostics if "stub" in json.dumps(d, ensure_ascii=False).lower()]
    assert not stub_alerts, f"stub-embedder detector fired on real embeddings: {stub_alerts}"
    state["diagnostics"] = diagnostics


def test_prescription_is_available_for_the_run(client, state):
    _need(state, "run_a")
    resp = client.get(f"/experiments/{state['run_a']}/prescription")
    assert resp.status_code == 200, resp.text


# ── 6. feedback becomes reusable knowledge ───────────────────────────────────

def test_leave_feedback_on_one_answer(client, state):
    _need(state, "run_a")
    detail = client.get(f"/experiments/{state['run_a']}").json()
    question_id = detail["question_results"][0]["question_id"]

    resp = client.put(
        f"/experiments/{state['run_a']}/questions/{question_id}/feedback",
        json={"rating": "bad", "comment": "The answer cites the wrong section.", "reviewer": "e2e"},
    )
    assert resp.status_code == 200, resp.text

    stored = client.get(f"/experiments/{state['run_a']}/feedback")
    assert stored.status_code == 200
    assert stored.json(), "feedback was written but reads back empty"
    state["question_id"] = question_id


def test_record_a_judgment(client, state, realm_id):
    """A judgment states what a reviewer observed, which outlives the run it
    came from and can be replayed against any later configuration.

    The chunk comes from the run's own retrieved sources rather than being
    invented, because a judgment naming a chunk the corpus does not contain
    could never be replayed. The endpoint enforces that a judgment names at
    least one chunk: a judgment about nothing states nothing.
    """
    _need(state, "run_a", "question_id")
    detail = client.get(f"/experiments/{state['run_a']}").json()
    result = next(q for q in detail["question_results"] if q["question_id"] == state["question_id"])
    sources = result.get("source_refs") or []
    if not sources:
        pytest.skip("the run retrieved no sources for this question, nothing to judge")

    # `relevant`/`irrelevant` are chunk ids; the router hydrates each into a
    # JudgedChunk from the corpus itself, so the caller cannot record evidence
    # the index does not actually hold.
    chunk_id = sources[0].get("chunk_id")
    assert chunk_id, f"the run's first source carries no chunk_id: {sources[0]}"

    resp = client.post("/judgments", json={
        "realm_id": realm_id,
        "corpus_id": CORPUS_ID,
        "question": result.get("question") or "Who approves a purchase of 12,000 EUR?",
        "relevant": [chunk_id],
        "irrelevant": [],
        "note": "recorded by the e2e walkthrough",
        "author": "e2e",
        "source_run_id": state["run_a"],
        "source_question_id": state["question_id"],
    })
    assert resp.status_code in (200, 201), resp.text
    judgment_id = resp.json().get("id") or resp.json().get("judgment_id")
    assert judgment_id, resp.json()

    # Both scopes are required rather than optional: a judgment is only
    # meaningful against the corpus it was made on, so an unscoped listing has
    # nothing meaningful to return.
    listing = client.get(f"/judgments?realm_id={realm_id}&corpus_id={CORPUS_ID}")
    assert listing.status_code == 200, listing.text
    assert judgment_id in json.dumps(listing.json(), ensure_ascii=False)
    state["judgment_id"] = judgment_id


# ── 7. the realm survives a round trip ───────────────────────────────────────

def test_export_the_realm(client, state, realm_id):
    _need(state, "realm_id")
    resp = client.get(f"/realms/{realm_id}/export")
    assert resp.status_code == 200, resp.text
    bundle = resp.json()
    assert bundle["format"] == "causa-realm/v1"
    assert bundle["realm"]["id"] == realm_id
    assert len(bundle["prompts"]) >= 1
    assert len(bundle["datasets"]) >= 1
    assert {c["corpus_id"] for c in bundle["corpora"]} == {CORPUS_ID}
    state["export"] = bundle


def test_summary(state, realm_id):
    """Prints what the walkthrough measured, so a quality regression in the
    demo material is visible as a number rather than only as a passing suite."""
    if "metrics_a" not in state:
        pytest.skip("no run completed, nothing to summarise")

    rows = [
        ("realm", realm_id),
        ("documents", state.get("ingest", {}).get("files", "—")),
        ("chunks", state.get("ingest", {}).get("chunks", "—")),
        ("questions", len(_questions())),
        ("diagnostics", len(state.get("diagnostics", []))),
    ]
    print("\n" + "─" * 62)
    print("  e2e walkthrough summary")
    print("─" * 62)
    for name, value in rows:
        print(f"  {name:<22} {value}")
    print("─" * 62)
    print(f"  {'metric':<28} {'run A':>10} {'run B':>10}")
    a, b = state["metrics_a"], state.get("metrics_b", {})
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        fa = f"{va:.3f}" if isinstance(va, (int, float)) else "—"
        fb = f"{vb:.3f}" if isinstance(vb, (int, float)) else "—"
        print(f"  {key:<28} {fa:>10} {fb:>10}")
    print("─" * 62)
