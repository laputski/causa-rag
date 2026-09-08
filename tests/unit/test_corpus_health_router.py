"""GET /corpus/{corpus_id}/health (near-dup wiring) and the three on-demand
POST /corpus/{corpus_id}/diagnostics/* endpoints (ragas/trulens/chunk-
coherence) — all mocked, no real Qdrant/Ollama/LLM-judge libraries needed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.api_gateway.routers.corpus as corpus
from core.models import Chunk
from services.api_gateway.routers.corpus import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d1", text=text)


# ── /health near-dup wiring ──────────────────────────────────────────────────

def test_health_includes_near_duplicate_item_when_found(client) -> None:
    chunks = [_chunk("a", "text one of a meaningful length for the check"), _chunk("b", "text two of a meaningful length")]
    fake_qdrant = MagicMock()
    fake_qdrant.scroll_all.return_value = chunks

    with patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=fake_qdrant), \
         patch("core.eval.corpus_health.detect_near_duplicates") as mock_detect:
        from core.eval.corpus_health import HealthItem
        mock_detect.return_value = HealthItem(
            id="near_duplicates", severity="info", title="x", detail="1 pair of chunks",
        )
        resp = client.get("/corpus/handbook/health")

    assert resp.status_code == 200
    body = resp.json()
    assert any(i["id"] == "near_duplicates" for i in body["items"])


def test_health_omits_near_duplicate_item_when_none_found(client) -> None:
    chunks = [_chunk("a", "text one of a meaningful length")]
    fake_qdrant = MagicMock()
    fake_qdrant.scroll_all.return_value = chunks

    with patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=fake_qdrant), \
         patch("core.eval.corpus_health.detect_near_duplicates", return_value=None):
        resp = client.get("/corpus/handbook/health")

    assert resp.status_code == 200
    assert not any(i["id"] == "near_duplicates" for i in resp.json()["items"])


def test_health_503_when_index_unavailable(client) -> None:
    with patch("services.api_gateway.routers.corpus._resolve_qdrant", side_effect=RuntimeError("down")):
        resp = client.get("/corpus/handbook/health")
    assert resp.status_code == 503


# ── deep diagnostics endpoints ───────────────────────────────────────────────

class _FakeRunnerResult:
    def __init__(self, metrics, per_question=None, per_chunk=None):
        self.metrics = metrics
        self.per_question = per_question or []
        self.per_chunk = per_chunk or []


def test_ragas_diagnostics_endpoint_returns_metrics(client) -> None:
    fake_runner = MagicMock()
    fake_runner.run.return_value = _FakeRunnerResult({"context_precision": 0.8})

    with patch("services.api_gateway.routers.corpus._build_diagnostics_pipeline", AsyncMock(return_value=(MagicMock(), MagicMock()))), \
         patch("services.api_gateway.routers.corpus._load_diagnostics_dataset", AsyncMock(return_value=MagicMock())), \
         patch("eval.ragas_runner.RagasRunner", return_value=fake_runner):
        resp = client.post("/corpus/handbook/diagnostics/ragas")

    assert resp.status_code == 200
    assert resp.json()["metrics"] == {"context_precision": 0.8}


def test_ragas_diagnostics_endpoint_503_when_not_installed(client) -> None:
    with patch("services.api_gateway.routers.corpus._build_diagnostics_pipeline", AsyncMock(return_value=(MagicMock(), MagicMock()))), \
         patch("services.api_gateway.routers.corpus._load_diagnostics_dataset", AsyncMock(return_value=MagicMock())), \
         patch("eval.ragas_runner.RagasRunner", side_effect=RuntimeError("ragas not installed")):
        resp = client.post("/corpus/handbook/diagnostics/ragas")
    assert resp.status_code == 503


def test_ragas_diagnostics_accepts_custom_dataset_pipeline_and_question_count(client) -> None:
    """Confirms the params are actually configurable, not just present with
    a default — pipeline_id/top_k/dataset/max_questions must reach
    _build_diagnostics_pipeline / EvalDataset.from_jsonl / runner.run."""
    fake_runner = MagicMock()
    fake_runner.run.return_value = _FakeRunnerResult({"context_precision": 0.5})

    with patch("services.api_gateway.routers.corpus._build_diagnostics_pipeline", AsyncMock(return_value=(MagicMock(), MagicMock()))) as mock_build, \
         patch("services.api_gateway.routers.corpus._load_diagnostics_dataset", AsyncMock(return_value=MagicMock())) as mock_load_dataset, \
         patch("eval.ragas_runner.RagasRunner", return_value=fake_runner):
        resp = client.post(
            "/corpus/handbook/diagnostics/ragas"
            "?pipeline_id=hybrid_rrf&top_k=10&dataset=handbook.v1.fast.jsonl&max_questions=25"
        )

    assert resp.status_code == 200
    mock_build.assert_called_once_with("handbook", "hybrid_rrf", 10, None)
    assert mock_load_dataset.call_args[0][0] == "handbook.v1.fast.jsonl"
    assert fake_runner.run.call_args.kwargs["max_questions"] == 25


def test_ragas_diagnostics_passes_realm_id_through(client) -> None:
    """Without this, deep diagnostics for a non-default Realm hit
    the gateway's own env-var Qdrant/OpenSearch regardless of which Realm's
    corpus was asked for (same leak _for_the_corpus closes for real runs)."""
    fake_runner = MagicMock()
    fake_runner.run.return_value = _FakeRunnerResult({})

    with patch("services.api_gateway.routers.corpus._build_diagnostics_pipeline", AsyncMock(return_value=(MagicMock(), MagicMock()))) as mock_build, \
         patch("services.api_gateway.routers.corpus._load_diagnostics_dataset", AsyncMock(return_value=MagicMock())), \
         patch("eval.ragas_runner.RagasRunner", return_value=fake_runner):
        resp = client.post("/corpus/handbook/diagnostics/ragas?realm_id=demo")

    assert resp.status_code == 200
    mock_build.assert_called_once_with("handbook", "naive", 5, "demo")


def test_ragas_diagnostics_404_on_unknown_dataset(client) -> None:
    # The dataset is looked for in the realm's database first and in
    # eval/golden/ second. Absent from both, a 404, and not a silent
    # substitution of the demonstration dataset: another dataset put in its
    # place computes metrics that look real.
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.post("/corpus/handbook/diagnostics/ragas?dataset=does-not-exist.jsonl")
    assert resp.status_code == 404


def test_ragas_diagnostics_caps_max_questions_at_ceiling(client) -> None:
    fake_runner = MagicMock()
    fake_runner.run.return_value = _FakeRunnerResult({})

    with patch("services.api_gateway.routers.corpus._build_diagnostics_pipeline", AsyncMock(return_value=(MagicMock(), MagicMock()))), \
         patch("services.api_gateway.routers.corpus._load_diagnostics_dataset", AsyncMock(return_value=MagicMock())), \
         patch("eval.ragas_runner.RagasRunner", return_value=fake_runner):
        resp = client.post("/corpus/handbook/diagnostics/ragas?max_questions=999999")

    assert resp.status_code == 200
    assert fake_runner.run.call_args.kwargs["max_questions"] == corpus._MAX_DIAGNOSTICS_QUESTIONS


def test_ragas_diagnostics_400_on_unknown_pipeline_id(client) -> None:
    with patch(
        "services.api_gateway.routers.corpus._build_diagnostics_pipeline",
        AsyncMock(side_effect=KeyError("Component not found: kind='pipeline' id='nope'")),
    ):
        resp = client.post("/corpus/handbook/diagnostics/ragas?pipeline_id=nope")
    assert resp.status_code == 400


def test_chunk_coherence_diagnostics_accepts_custom_sample_size(client) -> None:
    fake_judge = MagicMock()
    fake_judge.run.return_value = _FakeRunnerResult({"avg_coherence": 0.5})
    fake_judge.run.return_value.sample_size = 80

    fake_qdrant = MagicMock()
    fake_qdrant.scroll_all.return_value = [_chunk("a", "some text")]

    with patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=fake_qdrant), \
         patch("eval.chunk_coherence_judge.ChunkCoherenceJudge", return_value=fake_judge) as mock_judge:
        resp = client.post("/corpus/handbook/diagnostics/chunk-coherence?sample_size=80")

    assert resp.status_code == 200
    mock_judge.assert_called_once_with(sample_size=80)


def test_chunk_coherence_diagnostics_caps_sample_size_at_ceiling(client) -> None:
    fake_judge = MagicMock()
    fake_judge.run.return_value = _FakeRunnerResult({})
    fake_judge.run.return_value.sample_size = 0
    fake_qdrant = MagicMock()
    fake_qdrant.scroll_all.return_value = []

    with patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=fake_qdrant), \
         patch("eval.chunk_coherence_judge.ChunkCoherenceJudge", return_value=fake_judge) as mock_judge:
        resp = client.post("/corpus/handbook/diagnostics/chunk-coherence?sample_size=999999")

    assert resp.status_code == 200
    mock_judge.assert_called_once_with(sample_size=corpus._MAX_CHUNK_COHERENCE_SAMPLE_SIZE)


def test_trulens_diagnostics_endpoint_returns_metrics(client) -> None:
    fake_runner = MagicMock()
    fake_runner.run.return_value = _FakeRunnerResult({"groundedness": 0.7})

    with patch("services.api_gateway.routers.corpus._build_diagnostics_pipeline", AsyncMock(return_value=(MagicMock(), MagicMock()))), \
         patch("services.api_gateway.routers.corpus._load_diagnostics_dataset", AsyncMock(return_value=MagicMock())), \
         patch("eval.trulens_runner.TruLensRunner", return_value=fake_runner):
        resp = client.post("/corpus/handbook/diagnostics/trulens")

    assert resp.status_code == 200
    assert resp.json()["metrics"] == {"groundedness": 0.7}


def test_chunk_coherence_diagnostics_endpoint_returns_metrics(client) -> None:
    fake_judge = MagicMock()
    fake_judge.run.return_value = _FakeRunnerResult({"avg_coherence": 0.9})
    fake_judge.run.return_value.sample_size = 42

    fake_qdrant = MagicMock()
    fake_qdrant.scroll_all.return_value = [_chunk("a", "some text")]

    with patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=fake_qdrant), \
         patch("eval.chunk_coherence_judge.ChunkCoherenceJudge", return_value=fake_judge):
        resp = client.post("/corpus/handbook/diagnostics/chunk-coherence")

    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"] == {"avg_coherence": 0.9}
    assert body["sample_size"] == 42


def test_chunk_coherence_diagnostics_returns_incoherent_chunks_sorted_by_score(client) -> None:
    per_chunk = [
        {"chunk_id": "a", "score": 5, "text_preview": "a good one"},
        {"chunk_id": "b", "score": 0, "text_preview": "a truncated one"},
        {"chunk_id": "c", "score": 1, "text_preview": "nearly truncated"},
        {"chunk_id": "d", "score": 3, "text_preview": "fine"},
    ]
    fake_judge = MagicMock()
    fake_judge.run.return_value = _FakeRunnerResult({"avg_coherence": 0.5}, per_chunk=per_chunk)
    fake_judge.run.return_value.sample_size = 4

    fake_qdrant = MagicMock()
    fake_qdrant.scroll_all.return_value = [_chunk("a", "some text")]

    with patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=fake_qdrant), \
         patch("eval.chunk_coherence_judge.ChunkCoherenceJudge", return_value=fake_judge):
        resp = client.post("/corpus/handbook/diagnostics/chunk-coherence")

    assert resp.status_code == 200
    incoherent = resp.json()["incoherent_chunks"]
    assert [c["chunk_id"] for c in incoherent] == ["b", "c"]


def test_chunk_coherence_diagnostics_503_on_failure(client) -> None:
    with patch("services.api_gateway.routers.corpus._resolve_qdrant", side_effect=RuntimeError("down")):
        resp = client.post("/corpus/handbook/diagnostics/chunk-coherence")
    assert resp.status_code == 503
