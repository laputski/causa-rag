"""Smoke tests for eval/ragas_runner.py and eval/trulens_runner.py against a
real pipeline + Ollama. Mirrors tests/eval/test_deepeval_suite.py's
gating/skip pattern (see that file's module docstring for why generator
and judge models are deliberately decoupled).

Intentionally light — these are a cross-check alongside DeepEval, not the
primary SLA gate (see eval/ragas_runner.py module docstring). Just confirms
each runner executes end-to-end and produces a plausible metrics dict.

Run:
    pytest tests/eval/test_ragas_trulens_smoke.py -v -m integration
"""
from __future__ import annotations

import os
import pathlib

import pytest

from eval.judge_model import JUDGE_MODEL as _JUDGE_MODEL

pytestmark = [pytest.mark.integration]

_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")


def _check_ollama(model: str) -> bool:
    try:
        import httpx
        resp = httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        names = {m.get("name") for m in resp.json().get("models", [])}
        return model in names
    except Exception:
        return False


def _check_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=3,
            check_compatibility=False,
        ).get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def pipeline_and_dataset():
    if not _check_ollama(_OLLAMA_MODEL):
        pytest.skip(f"Ollama unreachable, or model {_OLLAMA_MODEL!r} not pulled")
    if not _check_ollama(_JUDGE_MODEL):
        pytest.skip(f"Ollama unreachable, or judge model {_JUDGE_MODEL!r} not pulled")
    if not _check_qdrant():
        pytest.skip("Qdrant unreachable. Run: docker compose up -d")

    from adapters.bge_m3 import BgeM3Embedder
    from adapters.ollama_generator import OllamaGenerator
    from adapters.qdrant import QdrantRetriever
    from core.models import Chunk
    from core.pipeline import NaivePipeline
    from eval.dataset import EvalDataset

    embedder = BgeM3Embedder(use_real_model=os.getenv("USE_REAL_BGE_M3", "").lower() == "true")
    if not embedder._use_real_model:
        pytest.skip("USE_REAL_BGE_M3 is not 'true'. With the stub embedder every metric is meaningless.")

    retriever = QdrantRetriever(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        strategy_id="ragas_trulens_smoke",
        embedder_id="bge_m3",
    )
    generator = OllamaGenerator(base_url=_OLLAMA_URL, model=_OLLAMA_MODEL)
    pipeline = NaivePipeline(retriever=retriever, embedder=embedder, generator=generator)

    seed_chunks = [
        Chunk(text="Article 44. What a team may decide. A team may decide anything within its own budget and headcount without a further approval.", doc_id="demo_handbook", structural_path="handbook/44", metadata={"status": "active"}),
        Chunk(text="Article 197. A claim must be raised within three years.", doc_id="demo_handbook", structural_path="handbook/197", metadata={"status": "active"}),
    ]
    vecs = embedder.embed([c.text for c in seed_chunks])
    retriever.upsert(seed_chunks, vecs)

    dataset_path = pathlib.Path("eval/golden/handbook.v1.fast.jsonl")
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    dataset = EvalDataset.from_jsonl(dataset_path)
    return pipeline, dataset


def test_ragas_runner_smoke(pipeline_and_dataset):
    try:
        import langchain_ollama  # noqa: F401
        import ragas  # noqa: F401
    except ImportError:
        pytest.skip("ragas/langchain-ollama not installed. Run: pip install -e '.[judges]'")

    from eval.ragas_runner import RagasRunner
    pipeline, dataset = pipeline_and_dataset
    runner = RagasRunner(pipeline=pipeline, ollama_model=_JUDGE_MODEL, ollama_base_url=_OLLAMA_URL)
    result = runner.run(dataset, max_questions=2)
    assert result.metrics
    assert all(0.0 <= v <= 1.0 for v in result.metrics.values())


def test_trulens_runner_smoke(pipeline_and_dataset):
    try:
        from trulens.providers.litellm import LiteLLM  # noqa: F401
    except ImportError:
        pytest.skip("trulens-providers-litellm not installed. Run: pip install -e '.[judges]'")

    from eval.trulens_runner import TruLensRunner
    pipeline, dataset = pipeline_and_dataset
    runner = TruLensRunner(pipeline=pipeline, ollama_model=_JUDGE_MODEL, ollama_base_url=_OLLAMA_URL)
    result = runner.run(dataset, max_questions=2)
    assert set(result.metrics) == {"context_relevance", "groundedness", "answer_relevance"}
