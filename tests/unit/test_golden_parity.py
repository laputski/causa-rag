"""golden-parity E2E: the same toy RAG, run (a) in_process
and (b) over HTTP via causa_rag_client.serve(), must produce matching
metrics on the same dataset. This is the regression test for exactly the bug
class this session already hit once for real (corpus_id silently defaulting
-> recall collapsed from ~0.97 to ~0.06, see the design notes) — a
platform-side wiring gap that quietly drops/distorts a metric without ever
raising an error.

Also covers the acceptance criteria's "no metric loss" (every expected key
present, not None, for answerable questions) and "attribution parity" (a
synthetic rerank-drop scenario must produce the same funnel.py verdict via
both paths).
"""
# @lat: [[connector-client#Golden-parity E2E — the regression test for "the platform silently loses/distorts a metric"]]
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from adapters.bge_m3 import BgeM3Embedder
from core.eval.funnel import diagnose_question
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.interfaces import Generator, Retriever
from core.models import Chunk, ScoredChunk
from core.pipeline import NaivePipeline
from core.registry import ComponentRegistry
from eval.dataset import EvalDataset

# clients/python/ is a standalone package (its own pyproject.toml, not pip
# installed into this monorepo's dev env) — add it to sys.path so this test
# can import causa_rag_client, mirroring clients/python/tests/conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "clients" / "python"))
from causa_rag_client import serve  # noqa: E402

# ── Shared toy "RAG" — ONE definition of retrieval/generation behaviour,
# wired into both an in-process NaivePipeline and a serve()'d HTTP app, so
# any difference in the resulting metrics is a platform wiring bug, not a
# difference between two independently-written toy RAGs. doc_id/source_code/
# article_no/structural_path follow the same "{source_code}/{article_no}"
# convention core/eval/retrieval_metrics.py:extract_ref_id expects. ────────

_DOC_META: dict[str, tuple[str, str]] = {  # doc_id -> (source_code, article_no)
    "art1": ("TOY", "1"),
    "art2": ("TOY", "2"),
}
_CORPUS: dict[str, str] = {
    "art1": "the right to rest is set out in article 1",
    "art2": "the right to training is set out in article 2",
}

# question -> doc_ids returned, in rank order (first = most relevant)
_RETRIEVAL: dict[str, list[str]] = {
    "What do the rules say about rest?": ["art1", "art2"],
    "What do the rules say about training?": ["art2", "art1"],
}

_MARKER_RE = re.compile(r"§(\S+?)§")


def _toy_retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    doc_ids = _RETRIEVAL.get(query, [])[:top_k]
    sources = []
    for d in doc_ids:
        source_code, article_no = _DOC_META[d]
        sources.append({
            "doc_id": d,
            "chunk_text": f"§{d}§ {_CORPUS[d]}",
            "structural_path": f"[{article_no}]",
            "source_code": source_code,
            "article_no": article_no,
        })
    return sources


def _toy_answer(doc_ids: list[str]) -> str:
    if not doc_ids:
        return "No answer found."
    cites = ", ".join(f"Article {_DOC_META[d][1]}" for d in doc_ids if d in _DOC_META)
    return f"An answer citing: {cites}."


def _toy_generate(query: str, sources: list[dict[str, Any]]) -> str:
    return _toy_answer([s["doc_id"] for s in sources])


class _ToyRetriever:
    """Wraps _toy_retrieve for the in_process path (core/interfaces.Retriever)."""

    retriever_id = "toy_retriever"

    def retrieve(self, query: str, k: int, filters: dict | None = None, **_: Any) -> list[ScoredChunk]:
        results = _toy_retrieve(query, k)
        return [
            ScoredChunk(
                chunk=Chunk(
                    doc_id=r["doc_id"], text=r["chunk_text"], structural_path=r["structural_path"],
                    metadata={"source_code": r["source_code"], "article_no": r["article_no"]},
                ),
                score=1.0 - i * 0.01,
                retriever_id=self.retriever_id,
            )
            for i, r in enumerate(results)
        ]


class _ToyGenerator:
    """Extracts the §doc_id§ markers _ToyRetriever embeds in chunk text and
    feeds them through the SAME _toy_answer() the HTTP side's _toy_generate
    uses — so the in_process and HTTP answer texts are byte-identical
    regardless of how NaivePipeline renders the prompt around them."""

    generator_id = "toy_generator"

    def generate(self, prompt: str, **params: Any) -> str:
        return _toy_answer(_MARKER_RE.findall(prompt))


assert isinstance(_ToyRetriever(), Retriever)
assert isinstance(_ToyGenerator(), Generator)


def _asgi_mock_transport(app: Any) -> httpx.MockTransport:
    """httpx.ASGITransport only implements the ASYNC transport interface
    (handle_async_request) — adapters/http_pipeline.py's HttpPipeline uses a
    plain sync httpx.Client, which needs handle_request and raises
    AttributeError on ASGITransport ("no attribute '__enter__'"). Routing
    each call through fastapi.testclient.TestClient (itself a sync wrapper
    around the ASGI app) inside a MockTransport sidesteps that mismatch
    without touching HttpPipeline's sync design.
    """
    client = TestClient(app)

    def handler(request: httpx.Request) -> httpx.Response:
        return client.request(
            request.method, request.url.path, content=request.content, headers=request.headers,
        )

    return httpx.MockTransport(handler)


def _make_in_process_runner() -> ExperimentRunner:
    reg = ComponentRegistry()
    embedder = BgeM3Embedder()
    pipeline = NaivePipeline(retriever=_ToyRetriever(), embedder=embedder, generator=_ToyGenerator())
    reg.register("embedder", "bge_m3", embedder)
    reg.register("pipeline", "naive", pipeline)
    return ExperimentRunner(registry=reg)


def _make_http_runner() -> tuple[ExperimentRunner, httpx.BaseTransport]:
    app = serve(_toy_retrieve, _toy_generate)
    transport = _asgi_mock_transport(app)
    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", BgeM3Embedder())

    def _resolver(_rag_id: str) -> dict[str, Any]:
        return {"url": "http://toy-rag.test/", "headers": {}, "retrieve_endpoint": None,
                "request_template": None, "response_mapping": None}

    runner = ExperimentRunner(registry=reg, external_rag_resolver=_resolver)
    return runner, transport


def _toy_dataset() -> EvalDataset:
    return EvalDataset.from_list([
        {
            "id": "q1", "question": "What do the rules say about rest?",
            "reference_answer": "the right to rest", "article_refs": ["TOY/1"],
            "answerability": "answerable",
        },
        {
            "id": "q2", "question": "What do the rules say about training?",
            "reference_answer": "the right to training", "article_refs": ["TOY/2"],
            "answerability": "answerable",
        },
    ])


def _evaluator(embedder: Any) -> Any:
    from services.api_gateway.routers.experiments import _CompositeEvaluator
    return _CompositeEvaluator(embedder=embedder, top_k=5, ref_resolver=None)


def _patch_http_pipeline_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.BaseTransport) -> None:
    """ExperimentRunner._build_pipeline constructs HttpPipeline itself (no
    transport injection point in the public API) — patch HttpPipeline's
    httpx.Client construction to use the mock transport instead of a real
    socket, exactly like adapters/http_pipeline.py's own tests do via the
    constructor's `transport=` param, just one layer further in."""
    import adapters.http_pipeline as http_pipeline_module

    original_init = http_pipeline_module.HttpPipeline.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("transport", transport)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(http_pipeline_module.HttpPipeline, "__init__", patched_init)


def test_golden_parity_metrics_match_between_in_process_and_http(monkeypatch: pytest.MonkeyPatch) -> None:
    in_process_runner = _make_in_process_runner()
    in_process_cfg = ExperimentConfig(
        name="golden_parity_in_process",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="toy_generator"),
        pipeline_id="naive",
    )
    in_process_result = in_process_runner.run(
        in_process_cfg, _toy_dataset(), evaluator=_evaluator(BgeM3Embedder()),
    )

    http_runner, transport = _make_http_runner()
    _patch_http_pipeline_transport(monkeypatch, transport)
    http_cfg = ExperimentConfig(
        name="golden_parity_http",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="toy_generator"),
        pipeline_source="http",
        external_rag_id="toy-rag",
    )
    http_result = http_runner.run(http_cfg, _toy_dataset(), evaluator=_evaluator(BgeM3Embedder()))

    # Answers and source_refs must be byte-identical — same retrieval, same
    # generation logic, only the transport differs.
    in_process_answers = [qr.generated_answer for qr in in_process_result.question_results]
    http_answers = [qr.generated_answer for qr in http_result.question_results]
    assert in_process_answers == http_answers

    in_process_doc_ids = [[sr["doc_id"] for sr in qr.source_refs] for qr in in_process_result.question_results]
    http_doc_ids = [[sr["doc_id"] for sr in qr.source_refs] for qr in http_result.question_results]
    assert in_process_doc_ids == http_doc_ids

    # The actual regression target: aggregate metrics must match within a
    # tight tolerance (exact equality for the deterministic ones; a small
    # epsilon covers any floating-point embedder noise on the semantic ones).
    assert in_process_result.aggregate_metrics.keys() == http_result.aggregate_metrics.keys()
    for key, in_process_value in in_process_result.aggregate_metrics.items():
        http_value = http_result.aggregate_metrics[key]
        assert in_process_value == pytest.approx(http_value, abs=1e-6), f"metric {key!r} diverged"


def test_no_metric_loss_for_answerable_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every expected metric key must be present and non-null for an
    answerable question, on BOTH paths — a silently-missing key is exactly
    how the corpus_id bug first went unnoticed (aggregate mean over fewer
    keys looked plausible, not obviously wrong)."""
    expected_keys = {
        "retrieval_recall_at_k", "retrieval_precision_at_k", "retrieval_average_precision",
        "answer_similarity", "answer_relevance", "context_support",
        "grounded_in_correct_source", "citation_number_coverage", "correct_refusal",
    }

    in_process_runner = _make_in_process_runner()
    in_process_result = in_process_runner.run(
        ExperimentConfig(
            name="no_loss_in_process",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="toy_generator"),
            pipeline_id="naive",
        ),
        _toy_dataset(),
        evaluator=_evaluator(BgeM3Embedder()),
    )
    for qr in in_process_result.question_results:
        assert expected_keys <= qr.metrics.keys(), f"missing keys for {qr.question_id}: {expected_keys - qr.metrics.keys()}"
        for key in expected_keys:
            assert qr.metrics[key] is not None

    http_runner, transport = _make_http_runner()
    _patch_http_pipeline_transport(monkeypatch, transport)
    http_result = http_runner.run(
        ExperimentConfig(
            name="no_loss_http",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="toy_generator"),
            pipeline_source="http",
            external_rag_id="toy-rag",
        ),
        _toy_dataset(),
        evaluator=_evaluator(BgeM3Embedder()),
    )
    for qr in http_result.question_results:
        assert expected_keys <= qr.metrics.keys(), f"missing keys for {qr.question_id}: {expected_keys - qr.metrics.keys()}"
        for key in expected_keys:
            assert qr.metrics[key] is not None


# ── Attribution parity — a synthetic rerank-drop must give the same
# funnel.py verdict via both paths. ─────────────────────

def _drop_target_rerank(query: str, sources: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Simulates a reranker that throws away the one correct chunk (art1),
    keeping only the decoy — retrieval found it, rerank lost it."""
    return [s for s in sources if s["doc_id"] != "art1"][:top_k]


class _DropTargetReranker:
    reranker_id = "drop_target"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return [sc for sc in candidates if sc.chunk.doc_id != "art1"]


def test_rerank_failure_attribution_matches_between_in_process_and_http(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = EvalDataset.from_list([
        {
            "id": "q1", "question": "What do the rules say about rest?",
            "reference_answer": "the right to rest", "article_refs": ["TOY/1"],
            "answerability": "answerable",
        },
    ])

    # in_process: NaivePipeline's reranker hook drops the target chunk.
    # ExperimentRunner._build_pipeline rebuilds a FRESH pipeline instance
    # from ExperimentConfig — it never reads the registered base pipeline's
    # own .reranker attribute when config.reranker is unset,
    # so the reranker must be registered separately and selected
    # via config.reranker, not passed directly to NaivePipeline here.
    reg = ComponentRegistry()
    embedder = BgeM3Embedder()
    in_process_pipeline = NaivePipeline(retriever=_ToyRetriever(), embedder=embedder, generator=_ToyGenerator())
    reg.register("embedder", "bge_m3", embedder)
    reg.register("pipeline", "naive", in_process_pipeline)
    reg.register("reranker", "drop_target", _DropTargetReranker())
    in_process_runner = ExperimentRunner(registry=reg)
    in_process_result = in_process_runner.run(
        ExperimentConfig(
            name="attribution_in_process",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="toy_generator"),
            pipeline_id="naive",
            reranker=ComponentRef(kind="reranker", component_id="drop_target"),
        ),
        dataset,
        evaluator=_evaluator(embedder),
    )

    # http: serve()'s rerank_fn drops the same target chunk.
    app = serve(_toy_retrieve, _toy_generate, rerank_fn=_drop_target_rerank)
    transport = _asgi_mock_transport(app)
    http_reg = ComponentRegistry()
    http_reg.register("embedder", "bge_m3", BgeM3Embedder())

    def _resolver(_rag_id: str) -> dict[str, Any]:
        return {"url": "http://toy-rag.test/", "headers": {}, "retrieve_endpoint": None,
                "request_template": None, "response_mapping": None}

    http_runner = ExperimentRunner(registry=http_reg, external_rag_resolver=_resolver)
    _patch_http_pipeline_transport(monkeypatch, transport)
    http_result = http_runner.run(
        ExperimentConfig(
            name="attribution_http",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="toy_generator"),
            pipeline_source="http",
            external_rag_id="toy-rag",
        ),
        dataset,
        evaluator=_evaluator(http_reg.resolve("embedder", "bge_m3")),
    )

    in_process_qr = in_process_result.question_results[0]
    http_qr = http_result.question_results[0]

    # Both retrieval (pre-rerank) found art1; both lost it to reranking —
    # the precondition for a "rerank" verdict, not "retrieval".
    assert in_process_qr.pre_rerank_source_refs and any(
        sr["doc_id"] == "art1" for sr in in_process_qr.pre_rerank_source_refs
    )
    assert http_qr.pre_rerank_source_refs and any(
        sr["doc_id"] == "art1" for sr in http_qr.pre_rerank_source_refs
    )

    in_process_pre_rerank_recall = in_process_qr.metrics.get("pre_rerank_recall_at_k")
    http_pre_rerank_recall = http_qr.metrics.get("pre_rerank_recall_at_k")
    assert in_process_pre_rerank_recall is not None
    assert http_pre_rerank_recall is not None

    in_process_verdict = diagnose_question("answerable", in_process_qr.metrics, in_process_pre_rerank_recall)
    http_verdict = diagnose_question("answerable", http_qr.metrics, http_pre_rerank_recall)

    assert in_process_verdict.layer == "rerank"
    assert http_verdict.layer == "rerank"
    assert in_process_verdict.layer == http_verdict.layer
