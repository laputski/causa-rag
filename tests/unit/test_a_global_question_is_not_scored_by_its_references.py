"""A question about the corpus as a whole is not scored by its references.

Such a question carries the documents it was drawn from, and no list of every
source that answers it exists. A recall against those references measures the
list and not the retrieval, and two metrics that score the answer against the
same references inherit the same flaw. The six are left out for it, the
declaration of each says why, and the detector for a metric without grounds
reads that declaration.
"""
from __future__ import annotations

from core.eval.detectors import detect_metric_without_grounds
from core.eval.funnel import diagnose_question
from core.models import Answer, SourceRef
from services.api_gateway.routers.experiments import _CompositeEvaluator

_BOUND_TO_REFERENCES = {
    "retrieval_recall_at_k", "retrieval_precision_at_k", "retrieval_average_precision",
    "pre_rerank_recall_at_k", "grounded_in_correct_source", "citation_number_coverage",
}


class _Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def _answer() -> Answer:
    ref = SourceRef(doc_id="d1", chunk_id="c1", chunk_text="context",
                    source_code="SRC001", article_no="44")
    return Answer(text="An answer drawing on several documents.",
                  source_refs=[ref], pre_rerank_source_refs=[ref])


def _question(scope: str | None) -> dict:
    q = {"question": "What do the documents require overall?", "ground_truth": "a reference",
         "article_refs": ["SRC001/44", "SRC002/3"], "answerability": "answerable"}
    if scope is not None:
        q["scope"] = scope
    return q


def test_a_global_question_carries_none_of_the_six() -> None:
    metrics = _CompositeEvaluator(embedder=_Embedder(), top_k=5).evaluate(_question("global"), _answer())
    assert not _BOUND_TO_REFERENCES & set(metrics)
    # What does not rest on the references is still measured.
    assert {"correct_refusal", "answer_similarity", "answer_relevance"} <= set(metrics)


def test_a_question_without_a_scope_is_scored_as_before() -> None:
    """Every question written before scopes existed is a local one."""
    metrics = _CompositeEvaluator(embedder=_Embedder(), top_k=5).evaluate(_question(None), _answer())
    assert "retrieval_recall_at_k" in metrics and "pre_rerank_recall_at_k" in metrics


def _run(scope: str) -> dict:
    return {"config": {}, "question_results": [{
        "question_id": "q1", "scope": scope, "answerability": "answerable",
        "metrics": {"retrieval_recall_at_k": 0.5},
    }]}


def test_a_recall_recorded_for_a_global_question_is_a_finding() -> None:
    finding = detect_metric_without_grounds(_run("global"))
    assert finding is not None
    assert finding.params["grounds"][0]["precondition"] == "bounded_by_its_sources"


def test_the_same_recall_for_a_local_question_is_not() -> None:
    assert detect_metric_without_grounds(_run("local")) is None


def test_the_funnel_gives_a_global_question_no_verdict_it_cannot_ground() -> None:
    """Without the scope, an answerable question with no recall fell through
    to the similarity branch, which reads the absence as retrieval having
    found the source."""
    verdict = diagnose_question("answerable", {"answer_similarity": 0.2}, None, scope="global")
    assert verdict.layer == "not_applicable"


def test_a_run_carries_each_question_s_scope_from_its_row() -> None:
    """Without it the stored question could not say which of its metrics
    were left out on purpose, and the detector would have nothing to read."""
    from adapters.bge_m3 import BgeM3Embedder
    from adapters.generator_stub import GeneratorStub
    from adapters.qdrant import QdrantRetrieverStub
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentRunner
    from core.pipeline import NaivePipeline
    from core.registry import ComponentRegistry
    from eval.dataset import EvalDataset

    reg = ComponentRegistry()
    emb, ret, gen = BgeM3Embedder(), QdrantRetrieverStub(), GeneratorStub()
    reg.register("embedder", "bge_m3", emb)
    reg.register("retriever", "qdrant_dense_stub", ret)
    reg.register("generator", "stub", gen)
    reg.register("pipeline", "naive", NaivePipeline(retriever=ret, embedder=emb, generator=gen))
    config = ExperimentConfig(
        name="scopes",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        retrievers=[ComponentRef(kind="retriever", component_id="qdrant_dense_stub")],
        generator=ComponentRef(kind="generator", component_id="stub"),
    )
    dataset = EvalDataset.from_list([
        {"id": "g", "question": "overall?", "article_refs": ["A/1"], "scope": "global"},
        {"id": "l", "question": "one fact?", "article_refs": ["A/1"]},
    ])
    result = ExperimentRunner(registry=reg).run(config, dataset)
    assert [qr.scope for qr in result.question_results] == ["global", "local"]
