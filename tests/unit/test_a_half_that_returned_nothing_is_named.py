"""A half the pipeline merges that returned nothing on every question.

Found live: a realm's dense collection had been empty for two months, the run
was hybrid throughout, the lexical half supplied every fragment, and the one
finding that fired at error said the corpus had been embedded with random
vectors.
"""
from __future__ import annotations

from core.eval.detectors import detect_half_returned_nothing, run_detectors

_REF = {"chunk_id": "c", "doc_id": "d"}


def _run(pipeline: str, dense: bool, sparse: bool, n: int = 5, graph: bool = False) -> dict:
    return {"config": {"pipeline_id": pipeline}, "question_results": [
        {"question_id": f"q{i}", "metrics": {},
         "dense_source_refs": [_REF] if dense else [],
         "sparse_source_refs": [_REF] if sparse else [],
         "graph_source_refs": [_REF] if graph else []} for i in range(n)]}


def test_an_empty_dense_half_is_named() -> None:
    finding = detect_half_returned_nothing(_run("hybrid_rrf", dense=False, sparse=True))
    assert finding is not None and finding.params == {"half": "dense", "questions": 5}
    assert "half_returned_nothing" in {d.id for d in run_detectors(_run("hybrid_rrf", False, True))}


def test_an_empty_graph_half_is_named_for_the_graph_pipeline() -> None:
    finding = detect_half_returned_nothing(_run("graph", dense=True, sparse=False, graph=False))
    assert finding is not None and finding.params["half"] == "graph"


def test_both_halves_answering_is_silent() -> None:
    assert detect_half_returned_nothing(_run("hybrid_rrf", dense=True, sparse=True)) is None


def test_a_half_empty_on_some_questions_is_silent() -> None:
    """Finding nothing relevant for some questions is what retrieval does."""
    run = _run("hybrid_rrf", dense=True, sparse=True)
    run["question_results"][0]["dense_source_refs"] = []
    assert detect_half_returned_nothing(run) is None


def test_a_run_stored_before_the_halves_were_recorded_is_silent() -> None:
    assert detect_half_returned_nothing(_run("hybrid_rrf", dense=False, sparse=False)) is None


def test_a_single_source_pipeline_is_silent() -> None:
    assert detect_half_returned_nothing(_run("naive", dense=True, sparse=False)) is None


def test_an_empty_dense_half_is_not_reported_as_a_stub_embedder() -> None:
    """Every merged fragment then carries a dense score of zero, which the stub
    check read as random vectors: found live at severity error."""
    run = _run("hybrid_rrf", dense=False, sparse=True)
    for q in run["question_results"]:
        q["source_refs"] = [{"chunk_id": "c", "doc_id": "d", "dense_score": 0.0, "sparse_score": 3.2}] * 3
    ids = {d.id for d in run_detectors(run)}
    assert "half_returned_nothing" in ids and "stub_embedder" not in ids
