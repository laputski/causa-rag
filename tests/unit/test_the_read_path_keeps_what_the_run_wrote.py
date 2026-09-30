"""Everything a run writes about itself survives being read back.

The catalogue holds an entry for a read path that loses data, and the entry
describes what happened here: three fields were written by a run, stored, and
dropped on the way back in. Two of them had just been added to catch three
other failures, so every check consulting the load record went silent the
moment a run was read back from the store. A silent check and a passing
one are the same thing on a screen.

Derived from the writer and never listed by hand. A list typed here would be
correct on the day it was typed, which is the same failure one field over.
"""
from __future__ import annotations

from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, QuestionResult
from services.api_gateway.routers.experiments import _parse_result


def _written() -> ExperimentResult:
    """A run with every field it can carry set to something recognisable."""
    result = ExperimentResult(
        config=ExperimentConfig(
            name="r", corpus_id="handbook", dataset_name="handbook.v1.jsonl",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="ollama"),
        ),
        run_id="run-1", started_at="2026-09-01T10:00:00Z", finished_at="2026-09-01T10:05:00Z",
        n_questions=1, dataset_name="handbook.v1.jsonl",
    )
    result.question_results = [QuestionResult(
        question_id="q1", question="?", reference_answer="", generated_answer="an answer",
        metrics={"retrieval_recall_at_k": 1.0}, answerability="answerable",
    )]
    result.aggregate_metrics = {"retrieval_recall_at_k": 1.0}
    result.realm_id = "demo"
    result.prompt_id = "p1"
    result.prompt_version = 3
    result.generator_model = "qwen3:8b"
    result.coverage_check = {"checked": True, "reason": ""}
    result.unavailable_components = ["reranker:cross_encoder"]
    result.applied = {"query_embedder_id": "bge_m3", "index_embedder_id": "bge_m3"}
    result.corpus_manifest = {"embedder_id": "bge_m3", "documents_digest": "a" * 64,
                              "embedder_is_real_model": True}
    return result


#: Written by the run and deliberately not read back. `stopped` is the only
#: one, and it is restored under its own name a few lines below the block
#: this checks; the entry exists so an addition to it has to be argued for.
_NOT_EXPECTED_BACK: set[str] = set()


def test_every_field_the_run_writes_comes_back() -> None:
    written = _written()
    read = _parse_result(written.to_dict())
    assert read is not None, "the run did not parse at all"

    lost: list[str] = []
    for name in written.to_dict():
        if name in _NOT_EXPECTED_BACK or name == "question_results":
            continue
        before = getattr(written, name, None)
        if before in (None, "", 0, [], {}):
            continue
        after = getattr(read, name, None)
        if not after:
            lost.append(name)
    assert lost == [], (
        f"written by the run and dropped by the read: {lost}. Every check reading one of "
        "these goes silent for a run that was stored, which looks exactly like a check "
        "that passed."
    )


def test_the_load_record_survives_in_particular() -> None:
    """Named on its own because three checks rest on it, and because this is
    the field the loss was found in."""
    read = _parse_result(_written().to_dict())
    assert read is not None
    assert read.corpus_manifest.get("embedder_is_real_model") is True
    assert read.applied.get("index_embedder_id") == "bge_m3"


def test_a_run_stored_before_these_fields_existed_still_parses() -> None:
    """The other direction. A stored run that never carried them reads back
    with them empty, and empty is the honest answer for a run made before
    anybody recorded it."""
    old: dict[str, Any] = _written().to_dict()
    for name in ("applied", "corpus_manifest", "unavailable_components"):
        old.pop(name)
    read = _parse_result(old)
    assert read is not None
    assert read.corpus_manifest == {}
    assert read.applied == {}
    assert read.unavailable_components == []


def _every_question_field_set() -> QuestionResult:
    """A question with each of its fields set to something recognisable."""
    ref = {"chunk_id": "c1", "doc_id": "d1", "chunk_text": "t", "score": 0.5}
    return QuestionResult(
        question_id="q1", question="?", reference_answer="r", generated_answer="a",
        source_refs=[ref], metrics={"retrieval_recall_at_k": 1.0},
        pre_rerank_source_refs=[ref], candidate_source_refs=[ref],
        dense_source_refs=[ref], sparse_source_refs=[ref], graph_source_refs=[ref],
        expected_refs=["d1"], computed_citations=["1"],
        stage_trace={"total_ms": 1.0}, error="e", answerability="answerable",
        root_cause={"cause": "retrieval"},
    )


def test_every_field_a_question_carries_comes_back() -> None:
    """The question's fields used to be typed out twice, once to write and
    once to read, and a field added to the class was written by every run and
    read by none until somebody typed it into both lists."""
    from dataclasses import fields

    written = _every_question_field_set()
    unset = [f.name for f in fields(QuestionResult) if getattr(written, f.name) in (None, "", [], {})]
    assert unset == [], f"the fixture leaves {unset} unset, so the check below cannot see them"

    run = _written()
    run.question_results = [written]
    read = _parse_result(run.to_dict())
    assert read is not None
    lost = [f.name for f in fields(QuestionResult)
            if getattr(read.question_results[0], f.name) != getattr(written, f.name)]
    assert lost == [], f"written with the question and changed or dropped by the read: {lost}"
