"""core/diagnostics.

The template is what real systems are grown from, so what it declares and
records every derived system inherits. These tests pin the two properties
that make the contract worth declaring at all: it is derived from the object
it describes rather than asserted, and recording traces can never cost a
caller their answer.
"""
from __future__ import annotations

from core.diagnostics import DiagnosticContract, TraceLog, contract_for_pipeline


class _Pipeline:
    def __init__(self, reranker=None) -> None:
        self._reranker = reranker


class _Answer:
    def __init__(self, text="an answer", refs=(), stage_trace=None) -> None:
        self.text = text
        self.source_refs = list(refs)
        self.stage_trace = stage_trace


class _Ref:
    def __init__(self, chunk_id, path, score) -> None:
        self.chunk_id, self.structural_path, self.score = chunk_id, path, score


# ── The contract describes the object, not the author's intentions ──────


def test_pre_rerank_support_follows_from_having_a_reranker() -> None:
    """Claiming a pre-rerank snapshot from a pipeline with no reranker would
    make the platform wait for something that can never arrive."""
    assert contract_for_pipeline(_Pipeline()).supports_pre_rerank is False
    assert contract_for_pipeline(_Pipeline(reranker=object())).supports_pre_rerank is True


def test_retrieval_only_support_follows_from_the_endpoint_being_exposed() -> None:
    assert contract_for_pipeline(_Pipeline()).supports_retrieval_only is False
    assert contract_for_pipeline(_Pipeline(), retrieve_endpoint="/retrieve").supports_retrieval_only is True


def test_the_contract_uses_the_same_field_names_external_rags_declare() -> None:
    """One shape whether a system is the platform's own or someone else's,
    so a reader never has to translate between two vocabularies."""
    data = contract_for_pipeline(_Pipeline()).to_dict()
    for key in (
        "supports_trace", "supports_retrieval_only", "retrieve_endpoint",
        "max_top_k", "source_ref_granularity", "supported_params",
    ):
        assert key in data


def test_declaring_no_params_is_different_from_declaring_none_known() -> None:
    # An empty list is a statement ("this system reads no params"), which is
    # what stops the platform from believing an ignored knob did something.
    assert contract_for_pipeline(_Pipeline()).to_dict()["supported_params"] == []


def test_an_omitted_capability_reads_as_not_offered() -> None:
    assert DiagnosticContract().supports_trace_export is False


# ── The trace log ───────────────────────────────────────────────────────


def test_nothing_is_recorded_until_a_deployment_switches_it_on() -> None:
    """A trace holds a user's question and fragments of retrieved documents,
    so recording is a deployment's decision rather than a default."""
    log = TraceLog()
    log.record("t1", "a question", _Answer(), created_at="2026-01-01")
    assert len(log) == 0
    assert log.export() == []


def test_a_recorded_trace_keeps_the_question_and_what_came_back() -> None:
    log = TraceLog(enabled=True)
    log.record("t1", "a question", _Answer(refs=[_Ref("c1", "Ch 1", 0.7)]), created_at="2026-01-01")
    entry = log.export()[0]
    assert entry["query"] == "a question"
    assert entry["sources"] == [{"chunk_id": "c1", "structural_path": "Ch 1", "score": 0.7}]


def test_recording_never_raises_on_a_malformed_answer() -> None:
    """A served system must not fail a user's request because its own
    diagnostics could not be written down."""
    log = TraceLog(enabled=True)
    log.record("t1", "a question", object(), created_at="2026-01-01")  # must not raise
    assert len(log) <= 1


def test_the_log_is_bounded_and_drops_the_oldest() -> None:
    # Unbounded, this is a memory leak with a delayed fuse inside every
    # system grown from the template.
    log = TraceLog(capacity=2, enabled=True)
    for i in range(5):
        log.record(f"t{i}", f"question {i}", _Answer(), created_at="2026-01-01")
    assert len(log) == 2
    assert [e["trace_id"] for e in log.export()] == ["t4", "t3"]


def test_export_returns_newest_first() -> None:
    # A collector polling periodically wants what it has not seen yet, and
    # that is always at the recent end.
    log = TraceLog(enabled=True)
    log.record("old", "q", _Answer(), created_at="2026-01-01")
    log.record("new", "q", _Answer(), created_at="2026-01-02")
    assert [e["trace_id"] for e in log.export()] == ["new", "old"]


def test_export_honours_a_limit() -> None:
    log = TraceLog(enabled=True)
    for i in range(4):
        log.record(f"t{i}", "q", _Answer(), created_at="2026-01-01")
    assert len(log.export(limit=2)) == 2


def test_the_answer_is_kept_as_a_bounded_preview_not_in_full() -> None:
    log = TraceLog(enabled=True)
    log.record("t1", "q", _Answer(text="x" * 5000), created_at="2026-01-01")
    assert len(log.export()[0]["answer_preview"]) < 5000
