"""ExperimentRunner.run() — per-question error isolation.

Found live: an external RAG (a slow/intermittently-hanging one, tested
against a real external service — see the design notes) timed out on one
question deep into a run. Before this fix, ExperimentRunner.run()'s loop had
no try/except around pipeline.run()/retrieve() at all — that one exception
aborted the whole run, discarding every already-answered question's results,
and the gateway reported the entire experiment as failed via _errors.

Now a single question's exception is recorded on that QuestionResult (empty
answer, metrics stay {} — same "absent ⇒ not aggregated" convention the
existing metric-averaging code already relies on) and the loop continues, so
the run still completes with partial results instead of losing everything.
"""
from __future__ import annotations

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.models import Answer, SourceRef
from core.registry import ComponentRegistry
from eval.dataset import make_stub_dataset


class _FlakyPipeline:
    """Bare duck-typed pipeline exposing only what ExperimentRunner actually
    calls (`.run`) — raises on chosen call indices, standing in for an
    external RAG that hangs/times out on some but not all questions.
    Deliberately not core/pipeline.py's NaivePipeline: that wraps generation
    in a Langfuse tracing span (a generator-based context manager) which
    mishandles an exception injected mid-span — an unrelated pre-existing
    rough edge, not what this file is about. The real failure this mirrors
    (adapters/http_pipeline.py's HttpPipeline.run() raising on a timed-out
    httpx call) has no such wrapping either.
    """

    def __init__(self, fail_at: set[int]) -> None:
        self._fail_at = fail_at
        self._calls = 0

    def run(self, req) -> Answer:
        i = self._calls
        self._calls += 1
        if i in self._fail_at:
            raise TimeoutError("[Errno 60] Operation timed out")
        return Answer(
            text=f"answer #{i}",
            source_refs=[SourceRef(doc_id="d1", chunk_id="c1")],
        )


def _make_runner(pipeline) -> ExperimentRunner:
    runner = ExperimentRunner(registry=ComponentRegistry())
    runner._build_pipeline = lambda config, *a, **kw: pipeline  # type: ignore[method-assign]
    return runner


def _base_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="flaky_rag_run",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        retrievers=[ComponentRef(kind="retriever", component_id="qdrant_dense_stub")],
        generator=ComponentRef(kind="generator", component_id="stub"),
    )


def test_one_failed_question_does_not_abort_the_whole_run():
    runner = _make_runner(_FlakyPipeline(fail_at={2}))  # 3rd of 5 questions
    ds = make_stub_dataset(n=5)

    result = runner.run(_base_config(), ds)  # must not raise

    assert len(result.question_results) == 5
    assert result.n_errors() == 1
    failed = result.question_results[2]
    assert failed.error is not None
    assert "timed out" in failed.error
    assert failed.generated_answer == ""
    assert failed.metrics == {}
    # The other 4 questions answered normally, unaffected by their sibling's failure.
    for i, qr in enumerate(result.question_results):
        if i != 2:
            assert qr.error is None
            assert qr.generated_answer != ""


def test_multiple_failures_are_all_isolated():
    runner = _make_runner(_FlakyPipeline(fail_at={0, 3}))
    ds = make_stub_dataset(n=4)

    result = runner.run(_base_config(), ds)

    assert result.n_errors() == 2
    assert len(result.question_results) == 4


def test_on_progress_still_advances_past_a_failed_question():
    runner = _make_runner(_FlakyPipeline(fail_at={1}))
    ds = make_stub_dataset(n=3)

    calls: list[tuple[int, int]] = []
    runner.run(_base_config(), ds, on_progress=lambda processed, total: calls.append((processed, total)))

    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_error_is_serialized_and_round_trips_through_to_dict():
    runner = _make_runner(_FlakyPipeline(fail_at={0}))
    ds = make_stub_dataset(n=2)

    result = runner.run(_base_config(), ds)
    payload = result.to_dict()

    assert payload["n_errors"] == 1
    assert payload["question_results"][0]["error"] is not None
    assert payload["question_results"][1]["error"] is None


def test_failed_question_is_excluded_from_aggregate_metrics_not_counted_as_zero():
    """An errored question has no metrics at all (not zero-valued ones) —
    it must not drag down aggregate_metrics as if it scored 0."""
    runner = _make_runner(_FlakyPipeline(fail_at={0}))
    ds = make_stub_dataset(n=2)

    class _Evaluator:
        def evaluate(self, question, answer: Answer):
            return {"answer_similarity": 1.0}

    result = runner.run(_base_config(), ds, evaluator=_Evaluator())

    # Only the one successful question contributed a score — if the failed
    # question were silently counted as 0.0, the mean would be 0.5, not 1.0.
    assert result.aggregate_metrics["answer_similarity"] == 1.0


# ── QuestionResult.answerability persistence (Eval Measurement
# Trustworthiness — found live: GET /experiments/{run_id} could only guess
# "answerable vs not" from metric-key presence, unable to distinguish
# "uncovered" from "out_of_scope" for display. Persisting the real class at
# eval time, when an evaluator exposes it, closes that gap.) ────────────────

def test_answerability_is_persisted_when_evaluator_exposes_it():
    runner = _make_runner(_FlakyPipeline(fail_at=set()))
    ds = make_stub_dataset(n=1)

    class _Evaluator:
        def evaluate(self, question, answer: Answer):
            return {"correct_refusal": 1.0}

        def resolve_answerability(self, question) -> str:
            return "out_of_scope"

    result = runner.run(_base_config(), ds, evaluator=_Evaluator())

    assert result.question_results[0].answerability == "out_of_scope"
    assert result.to_dict()["question_results"][0]["answerability"] == "out_of_scope"


def test_answerability_stays_none_when_evaluator_does_not_expose_it():
    """An evaluator without resolve_answerability (e.g. eval/gate.py's own
    generic usage) must not break — the field just stays unset, same as
    before this existed."""
    runner = _make_runner(_FlakyPipeline(fail_at=set()))
    ds = make_stub_dataset(n=1)

    class _Evaluator:
        def evaluate(self, question, answer: Answer):
            return {"answer_similarity": 1.0}

    result = runner.run(_base_config(), ds, evaluator=_Evaluator())

    assert result.question_results[0].answerability is None
    assert result.to_dict()["question_results"][0]["answerability"] is None


def test_answerability_stays_none_when_no_evaluator_at_all():
    runner = _make_runner(_FlakyPipeline(fail_at=set()))
    ds = make_stub_dataset(n=1)

    result = runner.run(_base_config(), ds)

    assert result.question_results[0].answerability is None
