"""Tests for ExperimentRunner — reproducibility + cache respect."""
from adapters.bge_m3 import BgeM3Embedder
from adapters.generator_stub import GeneratorStub
from adapters.qdrant import QdrantRetrieverStub
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, ExperimentRunner
from core.pipeline import NaivePipeline
from core.registry import ComponentRegistry
from eval.dataset import make_stub_dataset


def _make_registry() -> ComponentRegistry:
    reg = ComponentRegistry()
    emb = BgeM3Embedder()
    ret = QdrantRetrieverStub()
    gen = GeneratorStub()
    pipeline = NaivePipeline(retriever=ret, embedder=emb, generator=gen)
    reg.register("embedder", "bge_m3", emb)
    reg.register("retriever", "qdrant_dense_stub", ret)
    reg.register("generator", "stub", gen)
    reg.register("pipeline", "naive", pipeline)
    return reg


def _base_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test_run",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        retrievers=[ComponentRef(kind="retriever", component_id="qdrant_dense_stub")],
        generator=ComponentRef(kind="generator", component_id="stub"),
    )


def test_runner_returns_result():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    ds = make_stub_dataset(n=3)
    result = runner.run(_base_config(), ds)
    assert isinstance(result, ExperimentResult)
    assert len(result.question_results) == 3


def test_on_progress_called_once_per_question_with_running_total():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    calls: list[tuple[int, int]] = []
    runner.run(_base_config(), make_stub_dataset(n=3), on_progress=lambda processed, total: calls.append((processed, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


# Found live: a run picking a slow model (or hitting a stuck external RAG)
# had no way to be interrupted short of waiting out every remaining
# question. should_stop is checked once per question, same granularity/
# honest-degradation convention as on_progress.
def test_should_stop_halts_the_loop_and_marks_result_stopped():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    calls: list[tuple[int, int]] = []
    result = runner.run(
        _base_config(), make_stub_dataset(n=5),
        on_progress=lambda processed, total: calls.append((processed, total)),
        should_stop=lambda: len(calls) >= 2,
    )
    assert result.stopped is True
    assert len(result.question_results) == 2
    assert calls == [(1, 5), (2, 5)]


# Found live: the configuration panel showed the fixed adapter-kind literal
# ("ollama"/"stub") instead of the actual model — config.generator is
# decorative (see the design notes), so the real model has to be
# captured from Answer.metadata, mirroring the existing prompt_id capture.
def test_generator_model_captured_from_answer_metadata():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=2))
    assert result.generator_model == "stub"


def test_no_should_stop_runs_to_completion_unchanged():
    """Honest degradation: omitting should_stop must not change
    behavior at all — same as on_progress=None already guarantees."""
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=3))
    assert result.stopped is False
    assert len(result.question_results) == 3


def test_should_stop_checked_before_first_question_stops_immediately():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=3), should_stop=lambda: True)
    assert result.stopped is True
    assert result.question_results == []
    # Still a well-formed, "finished" result — not a raised exception — so
    # the caller can save/display it like any other completed run.
    assert result.finished_at != ""


def test_stage_trace_persisted_per_question_and_averaged():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=3))

    # Every in-process question carries the per-stage trace captured on its Answer.
    for qr in result.question_results:
        assert qr.stage_trace is not None
        assert "total_ms" in qr.stage_trace

    data = result.to_dict()
    assert data["question_results"][0]["stage_trace"] is not None
    # Dataset-averaged trace is present and covers the same stage keys.
    avg = data["avg_stage_trace"]
    assert avg is not None
    assert "total_ms" in avg


def test_avg_stage_trace_none_when_no_traces():
    # A run whose questions produced no stage_trace (e.g. an external RAG with
    # no ExternalTrace) must not fabricate a zero-latency average.
    cfg = _base_config()
    result = ExperimentResult(config=cfg)
    from core.experiment.runner import QuestionResult
    result.question_results = [
        QuestionResult(question_id="q1", question="q", reference_answer="", generated_answer="a"),
    ]
    assert result.avg_stage_trace() is None
    assert result.to_dict()["avg_stage_trace"] is None


def test_on_progress_optional_no_callback_no_error():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=2))
    assert len(result.question_results) == 2


def test_nfr_3_1_reproducibility():
    """Same config + same stub pipeline → same answer texts on two runs."""
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    ds = make_stub_dataset(n=3)
    cfg = _base_config()

    result_a = runner.run(cfg, ds)
    result_b = runner.run(cfg, ds)

    texts_a = [qr.generated_answer for qr in result_a.question_results]
    texts_b = [qr.generated_answer for qr in result_b.question_results]
    assert texts_a == texts_b, " same config must produce identical answers"


def test_nfr_3_2_cache_respect():
    """Second run of same questions → embed cache hit_ratio ≈ 1 (no recompute)."""
    emb = BgeM3Embedder()
    reg = ComponentRegistry()
    ret = QdrantRetrieverStub()
    gen = GeneratorStub()
    pipeline = NaivePipeline(retriever=ret, embedder=emb, generator=gen)
    reg.register("pipeline", "naive", pipeline)
    runner = ExperimentRunner(registry=reg)
    ds = make_stub_dataset(n=5)
    cfg = _base_config()

    runner.run(cfg, ds)   # first run — all misses
    runner.run(cfg, ds)   # second run — all hits

    # After two runs over same questions: hits ≥ misses
    assert emb.cache_hit_ratio >= 0.5, (
        f" expected hit_ratio ≥ 0.5, got {emb.cache_hit_ratio:.2f}"
    )


def test_result_to_dict():
    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=2))
    d = result.to_dict()
    assert "config_hash" in d
    assert "question_results" in d
    assert len(d["question_results"]) == 2


def test_computed_citations_carried_through_to_dict_and_back():
    """Real bug found live: core/pipeline.py already computes
    Answer.computed_citations (deterministic, from retrieved chunk
    metadata — not the LLM's own in-text citation, see core/citation.py)
    on every run, but neither QuestionResult nor ExperimentResult.to_dict()
    captured it, nor did the API's read-path (_parse_result in
    experiments.py) reconstruct it — the field existed on the Answer and
    was silently dropped on the way into storage."""
    from services.api_gateway.routers.experiments import _parse_result

    reg = _make_registry()
    runner = ExperimentRunner(registry=reg)
    result = runner.run(_base_config(), make_stub_dataset(n=1))

    qr = result.question_results[0]
    assert isinstance(qr.computed_citations, list)

    d = result.to_dict()
    assert "computed_citations" in d["question_results"][0]

    parsed = _parse_result(d)
    assert parsed is not None
    assert parsed.question_results[0].computed_citations == qr.computed_citations
