import pytest

from core.experiment.compare import compare
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult


def _cfg(top_k: int = 5, name: str = "test") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        top_k=top_k,
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="stub"),
    )


def _result(cfg: ExperimentConfig, metrics: dict) -> ExperimentResult:
    return ExperimentResult(config=cfg, aggregate_metrics=metrics)


def test_compare_identical_configs():
    a = _result(_cfg(top_k=5), {"faithfulness": 0.8})
    b = _result(_cfg(top_k=5), {"faithfulness": 0.85})
    report = compare(a, b)
    assert report.config_diff == {}


def test_compare_detects_config_diff():
    a = _result(_cfg(top_k=5), {})
    b = _result(_cfg(top_k=10), {})
    report = compare(a, b)
    assert "top_k" in report.config_diff
    assert report.config_diff["top_k"]["before"] == 5


def test_compare_metric_delta():
    a = _result(_cfg(), {"faithfulness": 0.80})
    b = _result(_cfg(), {"faithfulness": 0.90})
    report = compare(a, b)
    delta = next(d for d in report.metric_deltas if d.metric == "faithfulness")
    assert abs(delta.delta - 0.10) < 1e-9
    assert delta.delta_pct == pytest.approx(12.5, abs=0.1)


def test_compare_new_metric_in_after():
    a = _result(_cfg(), {"faithfulness": 0.8})
    b = _result(_cfg(), {"faithfulness": 0.85, "recall": 0.7})
    report = compare(a, b)
    metrics = {d.metric for d in report.metric_deltas}
    assert "recall" in metrics


def test_summary_contains_key_info():
    a = _result(_cfg(top_k=5), {"faithfulness": 0.8})
    b = _result(_cfg(top_k=10), {"faithfulness": 0.9})
    report = compare(a, b)
    summary = report.summary()
    assert "top_k" in summary
    assert "faithfulness" in summary
