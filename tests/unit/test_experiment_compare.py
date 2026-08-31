import pytest

from core.eval.regression import PairedDiffReport, paired_diff
from core.experiment.compare import check_comparability, compare
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, QuestionResult


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


# ── A metric measured in only one of the two runs ─────────────────────────────

def test_metric_absent_from_one_run_has_no_value_and_no_delta():
    # Filling the gap with 0.0 fabricated a verdict: the UI reads a move away
    # from zero as an improvement, so a metric that was never measured in the
    # "before" run rendered as a green rise from 0.000.
    a = _result(_cfg(), {"faithfulness": 0.8})
    b = _result(_cfg(), {"faithfulness": 0.85, "recall": 0.7})
    report = compare(a, b)
    recall = next(d for d in report.metric_deltas if d.metric == "recall")
    assert recall.before is None
    assert recall.after == 0.7
    assert recall.delta is None
    assert recall.delta_pct is None


def test_summary_says_a_metric_was_not_measured_rather_than_printing_a_number():
    a = _result(_cfg(), {"faithfulness": 0.8})
    b = _result(_cfg(), {"faithfulness": 0.85, "recall": 0.7})
    summary = compare(a, b).summary()
    assert "recall" in summary
    assert "0.0000" not in summary


# ── check_comparability ───────────────────────────────────────────────────────

def _ids(warnings) -> set[str]:
    """The stable identifiers, never the rendered sentences (CONTRIBUTING)."""
    return {w.id for w in warnings}


def _sev(warnings, warning_id: str) -> str:
    return next(w.severity for w in warnings if w.id == warning_id)


def _qr(question_id: str, **kw) -> QuestionResult:
    return QuestionResult(
        question_id=question_id, question=f"question {question_id}",
        reference_answer="ref", generated_answer="gen", **kw,
    )


def _run(cfg: ExperimentConfig | None = None, *, ids=("q1", "q2"), **kw) -> ExperimentResult:
    return ExperimentResult(
        config=cfg or _cfg(),
        question_results=[_qr(i) for i in ids],
        **kw,
    )


def _paired(before: ExperimentResult, after: ExperimentResult) -> PairedDiffReport:
    """The real pairing, so the tests exercise what the endpoint passes in."""
    def payload(run):
        return [{"question_id": qr.question_id, "metrics": {}, "funnel": {"layer": "ok"}}
                for qr in run.question_results]
    return paired_diff(payload(before), payload(after))


def test_two_runs_of_the_same_shape_raise_nothing():
    a, b = _run(), _run()
    assert check_comparability(a, b, _paired(a, b)) == []


def test_different_datasets_is_an_error():
    a = _run(_cfg(name="a"))
    b = _run(_cfg(name="b"))
    b.config.dataset_name = "other.jsonl"
    warnings = check_comparability(a, b, _paired(a, b))
    assert "different_dataset" in _ids(warnings)
    assert _sev(warnings, "different_dataset") == "error"


def test_no_shared_question_is_an_error_of_its_own():
    a = _run(ids=("a1", "a2"))
    b = _run(ids=("b1",))
    warnings = check_comparability(a, b, _paired(a, b))
    assert "no_overlap" in _ids(warnings)
    assert _sev(warnings, "no_overlap") == "error"
    # An empty overlap is a different statement from a partial one.
    assert "partial_overlap" not in _ids(warnings)


def test_same_dataset_name_with_a_changed_question_set_warns():
    # The case nothing surfaced before: the dataset was edited in place, so
    # the config diff is empty while the question sets have drifted apart.
    a = _run(ids=("q1", "q2", "q3"))
    b = _run(ids=("q1", "q2", "q4"))
    warnings = check_comparability(a, b, _paired(a, b))
    assert "partial_overlap" in _ids(warnings)
    assert _sev(warnings, "partial_overlap") == "warn"
    assert "different_dataset" not in _ids(warnings)


def test_a_run_with_no_questions_says_so_instead_of_blaming_the_overlap():
    a = _run(ids=())
    b = _run()
    assert "empty_run" in _ids(check_comparability(a, b, _paired(a, b)))


def test_a_changed_prompt_is_reported_although_the_config_diff_is_empty():
    a = _run(prompt_id="p1", prompt_version=1)
    b = _run(prompt_id="p1", prompt_version=2)
    assert compare(a, b).config_diff == {}
    assert "different_prompt" in _ids(check_comparability(a, b, _paired(a, b)))


def test_a_prompt_recorded_on_one_side_only_is_not_called_a_difference():
    # Found by walking every pair of a real store: 80% of the prompt findings
    # came from one run holding no record at all, where "they used different
    # prompts" is a claim the stored data does not support.
    a = _run(prompt_id="", prompt_version=0)
    b = _run(prompt_id="p1", prompt_version=2)
    warnings = check_comparability(a, b, _paired(a, b))
    assert "different_prompt" not in _ids(warnings)
    assert "prompt_unrecorded" in _ids(warnings)
    assert next(w for w in warnings if w.id == "prompt_unrecorded").params["side"] == "A"


def test_a_model_recorded_on_one_side_only_is_not_called_a_difference():
    a = _run(generator_model="qwen3:8b")
    b = _run(generator_model="")
    warnings = check_comparability(a, b, _paired(a, b))
    assert "different_generator_model" not in _ids(warnings)
    assert "generator_model_unrecorded" in _ids(warnings)
    assert next(w for w in warnings if w.id == "generator_model_unrecorded").params["side"] == "B"


def test_neither_side_recording_a_prompt_says_nothing_at_all():
    a, b = _run(), _run()
    warnings = _ids(check_comparability(a, b, _paired(a, b)))
    assert "prompt_unrecorded" not in warnings
    assert "generator_model_unrecorded" not in warnings


def test_a_changed_generator_model_is_reported_although_the_config_diff_is_empty():
    a = _run(generator_model="qwen2.5:7b")
    b = _run(generator_model="llama3.1:8b")
    assert compare(a, b).config_diff == {}
    assert "different_generator_model" in _ids(check_comparability(a, b, _paired(a, b)))


def test_a_different_realm_is_reported_although_it_is_absent_from_the_config():
    a = _run(realm_id="demo")
    b = _run(realm_id="other-realm")
    assert compare(a, b).config_diff == {}
    assert "different_realm" in _ids(check_comparability(a, b, _paired(a, b)))


def test_questions_that_failed_to_be_answered_are_reported():
    a = _run()
    b = ExperimentResult(
        config=_cfg(),
        question_results=[_qr("q1"), _qr("q2", error="connection refused")],
    )
    assert "errored_questions" in _ids(check_comparability(a, b, _paired(a, b)))


def test_a_stopped_run_is_reported():
    a = _run()
    b = _run(stopped=True)
    assert "partial_run" in _ids(check_comparability(a, b, _paired(a, b)))


def test_unverified_coverage_is_reported_and_an_empty_record_stays_silent():
    a = _run()
    b = _run(coverage_check={"checked": False, "reason": "index unreachable"})
    assert "coverage_unverified" in _ids(check_comparability(a, b, _paired(a, b)))
    quiet = _run(coverage_check={})
    assert "coverage_unverified" not in _ids(check_comparability(a, quiet, _paired(a, quiet)))


def test_a_metric_measured_in_only_one_run_is_reported():
    a = ExperimentResult(config=_cfg(), question_results=[_qr("q1")],
                         aggregate_metrics={"faithfulness": 0.8})
    b = ExperimentResult(config=_cfg(), question_results=[_qr("q1")],
                         aggregate_metrics={"faithfulness": 0.8, "recall": 0.7})
    assert "metric_missing_in_one_run" in _ids(check_comparability(a, b, _paired(a, b)))


def test_a_run_compared_against_itself_is_information_and_not_a_problem():
    a = _run(run_id="same")
    b = _run(run_id="same")
    warnings = check_comparability(a, b, _paired(a, b))
    assert _sev(warnings, "same_run") == "info"


def test_errors_are_ordered_before_warnings():
    a = _run(ids=("a1",), realm_id="demo")
    b = _run(ids=("b1",), realm_id="other-realm")
    severities = [w.severity for w in check_comparability(a, b, _paired(a, b))]
    assert severities == sorted(severities, key=["error", "warn", "info"].index)


def test_without_a_pairing_no_rule_about_the_overlap_is_invented():
    # The pre-flight path may hold metadata alone. Staying silent about the
    # overlap is the honest answer there.
    a = _run(ids=("a1",))
    b = _run(ids=("b1",))
    warnings = check_comparability(a, b)
    assert "no_overlap" not in _ids(warnings)
    assert "partial_overlap" not in _ids(warnings)


def test_the_warnings_lead_the_exported_summary():
    a = _run(realm_id="demo")
    b = _run(realm_id="other-realm")
    report = compare(a, b)
    report.compatibility = check_comparability(a, b, _paired(a, b))
    summary = report.summary()
    assert summary.index("Comparability") < summary.index("Config")

