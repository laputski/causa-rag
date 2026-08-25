from eval.gate import check_gate


def test_gate_passes_when_current_equals_baseline():
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.85},
        baseline_metrics={"faithfulness": 0.85},
    )
    assert passed
    assert violations == []


def test_gate_passes_when_current_above_baseline():
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.90},
        baseline_metrics={"faithfulness": 0.85},
    )
    assert passed


def test_gate_blocks_on_significant_drop():
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.70},
        baseline_metrics={"faithfulness": 0.85},
        thresholds={"faithfulness": 0.05},
    )
    assert not passed
    assert len(violations) == 1
    assert "faithfulness" in violations[0]


def test_gate_allows_small_drop_within_threshold():
    # 3% drop with 5% threshold → should pass
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.824},
        baseline_metrics={"faithfulness": 0.85},
        thresholds={"faithfulness": 0.05},
    )
    assert passed


def test_gate_multiple_metrics_one_fails():
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.9, "recall": 0.5},
        baseline_metrics={"faithfulness": 0.85, "recall": 0.8},
        thresholds={"faithfulness": 0.05, "recall": 0.05},
    )
    assert not passed
    assert any("recall" in v for v in violations)


def test_gate_ignores_missing_current_metric():
    # If current doesn't have a metric → skip it (not a violation)
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.9},
        baseline_metrics={"faithfulness": 0.85, "recall": 0.8},
    )
    assert passed


def test_gate_empty_baseline_passes():
    passed, violations = check_gate(
        current_metrics={"faithfulness": 0.5},
        baseline_metrics={},
    )
    assert passed
    assert violations == []
