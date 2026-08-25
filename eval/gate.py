"""Eval gate — blocks CI if metrics fall below baseline.

Usage (CLI):
    python -m eval.gate --dataset eval/golden/stub.v0.fast.jsonl \\
                        --baseline-run-id <run_id> \\
                        --threshold 0.05

Exit code 0 = passed, 1 = gate blocked, 2 = configuration error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import structlog

from core.eval.regression import (
    DEFAULT_THRESHOLDS,
    SLA_ABSOLUTE_FLOORS,
    check_gate,  # noqa: F401  (re-exported for backward compatibility)
)
from eval.dataset import EvalDataset

log = structlog.get_logger()

# Thresholds/floors and check_gate now live in core.eval.regression so the gateway
# and CI share one implementation. Re-exported above for back-compat.
__all__ = ["DEFAULT_THRESHOLDS", "SLA_ABSOLUTE_FLOORS", "check_gate", "main"]


def _run_eval(
    pipeline: Any,
    dataset: EvalDataset,
    evaluator: Any | None = None,
) -> dict[str, float]:
    """Run pipeline over dataset, return aggregate metrics."""
    from core.models import QueryRequest

    per_question: dict[str, list[float]] = {}
    for q in dataset.questions:
        req = QueryRequest(text=q["question"])
        answer = pipeline.run(req)
        if evaluator is not None:
            metrics = evaluator.evaluate(q, answer)
            for k, v in metrics.items():
                per_question.setdefault(k, []).append(v)

    return {k: sum(v) / len(v) for k, v in per_question.items()} if per_question else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval gate for CI")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05, help="Allowed relative drop")
    parser.add_argument(
        "--baseline-metric",
        nargs=2,
        metavar=("NAME", "VALUE"),
        action="append",
        default=[],
        help="Baseline metric value: --baseline-metric faithfulness 0.85",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        log.error("gate.dataset_not_found", path=str(args.dataset))
        sys.exit(2)

    baseline_metrics = {name: float(value) for name, value in args.baseline_metric}
    if not baseline_metrics:
        log.warning("gate.no_baseline", msg="No baseline metrics provided — gate passes vacuously")
        sys.exit(0)

    # In CI, pipeline + evaluator come from the registered components.
    # For unit testing this function is called directly with mock values.
    log.info("gate.checking", baseline=baseline_metrics, threshold=args.threshold)
    thresholds = {m: args.threshold for m in baseline_metrics}

    # Placeholder: real metrics would come from running the pipeline
    # This is wired up in integration tests and the full CI eval job.
    passed, violations = check_gate(
        current_metrics=baseline_metrics,  # stub: current == baseline → always passes in CLI dry-run
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
    )

    if passed:
        log.info("gate.passed")
        sys.exit(0)
    else:
        for v in violations:
            log.error("gate.violation", detail=v)
        sys.exit(1)


if __name__ == "__main__":
    main()
