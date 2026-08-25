"""Experiment comparison: config diff + metric deltas between two runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.experiment.runner import ExperimentResult


@dataclass
class MetricDelta:
    metric: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def delta_pct(self) -> float:
        if self.before == 0:
            return float("inf") if self.after > 0 else 0.0
        return (self.after - self.before) / abs(self.before) * 100


@dataclass
class CompareReport:
    config_diff: dict[str, dict[str, Any]]
    metric_deltas: list[MetricDelta]

    def summary(self) -> str:
        lines: list[str] = []
        if self.config_diff:
            lines.append("Config changes:")
            for field, change in self.config_diff.items():
                lines.append(f"  {field}: {change['before']!r} → {change['after']!r}")
        else:
            lines.append("Config: identical")

        lines.append("\nMetric deltas (after - before):")
        for d in sorted(self.metric_deltas, key=lambda x: x.metric):
            sign = "+" if d.delta >= 0 else ""
            lines.append(f"  {d.metric}: {d.before:.4f} → {d.after:.4f}  ({sign}{d.delta:.4f})")
        return "\n".join(lines)


def compare(before: ExperimentResult, after: ExperimentResult) -> CompareReport:
    """Compare two experiment results: config diff + metric deltas."""
    config_diff = before.config.diff(after.config)

    all_metrics = set(before.aggregate_metrics) | set(after.aggregate_metrics)
    deltas = [
        MetricDelta(
            metric=m,
            before=before.aggregate_metrics.get(m, 0.0),
            after=after.aggregate_metrics.get(m, 0.0),
        )
        for m in all_metrics
    ]
    return CompareReport(config_diff=config_diff, metric_deltas=deltas)
