"""The published Configuration Report says what its own measurements say.

The document under docs/reports/ is generated from eval/results/miracl/*.json
by `python3 -m eval.miracl.report`, and both are committed. Once committed
they are two copies of the same numbers, and a markdown table is the easy
one to adjust by hand: it is prose, it is where a rounding looks tidier, and
nothing about a changed digit looks wrong.

That is the failure this guards. A published number that no longer matches
the run behind it is worse than a missing one, because it carries the run's
authority. These checks read both files and compare.

They skip when the report has not been produced yet, so a fresh clone does
not fail on work that has not run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_RESULTS = _ROOT / "eval" / "results" / "miracl"
_DOCUMENT = _ROOT / "docs" / "reports" / "miracl-configuration-report.md"

_ROW = re.compile(
    r"^\|\s*(?P<lang>\w+)\s*\|\s*(?P<mode>\S+)\s*\|\s*(?P<k>\d+)\s*\|"
    r"\s*(?P<rerank>on|off)\s*\|\s*(?P<recall>[\d.]+|not measured)\s*\|"
    r"\s*(?P<precision>[\d.]+|not measured)\s*\|\s*(?P<map>[\d.]+|not measured)\s*\|"
    r"\s*(?P<n>\d+)\s*\|\s*(?P<failed>\d+)\s*\|\s*$"
)


def _rows_from_json() -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for path in sorted(_RESULTS.glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            out[(row["language"], row["mode"], row["top_k"], row["reranker"])] = row
    return out


def _rows_from_document() -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for line in _DOCUMENT.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            key = (m["lang"], m["mode"], int(m["k"]), m["rerank"] == "on")
            out[key] = m.groupdict()
    return out


pytestmark = pytest.mark.skipif(
    not _DOCUMENT.exists() or not any(_RESULTS.glob("*.json")),
    reason="the report has not been run yet",
)


def test_the_document_has_a_row_for_every_measurement():
    missing = sorted(set(_rows_from_json()) - set(_rows_from_document()))
    assert not missing, f"measured, yet absent from the document: {missing}"


def test_the_document_invents_no_rows():
    """The direction that matters more. A row with no measurement behind it
    is a number the run never produced."""
    extra = sorted(set(_rows_from_document()) - set(_rows_from_json()))
    assert not extra, f"published with nothing behind it: {extra}"


@pytest.mark.parametrize("metric,column", [
    ("retrieval_recall_at_k", "recall"),
    ("retrieval_precision_at_k", "precision"),
    ("retrieval_average_precision", "map"),
])
def test_every_published_number_matches_the_run(metric: str, column: str):
    measured = _rows_from_json()
    wrong = []
    for key, cell in _rows_from_document().items():
        value = measured[key]["metrics"].get(metric)
        printed = cell[column]
        if value is None:
            if printed != "not measured":
                wrong.append((key, "not measured", printed))
        elif printed == "not measured" or f"{value:.3f}" != printed:
            wrong.append((key, f"{value:.3f}", printed))
    assert not wrong, f"{metric} differs between the run and the document: {wrong}"


def test_question_and_failure_counts_match():
    measured = _rows_from_json()
    wrong = [
        (key, measured[key]["n_questions"], measured[key]["n_failed"], cell["n"], cell["failed"])
        for key, cell in _rows_from_document().items()
        if str(measured[key]["n_questions"]) != cell["n"]
        or str(measured[key]["n_failed"]) != cell["failed"]
    ]
    assert not wrong, f"counts differ between the run and the document: {wrong}"


def test_the_conditions_the_document_states_are_the_ones_the_grid_uses():
    """The prose above the table is generated too, and drifts the same way
    a number does once someone edits either side."""
    from eval.miracl import report

    text = _DOCUMENT.read_text(encoding="utf-8")
    assert f"fixed, {report.CHUNK_SIZE} characters" in text
    assert f"Candidate window: {report.FETCH_K}" in text
