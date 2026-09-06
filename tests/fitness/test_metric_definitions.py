"""Every number the platform reports declares what it computes.

A metric reaches the reader carrying a name and nothing else, so a
definition that drifts from its name is invisible: the name is all there is
on the screen, and the name still says what it always said. The declarations
in `core/eval/metric_definitions.py` are what make the drift askable, and
this is what keeps them from falling behind the evaluator.

Derived, never listed. A list typed out here would be correct on the day it
was typed and silently wrong afterwards, which is the same failure one
directory over: the interface's localisation test carried a hand-written
list of detectors and covered nothing added after it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.eval.metric_definitions import DEFINITIONS, definition_of

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "services" / "api_gateway" / "routers" / "experiments.py"

# `metrics["name"] = ...` — the evaluator's only way of putting a number on a
# question, so reading the assignments is reading the full set.
_WRITTEN = re.compile(r'metrics\["([a-z_0-9]+)"\]\s*=')


def _metrics_the_evaluator_writes() -> set[str]:
    return set(_WRITTEN.findall(EVALUATOR.read_text(encoding="utf-8")))


@pytest.mark.fitness
def test_the_evaluator_writes_metrics_this_can_find() -> None:
    """The premise. A pattern that matched nothing would make every
    assertion below pass by describing an empty set."""
    written = _metrics_the_evaluator_writes()
    assert len(written) >= 8, f"only found {sorted(written)} in {EVALUATOR.name}"


@pytest.mark.fitness
def test_every_metric_the_evaluator_writes_is_declared() -> None:
    undeclared = sorted(
        name for name in _metrics_the_evaluator_writes() if definition_of(name) is None
    )
    assert undeclared == [], (
        f"metrics written with no declaration of what they compute: {undeclared}. "
        "Declare each in core/eval/metric_definitions.py, or the name is all a reader has."
    )


@pytest.mark.fitness
def test_no_declaration_describes_a_metric_nobody_writes() -> None:
    """The other direction. A declaration outliving its metric describes a
    number nobody will ever see, and reads as coverage all the same."""
    written = _metrics_the_evaluator_writes()
    orphaned = sorted(d.name for d in DEFINITIONS if d.name not in written)
    assert orphaned == [], (
        f"declared and written nowhere: {orphaned}. Either the evaluator stopped writing "
        "them, or they were declared under a name it never used."
    )


@pytest.mark.fitness
def test_each_declaration_says_what_it_computes_and_over_which_questions() -> None:
    """A declaration that says nothing costs a reader the same as none.

    The population is checked as hard as the operation: a mean over the
    answerable questions and a mean over all of them are different numbers
    with the same name, which is the drift this exists to make visible.
    """
    thin = [
        d.name for d in DEFINITIONS
        if len(d.says.split()) < 5 or len(d.over.split()) < 2 or "question" not in d.over
    ]
    assert thin == [], f"declarations too thin to check a name against: {sorted(thin)}"
