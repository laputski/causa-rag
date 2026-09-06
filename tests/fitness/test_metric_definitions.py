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


@pytest.mark.fitness
def test_every_metric_says_what_must_hold_before_it_means_anything() -> None:
    """An empty list of preconditions is a claim, and it has to be made.

    The same ambiguity the catalogue's coordinates carry: a field left at its
    default reads as "computable anywhere" and as "nobody thought about it",
    and only one of those is worth trusting. Every metric written today has
    grounds, so an empty one is an omission until someone argues otherwise
    here.
    """
    unconditioned = sorted(d.name for d in DEFINITIONS if not d.requires)
    assert unconditioned == [], (
        f"declared with no preconditions at all: {unconditioned}. If a metric really is "
        "computable wherever it is written, say so here and name it."
    )


@pytest.mark.fitness
def test_each_precondition_says_what_must_hold_in_words_a_reader_can_use() -> None:
    """The sentence is printed inside the finding, so it has to survive
    being read away from the code that produced it."""
    thin = sorted({
        precondition.says
        for d in DEFINITIONS for precondition in d.requires
        if len(precondition.says.split()) < 4
    })
    assert thin == [], f"preconditions too terse to read in a finding: {thin}"


@pytest.mark.fitness
def test_a_precondition_can_actually_fail() -> None:
    """A predicate that is true of everything guards nothing.

    Each is checked against a payload built to violate it, because a
    precondition nobody can breach passes every run and proves nothing,
    which is the shape of a guard that has quietly stopped working.
    """
    breaches = {
        "the run reached the generator": ({"config": {"retrieval_only": True}}, {}),
        "the question is one the corpus covers": ({}, {"answerability": "out_of_scope"}),
        "retrieval found at least one source the question needs": (
            {}, {"metrics": {"retrieval_recall_at_k": 0.0}}),
    }
    for d in DEFINITIONS:
        for precondition in d.requires:
            assert precondition.says in breaches, (
                f"{d.name} declares {precondition.says!r} and no breach of it is written here, "
                "so nothing shows it can fail"
            )
            run, question = breaches[precondition.says]
            assert precondition.holds(run, question) is False, (
                f"{precondition.says!r} holds even on a payload built to violate it"
            )


@pytest.mark.fitness
def test_every_segmentation_strategy_says_what_it_promises() -> None:
    """A strategy with no declared promise cannot be caught doing nothing.

    Declared here and not in a rule about names, because only the
    strategy knows what its name committed it to, and a rule written
    elsewhere would be somebody's reading of that name.
    """
    from core.chunking.post_conditions import PROMISES

    strategies = {
        path.stem for path in (ROOT / "core" / "chunking").glob("*.py")
        if path.stem not in ("__init__", "post_conditions")
    }
    silent = sorted(strategies - set(PROMISES))
    assert silent == [], (
        f"segmentation strategies promising nothing checkable: {silent}. Declare the promise "
        "in core/chunking/post_conditions.py, or say there why this one makes none."
    )


@pytest.mark.fitness
def test_every_promise_can_actually_be_broken() -> None:
    """A promise nothing can breach is a promise nobody is keeping.

    Each is checked against output built to break it, for the same reason
    the preconditions above are: a check that passes on everything is
    indistinguishable from a check that has quietly stopped running.
    """
    from core.chunking.post_conditions import PROMISES, unmet

    class _Chunk:
        def __init__(self, text: str, structural_path: str = "document/section[1 A]") -> None:
            self.text = text
            self.structural_path = structural_path

    breaches = {
        "structure_aware": ([_Chunk("prose", "root")] * 4, 0),
        "fixed": ([_Chunk("x" * 200)] * 4, 100),
        "sentence": ([_Chunk("a fragment cut off mid")] * 4, 0),
        "paragraph": ([_Chunk("a fragment cut off mid")] * 4, 0),
    }
    for strategy in PROMISES:
        assert strategy in breaches, (
            f"{strategy} declares a promise and no way of breaking it is written here"
        )
        chunks, size = breaches[strategy]
        assert unmet(strategy, chunks, size), (
            f"{strategy}'s promise holds even on output built to break it"
        )
