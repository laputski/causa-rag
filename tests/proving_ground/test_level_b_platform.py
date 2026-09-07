"""Failures of the platform itself, staged on a run the platform produced.

Four entries name the platform as their instrument, which means the failure
is in this code and not in anybody's documents or settings. A pair for one of
those cannot break a corpus; it has to break the platform. What it breaks
here is the run the platform just made: the control is that run untouched and
the broken half is the same run with the defect applied to it exactly as the
defect would apply.

That is worth distinguishing from a fixture. A fixture is a payload written
by hand to look like a run; these are runs, made against the loaded indexes
with the real embedding model, and the control half is asserted silent before
anything is done to it. What the pair proves is that the detector reads what
the platform actually produces, which is the half a fixture cannot show.

One entry could not be staged at all, and its measurement is recorded here
too, because a blocker somebody has measured is worth more than a row that
merely says "not staged".
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.proving_ground.conftest import LANGUAGE, record, retrieval_only, run_on

pytestmark = pytest.mark.proving_ground

CORPUS = "base-ru"


def _run(embedder: Any, name: str, **overrides: Any) -> dict[str, Any]:
    from core.experiment.config import ExperimentConfig
    from tools.seed_proving_ground import control_config

    base = control_config(CORPUS).model_dump(exclude={"config_hash"})
    base["name"] = f"proving-ground-{name}"
    base["chunking_strategy"] = {"kind": "chunker", "component_id": "structure_aware",
                                 "params": {}}
    base.update(overrides)
    return run_on(embedder, retrieval_only(ExperimentConfig(**base)), CORPUS, LANGUAGE)


def _signals(run: dict[str, Any]) -> set[str]:
    from core.eval.detectors import run_detectors
    return {f"detector:{item.id}" for item in run_detectors(run)}


@pytest.fixture(scope="module")
def honest(embedder: Any) -> dict[str, Any]:
    """One real run, and the premise of every pair below."""
    return _run(embedder, "platform-control")


def test_the_control_run_is_reported_as_healthy(honest: dict[str, Any]) -> None:
    """Every pair below is this run with one thing done to it, so a control
    already carrying a finding would make each of them prove two things."""
    assert honest["question_results"], "the control run answered no questions"
    assert _signals(honest) == set(), f"the control run is not clean: {sorted(_signals(honest))}"


def test_F34_a_question_lost_on_the_way_out_moves_the_run_s_numbers(
    honest: dict[str, Any],
) -> None:
    """The read path losing a question, done to a run that has one to lose.

    This is the defect that was live in this repository days ago and is the
    entry's own description: the aggregate is a mean of the values its
    questions carry, so a question dropped between writing and reading moves
    the number on screen and leaves everything else looking correct.
    """
    lost = copy.deepcopy(honest)
    # A question that carries metrics. The last one of this run is out of
    # scope and carries none, so dropping it moves no aggregate at all and
    # the pair would have proved the detector silent on a real loss.
    carrying = next(
        i for i, q in enumerate(lost["question_results"]) if (q.get("metrics") or {})
    )
    dropped = lost["question_results"].pop(carrying)
    assert dropped.get("metrics"), "the question dropped carried no metrics to lose"

    spoke = _signals(lost)
    assert "detector:aggregate_disagrees" in spoke, (
        f"a question vanished between the run and the read and the numbers still agreed: "
        f"{sorted(spoke)}"
    )
    record("F34", "a question lost on the way out of a real run, and its numbers moved",
           questions_before=len(honest["question_results"]),
           questions_after=len(lost["question_results"]),
           signals=sorted(spoke), signals_on_the_control=sorted(_signals(honest)))


def test_F33_a_metric_arriving_with_no_declaration_is_reported(
    honest: dict[str, Any],
) -> None:
    """A number reaching the reader with a name and nothing else.

    Added to a real run under a name nobody declared, and carried by its
    questions as well, so the only thing wrong with it is that whether the
    name still matches what it computes cannot be asked.
    """
    undeclared = copy.deepcopy(honest)
    undeclared["aggregate_metrics"]["answer_precision"] = 0.9
    for question in undeclared["question_results"]:
        (question.setdefault("metrics", {}))["answer_precision"] = 0.9

    spoke = _signals(undeclared)
    assert "detector:undeclared_metric" in spoke, (
        f"a metric arrived with no declaration and nothing said so: {sorted(spoke)}"
    )
    assert "detector:aggregate_disagrees" not in spoke, (
        "the added metric also disagrees with its questions, so this pair proves two things"
    )
    record("F33", "a metric added to a real run with no declaration of what it computes",
           metrics=sorted(undeclared["aggregate_metrics"]),
           signals=sorted(spoke), signals_on_the_control=sorted(_signals(honest)))


def test_F35_a_metric_kept_past_its_own_precondition_is_reported(
    honest: dict[str, Any],
) -> None:
    """Retrieval metrics kept on questions the corpus does not cover.

    The evaluator will not write them there, which is the guard the entry
    describes as enforced in one place by hand. Applying the failure means
    doing what a platform without that guard does: leaving the numbers on a
    question whose answerability says they have nothing to be about.
    """
    ungrounded = copy.deepcopy(honest)
    for question in ungrounded["question_results"]:
        question["answerability"] = "out_of_scope"

    spoke = _signals(ungrounded)
    assert "detector:metric_without_grounds" in spoke, (
        f"every retrieval metric is recorded against a question the corpus does not cover "
        f"and nothing said so: {sorted(spoke)}"
    )
    record("F35", "a real run's retrieval metrics kept on questions with nothing to retrieve",
           questions=len(ungrounded["question_results"]),
           signals=sorted(spoke), signals_on_the_control=sorted(_signals(honest)))


def test_F02_one_identifier_over_two_fragments_of_a_real_run_is_reported(
    honest: dict[str, Any],
) -> None:
    """A lost source path collapsing two derivations onto one value.

    Applied to fragments a run actually returned, so the two texts under the
    single identifier are two real fragments of this corpus and not two
    strings invented to differ.
    """
    collided = copy.deepcopy(honest)
    moved = 0
    for question in collided["question_results"]:
        refs = question.get("source_refs") or []
        if len(refs) < 2 or refs[0]["chunk_text"] == refs[1]["chunk_text"]:
            continue
        refs[1]["chunk_id"] = refs[0]["chunk_id"]
        moved += 1
    assert moved > 0, "no question returned two different fragments to collide"

    spoke = _signals(collided)
    assert "detector:chunk_id_collision" in spoke, (
        f"two fragments of this run share one identifier and nothing said so: {sorted(spoke)}"
    )
    record("F02", "two fragments of a real run under one identifier",
           questions_affected=moved,
           signals=sorted(spoke), signals_on_the_control=sorted(_signals(honest)))


def test_F27_a_stage_that_ran_and_left_no_duration_is_reported(
    embedder: Any, honest: dict[str, Any],
) -> None:
    """The reranker running and reporting nothing.

    The platform records `rerank_ms` now, so a run of its own cannot show
    this: what a system without that recording looks like is a trace that is
    present, says the reranker ran, and carries no duration for it. That is
    what is done here, to a real run that did rerank.
    """
    reranked = _run(embedder, "platform-reranked")
    traced = [q for q in reranked["question_results"] if (q.get("stage_trace") or {}).get("n_reranked")]
    assert traced, "the control configuration did not rerank, so this pair has nothing to silence"
    assert any((q["stage_trace"] or {}).get("rerank_ms") for q in traced), (
        "the reranker ran and the platform recorded no time for it, which is the failure "
        "itself and means the control half is already broken"
    )

    silent = copy.deepcopy(reranked)
    for question in silent["question_results"]:
        trace = question.get("stage_trace") or {}
        # Removed and not zeroed. A recorded zero is a stage measured and
        # found fast, which a merge of fifty candidates genuinely is; the
        # failure is a field that was never written.
        trace.pop("rerank_ms", None)

    spoke = _signals(silent)
    assert "detector:unmeasured_stage_cost" in spoke, (
        f"the reranker ran on every question and left no duration, and nothing said so: "
        f"{sorted(spoke)}"
    )
    assert "detector:unmeasured_stage_cost" not in _signals(reranked), (
        "the untouched run is already reported as leaving a stage unmeasured"
    )
    record("F27", "a real run whose reranker ran and reported no time",
           questions_with_a_trace=len(traced),
           signals=sorted(spoke), signals_on_the_control=sorted(_signals(reranked)))


def test_F17_a_frequent_word_document_damages_retrieval_and_the_named_signal_is_silent(
    embedder: Any,
) -> None:
    """The measured blocker, recorded because a blocker nobody measured is
    worth less than a row saying "not staged".

    A document repeating the corpus's commonest words is added, loaded, and
    queried through a hybrid retrieval. It costs retrieval a third of its
    recall, so the failure is unmistakably present. The signal the entry
    names reads which half of the merge a fragment came from, and this
    document is found by both halves: it is made of the corpus's own words,
    so the embedding places it near the same questions the lexical index
    does. Nothing about the merge is imbalanced, and the entry's signal is
    right to stay silent.

    Skipped, loudly, until the stuffed corpus is loaded, because the load
    costs a pass of the real model over two hundred and twenty fragments and
    belongs to whoever is staging this, and not to every run of the suite.
    """
    from tests.proving_ground.conftest import _index_exists

    if not _index_exists("base-ru-stuffed", "structure_aware", embedder.embedder_id):
        pytest.skip(
            "NOT RUN: no index for 'base-ru-stuffed'. Create it with the plan from "
            "`python3 -m tools.corpus_mutate corpus/proving-ground/base-ru "
            "--defect stuff_a_document_with_the_corpus_own_words --out <dir>/base-ru`, "
            "then load that directory as corpus base-ru-stuffed."
        )

    control = _run(embedder, "F17-control", pipeline_id="hybrid_rrf", reranker=None)
    stuffed = _run(embedder, "F17-stuffed", pipeline_id="hybrid_rrf", reranker=None,
                   corpus_id="base-ru-stuffed")

    def _recall(run: dict[str, Any]) -> float:
        return float(run["aggregate_metrics"].get("retrieval_recall_at_k", -1.0))

    def _from_one_half(run: dict[str, Any]) -> float:
        refs = [s for q in run["question_results"] for s in (q.get("source_refs") or [])]
        scored = [(s.get("dense_score") or 0.0, s.get("sparse_score") or 0.0) for s in refs]
        scored = [(d, s) for d, s in scored if d > 0 or s > 0]
        if not scored:
            return 0.0
        return sum(1 for d, s in scored if d == 0 and s > 0) / len(scored)

    assert _recall(stuffed) < _recall(control), (
        f"the added document cost retrieval nothing: {_recall(stuffed)} against "
        f"{_recall(control)}, so there is no failure here to be silent about"
    )
    assert "detector:bm25_dominance" not in _signals(stuffed), (
        "the entry's own signal fired, so this is no longer a blocker and the pair should "
        "assert the reproduction and stop recording the absence"
    )
    record("F17", "not staged: the document is found by both halves, so the merge is balanced "
                  "and the signal the entry names reads only which half a fragment came from",
           reproduced=False,
           recall_control=_recall(control), recall_stuffed=_recall(stuffed),
           from_the_lexical_half_alone_control=_from_one_half(control),
           from_the_lexical_half_alone_stuffed=_from_one_half(stuffed),
           signals_stuffed=sorted(_signals(stuffed)))
