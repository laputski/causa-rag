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
from pathlib import Path
from typing import Any

import pytest

from tests.proving_ground.conftest import LANGUAGE, record, retrieval_only, run_on

ROOT = Path(__file__).resolve().parents[2]

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


def test_F17_a_document_of_frequent_words_is_raised_by_one_half(
    embedder: Any,
) -> None:
    """A document made of the corpus's own commonest words, and which half
    raises it.

    Measured per half and never on the share of the whole context, which is
    what the first attempt did and what left this entry blocked for so long.
    The share of a context that came from one half is a statistic about a
    merge; this failure is one document, and one document moves that share by
    a fragment. Asked the other way, the document enters the keyword half's
    window on six questions of nineteen and the semantic half's on one.

    So the signal the entry used to name is silent here, correctly: it reports
    a half that has stopped contributing, and nothing here has stopped. The
    entry no longer names it, and this is the pair that says why: a fixture
    that made every fragment come from the keyword half was proving a whole
    context ruled by one half, which two neighbouring entries are about, under
    this entry's name.

    Skipped, loudly, until the stuffed corpus is loaded, because the load
    costs a pass of the real model over two hundred and twenty fragments and
    belongs to whoever is staging this, and not to every run of the suite.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from tests.proving_ground.conftest import _index_exists
    from tools.corpus_mutate import _commonest_words, read_corpus

    corpus_id = "base-ru-stuffed"
    if not _index_exists(corpus_id, "structure_aware", embedder.embedder_id):
        pytest.skip(
            f"NOT RUN: no index for {corpus_id!r}. Create it with the plan from "
            "`python3 -m tools.corpus_mutate corpus/proving-ground/base-ru "
            "--defect stuff_a_document_with_the_corpus_own_words --out <dir>/base-ru`, "
            f"then load that directory as corpus {corpus_id}."
        )

    control = _run(embedder, "F17-control", pipeline_id="hybrid_rrf", reranker=None)
    stuffed = _run(embedder, "F17-stuffed", pipeline_id="hybrid_rrf", reranker=None,
                   corpus_id=corpus_id)

    def _recall(run: dict[str, Any]) -> float:
        return float(run["aggregate_metrics"].get("retrieval_recall_at_k", -1.0))

    assert _recall(stuffed) < _recall(control), (
        f"the added document cost retrieval nothing: {_recall(stuffed)} against "
        f"{_recall(control)}, so there is no failure here to be raised by anything"
    )

    dense = QdrantRetriever(host="localhost", port=6333, strategy_id="structure_aware",
                            embedder_id=embedder.embedder_id, corpus_id=corpus_id,
                            realm_id="proving-ground")
    sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id="structure_aware",
                                 corpus_id=corpus_id, realm_id="proving-ground",
                                 language=LANGUAGE)
    healthy = read_corpus(ROOT / "corpus" / "proving-ground" / "base-ru")
    filler = " ".join(_commonest_words(healthy, 12))
    questions = [q["question"] for q in stuffed["question_results"] if q.get("question")]
    assert len(questions) > 10, "too few questions to measure a half by"

    seen = {"the keyword half only": 0, "the semantic half only": 0, "both": 0, "neither": 0}
    for question in questions:
        vector = embedder.embed([question])[0]
        by_meaning = any(filler in h.chunk.text
                         for h in dense.retrieve(query=question, k=10, query_vector=vector))
        by_words = any(filler in h.chunk.text
                       for h in sparse.retrieve(query=question, k=10))
        if by_words and not by_meaning:
            seen["the keyword half only"] += 1
        elif by_meaning and not by_words:
            seen["the semantic half only"] += 1
        elif by_meaning:
            seen["both"] += 1
        else:
            seen["neither"] += 1

    assert seen["the keyword half only"] > seen["the semantic half only"], (
        f"the semantic half raises this document as readily as the keyword half does: {seen}, "
        "so it is not the keyword search that raises it and this entry is not staged"
    )
    spoke = _signals(stuffed)
    assert "detector:bm25_dominance" not in spoke, (
        "the signal this entry used to name fired, so it does read this after all and the "
        "entry should name it again"
    )

    record("F17", "a document of the corpus's own frequent words enters the keyword half's "
                  "window and not the semantic half's, and nothing reads which half raised it",
           recall_control=_recall(control), recall_stuffed=_recall(stuffed),
           questions=len(questions), reached_the_window=seen,
           signals_stuffed=sorted(spoke),
           why_nothing_reads_it="the only judgement about the halves counts the share of a "
                                "whole context that came from one of them and reports a half "
                                "that has stopped contributing; one document moves that share "
                                "by a fragment",
           the_entry_was_corrected="it claimed a detector until this pair ran")
