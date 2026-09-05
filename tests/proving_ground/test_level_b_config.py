"""Paired baits for the failures a run's own configuration stages.

Five entries, five pairs. Each runs the healthy configuration and the distorted
one against the same loaded index, and asserts on both halves.

Two of the five expect silence. That is not a weaker claim: those entries are
recorded as undetectable, and a bait showing the defect present while every
signal stays quiet is what makes "undetectable" an observation instead of an
admission. The day a signal is written for them, these two fail, which is
exactly when somebody should look.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.proving_ground.conftest import (
    detector_signals,
    recall,
    recall_before_rerank,
    record,
    retrieval_only,
    run_on,
)
from tools.config_distort import distort
from tools.seed_proving_ground import control_config

pytestmark = pytest.mark.proving_ground

CORPUS = "base-ru"
LANGUAGE = "ru_be"


@pytest.fixture(scope="module")
def control(embedder: Any) -> dict[str, Any]:
    """The healthy half, run once for the module.

    Every distorted half is measured against this one, so running it per test
    would pay for the same fifteen questions five times over and, worse, would
    let two tests disagree about what healthy means.
    """
    return run_on(embedder, retrieval_only(control_config(CORPUS)), CORPUS, LANGUAGE)


def _distorted(embedder: Any, name: str) -> dict[str, Any]:
    config = distort(control_config(CORPUS), name, corpus_language=LANGUAGE)
    return run_on(embedder, retrieval_only(config), CORPUS, LANGUAGE)


def test_the_control_is_healthy(control: dict[str, Any]) -> None:
    """The premise of every pair below. A control that is already broken makes
    each of them compare two broken halves and report a silent signal as
    evidence of nothing."""
    assert recall_before_rerank(control) == 1.0, (
        f"retrieval itself misses something on the healthy half: {recall_before_rerank(control)}"
    )
    assert detector_signals(control) == set(), (
        f"the healthy half is not silent: {sorted(detector_signals(control))}"
    )
    # Retrieval finds every answer; the reranker then drops one of the thirteen.
    # Asserted on the retrieval figure and not on the final one, because the
    # final one is a statement about the reranker. It was pinned at exactly one
    # on a corpus of fifteen general documents, where nothing competed; twenty
    # five instrument cards later the reranker prefers a card's reporting
    # section to the general one for a question about report deadlines, which is
    # the competition the cards were added to create.
    assert recall(control) >= 0.9, f"the reranker now loses more than one answer: {recall(control)}"


def test_F18_a_selection_of_one_loses_what_ranked_second(
    embedder: Any, control: dict[str, Any]
) -> None:
    """Narrowing the selection to a single fragment.

    Read on retrieval and not on a detector: the entry's named signal is a root
    cause, which is computed with the index behind it and not from the stored
    run. What the pair shows here is the loss itself, measured.
    """
    broken = _distorted(embedder, "narrow_the_selection")
    if recall(control) <= recall(broken):
        pytest.skip(
            f"NOT STAGED: the selection width changes nothing. Measured across widths of one, "
            f"three, five, eight and ten, recall was {recall(control):.3f} at every one of them. "
            "The ranking is in two modes and nothing sits between them: whenever the reranker "
            "keeps the right answer it puts it first, and the one answer it loses is below ten. "
            "A cut-off has nothing to cut. This did stage once, on a corpus of twenty documents, "
            "on the strength of a single question whose answer sat at rank two; twenty-five "
            "instrument cards moved that answer out of the top ten entirely. Staging it needs "
            "questions whose answers legitimately sit at ranks two and three, which is a "
            "question set and not a corpus."
        )
    # The other branch, which the first version of this test left empty: a pair
    # that stages is a pair that records what it saw, and a test that passes in
    # silence looks the same as one that never ran.
    record("F18", "narrowing the selection to one fragment loses what ranked below it",
           recall_control=recall(control), recall_broken=recall(broken))


def test_one_half_deciding_is_not_stageable_by_a_setting(
    embedder: Any, control: dict[str, Any]
) -> None:
    """Not staged, which is a third state and not a signal that failed.

    The signal reads where each chunk came from, so it needs the two halves to
    disagree about what to return. On this corpus they agree almost entirely:
    measured on the healthy half, sixty of seventy-five retrieved sources carry
    both a dense and a lexical score. Weighting the merge onto one half changes
    how the agreed set is ordered and not which chunks it contains, so nothing
    for the signal to see is created.

    This passed once, before the two halves were able to recognise a chunk as
    one chunk at all. It was reading a merge of two disjoint lists, where every
    chunk had exactly one provenance by construction, so it was evidence about
    a defect and not about the distortion under test.

    Staging it needs a corpus large enough that the candidate window does not
    cover it, and questions whose phrasing one half misses. Recorded here as an
    open gap, so it does not sit as a red test somebody would learn to ignore.
    """
    broken = _distorted(embedder, "let_one_half_decide")
    merged = [s for q in broken["question_results"] for s in (q.get("pre_rerank_source_refs") or [])]
    dense_only = [s for s in merged
                  if s.get("dense_score", 0) > 0 and not s.get("sparse_score", 0)]
    share = len(dense_only) / len(merged)
    assert share < 0.8, (
        "one half now supplies the merged list, so this failure is stageable here after all"
    )
    pytest.skip(
        f"NOT STAGED BY A SETTING: the signal reads which half supplied each chunk, and "
        f"putting the whole weight on one half moves the order of the merged list without "
        f"moving its membership. Measured on the merged list before the reranker: "
        f"{len(dense_only)} of {len(merged)} chunks came from the semantic half alone, "
        f"{share:.0%} against the signal's threshold of 80 per cent. Both halves draw on the "
        "same corpus, so no weight shuts one out. It is staged by a load instead, and proved "
        "there: see test_level_b_ingest.py, where a lexical index that was never built leaves "
        "one half supplying every chunk of every context. The first reason recorded here "
        "blamed the size of the corpus, which measuring on one of twice the size disproved."
    )


def test_F22_pinning_the_fusion_constant_changes_the_order_and_nothing_speaks(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The rank-fusion constant at one.

    The reverse bait: the defect is staged, the ranking demonstrably moves, and
    no signal says a word. The entry records that no check reads the run
    history to ask whether the constant was ever varied, and this is that
    record made observable.
    """
    broken = _distorted(embedder, "pin_the_fusion_constant")
    assert detector_signals(broken) == set(), (
        f"a signal now speaks about this, and the catalogue says none does: "
        f"{sorted(detector_signals(broken))}"
    )

    def merged(run: dict[str, Any], key: str) -> list[list[str]]:
        return [[s["chunk_id"] for s in (q.get(key) or [])] for q in run["question_results"]]

    # Read before the reranker, and this is the whole of why the pair reads
    # anything. The merge is what the constant governs; the reranker then
    # reorders fifty candidates and hands back five, and by then the constant
    # has been absorbed. Measured: the merged list differs on fourteen of the
    # fifteen questions, and the five that reach the answer differ on none.
    # A first version compared the five and concluded the corpus was too small
    # to stage this, which was wrong: the instrument was reading past the step
    # under test.
    before_control, before_broken = merged(control, "pre_rerank_source_refs"), merged(broken, "pre_rerank_source_refs")
    assert any(before_control), "the run recorded no pre-rerank list, so there is nothing to compare"
    moved = sum(1 for a, b in zip(before_control, before_broken, strict=True) if a != b)
    assert moved > len(before_control) // 2, (
        f"the constant moved the merge on {moved} of {len(before_control)} questions, so little "
        "that nothing was staged"
    )
    record("F22", "the rank-fusion constant reorders the merge, and no signal speaks",
           questions_whose_merge_moved=moved, questions=len(before_control),
           final_context_moved=sum(1 for a, b in zip(merged(control, "source_refs"),
                                                     merged(broken, "source_refs"), strict=True) if a != b),
           signals=sorted(detector_signals(broken)))


def test_F25_a_window_equal_to_the_selection_leaves_nothing_to_rescue(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The candidate window closed down to the selection.

    The other reverse bait. The entry records that no check compares the window
    against the selection, and what the run stores shows the same thing from
    the other side: the wider window is simply absent.
    """
    broken = _distorted(embedder, "close_the_candidate_window")
    widened = [q for q in control["question_results"] if q.get("candidate_source_refs")]
    assert widened, "the healthy half recorded no candidate window, so there is nothing to close"
    closed = [q for q in broken["question_results"] if q.get("candidate_source_refs")]
    assert not closed, "the window is still wider than the selection"
    assert detector_signals(broken) == set(), (
        f"a signal now speaks about this, and the catalogue says none does: "
        f"{sorted(detector_signals(broken))}"
    )
    record("F25", "the candidate window closed to the selection, and no signal speaks",
           questions_with_a_window_healthy=len(widened), questions_with_a_window_broken=len(closed),
           signals=sorted(detector_signals(broken)))
