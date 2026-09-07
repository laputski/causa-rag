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


def test_F21_the_whole_weight_on_one_half_leaves_the_other_contributing_nothing(
    embedder: Any,
) -> None:
    """One half deciding the merge, staged by the setting that decides it.

    Recorded twice as a blocker before this, and each reason was true of the
    corpus it was measured on. The first said the corpus was too small, which
    measuring on one of twice the size disproved. The second said the whole
    weight on one half moves the order of the merged list and not its
    membership, and on two hundred and twenty fragments that is what happens:
    the two halves return nearly the same twenty candidates, so re-weighting
    reorders an agreed set.

    On three thousand fragments they stop agreeing. With the whole weight on
    the semantic half it supplies about three quarters of the context on its
    own and the keyword half supplies nothing of its own at all, against
    roughly thirty and fifteen per cent when the weight is even. The half
    being paid for has stopped deciding anything.

    The signal the entry names stays silent, and this pair measures why rather
    than asserting it. Its threshold is four fifths of the context from one
    half alone, and the deciding half's own share stops at about three
    quarters at every window size tried, because the remaining quarter is what
    the two halves agreed on. Lowering the threshold is not the fix: measured
    over the runs stored here, a healthy small-corpus run has the keyword half
    contributing nothing of its own too, agreeing with the semantic half on
    fifty fragments of fifty-seven, so a rule reading "nothing of its own" as
    idleness reports five healthy runs as broken. That was written, measured
    and withdrawn.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.retrieval.hybrid import HybridRetriever
    from tests.proving_ground.conftest import _index_exists

    corpus, strategy, language = "miracl-ru-coded", "fixed", "ru_be"
    if not _index_exists(corpus, strategy, embedder.embedder_id):
        pytest.skip(
            f"NOT RUN: no index for {corpus!r}, which this pair needs because the base corpus "
            "is too small for the two halves to disagree. Build and load it with "
            "`python3 -m tools.corpus_mutate corpus/miracl-ru "
            f"--defect hide_a_code_in_one_document --out /tmp/{corpus} --limit 1500` and "
            f"`USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest /tmp/{corpus} "
            f"--strategy {strategy} --corpus-id {corpus} --language {language} "
            "--realm-id proving-ground`."
        )

    import json
    import pathlib

    dense = QdrantRetriever(host="localhost", port=6333, strategy_id=strategy,
                            embedder_id=embedder.embedder_id, corpus_id=corpus,
                            realm_id="proving-ground")
    sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id=strategy,
                                 corpus_id=corpus, realm_id="proving-ground", language=language)
    questions = [json.loads(line)["question"] for line
                 in pathlib.Path("eval/golden/miracl-ru.v1.fast.jsonl")
                 .read_text(encoding="utf-8").splitlines() if line.strip()][:40]

    def shares(alpha: float, window: int = 5) -> dict[str, float]:
        merge = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                                embedder=embedder, merge="weighted", alpha=alpha)
        counted = {"semantic only": 0, "keyword only": 0, "both": 0}
        for question in questions:
            vector = embedder.embed([question])[0]
            fetch = max(window * 2, 20)
            by_meaning = {h.chunk.chunk_id
                          for h in dense.retrieve(query=question, k=fetch, query_vector=vector)}
            by_words = {h.chunk.chunk_id for h in sparse.retrieve(query=question, k=fetch)}
            for hit in merge.retrieve(query=question, k=window, query_vector=vector):
                where = hit.chunk.chunk_id
                if where in by_meaning and where in by_words:
                    counted["both"] += 1
                elif where in by_meaning:
                    counted["semantic only"] += 1
                elif where in by_words:
                    counted["keyword only"] += 1
        total = sum(counted.values())
        assert total, "the merge returned nothing, so this measures nothing"
        return {name: round(n / total, 3) for name, n in counted.items()}

    even = shares(0.5)
    all_on_one = shares(1.0)
    assert all_on_one["keyword only"] == 0.0, (
        f"the keyword half still contributes fragments of its own with no weight at all: "
        f"{all_on_one}"
    )
    assert all_on_one["semantic only"] > even["semantic only"] * 2, (
        f"the weight changed nothing about which fragments reach the answer: {even} against "
        f"{all_on_one}, so this is the membership finding the earlier blocker recorded"
    )
    assert even["keyword only"] > 0.05, (
        f"the keyword half contributes nothing of its own on an even weighting either: {even}, "
        "so this corpus cannot show the difference and the blocker stands"
    )
    assert all_on_one["semantic only"] < 0.8, (
        f"the deciding half's own share reached {all_on_one['semantic only']}, so the signal "
        "the entry names does fire and the entry should say it is caught"
    )

    record("F21", "with the whole weight on one half it supplies three quarters of the context "
                  "on its own and the other supplies nothing of its own, and the named signal "
                  "stays under its threshold",
           corpus=corpus, questions=len(questions), window=5,
           even_weighting=even, whole_weight_on_the_semantic_half=all_on_one,
           the_signals_threshold=0.8,
           why_it_stays_silent="a quarter of the context is what the two halves agreed on, so "
                               "the deciding half's own share stops at about three quarters at "
                               "every window size tried",
           lowering_it_was_tried="a rule reading 'nothing of its own' as idleness reports five "
                                 "healthy stored runs as broken, one of them agreeing on fifty "
                                 "fragments of fifty-seven")


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


def test_F24_a_reranker_that_does_not_know_the_language_drops_what_retrieval_found(
    embedder: Any, control: dict[str, Any]
) -> None:
    """The corpus is Russian and the reranker reads English only.

    Both models are cross-encoders, both take a pair of texts, both return a
    score, and nothing in the configuration or in the run says one of them
    has never seen this alphabet. What says it is the funnel: retrieval finds
    the reference source for every question before the reranker, and the
    reranker gives back a context without it.

    Staged by a setting, because the model name is one: this is the entry's
    own instrument and not an argument about the corpus.
    """
    from core.eval.funnel import diagnose_question
    from core.experiment.config import ExperimentConfig

    def _reranked_by(name: str, model: str) -> dict[str, Any]:
        payload = control_config(CORPUS).model_dump(exclude={"config_hash"})
        payload["name"] = f"proving-ground-{name}"
        payload["chunking_strategy"] = {"kind": "chunker", "component_id": "structure_aware",
                                        "params": {}}
        payload["reranker"] = {"kind": "reranker", "component_id": "cross_encoder_local",
                               "params": {"model_name": model}}
        return run_on(embedder, retrieval_only(ExperimentConfig(**payload)), CORPUS, LANGUAGE)

    multilingual = _reranked_by("F24-multilingual", "BAAI/bge-reranker-v2-m3")
    english_only = _reranked_by("F24-english-only", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    assert recall_before_rerank(english_only) == recall_before_rerank(multilingual), (
        "retrieval itself moved, so this pair changed two things and neither is the reranker"
    )
    assert recall(english_only) < recall(multilingual), (
        f"the English-only reranker cost nothing: {recall(english_only)} against "
        f"{recall(multilingual)}"
    )

    def _lost_at_the_reranker(run: dict[str, Any]) -> int:
        return sum(
            1 for q in run["question_results"]
            if diagnose_question(q.get("answerability") or "answerable", q.get("metrics") or {},
                                 (q.get("metrics") or {}).get("pre_rerank_recall_at_k")).layer
            == "rerank"
        )

    lost = _lost_at_the_reranker(english_only)
    assert lost > _lost_at_the_reranker(multilingual), (
        f"the funnel blames the reranker on {lost} questions here and on "
        f"{_lost_at_the_reranker(multilingual)} with a model that reads the language, so the "
        "verdict does not separate the two"
    )
    record("F24", "a cross-encoder that has never seen this alphabet, ranking its corpus",
           recall_multilingual=recall(multilingual), recall_english_only=recall(english_only),
           before_the_reranker=recall_before_rerank(english_only),
           questions_the_funnel_blames_on_the_reranker=lost,
           the_same_with_a_multilingual_model=_lost_at_the_reranker(multilingual))
