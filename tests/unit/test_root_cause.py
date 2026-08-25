"""core/eval/root_cause.py.

The module exists to stop "retrieval" from being the end of the story. These
tests pin the ordering of the evidence, because getting it wrong is what
sends an engineer to tune a ranker over data that is not there.
"""
from __future__ import annotations

from core.eval.root_cause import classify_retrieval_cause, count_causes


def _miss(found: bool, rank: int | None = None, widened_k: int = 50) -> dict:
    return {"found": found, "rank": rank, "score": 0.4, "structural_path": "", "widened_k": widened_k}


# ── Absence beats every other observation ───────────────────────────────


def test_an_absent_source_is_a_data_problem_not_a_ranking_one() -> None:
    cause = classify_retrieval_cause({"S/1": "absent"}, _miss(found=False), top_k=5)
    assert cause.cause == "data_missing"
    assert cause.evidence["absent_refs"] == ["S/1"]


def test_absence_outranks_a_successful_widened_lookup() -> None:
    """Contrived but load-bearing: if the resolver says absent, no retrieval
    observation can make it a ranking problem, because every observation
    about a missing document is a consequence rather than a cause."""
    cause = classify_retrieval_cause({"S/1": "absent"}, _miss(found=True, rank=7), top_k=5)
    assert cause.cause == "data_missing"


def test_one_absent_ref_among_present_ones_still_reports_data_missing() -> None:
    cause = classify_retrieval_cause({"S/1": "present", "S/2": "absent"}, _miss(True, 3), top_k=5)
    assert cause.cause == "data_missing"
    assert cause.evidence["absent_refs"] == ["S/2"]


# ── that change's distinction, carried through ─────────────────────────────


def test_an_unverified_index_yields_unknown_not_a_confident_cause() -> None:
    """Naming a ranking cause against an index nobody could read is exactly
    the false confidence that was removed from answerability."""
    cause = classify_retrieval_cause({"S/1": "unknown"}, _miss(found=True, rank=9), top_k=5)
    assert cause.cause == "unknown"
    assert cause.evidence["unknown_refs"] == ["S/1"]


def test_a_question_with_no_expected_refs_states_no_cause() -> None:
    assert classify_retrieval_cause({}, _miss(True, 1), top_k=5).cause == "unknown"


def test_a_failed_widened_query_is_unknown_not_not_retrievable() -> None:
    """"The re-query could not run" and "the re-query found nothing" mean
    opposite things and must not collapse into one verdict."""
    cause = classify_retrieval_cause({"S/1": "present"}, None, top_k=5)
    assert cause.cause == "unknown"


# ── The two causes that need the widened re-query ───────────────────────


def test_found_at_a_wider_k_means_the_cut_off_is_the_problem() -> None:
    cause = classify_retrieval_cause({"S/1": "present"}, _miss(found=True, rank=12), top_k=5)
    assert cause.cause == "ranking"
    assert cause.evidence["rank"] == 12
    assert cause.evidence["top_k"] == 5


def test_not_found_even_at_a_wider_k_means_query_and_text_are_far_apart() -> None:
    cause = classify_retrieval_cause({"S/1": "present"}, _miss(found=False), top_k=5)
    assert cause.cause == "not_retrievable"
    assert cause.evidence["widened_k"] == 50


def test_the_third_bucket_is_not_called_chunking() -> None:
    """the design notes task list names this bucket `chunking`. The
    evidence only establishes that the text is not close to the query, and a
    split citable unit is one explanation among several — a vocabulary gap
    is another. Split-unit detection can carve a genuine
    `chunking` cause out of this one; until then the name must not assert
    more than is known."""
    assert classify_retrieval_cause({"S/1": "present"}, _miss(False), top_k=5).cause != "chunking"


# ── The aggregate, which is why this runs for every failure ─────────────


def test_counts_group_a_run_into_kinds_of_work() -> None:
    results = [
        {"root_cause": {"cause": "data_missing"}},
        {"root_cause": {"cause": "data_missing"}},
        {"root_cause": {"cause": "ranking"}},
        {"root_cause": None},
        {},
    ]
    assert count_causes(results) == {"data_missing": 2, "ranking": 1}


def test_counting_a_run_with_no_failures_yields_nothing_rather_than_zeros() -> None:
    assert count_causes([{"root_cause": None}, {}]) == {}


# ── The lever a cause points at ──────────────────────────────


def test_every_cause_names_a_lever() -> None:
    """The lever is derived in __post_init__ rather than passed in, so a
    cause added later cannot ship without one attached."""
    from core.eval.root_cause import RootCause

    assert RootCause(cause="data_missing", detail="").lever == "ingest"
    assert RootCause(cause="ranking", detail="").lever == "ranking"
    assert RootCause(cause="not_retrievable", detail="").lever == "chunking_or_vocabulary"
    assert RootCause(cause="unknown", detail="").lever == "verify_index"


def test_an_unrecognised_cause_points_at_verifying_rather_than_changing() -> None:
    """Not knowing what went wrong is a reason to establish the ground truth,
    never a reason to start editing something."""
    from core.eval.root_cause import lever_for

    assert lever_for("something_new") == "verify_index"


def test_the_lever_is_serialised_alongside_the_cause() -> None:
    from core.eval.root_cause import RootCause

    assert RootCause(cause="data_missing", detail="d").to_dict()["lever"] == "ingest"


# ── A split unit is its own cause ────────────────────────────


def test_a_unit_split_across_chunks_is_a_chunking_problem() -> None:
    cause = classify_retrieval_cause(
        {"S/1": "present"}, _miss(found=False), top_k=5, chunk_counts={"S/1": 4},
    )
    assert cause.cause == "chunking"
    assert cause.lever == "chunking"
    assert cause.evidence["split_into"] == 4


def test_a_single_chunk_unit_rules_chunking_out_and_points_at_wording() -> None:
    """All of the unit's text sits in one place and retrieval still does not
    surface it, so the splitting cannot be what went wrong."""
    cause = classify_retrieval_cause(
        {"S/1": "present"}, _miss(found=False), top_k=5, chunk_counts={"S/1": 1},
    )
    assert cause.cause == "not_retrievable"
    assert cause.lever == "vocabulary"


def test_an_uncounted_unit_stays_undecided_rather_than_assumed_single() -> None:
    """Not knowing how many chunks a unit occupies must not be reported as
    knowing it occupies one — that would assert wording as the cause on no
    evidence at all."""
    cause = classify_retrieval_cause(
        {"S/1": "present"}, _miss(found=False), top_k=5, chunk_counts={"S/1": None},
    )
    assert cause.cause == "not_retrievable"
    assert cause.lever == "chunking_or_vocabulary"


def test_omitting_counts_entirely_behaves_as_uncounted() -> None:
    cause = classify_retrieval_cause({"S/1": "present"}, _miss(found=False), top_k=5)
    assert cause.cause == "not_retrievable"
    assert cause.lever == "chunking_or_vocabulary"


def test_one_split_unit_among_several_expected_refs_is_enough() -> None:
    # A single split unit already explains the miss, so the maximum is what
    # matters rather than every ref having to be split.
    cause = classify_retrieval_cause(
        {"S/1": "present", "S/2": "present"}, _miss(found=False), top_k=5,
        chunk_counts={"S/1": 1, "S/2": 3},
    )
    assert cause.cause == "chunking"
    assert cause.evidence["split_into"] == 3


def test_a_split_unit_that_ranks_within_a_wider_k_is_still_a_ranking_problem() -> None:
    """Splitting only explains a miss when nothing of the unit surfaces. If a
    fragment does rank, the cut-off is what kept it out."""
    cause = classify_retrieval_cause(
        {"S/1": "present"}, _miss(found=True, rank=11), top_k=5, chunk_counts={"S/1": 4},
    )
    assert cause.cause == "ranking"
