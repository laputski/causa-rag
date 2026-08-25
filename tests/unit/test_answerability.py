"""core/eval/answerability.py — Eval Measurement Trustworthiness, Phase 0;
reworked later.

These were previously driven by a real on-disk corpus directory,
which is gitignored — so on a fresh checkout they silently exercised an
EMPTY corpus and every assertion about "answerable" would have been wrong.
They now run against in-memory resolvers: same logic, no external data, and
the previously untestable "could not check" path is finally reachable.

Deliberately dropped: the four assertions that counted classification
buckets across real golden datasets (e.g. "one full set is exactly
84/11/5"). Those asserted the state of one corpus snapshot on one machine,
not the behaviour of this module, and they are the reason this file could
not run without 42 MB of data present. Coverage counts against a real
indexed corpus are an integration concern, not a unit one.
"""
from __future__ import annotations

from core.eval.answerability import (
    classify_answerability,
    classify_dataset,
    resolve_answerability,
    resolve_answerability_verdict,
)
from core.eval.ref_resolution import IndexRefResolver, UnknownRefResolver

# Stands in for an index that contains these units and nothing else.
_INDEXED = IndexRefResolver(
    known_ref_ids=frozenset({"SRC001/44", "SRC001/47", "DOC1#Section > Subsection", "DOC2"})
)


# ── Core classification ──────────────────────────────────────────────────

# @lat: [[external-rag#Generic answerability gating]]
def test_no_refs_is_out_of_scope() -> None:
    verdict = classify_answerability([], _INDEXED)
    assert verdict.answerability == "out_of_scope"
    # No refs means nothing to look up, so this verdict IS fully determined.
    assert verdict.coverage_checked is True


def test_indexed_ref_is_answerable() -> None:
    assert classify_answerability(["SRC001/44"], _INDEXED).answerability == "answerable"


def test_ref_absent_from_index_is_uncovered() -> None:
    assert classify_answerability(["SRC003/101"], _INDEXED).answerability == "uncovered"


def test_multi_ref_all_present_is_answerable() -> None:
    refs = ["SRC001/44", "SRC001/47"]
    assert classify_answerability(refs, _INDEXED).answerability == "answerable"


def test_multi_ref_one_absent_is_uncovered() -> None:
    refs = ["SRC001/44", "SRC003/101"]
    verdict = classify_answerability(refs, _INDEXED)
    assert verdict.answerability == "uncovered"
    # The reviewer is told WHICH unit is missing, not just that one is.
    assert verdict.missing_refs == ("SRC003/101",)


def test_structural_path_and_bare_doc_id_refs_resolve_like_any_other() -> None:
    # A corpus with no external document-code numbering produces these two
    # ref shapes; they go through the same membership check as any other.
    assert classify_answerability(["DOC1#Section > Subsection"], _INDEXED).answerability == "answerable"
    assert classify_answerability(["DOC2"], _INDEXED).answerability == "answerable"
    assert classify_answerability(["DOC3"], _INDEXED).answerability == "uncovered"


# ── The bug this rework exists to remove ─────────────────────────────────
# An unavailable check must never masquerade as a negative check result.

def test_unavailable_resolver_yields_answerable_not_uncovered() -> None:
    """THE regression guard for that change.

    Before: an absent/unreachable corpus made every ref fail to resolve, so
    every question became "uncovered" — which excludes it from retrieval
    metrics entirely, silently zeroing measurement with no error anywhere.
    After: inability to check degrades to "answerable, not verified".
    """
    verdict = classify_answerability(["SRC001/44"], UnknownRefResolver(reason="index unreachable"))
    assert verdict.answerability == "answerable"
    assert verdict.coverage_checked is False


def test_no_resolver_at_all_behaves_the_same_as_an_unavailable_one() -> None:
    verdict = classify_answerability(["anything/1"], None)
    assert verdict.answerability == "answerable"
    assert verdict.coverage_checked is False


def test_verified_coverage_is_marked_as_checked() -> None:
    assert classify_answerability(["SRC001/44"], _INDEXED).coverage_checked is True


def test_confirmed_gap_outranks_an_unverifiable_ref() -> None:
    """Rule 2 beats rule 3: one ref definitely absent decides the verdict
    even when another could not be checked, because a confirmed miss is
    information and an unverifiable ref is not."""

    class _Partial:
        def presence(self, ref: str) -> str:
            return "absent" if ref == "GONE/1" else "unknown"

        @property
        def checked(self) -> bool:
            return True

    verdict = classify_answerability(["GONE/1", "MAYBE/2"], _Partial())  # type: ignore[arg-type]
    assert verdict.answerability == "uncovered"
    assert verdict.missing_refs == ("GONE/1",)


# ── Row-level resolution ─────────────────────────────────────────────────

def test_explicit_row_field_wins_over_the_index() -> None:
    # An external RAG's dataset is classified by its own author; the
    # platform never sees that corpus, so its own index must not override.
    question = {"article_refs": ["SRC003/101"], "answerability": "answerable"}
    verdict = resolve_answerability_verdict(question, _INDEXED)
    assert verdict.answerability == "answerable"
    assert verdict.coverage_checked is True


def test_row_without_explicit_field_falls_back_to_the_resolver() -> None:
    assert resolve_answerability({"article_refs": ["SRC001/44"]}, _INDEXED) == "answerable"
    assert resolve_answerability({"article_refs": ["SRC003/101"]}, _INDEXED) == "uncovered"


def test_row_with_no_refs_key_at_all_is_out_of_scope() -> None:
    assert resolve_answerability({}, _INDEXED) == "out_of_scope"


def test_classify_dataset_maps_every_row_by_id() -> None:
    rows = [
        {"id": "q1", "article_refs": ["SRC001/44"]},
        {"id": "q2", "article_refs": ["SRC003/101"]},
        {"id": "q3", "article_refs": []},
        {"id": "q4", "article_refs": ["SRC003/101"], "answerability": "answerable"},
    ]
    assert classify_dataset(rows, _INDEXED) == {
        "q1": "answerable", "q2": "uncovered", "q3": "out_of_scope", "q4": "answerable",
    }
