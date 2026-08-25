"""core/citation.py — programmatic citation labels (not LLM-transcribed).

See the module docstring for why: generated text unreliably transcribes
citation numbers (index confusion, outright hallucination), while
source_refs metadata is always correct (it's what retrieval actually found).
"""
from __future__ import annotations

from core.citation import (
    answer_has_any_number,
    citation_number_coverage,
    compute_citation_labels,
    substitute_fragment_markers,
)
from core.models import SourceRef


def _ref(
    structural_path: str, doc_id: str = "d1",
    source_code: str | None = None, article_no: str | None = None,
) -> SourceRef:
    return SourceRef(
        doc_id=doc_id, chunk_id="c1", structural_path=structural_path,
        source_code=source_code, article_no=article_no,
    )


def test_no_source_refs_returns_empty() -> None:
    assert compute_citation_labels("some answer", []) == []


def test_no_fragment_marker_falls_back_to_top_ranked() -> None:
    refs = [
        _ref("document/article[Article 210.5. Closing an escrow account]"),
        _ref("document/article[Article 210.1. The escrow account agreement]"),
    ]
    # The model hallucinated an article 1 with no "Фрагмент N" marker at all —
    # must not trust that, must use the top-ranked source instead.
    assert compute_citation_labels("The escrow agent. (Article 1)", refs) == [
        "Article 210.5. Closing an escrow account",
    ]


def test_fragment_marker_maps_to_correct_ref_not_top_ranked() -> None:
    refs = [
        _ref("document/article[Article 210.5. Closing an escrow account]"),
        _ref("document/article[Article 210.1. The escrow account agreement]"),
    ]
    assert compute_citation_labels("Answer text (Фрагмент 2).", refs) == [
        "Article 210.1. The escrow account agreement",
    ]


def test_multiple_distinct_fragment_markers_preserve_first_mention_order() -> None:
    refs = [_ref(f"document/article[Article {n}]") for n in (10, 20, 30)]
    text = "See Фрагмент 3 and Фрагмент 1, and Фрагмент 3 again."
    assert compute_citation_labels(text, refs) == ["Article 30", "Article 10"]


def test_out_of_range_fragment_index_is_ignored() -> None:
    refs = [_ref("document/article[Article 5]")]
    # Model said "Фрагмент 9" but there's only 1 retrieved chunk — ignore
    # the bogus index rather than crash or fabricate something.
    assert compute_citation_labels("An answer (Фрагмент 9).", refs) == ["Article 5"]


def test_falls_back_to_doc_id_when_structural_path_has_no_bracket_label() -> None:
    refs = [SourceRef(doc_id="doc-42", chunk_id="c1", structural_path="document/paragraph")]
    assert compute_citation_labels("an answer with no markers", refs) == ["document/paragraph"]


def test_falls_back_to_doc_id_when_structural_path_is_empty() -> None:
    refs = [SourceRef(doc_id="doc-42", chunk_id="c1", structural_path="")]
    assert compute_citation_labels("an answer", refs) == ["doc-42"]


# ── citation_number_coverage / answer_has_any_number ─────────────────────────
# Domain-agnostic: never assumes a citation grammar ("Article N", "Section N",
# ...) — only checks whether the retrieved label's OWN digits occur anywhere
# in the answer text, since the candidate numbers are already known from
# source_refs, not guessed from the answer's wording.

def test_wrong_cited_number_gives_zero_coverage() -> None:
    """The real bug this metric exists to catch: model wrote article 1
    when the retrieved chunk was actually "Article 210.5" — no fragment
    marker confusion needed to reproduce, just the literal wrong digits."""
    refs = [_ref("document/article[Article 210.5. Closing an escrow account]")]
    assert citation_number_coverage("The escrow agent. Article 1.", refs) == 0.0


def test_correct_cited_number_gives_full_coverage() -> None:
    refs = [_ref("document/article[Article 210.5. Closing an escrow account]")]
    assert citation_number_coverage("The escrow agent. Article 210.5.", refs) == 1.0


def test_comma_as_decimal_separator_still_matches() -> None:
    # The label says "210.5" but the model wrote "210,5" — different
    # punctuation convention, same number; must not count as a miss.
    refs = [_ref("document/article[Article 210.5. Closing an escrow account]")]
    assert citation_number_coverage("See article 210,5.", refs) == 1.0


def test_no_number_in_answer_gives_zero_coverage_not_none() -> None:
    refs = [_ref("document/article[Article 210.5. Closing an escrow account]")]
    assert citation_number_coverage("Yes, they apply.", refs) == 0.0


def test_multi_ref_coverage_is_fractional_not_top1_only() -> None:
    refs = [
        _ref("document/article[Article 210.5. Closing an escrow account]"),
        _ref("document/article[Article 774.1. The escrow account agreement]"),
    ]
    assert citation_number_coverage("See article 210.5.", refs) == 0.5
    assert citation_number_coverage("See articles 210.5 and 774.1.", refs) == 1.0


def test_none_when_no_source_ref_has_a_bracketed_label() -> None:
    """Flat-chunked corpus / no structure_parser — there's no extractable
    structural number to check against at all. Must return None, not 0.0,
    so this question is excluded from the aggregate rather than silently
    counted as "wrong citation"."""
    refs = [SourceRef(doc_id="doc-42", chunk_id="c1", structural_path="document/paragraph")]
    assert citation_number_coverage("some answer with no markers", refs) is None


def test_none_when_no_source_refs_at_all() -> None:
    assert citation_number_coverage("an answer", []) is None


def test_substring_guard_does_not_match_inside_a_longer_number() -> None:
    """"5" must not spuriously match inside "2025" — a citation-coverage
    check on "Article 5" shouldn't be fooled by an unrelated year."""
    refs = [_ref("document/article[Article 5. Application]")]
    assert citation_number_coverage("As amended in 2025.", refs) == 0.0


# ── article_refs-filtered coverage (real fix: was capped near 1/top_k) ──────

def test_article_refs_filter_restricts_candidates_to_the_correct_chunk() -> None:
    """The actual bug this filter fixes: at top_k=10, a perfectly-correct
    single-citation answer was scored against ALL retrieved chunks'
    numbers, capping near 1/k even when correct. Filtering to only the
    chunk(s) matching article_refs makes a correct citation score 1.0."""
    refs = [
        _ref(
            "document/article[Article 210.5. Closing an escrow account]",
            source_code="SRC001", article_no="210.5",
        ),
        _ref(
            "document/article[Article 774.1. The escrow account agreement]",
            source_code="SRC001", article_no="774.1",
        ),
        _ref(
            "document/article[Article 99. An unrelated article]",
            source_code="SRC001", article_no="99",
        ),
    ]
    answer = "The escrow agent (Article 210.5. Closing an escrow account)"
    # Without the filter: 1 of 3 distinct numbers present -> 1/3.
    assert citation_number_coverage(answer, refs) == 1 / 3
    # With the filter: only 210.5 is the ground-truth chunk -> 1.0.
    assert citation_number_coverage(answer, refs, article_refs=["SRC001/210.5"]) == 1.0


def test_article_refs_filter_returns_none_when_no_chunk_matches_ground_truth() -> None:
    """recall_at_k=0 case — nothing relevant was even retrieved, so there's
    no number to check against. Must be None, not a misleading 0.0 (same
    convention as grounded_in_correct_source)."""
    refs = [_ref("document/article[Article 99]", source_code="SRC001", article_no="99")]
    assert citation_number_coverage("some answer", refs, article_refs=["SRC001/210.5"]) is None


def test_article_refs_filter_still_fractional_for_multi_ref_questions() -> None:
    refs = [
        _ref("document/article[Article 210.5]", source_code="SRC001", article_no="210.5"),
        _ref("document/article[Article 774.1]", source_code="SRC001", article_no="774.1"),
        _ref("document/article[Article 99]", source_code="SRC001", article_no="99"),
    ]
    article_refs = ["SRC001/210.5", "SRC001/774.1"]
    assert citation_number_coverage("See article 210.5.", refs, article_refs) == 0.5
    assert citation_number_coverage("See articles 210.5 and 774.1.", refs, article_refs) == 1.0


# ── Corpus with no external document-code numbering scheme (found live: the
# filter used to hardcode a "{source_code}/{article_no}" match, so it never
# matched anything for a corpus whose chunks have no source_code at all —
# every candidate number vanished, None, for every question of such a run) ──

def test_article_refs_filter_matches_doc_id_structural_path_refs() -> None:
    refs = [
        _ref("document/article[Article 210.5]", doc_id="d1"),
        _ref("document/article[Article 99]", doc_id="d2"),
    ]
    answer = "See article 210.5."
    assert citation_number_coverage(answer, refs, article_refs=["d1#document/article[Article 210.5]"]) == 1.0


def test_answer_has_any_number_true_and_false() -> None:
    assert answer_has_any_number("Article 1.") is True
    assert answer_has_any_number("Yes, they apply.") is False


# ── substitute_fragment_markers ───────────────────────────────────────────────
# prompts/demo_prompt_v1 — the model cites ONLY by position ("Фрагмент N"), never
# transcribes the real number itself; this is what turns that position into
# the real label before the answer reaches the user (the "cite by index, not
# by content" pattern, see GuidePage discussion this session).

def test_substitutes_parenthetical_marker_with_real_label() -> None:
    refs = [_ref("document/article[Article 210.5. Closing an escrow account]")]
    assert substitute_fragment_markers("The escrow agent. (Фрагмент 1).", refs) == (
        "The escrow agent. (Article 210.5. Closing an escrow account)."
    )


def test_substitutes_bare_marker_with_no_parens() -> None:
    refs = [_ref("document/article[Article 210.5. Closing an escrow account]")]
    assert substitute_fragment_markers("Yes, it is required. Фрагмент 1", refs) == (
        "Yes, it is required. Article 210.5. Closing an escrow account"
    )


def test_out_of_range_marker_is_removed_not_left_dangling() -> None:
    """Real bug this guards against: a raw, unresolvable "(Фрагмент 9)"
    leaking to the user is worse than no citation at all — must be
    stripped cleanly, including the space it leaves before punctuation."""
    refs = [_ref("document/article[Article 5. Application]")]
    assert substitute_fragment_markers("Something odd (Фрагмент 9).", refs) == "Something odd."


def test_multiple_distinct_markers_each_substituted() -> None:
    refs = [
        _ref("document/article[Article 210.5. Closing an escrow account]"),
        _ref("document/article[Article 774.1. The escrow account agreement]"),
    ]
    result = substitute_fragment_markers("See Фрагмент 1 and Фрагмент 2.", refs)
    assert "Article 210.5. Closing an escrow account" in result
    assert "Article 774.1. The escrow account agreement" in result
    assert "Фрагмент" not in result


def test_no_marker_present_leaves_text_unchanged() -> None:
    refs = [_ref("document/article[Article 5. Application]")]
    assert substitute_fragment_markers("Yes, they apply.", refs) == "Yes, they apply."


def test_no_source_refs_at_all_strips_any_marker() -> None:
    assert substitute_fragment_markers("An answer (Фрагмент 1).", []) == "An answer."


def test_latin_cyrillic_code_switched_marker_is_still_caught() -> None:
    """Real bug found live: the model wrote "Фragment 7" — Cyrillic Ф +
    Latin "ragment" — which the original Cyrillic-only regex missed
    entirely, leaking the raw marker to the user untouched."""
    refs = [_ref("document/article[Article 105. Subsidiary companies]")]
    assert substitute_fragment_markers("An answer. (Фragment 1)", refs) == (
        "An answer. (Article 105. Subsidiary companies)"
    )


def test_full_latin_fragment_marker_is_also_caught() -> None:
    refs = [_ref("document/article[Article 5. Application]")]
    assert substitute_fragment_markers("An answer (Fragment 1).", refs) == (
        "An answer (Article 5. Application)."
    )
