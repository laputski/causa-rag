"""core/eval/retrieval_metrics.py — Eval Measurement Trustworthiness, Phase 0."""
from __future__ import annotations

from core.eval.retrieval_metrics import (
    average_precision,
    extract_ref_id,
    extracted_refs,
    mrr,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_full_match() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/44", "SRC001/99"]
    assert recall_at_k(expected, retrieved) == 1.0


def test_recall_at_k_no_match() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/99", "SRC001/100"]
    assert recall_at_k(expected, retrieved) == 0.0


def test_recall_at_k_partial_multi_ref() -> None:
    """Comparative question citing two articles — finding only one is
    partial credit, not a binary pass/fail."""
    expected = ["SRC001/44", "SRC001/47"]
    retrieved = ["SRC001/44", "SRC001/99"]
    assert recall_at_k(expected, retrieved) == 0.5


def test_recall_at_k_respects_k_cutoff() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/99", "SRC001/100", "SRC001/44"]
    assert recall_at_k(expected, retrieved, k=2) == 0.0
    assert recall_at_k(expected, retrieved, k=3) == 1.0


def test_recall_at_k_disambiguates_gk_tk_article_collision() -> None:
    """'Статья 5' exists in both ГК (SRC001) and ТК (SRC002) — without
    source_code, matching by article number alone would falsely match."""
    expected = ["SRC001/5"]  # ГК Статья 5
    retrieved_wrong_code = ["SRC002/5"]  # ТК Статья 5 — different article entirely
    assert recall_at_k(expected, retrieved_wrong_code) == 0.0


def test_recall_at_k_empty_expected_returns_zero() -> None:
    assert recall_at_k([], ["SRC001/5"]) == 0.0


def test_precision_at_k_basic() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/44", "SRC001/99", "SRC001/100"]
    assert precision_at_k(expected, retrieved) == 1 / 3


def test_precision_at_k_dedupes_repeated_article_chunks() -> None:
    """Multiple chunks from the same relevant article shouldn't inflate
    precision — dedupe by article before scoring."""
    expected = ["SRC001/44"]
    retrieved = ["SRC001/44", "SRC001/44", "SRC001/99"]
    assert precision_at_k(expected, retrieved) == 1 / 2


def test_precision_at_k_empty_retrieved_returns_zero() -> None:
    assert precision_at_k(["SRC001/44"], []) == 0.0


def test_mrr_first_hit() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/44", "SRC001/99"]
    assert mrr(expected, retrieved) == 1.0


def test_mrr_second_hit() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/99", "SRC001/44"]
    assert mrr(expected, retrieved) == 0.5


def test_mrr_no_hit() -> None:
    assert mrr(["SRC001/44"], ["SRC001/99"]) == 0.0


def test_multi_article_chapter_coverage() -> None:
    """Глава 25 expands to 14 real articles (364-377) — recall@k over the
    full expanded set, not the chapter label."""
    expected = [f"SRC001/{n}" for n in range(364, 378)]
    retrieved = ["SRC001/364", "SRC001/365", "SRC001/999"]
    assert recall_at_k(expected, retrieved) == 2 / 14


def test_average_precision_relevant_ranked_first_scores_full() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/44", "SRC001/99", "SRC001/100"]
    assert average_precision(expected, retrieved) == 1.0


def test_average_precision_relevant_buried_scores_lower_than_ranked_first() -> None:
    """The whole point of this metric over precision_at_k: same top-k
    membership, different rank — must score differently."""
    expected = ["SRC001/44"]
    buried = average_precision(expected, ["SRC001/99", "SRC001/100", "SRC001/44"])
    first = average_precision(expected, ["SRC001/44", "SRC001/99", "SRC001/100"])
    assert buried < first
    assert buried == 1 / 3  # precision at rank 3 (the only hit), averaged over 1 expected ref


def test_average_precision_multi_ref_rewards_early_ranking_of_both() -> None:
    expected = ["SRC001/44", "SRC001/47"]
    both_early = average_precision(expected, ["SRC001/44", "SRC001/47", "SRC001/99"])
    one_late = average_precision(expected, ["SRC001/44", "SRC001/99", "SRC001/47"])
    assert both_early == 1.0
    assert one_late < both_early


def test_average_precision_dedupes_repeated_article_chunks() -> None:
    expected = ["SRC001/44"]
    retrieved = ["SRC001/44", "SRC001/44", "SRC001/99"]
    assert average_precision(expected, retrieved) == 1.0


def test_average_precision_no_hit_is_zero() -> None:
    assert average_precision(["SRC001/44"], ["SRC001/99"]) == 0.0


def test_average_precision_empty_expected_returns_zero() -> None:
    assert average_precision([], ["SRC001/44"]) == 0.0


def test_extract_ref_id_builds_composite_key() -> None:
    sr = {"source_code": "SRC001", "article_no": "44", "chunk_text": "..."}
    assert extract_ref_id(sr) == "SRC001/44"


def test_extract_ref_id_missing_fields_returns_none() -> None:
    assert extract_ref_id({"chunk_text": "no metadata here"}) is None
    assert extract_ref_id({"source_code": "SRC001"}) is None


def test_extract_ref_id_falls_back_to_structural_path_without_article_no() -> None:
    """Realm-agnostic: a corpus not laid out as one numbered file per
    article (e.g. a manual chunked by heading, not by <code>/<number>.txt)
    still resolves to a usable ref id instead of None."""
    sr = {"source_code": "install-guide", "structural_path": "section/subsection[Montage]"}
    assert extract_ref_id(sr) == "install-guide#section/subsection[Montage]"


def test_extract_ref_id_prefers_article_no_over_structural_path_when_both_present() -> None:
    sr = {"source_code": "SRC001", "article_no": "44", "structural_path": "chapter/article"}
    assert extract_ref_id(sr) == "SRC001/44"


def test_extract_ref_id_none_when_neither_source_code_nor_doc_id_present() -> None:
    assert extract_ref_id({"structural_path": "section/subsection[Montage]"}) is None


# ── doc_id fallback for corpora with no source_code concept at all (found
# live: a general document corpus — technical manuals, no legal-document-
# code system — left every question generated against it with a completely
# empty article_refs, universally scored "out_of_scope" regardless of
# whether the corpus actually covers the content) ──────────────────────────

def test_extract_ref_id_falls_back_to_doc_id_and_structural_path_without_source_code() -> None:
    sr = {"doc_id": "6a50c2c2838af6f9c9765f4c", "structural_path": "ОСНОВНЫЕ ТРЕБОВАНИЯ > ЭМС"}
    assert extract_ref_id(sr) == "6a50c2c2838af6f9c9765f4c#ОСНОВНЫЕ ТРЕБОВАНИЯ > ЭМС"


def test_extract_ref_id_falls_back_to_bare_doc_id_without_source_code_or_structural_path() -> None:
    sr = {"doc_id": "6a50c2c2838af6f9c9765f4c"}
    assert extract_ref_id(sr) == "6a50c2c2838af6f9c9765f4c"


def test_extract_ref_id_source_code_still_wins_over_doc_id_when_both_present() -> None:
    """A corpus that DOES have source_code metadata must keep using it —
    the doc_id fallback only ever applies when source_code is absent."""
    sr = {"source_code": "SRC001", "article_no": "44", "doc_id": "some-internal-uuid"}
    assert extract_ref_id(sr) == "SRC001/44"


def test_extracted_refs_preserves_rank_order_and_skips_unresolvable() -> None:
    source_refs = [
        {"source_code": "SRC001", "article_no": "44"},
        {"chunk_text": "legacy chunk, no metadata"},
        {"source_code": "SRC001", "article_no": "47"},
    ]
    assert extracted_refs(source_refs) == ["SRC001/44", "SRC001/47"]
