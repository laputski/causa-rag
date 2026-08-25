"""Retrieval-quality metrics against article_refs ground truth (Eval
Measurement Trustworthiness, Phase 0).

Pure functions over ref-id strings — "{source_code}/{article_no}" (e.g.
"SRC001/5") when the corpus is laid out as one numbered file per citable
unit, falling back to "{source_code}#{structural_path}" (e.g.
"install-guide#section/subsection[Montage]") otherwise — see
extract_ref_id. The source_code disambiguates collisions between corpus
docs that share the same suffix (two different codes both with an article 5, or two manuals
both having an "Overview" section); without it, matching by article number
or structural_path alone is unsound (see core/eval/answerability.py
docstring for why this exists).

These metrics only make sense on "answerable" questions (see
answerability.py) — on "uncovered"/"out_of_scope" questions there is no
correct retrieval target, so recall/precision/MRR are undefined there.
"""
from __future__ import annotations

from typing import Any


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Multiple chunks from the same article shouldn't inflate precision —
    a "hit" is about whether the right article was retrieved at all, not
    how many of its chunks landed in top-k."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(expected: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Fraction of expected refs found anywhere in the top-k retrieved list.

    Multi-ref questions (comparative questions citing several articles)
    need ALL expected refs found to reach 1.0 — partial credit for partial
    coverage, not an all-or-nothing match.
    """
    if not expected:
        return 0.0
    top = retrieved[:k] if k is not None else retrieved
    found = set(top) & set(expected)
    return len(found) / len(expected)


def precision_at_k(expected: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Fraction of the top-k retrieved (article-deduplicated) that are relevant."""
    top = _dedupe_preserve_order(retrieved[:k] if k is not None else retrieved)
    if not top:
        return 0.0
    relevant = sum(1 for item in top if item in expected)
    return relevant / len(top)


def mrr(expected: list[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first relevant hit (1-based), 0.0 if none found."""
    expected_set = set(expected)
    for rank, item in enumerate(retrieved, start=1):
        if item in expected_set:
            return 1.0 / rank
    return 0.0


def average_precision(expected: list[str], retrieved: list[str]) -> float:
    """Ranking-aware precision — are relevant articles ranked ABOVE
    irrelevant ones, not just present somewhere in top-k.

    Deterministic counterpart to deepeval's ContextualPrecisionMetric (LLM
    judge): precision_at_k only asks "what fraction of top-k is relevant",
    blind to order — a retriever that buries the one relevant chunk at rank
    10 of 10 scores the same as one that puts it at rank 1. Standard
    information-retrieval AP: precision_at_k computed at each rank where a
    relevant item appears, averaged over the relevant items found (article-
    deduplicated, same rationale as precision_at_k — a chunk landing twice
    isn't "more precise"). 0.0 when no expected ref is retrieved at all.
    """
    if not expected:
        return 0.0
    deduped = _dedupe_preserve_order(retrieved)
    expected_set = set(expected)
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(deduped, start=1):
        if item in expected_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(expected_set) if hits else 0.0


def extract_ref_id(source_ref: dict[str, Any]) -> str | None:
    """Build a ref id from a source_ref dict's metadata — realm-agnostic by
    design: '{source_code}/{article_no}' when the corpus follows the
    one-numbered-file-per-unit layout (services/ingestion/cli.py's
    `_ARTICLE_NO_RE`), otherwise '{source_code}#{structural_path}' — the
    chunking-derived heading breadcrumb every corpus gets regardless of file
    layout (core/chunking/structure_aware.py). Neither requires a corpus to
    be organized as one numbered file per citable unit in the first place.

    Returns None when source_code is present but both article_no and
    structural_path are absent (e.g. chunks ingested before this metadata
    existed, or fixed-size chunking with no structure tree at all) —
    degrades gracefully rather than raising, so old runs/corpora don't
    crash new evaluation.

    Found live: a general document corpus (technical manuals, no external
    document-code numbering scheme at all) has NO source_code concept
    whatsoever — every single chunk hit the old `if not source_code: return
    None` and every question generated against it ended up with a
    completely empty article_refs, universally classified "out_of_scope"
    regardless of whether the corpus actually covers the content. `doc_id`
    (SourceRef's own required field — always present, unlike source_code)
    is the fallback identity for a corpus like this:
    '{doc_id}#{structural_path}' when available, else `doc_id` alone.

    Ambiguity worth flagging (two, now): (1) unlike article_no (a stable
    citation that rarely changes once a law is published), structural_path
    is derived from the document's own heading titles at ingest time —
    re-ingesting a manual after its headings were edited/reorganized can
    shift a chunk's structural_path, changing its ref id even though "the
    same section" still exists. A golden dataset's structural_path-based
    article_refs can go stale after a corpus re-ingest in a way
    article_no-based ones don't; there's no version-pinning here, just the
    plain current path string. (2) a bare `doc_id` ref (no structural_path
    at all) is document-level, not section-level — coarser than every other
    tier here. recall_at_k against a doc_id-only ref credits ANY chunk from
    the same document, even the wrong section of a large multi-section
    manual — a real, accepted precision/recall trade-off for corpora with
    no finer-grained identity available at all, not a bug.
    """
    source_code = source_ref.get("source_code")
    article_no = source_ref.get("article_no")
    structural_path = source_ref.get("structural_path")
    if source_code:
        if article_no:
            return f"{source_code}/{article_no}"
        if structural_path:
            return f"{source_code}#{structural_path}"
        return None
    doc_id = source_ref.get("doc_id")
    if not doc_id:
        return None
    return f"{doc_id}#{structural_path}" if structural_path else doc_id


def extracted_refs(source_refs: list[dict[str, Any]]) -> list[str]:
    """Map a ranked list of source_refs to ranked ref-id strings, dropping
    any that can't be resolved (see extract_ref_id)."""
    return [ref_id for sr in source_refs if (ref_id := extract_ref_id(sr)) is not None]
