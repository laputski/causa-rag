"""core/eval/ref_resolution.py.

Pins the two properties the whole rework rests on: a ref index built from
indexed chunks agrees with what retrieval metrics call "the same source
unit", and inability to check is reported as `unknown` rather than as
`absent`.
"""
from __future__ import annotations

from core.eval.ref_resolution import (
    IndexRefResolver,
    UnknownRefResolver,
    build_ref_index,
    ref_source_from_chunk,
    resolver_from_chunks,
)
from core.models import Chunk


def _chunk(doc_id="d1", structural_path="", source_code=None, article_no=None) -> Chunk:
    metadata = {}
    if source_code:
        metadata["source_code"] = source_code
    if article_no:
        metadata["article_no"] = article_no
    return Chunk(doc_id=doc_id, text="x", structural_path=structural_path, metadata=metadata)


# ── Flattening: source_code/article_no live in metadata, not top level ───

def test_flattens_source_code_and_article_no_out_of_chunk_metadata() -> None:
    # The lift core/pipeline.py#_to_source_refs performs when building
    # SourceRefs — mirrored so a chunk read straight from the index yields
    # the same ref id it would after passing through the pipeline.
    src = ref_source_from_chunk(_chunk(source_code="SRC001", article_no="44"))
    assert src["source_code"] == "SRC001"
    assert src["article_no"] == "44"


def test_accepts_plain_dicts_so_tests_need_no_model_objects() -> None:
    src = ref_source_from_chunk({"doc_id": "d1", "metadata": {"source_code": "S", "article_no": "1"}})
    assert (src["source_code"], src["article_no"]) == ("S", "1")


# ── Index construction agrees with extract_ref_id's three tiers ──────────

def test_index_uses_the_same_ref_ids_retrieval_metrics_use() -> None:
    ids = build_ref_index([
        _chunk(source_code="SRC001", article_no="44"),          # tier 1
        _chunk(doc_id="D1", structural_path="Ch 2 > Art 10"),        # tier 2 (no source_code)
        _chunk(doc_id="D2"),                                        # tier 3 (bare doc_id)
    ])
    assert ids == {"SRC001/44", "D1#Ch 2 > Art 10", "D2"}


def test_chunks_with_no_identity_are_skipped_not_stored_as_empty() -> None:
    # A chunk yielding no ref id must not make an unrelated ref look present.
    ids = build_ref_index([Chunk(doc_id="", text="x")])
    assert ids == frozenset()


def test_duplicate_chunks_of_one_unit_collapse_to_one_ref_id() -> None:
    # One article usually spans several chunks; the index is a set.
    ids = build_ref_index([
        _chunk(source_code="S", article_no="1"),
        _chunk(source_code="S", article_no="1"),
    ])
    assert ids == {"S/1"}


# ── Presence semantics ──────────────────────────────────────────────────

def test_present_and_absent_against_a_built_index() -> None:
    resolver = resolver_from_chunks([_chunk(source_code="S", article_no="1")])
    assert resolver.presence("S/1") == "present"
    assert resolver.presence("S/2") == "absent"
    assert resolver.checked is True


def test_unknown_resolver_never_says_absent() -> None:
    """The distinction the whole rework exists for: an unreadable index
    must not be indistinguishable from a genuinely missing document."""
    resolver = UnknownRefResolver(reason="index unreachable")
    assert resolver.presence("anything") == "unknown"
    assert resolver.checked is False


def test_empty_index_reports_absent_not_unknown() -> None:
    # A genuinely empty (but readable) index IS a real coverage gap, and is
    # reported as such — the honest counterpart to the case above.
    resolver = IndexRefResolver(known_ref_ids=frozenset())
    assert resolver.presence("S/1") == "absent"
    assert resolver.checked is True


# ── How many chunks a source unit occupies ───────────────────

def test_chunk_counts_are_built_in_the_same_pass_as_the_index() -> None:
    resolver = resolver_from_chunks([
        _chunk(source_code="S", article_no="1"),
        _chunk(source_code="S", article_no="1"),
        _chunk(source_code="S", article_no="1"),
        _chunk(source_code="S", article_no="2"),
    ])
    assert resolver.chunk_count("S/1") == 3
    assert resolver.chunk_count("S/2") == 1


def test_an_uncounted_resolver_reports_none_rather_than_zero() -> None:
    """None and 0 are different answers: None means nobody counted, 0 would
    mean the unit is not indexed at all, which `presence` already says."""
    resolver = IndexRefResolver(known_ref_ids=frozenset({"S/1"}))
    assert resolver.chunk_count("S/1") is None


def test_a_ref_absent_from_a_counted_index_reports_zero() -> None:
    resolver = resolver_from_chunks([_chunk(source_code="S", article_no="1")])
    assert resolver.chunk_count("S/9") == 0


def test_counting_does_not_change_which_refs_are_present() -> None:
    chunks = [_chunk(source_code="S", article_no="1"), _chunk(doc_id="D2")]
    assert build_ref_index(chunks) == resolver_from_chunks(chunks).known_ref_ids
