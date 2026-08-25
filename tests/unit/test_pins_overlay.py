"""core/pins/overlay.py — pure pin-matching and injection logic. No I/O anywhere in this module; every fixture below
is a plain in-memory ScoredChunk/Chunk, mirroring test_paired_diff.py's
own "pure core logic, tested on fixtures" convention.
"""
from __future__ import annotations

from core.models import Chunk, ScoredChunk
from core.pins.overlay import Pin, apply_overlay, cosine_similarity, select_matching_pins


def _chunk(chunk_id: str, text: str = "text", **metadata) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc1", text=text, structural_path=chunk_id, metadata=metadata)


def _sc(chunk_id: str, score: float, **metadata) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, **metadata), score=score, retriever_id="hybrid")


class TestCosineSimilarity:
    def test_identical_vectors_are_maximally_similar(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors_are_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_are_negative_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_empty_vector_is_zero_not_a_crash(self) -> None:
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_mismatched_length_is_zero_not_a_crash(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_zero_magnitude_vector_is_zero_not_a_division_error(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestSelectMatchingPins:
    def test_pin_at_or_above_its_own_threshold_matches(self) -> None:
        pin = Pin(id="p1", query_vec=[1.0, 0.0], threshold=0.99)
        assert select_matching_pins([1.0, 0.0], [pin]) == [pin]

    def test_pin_below_its_own_threshold_does_not_match(self) -> None:
        pin = Pin(id="p1", query_vec=[0.0, 1.0], threshold=0.5)
        assert select_matching_pins([1.0, 0.0], [pin]) == []

    def test_multiple_pins_can_match_the_same_query_independently(self) -> None:
        p1 = Pin(id="p1", query_vec=[1.0, 0.0], threshold=0.9)
        p2 = Pin(id="p2", query_vec=[0.9, 0.1], threshold=0.9)
        matched = select_matching_pins([1.0, 0.0], [p1, p2])
        assert p1 in matched and p2 in matched


class TestApplyOverlay:
    def test_no_matching_pins_leaves_the_list_unchanged(self) -> None:
        scored = [_sc("c1", 0.9), _sc("c2", 0.5)]
        result = apply_overlay(scored, [])
        assert [sc.chunk.chunk_id for sc in result] == ["c1", "c2"]

    def test_original_list_is_never_mutated(self) -> None:
        scored = [_sc("c1", 0.9)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, demote_chunk_ids=["c1"])
        apply_overlay(scored, [pin])
        assert len(scored) == 1  # untouched

    def test_demote_removes_the_chunk_entirely(self) -> None:
        scored = [_sc("c1", 0.9), _sc("c2", 0.5)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, demote_chunk_ids=["c1"])
        result = apply_overlay(scored, [pin])
        assert [sc.chunk.chunk_id for sc in result] == ["c2"]

    def test_pin_injects_a_new_chunk_ranked_above_every_original(self) -> None:
        scored = [_sc("c1", 0.9), _sc("c2", 0.5)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, pin_chunks=[_chunk("pinned-1")])
        result = apply_overlay(scored, [pin])
        assert result[0].chunk.chunk_id == "pinned-1"
        assert result[0].score > 0.9

    def test_injected_chunk_is_tagged_pinned_with_its_pin_id(self) -> None:
        scored = [_sc("c1", 0.9)]
        pin = Pin(id="the-pin-id", query_vec=[], threshold=0.9, pin_chunks=[_chunk("pinned-1")])
        result = apply_overlay(scored, [pin])
        pinned = next(sc for sc in result if sc.chunk.chunk_id == "pinned-1")
        assert pinned.chunk.metadata["pinned"] is True
        assert pinned.chunk.metadata["pin_id"] == "the-pin-id"

    def test_pin_chunk_already_present_is_not_duplicated(self) -> None:
        scored = [_sc("c1", 0.9)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, pin_chunks=[_chunk("c1")])
        result = apply_overlay(scored, [pin])
        assert [sc.chunk.chunk_id for sc in result] == ["c1"]
        assert result[0].chunk.metadata.get("pinned") is not True  # the ORIGINAL c1, not injected

    def test_pin_and_demote_from_two_different_pins_both_apply(self) -> None:
        scored = [_sc("c1", 0.9), _sc("c2", 0.5)]
        demote_pin = Pin(id="p1", query_vec=[], threshold=0.9, demote_chunk_ids=["c2"])
        pin_pin = Pin(id="p2", query_vec=[], threshold=0.9, pin_chunks=[_chunk("pinned-1")])
        result = apply_overlay(scored, [demote_pin, pin_pin])
        ids = [sc.chunk.chunk_id for sc in result]
        assert "c2" not in ids
        assert "pinned-1" in ids
        assert "c1" in ids

    def test_rewrite_merges_extra_results_via_rrf(self) -> None:
        scored = [_sc("c1", 0.9)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, rewrite="a different phrasing")
        extra = [_sc("c-from-rewrite", 0.8)]
        result = apply_overlay(scored, [pin], rewrite_results={"p1": extra})
        ids = {sc.chunk.chunk_id for sc in result}
        assert "c-from-rewrite" in ids
        assert "c1" in ids

    def test_rewrite_set_but_no_results_supplied_is_silently_skipped(self) -> None:
        scored = [_sc("c1", 0.9)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, rewrite="a different phrasing")
        result = apply_overlay(scored, [pin], rewrite_results=None)
        assert [sc.chunk.chunk_id for sc in result] == ["c1"]

    def test_rewrite_overlapping_chunk_is_deduplicated_not_doubled(self) -> None:
        scored = [_sc("c1", 0.9)]
        pin = Pin(id="p1", query_vec=[], threshold=0.9, rewrite="a different phrasing")
        extra = [_sc("c1", 0.7)]  # same chunk the original retrieval already found
        result = apply_overlay(scored, [pin], rewrite_results={"p1": extra})
        assert [sc.chunk.chunk_id for sc in result] == ["c1"]
