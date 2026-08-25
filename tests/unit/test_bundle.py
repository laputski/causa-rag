"""core/bundle —.

The bundle is what lets a correction reach a served system without the
platform being in its request path. These tests pin the properties that
distinguish it from the store lookup that was removed: it names content
rather than positions, it is resolved once at load, it carries text rather
than vectors, and it says out loud what it could not resolve.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.bundle import Anchor, FixEntry, load_bundle, resolve_anchors
from core.bundle.anchor import resolution_report, text_hash
from core.bundle.bundle import UnsupportedBundleFormat, bundle_from_judgments, receive_bundle
from core.bundle.calibration import calibrate_threshold
from core.judgments import JudgedChunk, RelevanceJudgment


class _Chunk:
    def __init__(self, chunk_id: str, ref_id: str, text: str) -> None:
        self.chunk_id, self.ref_id, self.text = chunk_id, ref_id, text


# ── The anchor names content, not a position ─────────────────


def test_the_same_text_hashes_the_same_despite_formatting_drift() -> None:
    """Two systems that ingested one document rarely produce byte-identical
    text. Hashing raw text would fail anchors for reasons unrelated to
    meaning — the fragility the anchor exists to avoid."""
    assert text_hash("Article  44.\n\nAppeal  deadline") == text_hash("article 44. appeal deadline")


def test_an_anchor_survives_a_chunk_id_that_changed_meaning() -> None:
    """The position-identifier debt this closes: re-indexing gives the same
    text a different chunk_id, and an anchor still finds it."""
    anchor = Anchor.from_chunk("S/44", "Ch 1", "the article text")
    resolutions = resolve_anchors([anchor], [_Chunk("completely-new-id", "S/44", "the article text")])
    assert resolutions[0].chunk_id == "completely-new-id"
    assert resolutions[0].match == "exact"


def test_a_recipient_that_chunked_differently_still_matches_the_unit() -> None:
    """`unit` is information rather than failure: it tells the reader their
    chunking differs from the publisher's."""
    anchor = Anchor.from_chunk("S/44", "Ch 1", "the original text in full")
    resolutions = resolve_anchors([anchor], [_Chunk("c1", "S/44", "only half of the text")])
    assert resolutions[0].match == "unit"
    assert resolutions[0].chunk_id == "c1"


def test_an_anchor_matching_nothing_is_reported_not_dropped() -> None:
    """A bundle that silently applies half of itself is worse than one that
    says so."""
    anchor = Anchor.from_chunk("GONE/1", "", "no text")
    report = resolution_report(resolve_anchors([anchor], [_Chunk("c1", "S/44", "something else")]))
    assert report["counts"]["unresolved"] == 1
    assert report["resolved_share"] == 0.0
    assert report["unresolved"][0]["ref_id"] == "GONE/1"


def test_the_report_carries_counts_beside_the_share() -> None:
    # Four anchors at 75% and four hundred at 75% call for different
    # reactions, and a percentage hides which you have.
    anchors = [Anchor.from_chunk(f"S/{i}", "", f"text {i}") for i in range(4)]
    chunks = [_Chunk(f"c{i}", f"S/{i}", f"text {i}") for i in range(3)]
    report = resolution_report(resolve_anchors(anchors, chunks))
    assert report["total"] == 4
    assert report["counts"]["exact"] == 3
    assert report["resolved_share"] == 0.75


# ── The artefact ─────────────────────────────────────────────


def _judgment(**overrides) -> RelevanceJudgment:
    base = dict(
        id="j1", realm_id="acme", corpus_id="handbook_01", question="What is the deadline?",
        relevant=(JudgedChunk(chunk_id="c1", ref_id="S/44", structural_path="Ch 1", text="some text"),),
        irrelevant=(),
    )
    base.update(overrides)
    return RelevanceJudgment(**base)


def test_a_bundle_carries_text_rather_than_vectors() -> None:
    """An embedding belongs to one model; shipping one would expire the
    artefact the day the recipient changed models."""
    data = bundle_from_judgments([_judgment()], "acme", "handbook_01").to_dict()
    entry = data["entries"][0]
    assert entry["question"] == "What is the deadline?"
    assert not any("vec" in k for k in entry)


def test_a_retired_judgment_is_not_published() -> None:
    """Shipping it would apply a correction its own author withdrew."""
    bundle = bundle_from_judgments([_judgment(status="retired")], "acme", "handbook_01")
    assert bundle.entries == ()


def test_a_judgment_ruling_on_nothing_is_not_published() -> None:
    bundle = bundle_from_judgments([_judgment(relevant=(), irrelevant=())], "acme", "handbook_01")
    assert bundle.entries == ()


def test_the_normalised_question_travels_so_a_repeat_needs_no_embedding() -> None:
    data = bundle_from_judgments([_judgment(question="  What  is the DEADLINE? ")], "acme", "d").to_dict()
    assert data["entries"][0]["question_normalised"] == "what is the deadline?"


def test_a_bundle_round_trips_through_a_file(tmp_path: Path) -> None:
    bundle = bundle_from_judgments([_judgment()], "acme", "handbook_01", created_at="2026-08-04")
    path = tmp_path / "bundle.json"
    bundle.write(path)
    loaded = load_bundle(path)
    assert loaded.corpus_id == "handbook_01"
    assert loaded.entries[0].relevant[0].ref_id == "S/44"


def test_an_unrecognised_format_refuses_the_whole_document(tmp_path: Path) -> None:
    """Everything else here degrades honestly, because the cost is a missing
    feature. Here the cost is applying corrections the recipient misread,
    which changes answers."""
    path = tmp_path / "bundle.json"
    path.write_text('{"format_version": 99, "entries": []}', encoding="utf-8")
    with pytest.raises(UnsupportedBundleFormat):
        load_bundle(path)


# ── The recipient ────────────────────────────────────────────


def test_a_bundle_is_resolved_once_into_the_recipients_own_ids() -> None:
    """Resolution at load, never per query: that is the whole difference
    from the store lookup that was removed."""
    bundle = bundle_from_judgments([_judgment()], "acme", "handbook_01")
    received = receive_bundle(bundle, [_Chunk("their-own-id", "S/44", "some text")])
    assert received["entries"][0]["relevant_chunk_ids"] == ["their-own-id"]
    assert received["report"]["resolved_share"] == 1.0


def test_a_recipient_missing_the_content_reports_it_rather_than_applying_nothing() -> None:
    bundle = bundle_from_judgments([_judgment()], "acme", "handbook_01")
    received = receive_bundle(bundle, [])
    assert received["entries"][0]["relevant_chunk_ids"] == []
    assert received["report"]["counts"]["unresolved"] == 1


# ── A procedure, not a number ────────────────────────────────


def test_the_threshold_sits_just_below_the_weakest_intended_match() -> None:
    cal = calibrate_threshold(positives=[0.91, 0.88], negatives=[0.40, 0.55])
    assert cal.separable is True
    assert cal.threshold == pytest.approx(0.87)


def test_overlapping_examples_yield_no_threshold_at_all() -> None:
    """Any threshold would either fire where it must not or fail where it
    must. Returning a best compromise would hide that — which is how a
    mechanism ends up firing on questions nobody intended."""
    cal = calibrate_threshold(positives=[0.70], negatives=[0.80])
    assert cal.separable is False
    assert cal.threshold is None
    assert cal.lowest_positive == 0.7
    assert cal.highest_negative == 0.8


def test_without_examples_no_threshold_is_asserted() -> None:
    assert calibrate_threshold([], [0.5]).threshold is None
    assert calibrate_threshold([0.9], []).threshold is None


def test_a_single_entry_bundle_carries_no_negative_examples() -> None:
    """Found live: a flat list of questions put a one-entry bundle's own
    question into its "must not match" set, telling a recipient a question
    must not match itself. No threshold satisfies that, so calibration would
    have declared the mechanism unusable on a bundle that was fine."""
    from core.bundle.calibration import calibration_material

    material = calibration_material([FixEntry(id="j1", question="What is the deadline?")])
    assert material["must_not_match_pairs"] == []


def test_two_entries_are_negatives_for_each_other() -> None:
    from core.bundle.calibration import calibration_material

    material = calibration_material([
        FixEntry(id="j1", question="What is the deadline?"),
        FixEntry(id="j2", question="What are the dimensions?"),
    ])
    assert material["must_not_match_pairs"] == [{"a": "What is the deadline?", "b": "What are the dimensions?"}]


def test_paraphrases_are_never_invented_by_the_publisher() -> None:
    """Without positives a recipient can establish an upper bound and no
    more; inventing them would put a guess into an artefact meant to carry
    only observations."""
    from core.bundle.calibration import calibration_material

    assert calibration_material([FixEntry(id="j1", question="x")])["must_match"] == []
