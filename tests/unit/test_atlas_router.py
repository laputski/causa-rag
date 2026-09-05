"""The catalogue, as the interface receives it.

Three properties are worth a test of their own, and each is a decision the
router makes and not a shape it happens to have.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from core.eval.atlas import FAILURES
from services.api_gateway.routers.atlas import (
    attach_failure_ids,
    read_atlas,
    read_failure,
    read_signals,
)


def _run(coro):
    return asyncio.run(coro)


def test_without_an_architecture_applicability_is_left_unanswered() -> None:
    """Applicability is a question about an architecture. Asked without one, the
    honest answer is none, and a `false` there would read as "ruled out"."""
    payload = _run(read_atlas())
    assert len(payload["entries"]) == len(FAILURES)
    assert {e["applies_here"] for e in payload["entries"]} == {None}


def test_a_coordinate_the_point_does_not_record_is_not_a_denial() -> None:
    """The distinction the whole scope mechanism rests on.

    True, false and null are three answers. Null says the point records nothing
    about a coordinate the entry depends on, so the failure is neither possible
    nor excluded. Folding it into false reports an exclusion nobody established.
    """
    payload = _run(read_atlas(point="dense"))
    verdicts = {e["applies_here"] for e in payload["entries"]}
    assert verdicts <= {True, False, None}
    assert True in verdicts, "no entry applies to a dense system, which cannot be right"


def test_the_gap_in_coverage_is_reported_and_not_left_silent() -> None:
    """A coordinate no entry speaks about is the state the graph point is in,
    and nothing anywhere used to say so."""
    payload = _run(read_atlas(point="graph"))
    codes = {g["code"] for g in payload["uncovered_coordinates"]}
    assert codes, (
        "the graph point is reported as fully covered, which would mean the "
        "catalogue has entries on graph coordinates. It has none."
    )
    for gap in payload["uncovered_coordinates"]:
        assert gap["url"].startswith("https://"), "a gap names no place to read about it"
        assert gap["dimension"], f"{gap['code']} resolves to no dimension"


def test_an_unknown_architecture_is_refused_and_the_known_ones_are_named() -> None:
    with pytest.raises(HTTPException) as raised:
        _run(read_atlas(point="quantum"))
    assert raised.value.status_code == 404
    assert "dense" in str(raised.value.detail), "the refusal does not say what would work"


def test_a_signal_standing_for_two_entries_says_so() -> None:
    """A signal that is evidence for two failures is evidence for neither in
    particular, and a reader not told that takes it for evidence about the entry
    in front of them."""
    payload = _run(read_signals())
    shared = [s for s in payload["signals"] if not s["singles_out"]]
    assert shared, "no signal is shared, which contradicts the catalogue's own declarations"
    for signal in shared:
        assert len(signal["failures"]) > 1


def test_every_signal_the_catalogue_names_is_listed() -> None:
    listed = {s["id"] for s in _run(read_signals())["signals"]}
    named = {s.id for f in FAILURES for s in f.signals}
    assert listed == named


def test_a_finding_is_joined_to_the_entries_it_evidences() -> None:
    items = [{"id": "duplicates"}, {"id": "bm25_dominance"}, {"id": "no_such_detector"}]
    attach_failure_ids(items, "detector")
    assert items[0]["failure_ids"] == ["F01"]
    assert items[1]["failure_ids"] == ["F17", "F21", "F40"], (
        "a finding standing for two entries must name both"
    )
    assert items[2]["failure_ids"] == [], (
        "a finding no entry names keeps an empty list: the platform is saying "
        "something the catalogue has no place for, and that is worth seeing"
    )


def test_the_join_is_read_from_the_catalogue_and_not_kept_beside_it() -> None:
    """A bait for the join. If it were a second hand-kept table, adding an entry
    would not change what a finding points at."""
    from core.eval.atlas import signal_index

    assert signal_index()["detector:duplicates"] == ["F01"]
    items = [{"id": "duplicates"}]
    attach_failure_ids(items, "detector")
    assert items[0]["failure_ids"] == signal_index()["detector:duplicates"]


def test_one_failure_is_readable_on_its_own() -> None:
    entry = _run(read_failure("F03"))
    assert entry["id"] == "F03"
    assert entry["severity"]["total"] == (
        entry["severity"]["quiet"] * entry["severity"]["cost"] * entry["severity"]["prevalence"]
    )
    assert entry["bait_level"] in ("unit", "proving_ground", None)


def test_an_unknown_failure_is_a_refusal_and_not_an_empty_answer() -> None:
    with pytest.raises(HTTPException) as raised:
        _run(read_failure("F99"))
    assert raised.value.status_code == 404


def test_the_schema_block_carries_both_dates() -> None:
    """The release the copy follows, and the day its values were checked.

    They differ whenever the published data has been rebuilt since the last
    release was cut, and on the day this was written they did: the copy holds
    two values the named release predates. One date cannot say both things, and
    a reader shown only the older one reads a copy as staler than it is.
    """
    from core.eval import rag_space

    payload = asyncio.run(read_atlas())
    assert payload["schema"]["release"] == rag_space.SCHEMA_RELEASE
    assert payload["schema"]["verified_on"] == rag_space.SCHEMA_VERIFIED_ON
