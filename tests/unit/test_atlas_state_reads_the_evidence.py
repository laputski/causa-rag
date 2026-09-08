"""An entry's state stops guessing about the proving ground and reads it.

The rule was written when the proving ground did not exist. It said: a bait
under `tests/proving_ground/` waits on a stand nobody has built, so nobody has
watched it fire, so the claim is unproven. Both halves of that were true then.

The stand exists now and every pair that runs on it files what it measured, so
the second half stopped being true without the sentence changing. Two entries
were reproduced on a live stack, with the numbers on file beside them, and the
report and the interface both went on telling a reader that nobody had proved
them. The report at least printed the count of reproductions underneath;
the interface showed the word alone.

So the word is still decided in one place, and the one fact an entry cannot
know about itself is asked for: whether a run happened. A caller with no
evidence passes False and gets the old answer for the honest reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.eval import proving_ground
from core.eval.atlas import FAILURES, get

ROOT = Path(__file__).resolve().parents[2]


def _with_a_proving_ground_bait() -> list:
    return [f for f in FAILURES if f.bait.startswith("tests/proving_ground/")]


def test_a_bait_that_only_the_stand_runs_is_unproven_until_a_run_says_otherwise() -> None:
    entries = _with_a_proving_ground_bait()
    assert entries, "no entry keeps its bait on the proving ground, so this checks nothing"
    for failure in entries:
        assert failure.state_given(staged=False) == "unproven"
        assert failure.state_given(staged=True) in ("caught", "visible")


def test_a_bait_the_build_runs_is_never_unproven() -> None:
    """A unit bait runs on every commit, so the evidence is the build itself
    and the answer does not depend on what any stand filed."""
    for failure in FAILURES:
        if failure.detection == "none" or failure.bait.startswith("tests/proving_ground/"):
            continue
        assert failure.state_given(staged=False) == failure.state_given(staged=True)
        assert failure.state_given(staged=False) != "unproven"


def test_an_entry_nothing_catches_says_so_whatever_the_stand_saw() -> None:
    """Staging a failure nothing detects is worth doing and does not turn it
    into a detected one: the pair proves the defect is there and the silence
    is the finding."""
    for failure in FAILURES:
        if failure.detection != "none":
            continue
        assert failure.state_given(staged=True) == "none"
        assert failure.state_given(staged=False) == "none"


def test_the_evidence_is_read_from_the_files_a_run_writes(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proving_ground, "EVIDENCE", tmp_path)
    assert proving_ground.reproduced("F01") is False

    (tmp_path / "F01.json").write_text(json.dumps({"reproduced": True}), encoding="utf-8")
    assert proving_ground.reproduced("F01") is True

    # A pair that measured why an entry cannot be staged files a record too,
    # and counting it would put the entry in the reproduced column for having
    # proved that it could not be reproduced.
    (tmp_path / "F02.json").write_text(json.dumps({"reproduced": False}), encoding="utf-8")
    assert proving_ground.reproduced("F02") is False

    # A file nobody can read is still a run that happened. Reading it as
    # "nobody looked" is the one answer that is certainly wrong.
    (tmp_path / "F03.json").write_text("{ not json", encoding="utf-8")
    assert proving_ground.reproduced("F03") is True


def test_the_two_readers_of_the_state_give_one_answer() -> None:
    """The drift this rule was moved into the catalogue to stop. It came back
    in another shape: one reader knew about the stand and the other did not,
    so they disagreed about two entries without either of them being wrong
    about the rule."""
    import asyncio

    from services.api_gateway.routers.atlas import read_atlas
    from tools.atlas_report import rows

    served = {e["id"]: e["state"] for e in asyncio.run(read_atlas())["entries"]}
    for point in ("hybrid", "dense", "graph"):
        for row in rows(point):
            printed = row["state"].removesuffix(" (unstaged)")
            assert printed == served[row["id"]], (
                f"{row['id']}: the report says {printed!r} and the interface says "
                f"{served[row['id']]!r}"
            )


def test_the_entries_this_was_found_on_are_no_longer_called_unproven() -> None:
    """Named, because they are what the fix was measured against: both are
    reproduced on the stand with their numbers filed, and both were being
    shown as claims nobody had proved."""
    for failure_id in ("F19", "F26"):
        failure = get(failure_id)
        assert failure is not None
        assert proving_ground.reproduced(failure_id), (
            f"{failure_id} is no longer reproduced on the stand, so this test now checks "
            "nothing; pick two entries that are, or delete it"
        )
        assert failure.state_given(proving_ground.reproduced(failure_id)) != "unproven"
