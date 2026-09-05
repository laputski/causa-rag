"""A fixture the tests read has to survive a fresh clone.

`corpus/*/` is ignored, because a scraped corpus is large and re-downloadable.
Two corpora underneath it are neither: the demo handbook and the proving
ground's two base corpora are written by this repository, shipped with it, and
read by the unit tests. The demo one carries an exception in `.gitignore`; the
proving ground's did not, and a tree without them fails twenty-four tests and
errors on three more, measured by moving the directory aside.

The same applies to what a proving-ground bait records: `tools/atlas_report.py`
reads those files to say whether a catalogue entry has been reproduced on a
live stack, and a report whose evidence exists only on the machine that made it
answers a different question on every machine.

Checked by asking git, and not by reading `.gitignore`: the rules interact, and
which one wins is git's answer to give. Asked with `--no-index`, because git
never ignores a file it already tracks: without that flag the question becomes
"is this file committed", every one of these passes the moment it is, and the
rule could then be deleted without a single test noticing. Measured: with the
demo corpus's exception commented out, the plain form answered "not ignored"
and the `--no-index` form named the rule that would have excluded it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Paths the test suite reads and which therefore have to be committed. Each is
#: named with the test that would fail without it, so a reader can check the
#: claim instead of taking it on trust.
SHIPPED = [
    ("corpus/demo_handbook", "tests/unit/test_demo_realm.py"),
    ("corpus/proving-ground/base-ru", "tests/unit/test_proving_ground_corpora.py"),
    ("corpus/proving-ground/base-en", "tests/unit/test_proving_ground_corpora.py"),
    ("eval/golden/base-ru.v1.fast.jsonl", "tests/unit/test_proving_ground_corpora.py"),
    ("eval/golden/base-en.v1.fast.jsonl", "tests/unit/test_proving_ground_corpora.py"),
]


def _ignored(relative: str) -> bool:
    """Whether git would leave this path out of a clone."""
    path = ROOT / relative
    sample = path if path.is_file() else next((f for f in sorted(path.rglob("*")) if f.is_file()), None)
    if sample is None:
        pytest.fail(f"{relative} holds no files, so there is nothing to ship")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", str(sample)],
        capture_output=True, check=False, timeout=30,
    )
    return result.returncode == 0


@pytest.mark.parametrize(("relative", "read_by"), SHIPPED, ids=lambda v: v.split("/")[-1])
def test_a_fixture_the_tests_read_reaches_a_clone(relative: str, read_by: str) -> None:
    assert (ROOT / relative).exists(), f"{relative} is missing from this tree"
    assert not _ignored(relative), (
        f"{relative} is ignored, so a clone would not carry it and {read_by} would fail there"
    )


def test_the_evidence_of_a_live_run_reaches_a_clone() -> None:
    """Empty is allowed: nobody may have run the proving ground yet. Ignored is
    not, because then running it would produce evidence that never travels."""
    evidence = ROOT / "eval" / "results" / "proving_ground"
    if not evidence.is_dir():
        pytest.skip("NOT RUN: no proving-ground evidence in this tree yet")
    files = sorted(evidence.glob("*.json"))
    if not files:
        pytest.skip("NOT RUN: the evidence directory is empty")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", str(files[0])],
        capture_output=True, check=False, timeout=30,
    )
    assert result.returncode != 0, (
        "what a proving-ground bait recorded is ignored, so the report would answer "
        "differently on every machine"
    )


def test_a_scraped_corpus_is_still_left_out() -> None:
    """The rule the exceptions are exceptions to. Losing it would put hundreds
    of megabytes of downloaded text into the history."""
    scraped = ROOT / "corpus" / "miracl-ru"
    if not scraped.is_dir():
        pytest.skip("NOT RUN: no scraped corpus in this tree to check the rule against")
    assert _ignored("corpus/miracl-ru"), "a downloaded corpus is no longer ignored"
