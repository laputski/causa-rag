"""Every test tree must collect, including the suites that do not run by default.

`make test` walks tests/unit and tests/contract; the end-to-end, integration and
evaluation suites are filtered out by marker. An import error in those is
therefore invisible to the ordinary round: it waits for the day the suite is
needed, and on that day it turns out the suite has not run for a week.

Which is what happened. Realm cleanup moved out of conftest.py into a
neighbouring module, the import became relative, and `tests/e2e/` was not a
package, so the whole end-to-end suite failed at collection. No green run said
so.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


# @lat: [[fitness#Fitness Functions#Дерево тестов собирается целиком]]
@pytest.mark.fitness
def test_every_test_file_imports() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-m", "", "-p", "no:cacheprovider"],
        cwd=_ROOT, capture_output=True, text=True, timeout=300,
    )
    # Collecting is not running: no test here executes and no service is
    # needed. All this checks is that every file and every conftest imports.
    assert result.returncode == 0, (
        "the test tree does not collect:\n"
        + (result.stdout or "")[-3000:] + (result.stderr or "")[-2000:]
    )


@pytest.mark.fitness
def test_the_suites_that_need_the_stack_are_not_selected_by_default() -> None:
    """A suite whose fixtures skip when the stack is down reads as passing.

    `proving_ground` was declared as a marker and left out of the default
    deselection, so a fresh clone in CI collected all twenty-eight of its
    tests: twenty-six skipped on the fixture that checks the stack, which
    nobody reads, and the two that reach the index directly failed with a
    refused connection. The build went red for the honest reason and the
    twenty-six were green for a dishonest one.

    This checks the selection, not the fixtures. A marked suite must not be
    collected when pytest runs with the options a person gets by typing
    `pytest`.
    """
    for suite in ("tests/proving_ground", "tests/e2e", "tests/integration"):
        if not (_ROOT / suite).is_dir():
            continue
        result = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            cwd=_ROOT, capture_output=True, text=True, timeout=300,
        )
        selected = [line for line in (result.stdout or "").splitlines()
                    if line.startswith(f"{suite}/")]
        assert selected == [], (
            f"{suite} is selected by the default options, so it runs wherever the stack "
            f"is not up: {selected[:3]}"
        )
