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
