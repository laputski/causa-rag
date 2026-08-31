"""tools/check_style.py: which em dashes count as the tic and which do not.

The counter is a ratchet, so a pattern that mistakes typography for the tic
inflates the budget permanently and one that mistakes the tic for typography
lets it through. Both directions are pinned here.
"""
from __future__ import annotations

import pytest

from tools.check_style import _em_dashes


@pytest.mark.parametrize("line", [
    "| value | — |",
    "  — a label opening a line",
    "the range 12—16 inclusive",
    "['Seed', run.config?.seed ?? '—'],",
    '("documents", state.get("files", "—")),',
    "return sec != null ? formatDuration(sec) : '—'",
])
def test_typography_is_not_counted(line: str) -> None:
    assert _em_dashes(line) == 0


@pytest.mark.parametrize("line", [
    "# the run is stored — and the file is only a backup",
    "see 'before' — 'after' for the pair",
    "A dash welding two clauses — this is the tic itself",
])
def test_a_connector_is_counted(line: str) -> None:
    assert _em_dashes(line) == 1


def test_a_placeholder_does_not_absolve_a_connector_on_the_same_line():
    # Subtraction is per occurrence, so the real connector still counts.
    assert _em_dashes("`${name} — ${endpoint ?? '—'}`") == 1
