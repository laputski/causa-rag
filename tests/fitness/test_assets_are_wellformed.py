"""Every asset the repository publishes must parse as XML.

A malformed SVG does not fail loudly. Nothing here reads these files: they are
served to a browser, embedded in the README, or handed to code hosting for the
tab icon, and each of those simply declines to draw and moves on. The
repository stays green, and the first page a visitor sees carries a broken
image.

Which is what happened. The mark's generator appended a `class` attribute to
every rectangle after drawing it, and when the thread's class was renamed the
guard in that step still tested the old name, so the thread took a second
`class`, the wordmark stopped being valid XML, and the README's first image
stopped rendering. It was reported by a person, not by a test.

The generator no longer appends anything; it writes one colour attribute per
rectangle. This test guards the outcome, not that particular mistake:
whatever a future generator does, the file it writes has to parse.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where published assets live. Anything under these is served or embedded
#: somewhere and nobody reads it back, so nobody would notice it break.
PUBLISHED = ("docs/assets", "ui/public")


def _svgs() -> list[Path]:
    found: list[Path] = []
    for folder in PUBLISHED:
        found.extend(sorted((ROOT / folder).rglob("*.svg")))
    return found


def test_there_are_assets_to_check() -> None:
    """A test that checks nothing passes for the wrong reason.

    If the folders move, this fails instead of quietly guarding an empty set.
    """
    assert _svgs(), f"no SVG found under {', '.join(PUBLISHED)}"


@pytest.mark.parametrize("path", _svgs(), ids=lambda p: str(p.relative_to(ROOT)))
def test_svg_parses(path: Path) -> None:
    try:
        ElementTree.parse(path)
    except ElementTree.ParseError as error:
        pytest.fail(f"{path.relative_to(ROOT)} is not well-formed XML: {error}")
