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


def _referenced_images(markdown: Path) -> list[str]:
    """Every local image path a markdown file points at.

    Both spellings are read: the markdown form and the HTML tag, because the
    README uses the tag wherever it needs a width and the markdown form
    elsewhere. Remote addresses are left alone; whether a shield renders is the
    business of the service that draws it.
    """
    import re

    text = markdown.read_text(encoding="utf-8")
    found = re.findall(r'<img[^>]+src="([^"]+)"', text)
    found += [m for _, m in re.findall(r"!\[([^\]]*)\]\(([^)\s]+)", text)]
    return [p for p in found if not p.startswith(("http://", "https://", "data:"))]


def test_every_image_the_readme_points_at_exists() -> None:
    """The first page a visitor sees carries these, and a missing one does not
    fail loudly: the browser declines to draw it and moves on.

    That has already happened here once, with a malformed wordmark, and it was
    reported by a person and not by a test. This is the other half of the same
    accident: a file renamed or removed while the reference stayed.
    """
    readme = ROOT / "README.md"
    referenced = _referenced_images(readme)
    assert referenced, "the README points at no local image at all, which it used to"
    missing = sorted(p for p in referenced if not (ROOT / p).is_file())
    assert missing == [], f"README points at files that are not here: {missing}"


def test_no_published_asset_is_pointed_at_by_nothing() -> None:
    """The other direction. An asset nothing references is either a leftover or
    a reference somebody forgot to write, and both are worth seeing.

    The social preview and the two logo variants are named by the code host and
    by the interface, and by no markdown file, so they are listed here as
    the exceptions they are.
    """
    named_elsewhere = {"social-preview.png", "logo.svg", "logo-dark.svg"}
    referenced = set()
    for markdown in sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md")):
        referenced |= {Path(p).name for p in _referenced_images(markdown)}
    stray = sorted(p.name for p in (ROOT / "docs" / "assets").iterdir()
                   if p.is_file() and p.name not in referenced | named_elsewhere)
    assert stray == [], f"assets nothing points at: {stray}"
