#!/usr/bin/env python3
"""Generate every file the mark appears in, from one layout.

Run after editing `mark_layout.py`. The component in `ui/src/components/Logo.tsx`
cannot import Python, so its rectangles are printed here for pasting: keeping
them in step is then a copy, never a redrawing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mark_layout import ACCENT, ACCENT_DARK, BOX, INK, INK_DARK, MODULE, cells  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs/assets"
PUBLIC = ROOT / "ui/public"

HEAD = """  <!-- The funnel the platform measures, with the thread that came through it:
       everything retrieved, what survived reranking, what reached the answer.
       The middle column in the accent colour is the part that made it all the
       way, and it is what makes the narrowing read as a funnel instead of as
       a tree.

       The module and the grid are the sibling registry's, module to pitch as
       0.8: the kinship is carried by the unit of measure, not by a shared
       palette, so each mark keeps its own colour and still reads as family. -->"""


def rects(indent: int = 2, thread_class: str = "thread", ink_class: str | None = None,
          thread_fill: str | None = None, ink_fill: str | None = None) -> str:
    """Every module as a rect, with exactly one colour attribute each.

    The attribute is chosen here and never appended afterwards. Appending is
    what produced a duplicate `class` once already: the thread's class name
    changed and the guard in the post-processing step still tested the old one,
    so every rect took a second class and the file stopped being valid XML.
    """
    pad = " " * indent
    out = []
    for x, y, is_thread in cells():
        if is_thread:
            attr = f' fill="{thread_fill}"' if thread_fill else f' class="{thread_class}"'
        elif ink_fill:
            attr = f' fill="{ink_fill}"'
        elif ink_class:
            attr = f' class="{ink_class}"'
        else:
            attr = ""
        out.append(f'{pad}<rect x="{x}" y="{y}" width="{MODULE}" height="{MODULE}"{attr}/>')
    return "\n".join(out)


def write_logo() -> None:
    style = f"""  <!-- A class, not `:root`: inlined into markup, `:root` reaches the
       containing document and colours far more than the mark. -->
  <style>
    .cr-mark {{ color: {INK} }}
    .cr-mark .thread {{ fill: {ACCENT} }}
    @media (prefers-color-scheme: dark) {{
      .cr-mark {{ color: {INK_DARK} }}
      .cr-mark .thread {{ fill: {ACCENT_DARK} }}
    }}
  </style>"""
    (ASSETS / "logo.svg").write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"
     class="cr-mark" role="img" aria-label="Causa RAG">
  <title>Causa RAG</title>
{HEAD}
{style}
  <g fill="currentColor">
{rects()}
  </g>
</svg>
""", encoding="utf-8")

    # The dark twin carries its colours outright, for the places a media query
    # never reaches: mail attachments, a README on a dark ground.
    (ASSETS / "logo-dark.svg").write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"
     role="img" aria-label="Causa RAG">
  <title>Causa RAG</title>
{HEAD}
  <g>
{rects(thread_fill=ACCENT_DARK, ink_fill=INK_DARK)}
  </g>
</svg>
""", encoding="utf-8")
    print("docs/assets/logo.svg, logo-dark.svg")


def write_wordmark() -> None:
    path = ASSETS / "wordmark.svg"
    s = path.read_text(encoding="utf-8")
    block = re.search(r'  <g transform="translate\(2, 6\) scale\(1\.5\)">.*?\n  </g>', s, re.S)
    if not block:
        raise SystemExit("wordmark.svg: mark group not found")
    body = rects(indent=4, thread_class="thread", ink_class="fg")
    s = s.replace(block.group(0),
                  '  <g transform="translate(2, 6) scale(1.5)">\n' + body + "\n  </g>")
    path.write_text(s, encoding="utf-8")
    print("docs/assets/wordmark.svg")


def write_favicon() -> None:
    """The tab icon, and the touch icon that cannot be transparent.

    The interface had no icon at all: the tab showed the browser's default
    globe. That was reasonable while there was nothing to put there.

    The tab icon follows the theme, because the browser paints the tab in it.
    The touch icon does not: a home screen gives it no ground, so its colours
    are fixed and it carries a plate of its own.
    """
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "favicon.svg").write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"
     class="cr-mark" role="img" aria-label="Causa RAG">
  <title>Causa RAG</title>
  <style>
    .cr-mark {{ color: {INK} }}
    .cr-mark .thread {{ fill: {ACCENT} }}
    @media (prefers-color-scheme: dark) {{
      .cr-mark {{ color: {INK_DARK} }}
      .cr-mark .thread {{ fill: {ACCENT_DARK} }}
    }}
  </style>
  <g fill="currentColor">
{rects()}
  </g>
</svg>
""", encoding="utf-8")

    # A rounded plate with room around the mark: a home screen crops the corners
    # and an icon drawn to the edge loses them.
    pad, box = 24, 180
    inner = box - pad * 2
    scale = inner / BOX
    body = "\n".join(
        f'  <rect x="{pad + x * scale:.1f}" y="{pad + y * scale:.1f}"'
        f' width="{MODULE * scale:.1f}" height="{MODULE * scale:.1f}"'
        f' fill="{ACCENT_DARK if t else INK_DARK}"/>'
        for x, y, t in cells())
    (PUBLIC / "apple-touch-icon.svg").write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {box} {box}"
     width="{box}" height="{box}" role="img" aria-label="Causa RAG">
  <title>Causa RAG</title>
  <rect width="{box}" height="{box}" rx="{box * 0.22:.0f}" fill="#0e1013"/>
{body}
</svg>
""", encoding="utf-8")
    # Safari takes the touch icon reliably only as a raster, so the same
    # geometry is drawn twice, because relying on a conversion failed once.
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ui/public/favicon.svg, apple-touch-icon.svg (PIL missing, no PNG)")
        return

    def rgb(v: str) -> tuple[int, int, int]:
        return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))

    png = Image.new("RGB", (box, box), rgb("#0e1013"))
    draw = ImageDraw.Draw(png)
    radius = int(box * 0.22)
    draw.rounded_rectangle([0, 0, box - 1, box - 1], radius=radius, fill=rgb("#0e1013"))
    for x, y, t in cells():
        draw.rectangle(
            [pad + x * scale, pad + y * scale,
             pad + (x + MODULE) * scale - 1, pad + (y + MODULE) * scale - 1],
            fill=rgb(ACCENT_DARK if t else INK_DARK))
    png.save(PUBLIC / "apple-touch-icon.png")
    print("ui/public/favicon.svg, apple-touch-icon.svg, apple-touch-icon.png")


def print_tsx() -> None:
    print("\npaste into ui/src/components/Logo.tsx, Cascade()")
    for x, y, is_thread in cells():
        fill = 'var(--color-primary)' if is_thread else 'currentColor'
        print(f'      <rect x="{x}" y="{y}" width="{MODULE}" height="{MODULE}" fill="{fill}" />')


if __name__ == "__main__":
    write_logo()
    write_favicon()
    write_wordmark()
    print_tsx()
