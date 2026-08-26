#!/usr/bin/env python3
"""Generate every file the mark appears in, from one layout.

Run after editing `mark_layout.py`. The component in `ui/src/components/Logo.tsx`
cannot import Python, so its rectangles are printed here for pasting: keeping
them in step is then a copy rather than a redrawing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mark_layout import ACCENT, ACCENT_DARK, INK, INK_DARK, MODULE, cells  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs/assets"

HEAD = """  <!-- The funnel the platform measures, with the thread that came through it:
       everything retrieved, what survived reranking, what reached the answer.
       The middle column in the accent colour is the part that made it all the
       way, and it is what makes the narrowing read as a funnel rather than as
       a tree.

       The module and the grid are the sibling registry's, module to pitch as
       0.8: the kinship is carried by the unit of measure, not by a shared
       palette, so each mark keeps its own colour and still reads as family. -->"""


def rects(indent: int = 2, thread_class: str | None = "thread",
          thread_fill: str | None = None, ink: str | None = None) -> str:
    pad = " " * indent
    out = []
    for x, y, is_thread in cells():
        attr = ""
        if is_thread:
            attr = f' class="{thread_class}"' if thread_fill is None else f' fill="{thread_fill}"'
        elif ink is not None:
            attr = f' fill="{ink}"'
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
{rects(thread_fill=ACCENT_DARK, ink=INK_DARK)}
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
    body = rects(indent=4, thread_class="thread").replace('<rect', '<rect class="fg"' , 0)
    body = "\n".join(
        line if 'class="accent"' in line else line.replace("/>", ' class="fg"/>')
        for line in body.splitlines())
    s = s.replace(block.group(0),
                  '  <g transform="translate(2, 6) scale(1.5)">\n' + body + "\n  </g>")
    path.write_text(s, encoding="utf-8")
    print("docs/assets/wordmark.svg")


def print_tsx() -> None:
    print("\n— paste into ui/src/components/Logo.tsx, Cascade() —")
    for x, y, is_thread in cells():
        fill = 'var(--color-primary)' if is_thread else 'currentColor'
        print(f'      <rect x="{x}" y="{y}" width="{MODULE}" height="{MODULE}" fill="{fill}" />')


if __name__ == "__main__":
    write_logo()
    write_wordmark()
    print_tsx()
