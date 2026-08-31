#!/usr/bin/env python3
"""The social preview card, 1280x640, the size code hosting expects.

**The typeface is a substitute and that has to be known.** The project's own
faces ship as woff2, PIL cannot read them, and no converter is installed here.
Georgia stands in: the same class of text serif as Source Serif 4, and at this
size the difference does not carry. When a converter is available, this script
is the place to swap the face.

The mark comes from `mark_layout.py`, so the card cannot drift from the mark.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mark_layout import ACCENT_DARK, BOX, MODULE, cells  # noqa: E402

W, H = 1280, 640
BG, INK, MUTED = (14, 16, 19), (232, 234, 237), (155, 161, 172)
GEORGIA = "/System/Library/Fonts/Supplemental/Georgia"
MARK_PX = 190


def rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def draw_card() -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    canvas = ImageDraw.Draw(image)

    title = ImageFont.truetype(f"{GEORGIA} Bold.ttf", 82)
    lead = ImageFont.truetype(f"{GEORGIA}.ttf", 38)
    small = ImageFont.truetype(f"{GEORGIA}.ttf", 26)

    # Centred, not flush left: previews are cropped differently in
    # different places, and a block against the edge loses half of itself.
    content = MARK_PX + 64 + 640
    left, top = (W - content) // 2, 168

    scale = MARK_PX / BOX
    for x, y, is_thread in cells():
        canvas.rectangle(
            [left + x * scale, top + y * scale,
             left + (x + MODULE) * scale - 1, top + (y + MODULE) * scale - 1],
            fill=rgb(ACCENT_DARK) if is_thread else INK)

    text_x = left + MARK_PX + 64
    canvas.text((text_x, 214), "Causa RAG", font=title, fill=INK)
    # The line from the README: it is the promise as well as the description.
    canvas.text((text_x, 318), "Shows which questions a change fixed,", font=lead, fill=MUTED)
    canvas.text((text_x, 366), "and which it broke.", font=lead, fill=MUTED)
    canvas.text((text_x, 442), "A diagnostic bench for retrieval-augmented generation.",
                font=small, fill=MUTED)
    canvas.text((text_x, 476), "Runs entirely on your own machine.", font=small, fill=MUTED)

    # The Okabe-Ito row, the palette the platform's own diagrams use. It sits
    # under the text, not in a corner: a lone element at the edge reads
    # as the offcut of something, not as part of the composition.
    for i, colour in enumerate(("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00")):
        canvas.rectangle([text_x + i * 40, 534, text_x + i * 40 + 28, 542], fill=rgb(colour))

    return image


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs/assets/social-preview.png"
    draw_card().save(out)
    print(out.name, draw_card().size)
