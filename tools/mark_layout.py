"""The layout of the mark, defined once.

The mark appears in four places: `docs/assets/logo.svg`, its dark twin, the
wordmark, and the sidebar component. A shape edited in one of them and
forgotten in the others stops being a mark and becomes four drawings. The
coordinates therefore live here, and every output is generated from them.

**What the mark says.** The funnel the platform measures, with the thread that
came through it. Everything retrieved, what survived reranking, what reached the
answer, and the column running down the middle in the accent colour is the part
that made it all the way. Narrowing alone was ambiguous: it read as a tree or an
arrow. The surviving thread is what turns it into a funnel.

The module and the grid are the ones the sibling registry's mark is built from,
module to pitch as 0.8. The kinship is carried by the unit of measure, which is
what lets each mark keep its own palette and still read as family.
"""

PITCH, MODULE = 4.8, 3.84
BOX = 24.0
COLUMNS, ROWS = 5, 5

#: Which cells are filled, by row. The middle column is the thread and is
#: listed separately because it takes the accent colour.
LOST = {1: (0, 1, 3, 4), 2: (1, 3)}
THREAD = (1, 2, 3)
THREAD_COLUMN = 2

#: The accent, and its counterpart for a dark ground. Okabe-Ito blue, the same
#: value the platform's own diagrams use.
ACCENT, ACCENT_DARK = "#0072B2", "#4EA3DC"
INK, INK_DARK = "#1f2328", "#e6edf3"


def _xy(column: int, row: int) -> tuple[float, float]:
    origin = (BOX - COLUMNS * PITCH) / 2 + PITCH / 2 - MODULE / 2
    return round(column * PITCH + origin, 2), round(row * PITCH + origin, 2)


def cells() -> list[tuple[float, float, bool]]:
    """Every module as ``(x, y, is_thread)``, top row first."""
    out = []
    for row in range(ROWS):
        for column in LOST.get(row, ()):
            out.append((*_xy(column, row), False))
        if row in THREAD:
            out.append((*_xy(THREAD_COLUMN, row), True))
    return out
