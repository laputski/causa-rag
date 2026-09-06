"""What a segmentation strategy promises about its own output.

A strategy named for structure that produces no structure has done exactly
what the plain fixed-window strategy does, under a different name, and
nothing about the run says so: the chunks are well-formed, the index accepts
them, retrieval works, and the setting on the screen still says
`structure_aware`. The failure is a silence, and the silence is where a name
and a behaviour part company.

The check belongs to the strategy because only the strategy knows what it
promised. A rule written elsewhere would be somebody's reading of the name,
which is the thing in dispute.

Evaluated once over a whole load and not per document. A single flat file in
a structured corpus is a property of that file; every file coming out flat is
a property of the strategy, and only the second is worth stopping for. The
line between them is a share of the corpus, stated here so it can be argued
with.

Pure: takes chunks, returns the promises they do not keep.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: The share of a corpus that has to breach a promise before the promise is
#: reported as broken. One flat file among forty is a fact about that file.
#: Half of them is a fact about the strategy.
_ENOUGH = 0.5


def _structural_paths_are_real(chunks: Sequence[Any]) -> float:
    """The share of chunks that fell back to the strategy's own root.

    `structure_aware` returns a single "root" path for a document whose
    headings it could not read, and that value is the fallback itself rather
    than a path anybody wrote.
    """
    if not chunks:
        return 0.0
    fell_back = sum(1 for c in chunks if (getattr(c, "structural_path", "") or "") in ("", "root"))
    return fell_back / len(chunks)


def _sizes_are_within_the_window(chunks: Sequence[Any], size: int) -> float:
    """The share of chunks wider than the window that was asked for."""
    if not chunks or size <= 0:
        return 0.0
    return sum(1 for c in chunks if len(getattr(c, "text", "")) > size) / len(chunks)


def _ends_are_sentence_ends(chunks: Sequence[Any]) -> float:
    """The share of fragments ending inside a sentence.

    The last fragment of a document is excluded from the judgement in spirit
    and not in code: a document whose own text ends without punctuation
    produces one such fragment, which is a fact about the document. The
    share is what decides, and one fragment in a corpus never reaches it.
    """
    if not chunks:
        return 0.0
    cut = sum(
        1 for c in chunks
        if (text := (getattr(c, "text", "") or "").rstrip()) and text[-1] not in ".!?:;»\"')]"
    )
    return cut / len(chunks)


#: One promise per strategy, in the words its name makes. A strategy absent
#: from here promises nothing checkable, and that is a claim somebody has to
#: make deliberately: the guard in tests/fitness reads this against the
#: strategies that exist.
PROMISES: dict[str, tuple[str, str]] = {
    "structure_aware": (
        "every fragment is placed by the document's own headings",
        "the headings could not be read, so this load produced the same fragments the "
        "plain fixed-window strategy would have produced under another name",
    ),
    "fixed": (
        "no fragment is wider than the window that was asked for",
        "fragments came back wider than the requested window, so the setting that names "
        "the window did not decide it",
    ),
    "sentence": (
        "no fragment ends inside a sentence",
        "most fragments end mid-sentence, so the boundaries came from somewhere other "
        "than the sentence segmenter this strategy is named for",
    ),
    "paragraph": (
        "no fragment ends inside a sentence, since a paragraph ends at one",
        "most fragments end mid-sentence, so neither the paragraph split nor the "
        "sentence fallback beneath it decided the boundaries",
    ),
}


def unmet(strategy_id: str, chunks: Sequence[Any], chunk_size: int = 0) -> list[str]:
    """The promises this load did not keep, each in the words of the promise.

    An unknown strategy returns nothing, and that is the honest answer: a
    strategy that never declared a promise has not broken one.
    """
    breached: list[str] = []
    if strategy_id == "structure_aware" and _structural_paths_are_real(chunks) > _ENOUGH:
        breached.append(PROMISES["structure_aware"][1])
    if strategy_id == "fixed" and _sizes_are_within_the_window(chunks, chunk_size) > 0:
        breached.append(PROMISES["fixed"][1])
    if strategy_id in ("sentence", "paragraph") and _ends_are_sentence_ends(chunks) > _ENOUGH:
        breached.append(PROMISES[strategy_id][1])
    return breached
