"""A portable way to name a piece of text.

A `chunk_id` here is derived from position, so re-indexing makes the same id
mean different text. Anything built on one is a fix that silently starts
pointing somewhere else, which is the position-identifier debt the old plan's
critical review recorded as point 8.

An anchor names content instead: the document it belongs to, where in the
document's structure it sits, a hash of its normalised text, and the text.
Four descriptions of the same thing, deliberately redundant — a recipient
whose chunking differs from the platform's will fail to match some of them
and can still match others, and knowing *which* level matched is what makes
the resolution report honest rather than a bare success count.

Resolution happens once, when a bundle is loaded, never per query. The
recipient turns anchors into its own identifiers up front, so the cost is
paid at startup and a query pays nothing.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

# How much text travels with an anchor. Enough for a human to recognise what
# was meant and for a recipient to match by prefix; short enough that a
# bundle stays a document rather than a copy of the corpus.
_QUOTE_CHARS = 300

_WHITESPACE = re.compile(r"\s+")

Match = Literal["exact", "unit", "unresolved"]


def normalise(text: str) -> str:
    """Whitespace-collapsed, case-folded, Unicode-normalised.

    Two systems that ingested the same document rarely produce byte-identical
    text: line wrapping, non-breaking spaces and quote characters all drift.
    Hashing the raw text would make anchors fail across systems for reasons
    that have nothing to do with meaning, which is exactly the fragility the
    anchor exists to avoid.
    """
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _WHITESPACE.sub(" ", folded).strip()


def text_hash(text: str) -> str:
    """Short digest of the normalised text. Truncated because it identifies
    within one corpus, not across the world, and a full digest would triple
    the size of a bundle for no gain in that job."""
    return hashlib.sha256(normalise(text).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Anchor:
    """One piece of text, described four ways."""

    ref_id: str
    structural_path: str = ""
    text_hash: str = ""
    quote: str = ""

    @classmethod
    def from_chunk(cls, ref_id: str, structural_path: str, text: str) -> Anchor:
        return cls(
            ref_id=ref_id,
            structural_path=structural_path,
            text_hash=text_hash(text),
            quote=(text or "")[:_QUOTE_CHARS],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "structural_path": self.structural_path,
            "text_hash": self.text_hash,
            "quote": self.quote,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Anchor:
        return cls(
            ref_id=str(data.get("ref_id") or ""),
            structural_path=str(data.get("structural_path") or ""),
            text_hash=str(data.get("text_hash") or ""),
            quote=str(data.get("quote") or ""),
        )


@dataclass(frozen=True)
class AnchorResolution:
    """What a recipient made of one anchor."""

    anchor: Anchor
    chunk_id: str | None
    match: Match

    def to_dict(self) -> dict[str, Any]:
        return {"anchor": self.anchor.to_dict(), "chunk_id": self.chunk_id, "match": self.match}


def resolve_anchors(anchors: list[Anchor], chunks: list[Any]) -> list[AnchorResolution]:
    """Turns anchors into the recipient's own chunk ids.

    Two levels, tried in order, and the level that matched is reported.

    `exact` means the normalised text hash matched: the recipient holds the
    same text, whatever it calls it. `unit` means only the source unit
    matched: the recipient has the right article or section but chunked it
    differently, so the anchor points at the right *content* and not at the
    same fragment. That distinction matters to whoever reads the report —
    a bundle resolving mostly at `unit` level is telling them their chunking
    differs from the publisher's, which is information, not a failure.

    Unresolved anchors are returned rather than dropped. A bundle that
    silently applies half of itself is worse than one that says so.
    """
    by_hash: dict[str, Any] = {}
    by_ref: dict[str, Any] = {}
    for chunk in chunks:
        digest = text_hash(getattr(chunk, "text", "") or "")
        by_hash.setdefault(digest, chunk)
        ref = getattr(chunk, "ref_id", "") or ""
        if ref:
            by_ref.setdefault(ref, chunk)

    resolutions = []
    for anchor in anchors:
        chunk = by_hash.get(anchor.text_hash) if anchor.text_hash else None
        if chunk is not None:
            resolutions.append(AnchorResolution(anchor, getattr(chunk, "chunk_id", None), "exact"))
            continue
        chunk = by_ref.get(anchor.ref_id) if anchor.ref_id else None
        if chunk is not None:
            resolutions.append(AnchorResolution(anchor, getattr(chunk, "chunk_id", None), "unit"))
            continue
        resolutions.append(AnchorResolution(anchor, None, "unresolved"))
    return resolutions


def resolution_report(resolutions: list[AnchorResolution]) -> dict[str, Any]:
    """Counts per match level, and the share that resolved at all.

    Reported as counts alongside the share rather than as a share alone: a
    bundle of four anchors at 75% and one of four hundred at 75% call for
    different reactions, and a percentage hides which one you have.
    """
    counts = {"exact": 0, "unit": 0, "unresolved": 0}
    for r in resolutions:
        counts[r.match] += 1
    total = len(resolutions)
    resolved = counts["exact"] + counts["unit"]
    return {
        "total": total,
        "counts": counts,
        "resolved_share": (resolved / total) if total else 0.0,
        "unresolved": [r.anchor.to_dict() for r in resolutions if r.match == "unresolved"],
    }
