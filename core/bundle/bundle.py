"""The artefact itself, and the library that receives it.

The bundle is what the platform publishes and a served system consumes. It is
deliberately a plain JSON document: a recipient has to be able to read it
with a standard library and nothing else, since requiring a dependency from
the platform would put the platform back in the dependency graph the whole
design removes it from.

The receiving half lives here too, in `core/`, so the template carries it and
every system grown from the template gets it for free. A recipient that wrote its own loader would be free to disagree
with the publisher about what a bundle means; one loader in the template
means there is nothing to disagree about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.bundle.anchor import Anchor, normalise, resolution_report, resolve_anchors

# Bumped only on a breaking change to the document shape. A recipient that
# does not recognise it refuses the whole bundle rather than applying the
# parts it happens to understand.
BUNDLE_FORMAT = 1


@dataclass(frozen=True)
class FixEntry:
    """One reviewer statement, in portable form.

    `question` is text, not a vector: an embedding belongs to one model, and
    shipping one would expire the artefact the day the recipient changed
    models. The recipient embeds it with its own.
    """

    id: str
    question: str
    relevant: tuple[Anchor, ...] = ()
    irrelevant: tuple[Anchor, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            # Stored alongside the text so a recipient can answer the common
            # case — the same question asked again, verbatim — without
            # embedding anything at all (PLAN section 5.4).
            "question_normalised": normalise(self.question),
            "relevant": [a.to_dict() for a in self.relevant],
            "irrelevant": [a.to_dict() for a in self.irrelevant],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixEntry:
        return cls(
            id=str(data.get("id") or ""),
            question=str(data.get("question") or ""),
            relevant=tuple(Anchor.from_dict(a) for a in data.get("relevant") or []),
            irrelevant=tuple(Anchor.from_dict(a) for a in data.get("irrelevant") or []),
            note=str(data.get("note") or ""),
        )


@dataclass(frozen=True)
class FixBundle:
    """A versioned, self-contained set of corrections."""

    realm_id: str
    corpus_id: str
    created_at: str = ""
    entries: tuple[FixEntry, ...] = ()
    format_version: int = BUNDLE_FORMAT
    # The calibration procedure, not a number. A similarity threshold depends
    # on the embedding model, so the publisher ships the material to derive
    # one — paraphrases that must match, and unrelated questions that must
    # not — and the recipient derives it with its own model (PLAN 5.3).
    calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "realm_id": self.realm_id,
            "corpus_id": self.corpus_id,
            "created_at": self.created_at,
            "n_entries": len(self.entries),
            "calibration": self.calibration,
            "entries": [e.to_dict() for e in self.entries],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def bundle_from_judgments(
    judgments: list[Any], realm_id: str, corpus_id: str, created_at: str = "",
    calibration: dict[str, Any] | None = None,
) -> FixBundle:
    """Publishes active judgments as a bundle.

    Retired and empty judgments are left out: a retired verdict is one the
    reviewer withdrew, and shipping it would apply a correction its own
    author no longer stands behind.

    An anchor is built from what the judgment recorded at the time — the
    ref id and the text snapshot — rather than by re-reading the corpus now.
    The snapshot is what the reviewer actually looked at, and a bundle
    should carry the evidence the verdict was made on, not a later version
    of it that may say something else.
    """
    entries = []
    for j in judgments:
        if not getattr(j, "is_active", True) or getattr(j, "is_empty", False):
            continue
        entries.append(FixEntry(
            id=getattr(j, "id", ""),
            question=getattr(j, "question", ""),
            relevant=tuple(
                Anchor.from_chunk(c.ref_id, c.structural_path, c.text) for c in j.relevant
            ),
            irrelevant=tuple(
                Anchor.from_chunk(c.ref_id, c.structural_path, c.text) for c in j.irrelevant
            ),
            note=getattr(j, "note", ""),
        ))
    return FixBundle(
        realm_id=realm_id, corpus_id=corpus_id, created_at=created_at,
        entries=tuple(entries), calibration=calibration or {},
    )


# ── The recipient library ────────────────────────────────────


class UnsupportedBundleFormat(ValueError):
    """Raised rather than degraded on purpose.

    Every other failure in this codebase degrades honestly, because the cost
    of degrading is a missing feature. Here the cost is applying corrections
    a recipient has misread, which changes answers — so an unrecognised
    format refuses the whole document instead of the parts it thinks it
    understands.
    """


def load_bundle(path: Path) -> FixBundle:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = int(data.get("format_version") or 0)
    if version != BUNDLE_FORMAT:
        raise UnsupportedBundleFormat(
            f"bundle format {version} is not {BUNDLE_FORMAT}; refusing to apply it partially"
        )
    return FixBundle(
        realm_id=str(data.get("realm_id") or ""),
        corpus_id=str(data.get("corpus_id") or ""),
        created_at=str(data.get("created_at") or ""),
        entries=tuple(FixEntry.from_dict(e) for e in data.get("entries") or []),
        format_version=version,
        calibration=dict(data.get("calibration") or {}),
    )


def receive_bundle(bundle: FixBundle, chunks: list[Any]) -> dict[str, Any]:
    """Resolves a whole bundle against a recipient's own index, once.

    Returns the resolution report and the per-entry resolved ids. Called at
    startup, never per query: that is the difference between a bundle and the
    store lookup this design sheds, and it is the entire reason a bundle can
    exist without violating the "stand outside the hot path" invariant.
    """
    all_anchors = [a for e in bundle.entries for a in (*e.relevant, *e.irrelevant)]
    resolutions = resolve_anchors(all_anchors, chunks)
    by_anchor = {(r.anchor.ref_id, r.anchor.text_hash): r for r in resolutions}

    entries = []
    for entry in bundle.entries:
        entries.append({
            "id": entry.id,
            "question": entry.question,
            "question_normalised": normalise(entry.question),
            "relevant_chunk_ids": [
                r.chunk_id for a in entry.relevant
                if (r := by_anchor.get((a.ref_id, a.text_hash))) and r.chunk_id
            ],
            "irrelevant_chunk_ids": [
                r.chunk_id for a in entry.irrelevant
                if (r := by_anchor.get((a.ref_id, a.text_hash))) and r.chunk_id
            ],
        })

    return {
        "realm_id": bundle.realm_id,
        "corpus_id": bundle.corpus_id,
        "report": resolution_report(resolutions),
        "entries": entries,
    }
