"""Reading and writing the judgments file.

Lives in ``core/`` rather than ``services/`` on purpose. The file is meant to
be a self-contained artefact a served system can load on its own, so the
loader must not sit behind the platform's HTTP layer. What core does *not*
decide is where the file lives: every function takes an explicit path, and the
caller supplies it — the same discipline that keeps ``qdrant_cfg`` and the ref
resolver out of ``core/``.

Format is JSON Lines, one judgment per line. Chosen over a single JSON array
because the file is append-mostly and is meant to be read as a diff: one new
judgment is one added line, which a reviewer can see at a glance in version
control. A whole-array rewrite would show every line as changed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.judgments.model import RelevanceJudgment, judgment_from_dict, judgment_to_dict

# realm_id and corpus_id reach this from user-supplied API input, so they are
# reduced to a conservative alphabet before touching the filesystem. Anything
# else collapses to an underscore: two different ids can in principle collapse
# to one filename, which is a visible collision rather than a path escape.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def judgments_path(base_dir: Path, realm_id: str, corpus_id: str) -> Path:
    """One file per Realm and corpus pair.

    Split this way because a Realm is an isolated tenant and a judgment is
    only meaningful against the corpus it was made on. Keeping them apart
    means a Realm's file can be handed to that Realm's served system on its
    own, with nothing belonging to anyone else inside it.
    """
    realm = _SAFE.sub("_", realm_id or "_")
    corpus = _SAFE.sub("_", corpus_id or "_")
    return base_dir / f"{realm}__{corpus}.jsonl"


def load_judgments(path: Path) -> list[RelevanceJudgment]:
    """Every judgment in the file, in write order.

    A missing file yields an empty list rather than raising: no judgments
    recorded yet is an ordinary state, not an error. A malformed line is
    skipped rather than taking the rest of the file with it, because the file
    is hand-editable by design and one bad edit must not cost every judgment
    already recorded.
    """
    if not path.exists():
        return []
    judgments: list[RelevanceJudgment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            judgments.append(judgment_from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return judgments


def append_judgment(path: Path, judgment: RelevanceJudgment) -> None:
    """Adds one line. Never rewrites what is already there, so an existing
    judgment cannot be lost to a bug in serialising a new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(judgment_to_dict(judgment), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def rewrite_judgments(path: Path, judgments: list[RelevanceJudgment]) -> None:
    """Replaces the file wholesale, for edits and deletions.

    Writes to a sibling temporary file and renames it over the original, so an
    interrupted write leaves the previous file intact instead of a truncated
    one. ``Path.replace`` is atomic within a filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = "".join(
        json.dumps(judgment_to_dict(j), ensure_ascii=False, sort_keys=True) + "\n"
        for j in judgments
    )
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
