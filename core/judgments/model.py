"""The judgment value object and the three things a judgment turns into.

Pure: no I/O, no store access, no framework types. The services layer owns
reading and writing the file, the same way it owns resolving `qdrant_cfg` and
the ref resolver and passing them in, which is what keeps the layers apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A judgment's own verdict about one chunk. Kept as an explicit pair of
# tuples on the judgment rather than a `relevant: bool` per chunk, because
# the two lists mean different things to a reader: "these should have been
# retrieved" is a coverage claim, "these should not have been" is a precision
# claim, and a judgment very often carries only one of the two.
_ACTIVE = "active"


@dataclass(frozen=True)
class JudgedChunk:
    """One chunk a reviewer ruled on.

    ``ref_id`` is carried alongside ``chunk_id`` deliberately. A chunk_id in
    this platform is derived from position, so re-indexing can silently make
    it mean a different piece of text, which is a known limitation. A ref_id
    built by
    ``core/eval/retrieval_metrics.py#extract_ref_id`` names the *source unit*
    instead, survives re-indexing, and is the form a golden question's
    ``article_refs`` already uses — so it is what makes
    :func:`to_golden_question` possible at all.

    ``text`` is a snapshot taken when the judgment was recorded. It is
    evidence of what the reviewer actually looked at, not a live lookup: if
    the corpus later changes, the discrepancy between snapshot and index is
    itself a finding worth surfacing rather than something to hide by
    re-reading.
    """

    chunk_id: str
    ref_id: str = ""
    doc_id: str = ""
    structural_path: str = ""
    text: str = ""


@dataclass(frozen=True)
class RelevanceJudgment:
    """A reviewer's statement about one question.

    Deliberately has no threshold, no similarity vector and no instruction
    fields. Those belonged to the retrieval-pin mechanism, which decided at
    query time *which future questions a statement should also apply to* —
    the part the objective review found unsound. A judgment applies to the
    question it was made about, and nothing else claims otherwise.
    """

    id: str
    realm_id: str
    corpus_id: str
    question: str
    relevant: tuple[JudgedChunk, ...] = field(default_factory=tuple)
    irrelevant: tuple[JudgedChunk, ...] = field(default_factory=tuple)
    note: str = ""
    author: str = ""
    created_at: str = ""
    status: str = _ACTIVE
    # Where the judgment came from, kept so a reader can walk back to the
    # evidence: the run whose result prompted it, the reviewer comment it was
    # triaged from, and the golden question if one already existed.
    source_run_id: str = ""
    source_feedback_id: str = ""
    source_question_id: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == _ACTIVE

    @property
    def is_empty(self) -> bool:
        """A judgment ruling on no chunk at all states nothing and is
        rejected at the boundary rather than stored as a silent no-op."""
        return not self.relevant and not self.irrelevant


def _chunk_from_dict(data: dict[str, Any]) -> JudgedChunk:
    return JudgedChunk(
        chunk_id=str(data.get("chunk_id") or ""),
        ref_id=str(data.get("ref_id") or ""),
        doc_id=str(data.get("doc_id") or ""),
        structural_path=str(data.get("structural_path") or ""),
        text=str(data.get("text") or ""),
    )


def _chunk_to_dict(chunk: JudgedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "ref_id": chunk.ref_id,
        "doc_id": chunk.doc_id,
        "structural_path": chunk.structural_path,
        "text": chunk.text,
    }


def judgment_from_dict(data: dict[str, Any]) -> RelevanceJudgment:
    """Tolerant of missing keys on purpose. The file is append-only and
    hand-editable, so a record written by an older version must still load
    rather than take the whole file down with it."""
    return RelevanceJudgment(
        id=str(data.get("id") or ""),
        realm_id=str(data.get("realm_id") or ""),
        corpus_id=str(data.get("corpus_id") or ""),
        question=str(data.get("question") or ""),
        relevant=tuple(_chunk_from_dict(c) for c in data.get("relevant") or []),
        irrelevant=tuple(_chunk_from_dict(c) for c in data.get("irrelevant") or []),
        note=str(data.get("note") or ""),
        author=str(data.get("author") or ""),
        created_at=str(data.get("created_at") or ""),
        status=str(data.get("status") or _ACTIVE),
        source_run_id=str(data.get("source_run_id") or ""),
        source_feedback_id=str(data.get("source_feedback_id") or ""),
        source_question_id=str(data.get("source_question_id") or ""),
    )


def judgment_to_dict(judgment: RelevanceJudgment) -> dict[str, Any]:
    return {
        "id": judgment.id,
        "realm_id": judgment.realm_id,
        "corpus_id": judgment.corpus_id,
        "question": judgment.question,
        "relevant": [_chunk_to_dict(c) for c in judgment.relevant],
        "irrelevant": [_chunk_to_dict(c) for c in judgment.irrelevant],
        "note": judgment.note,
        "author": judgment.author,
        "created_at": judgment.created_at,
        "status": judgment.status,
        "source_run_id": judgment.source_run_id,
        "source_feedback_id": judgment.source_feedback_id,
        "source_question_id": judgment.source_question_id,
    }


# ── The three things a judgment turns into ──────────────────────────────


def to_golden_question(judgment: RelevanceJudgment) -> dict[str, Any] | None:
    """First use: the judgment becomes a permanent test.

    A chunk the reviewer called relevant becomes an expected source. From
    then on, losing it is caught by the next run rather than by the next
    complaint — which is the difference between a mechanism that accumulates
    exceptions and one that accumulates coverage.

    Returns ``None`` when no relevant chunk carries a ref_id: without one
    there is nothing a retrieval metric could check against, and emitting a
    question with an empty ``article_refs`` would quietly classify it
    ``out_of_scope`` (see core/eval/answerability.py) — a test that can never
    fail, which is worse than no test.
    """
    refs = [c.ref_id for c in judgment.relevant if c.ref_id]
    if not refs:
        return None
    # dict.fromkeys preserves the reviewer's ordering while dropping repeats,
    # which matter because one source unit usually spans several chunks.
    return {
        "id": f"j_{judgment.id}",
        "question": judgment.question,
        "article_refs": list(dict.fromkeys(refs)),
        "origin": "relevance_judgment",
        "judgment_id": judgment.id,
    }


def preference_pairs(judgment: RelevanceJudgment) -> list[tuple[str, str, str]]:
    """Second use: training material for ranking work.

    Returns ``(question, positive_chunk_id, negative_chunk_id)`` for every
    relevant/irrelevant combination. This is the form a cross-encoder or a
    feature-based ranker consumes directly, and it is why a judgment carrying
    both lists is worth more than two judgments carrying one each.

    Empty when the judgment names only one side: a pair needs both.
    """
    return [
        (judgment.question, pos.chunk_id, neg.chunk_id)
        for pos in judgment.relevant
        for neg in judgment.irrelevant
    ]
