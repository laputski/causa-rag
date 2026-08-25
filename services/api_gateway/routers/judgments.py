"""Relevance judgments CRUD.

Replaces `pins.py`. A reviewer states what is correct ("for this question,
this chunk is relevant, that one is not") and the platform records it. Nothing
here changes retrieval behaviour: a judgment is an observation, and the three
things it turns into (a permanent test case, a preference pair, and — only if
ever wanted — a curation rule) all act later and elsewhere.

The store is a file, not MongoDB. That is the point rather than an
implementation preference. Nothing a served system needs at query time may
live in the platform's database,
and a versioned JSON Lines file is an artefact that can simply be handed over.
It also means the form the reviewer authors in and the form a served system
would consume are the same file, with no export step to drift.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.judgments import (
    JudgedChunk,
    RelevanceJudgment,
    judgment_to_dict,
    preference_pairs,
    to_golden_question,
)
from core.judgments.store import (
    append_judgment,
    judgments_path,
    load_judgments,
    rewrite_judgments,
)

router = APIRouter(prefix="/judgments", tags=["judgments"])

# Sits beside eval/golden/, the platform's other curated evaluation data, for
# the same reason: both are reviewed artefacts rather than runtime state.
_JUDGMENTS_DIR = Path(__file__).parents[3] / "eval" / "judgments"


class JudgmentWriteRequest(BaseModel):
    """`relevant`/`irrelevant` are chunk_ids. At least one of the two must be
    non-empty, because a judgment ruling on nothing states nothing."""

    realm_id: str
    corpus_id: str
    question: str = Field(min_length=1)
    relevant: list[str] = Field(default_factory=list)
    irrelevant: list[str] = Field(default_factory=list)
    note: str = ""
    author: str = ""
    source_run_id: str = ""
    source_feedback_id: str = ""
    source_question_id: str = ""


class JudgmentPatchRequest(BaseModel):
    """Only the fields a human revises after the fact. The verdict itself is
    not patchable: changing which chunks a judgment names would silently
    rewrite the evidence a golden question was already derived from. A
    corrected verdict is a new judgment, and the old one is retired."""

    status: str | None = None
    note: str | None = None


class GoldenQuestionRequest(BaseModel):
    """`reference_answer` is required and deliberately not derived from the
    judgment. A judgment states which sources are right, not what the answer
    should say, and a golden question with an invented reference answer would
    score generation against a fabrication."""

    dataset_id: str
    reference_answer: str = Field(min_length=1)


def _path_for(realm_id: str, corpus_id: str) -> Path:
    return judgments_path(_JUDGMENTS_DIR, realm_id, corpus_id)


async def _resolve_chunks(
    corpus_id: str, realm_id: str, chunk_ids: list[str],
) -> list[JudgedChunk]:
    """Turns chunk_ids into judged chunks carrying a ref_id and a text
    snapshot.

    The ref_id is what makes a judgment outlive the corpus: chunk_id in this
    platform is position-derived, so re-indexing can quietly change what it
    points at, while the ref_id built by `extract_ref_id` names the source
    unit and is the exact form a golden question's `article_refs` uses.

    Degrades to bare chunk_ids (never raises) when the index is unreachable.
    The reviewer's verdict is worth recording even if the platform cannot
    enrich it right now; what is lost is the ability to derive a test case
    from it, and that shows up plainly as an empty ref_id rather than as a
    failed request.
    """
    if not chunk_ids:
        return []
    try:
        from core.eval.ref_resolution import ref_source_from_chunk
        from core.eval.retrieval_metrics import extract_ref_id
        from core.experiment.runner import _rebind_corpus_id
        from core.registry import registry

        base = registry.resolve("pipeline", "naive")
        qdrant_cfg = None
        if realm_id:
            from services.api_gateway.routers.corpus import _get_realm_resource
            qdrant_cfg = await _get_realm_resource(realm_id, "qdrant")
        retriever = _rebind_corpus_id(base._retriever, corpus_id, realm_id, qdrant_cfg, None)
        get_by_ids = getattr(retriever, "get_chunks_by_ids", None)
        if get_by_ids is None:
            raise RuntimeError("retriever cannot fetch chunks by id")
        chunks = get_by_ids(chunk_ids)
    except Exception:
        return [JudgedChunk(chunk_id=cid) for cid in chunk_ids]

    resolved = {
        c.chunk_id: JudgedChunk(
            chunk_id=c.chunk_id,
            ref_id=extract_ref_id(ref_source_from_chunk(c)) or "",
            doc_id=c.doc_id,
            structural_path=c.structural_path,
            text=c.text,
        )
        for c in chunks
    }
    # Preserve the reviewer's own ordering, and keep an id the index could
    # not resolve rather than dropping it — a chunk that has vanished from
    # the corpus is itself a finding.
    return [resolved.get(cid, JudgedChunk(chunk_id=cid)) for cid in chunk_ids]


def _find(judgments: list[RelevanceJudgment], judgment_id: str) -> RelevanceJudgment:
    for judgment in judgments:
        if judgment.id == judgment_id:
            return judgment
    raise HTTPException(status_code=404, detail=f"Judgment {judgment_id!r} not found")


def _public(judgment: RelevanceJudgment) -> dict[str, Any]:
    data = judgment_to_dict(judgment)
    data["preference_pairs"] = len(preference_pairs(judgment))
    data["can_become_test"] = to_golden_question(judgment) is not None
    return data


@router.post("", status_code=201)
async def create_judgment(body: JudgmentWriteRequest) -> dict[str, Any]:
    if not body.relevant and not body.irrelevant:
        raise HTTPException(
            status_code=400,
            detail="a judgment must name at least one relevant or irrelevant chunk",
        )
    judgment = RelevanceJudgment(
        id=str(uuid.uuid4())[:8],
        realm_id=body.realm_id,
        corpus_id=body.corpus_id,
        question=body.question,
        relevant=tuple(await _resolve_chunks(body.corpus_id, body.realm_id, body.relevant)),
        irrelevant=tuple(await _resolve_chunks(body.corpus_id, body.realm_id, body.irrelevant)),
        note=body.note,
        author=body.author,
        created_at=datetime.now(UTC).isoformat(),
        source_run_id=body.source_run_id,
        source_feedback_id=body.source_feedback_id,
        source_question_id=body.source_question_id,
    )
    append_judgment(_path_for(body.realm_id, body.corpus_id), judgment)
    return _public(judgment)


@router.get("")
async def list_judgments(realm_id: str, corpus_id: str) -> list[dict[str, Any]]:
    """`realm_id` and `corpus_id` are required rather than optional filters.
    A judgment is only meaningful against the corpus it was made on, and the
    file layout is per-pair, so there is no meaningful unscoped listing to
    return — asking for one is a caller mistake worth surfacing."""
    return [_public(j) for j in load_judgments(_path_for(realm_id, corpus_id))]


# Declared before the `/{judgment_id}` routes below: a path parameter
# would otherwise match the literal "bundle" and look up a judgment by
# that id. Found live, as a 404 naming a judgment nobody asked for.
@router.get("/bundle")
async def publish_bundle(realm_id: str, corpus_id: str) -> dict[str, Any]:
    """Phase 6 — the artefact a served system loads at startup.

    Published rather than applied: the platform never reaches into a served
    system, and a system that has loaded this holds it in memory and consults
    nothing at query time. That is what makes a correction possible at all
    without breaking that invariant.

    Built from the same judgments file the page reads, so what is handed over
    is exactly what a reviewer recorded — no separate export state to drift.
    """
    from datetime import UTC, datetime

    from core.bundle import bundle_from_judgments
    from core.bundle.calibration import calibration_material

    judgments = load_judgments(_path_for(realm_id, corpus_id))
    bundle = bundle_from_judgments(
        judgments, realm_id=realm_id, corpus_id=corpus_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    # Built after the entries, from the entries: the negatives are the
    # bundle's own other questions, so the material cannot drift away from
    # what the bundle actually contains.
    return FixBundleResponse(bundle, calibration_material(list(bundle.entries))).payload


class FixBundleResponse:
    """Assembles the published document. A tiny class rather than two lines
    inline, so the calibration material is attached in exactly one place and
    a future caller cannot publish a bundle without it."""

    def __init__(self, bundle: Any, calibration: dict[str, Any]) -> None:
        self._payload = {**bundle.to_dict(), "calibration": calibration}

    @property
    def payload(self) -> dict[str, Any]:
        return self._payload


@router.get("/{judgment_id}")
async def get_judgment(judgment_id: str, realm_id: str, corpus_id: str) -> dict[str, Any]:
    return _public(_find(load_judgments(_path_for(realm_id, corpus_id)), judgment_id))


@router.patch("/{judgment_id}")
async def update_judgment(
    judgment_id: str, body: JudgmentPatchRequest, realm_id: str, corpus_id: str,
) -> dict[str, Any]:
    path = _path_for(realm_id, corpus_id)
    judgments = load_judgments(path)
    existing = _find(judgments, judgment_id)
    if body.status is not None and body.status not in ("active", "retired"):
        raise HTTPException(status_code=400, detail="status must be 'active' or 'retired'")

    from dataclasses import replace

    updated = replace(
        existing,
        status=body.status if body.status is not None else existing.status,
        note=body.note if body.note is not None else existing.note,
    )
    rewrite_judgments(path, [updated if j.id == judgment_id else j for j in judgments])
    return _public(updated)


@router.delete("/{judgment_id}", status_code=204)
async def delete_judgment(judgment_id: str, realm_id: str, corpus_id: str) -> None:
    path = _path_for(realm_id, corpus_id)
    judgments = load_judgments(path)
    _find(judgments, judgment_id)
    rewrite_judgments(path, [j for j in judgments if j.id != judgment_id])


@router.post("/{judgment_id}/golden-question", status_code=201)
async def promote_to_golden_question(
    judgment_id: str, body: GoldenQuestionRequest, realm_id: str, corpus_id: str,
) -> dict[str, Any]:
    """The first and most valuable use of a judgment: it becomes a permanent
    test, so losing the chunk again is caught by the next run instead of by
    the next complaint."""
    judgment = _find(load_judgments(_path_for(realm_id, corpus_id)), judgment_id)
    question = to_golden_question(judgment)
    if question is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "no relevant chunk of this judgment resolved to a ref id, so the "
                "question would carry no expected sources and could never fail"
            ),
        )

    from services.api_gateway.routers.datasets import QuestionWriteRequest, add_question

    payload = QuestionWriteRequest(
        question=question["question"],
        reference_answer=body.reference_answer,
        article_refs=question["article_refs"],
        origin=question["origin"],
        judgment_id=question["judgment_id"],
    )
    return await add_question(body.dataset_id, payload, realm_id=realm_id)
