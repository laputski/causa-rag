"""Datasets router — exposes golden evaluation sets.

Read order: MongoDB primary → file fallback (the older platform golden
sets, migrated into Mongo once by tools/migrate_to_mongodb.py and otherwise
managed as JSONL files on disk).

POST/DELETE below are the first LIVE write path this collection
has ever had. Previously the only way to register a dataset at all was the
external-RAG-specific `external_rags.py` (`external_rag_datasets`, a
separate collection) — which made a RAG's own uploaded questions look like a
second-class citizen next to the platform's "golden" sets, a distinction
that was never a deliberate design decision, just a byproduct of `datasets`
having no write endpoint. Both are the same `datasets` collection now; the
optional `source_rag_id` is pure provenance (who to ask if a dataset's
quality looks off), not a UI-visible tier.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/datasets", tags=["datasets"])

_GOLDEN_DIR = Path(__file__).parents[3] / "eval" / "golden"


class DatasetCreateRequest(BaseModel):
    # "<name>.<version>.<speed>.jsonl" — same convention _parse_filename
    # already derives display metadata from for every dataset, platform or
    # RAG-authored alike (e.g. "my_rag.v0.fast.jsonl").
    filename: str
    realm_id: str
    questions: list[dict[str, Any]]
    # Provenance only (see module docstring) — which ExternalRag's own
    # ingest pipeline registered this, if any. None for a platform-authored
    # dataset. Not used for access control or UI filtering by default.
    source_rag_id: str | None = None


class QuestionWriteRequest(BaseModel):
    """One control question. `id` is never client-supplied — always
    server-generated (see add_question/add_questions_batch below), same
    short-uuid convention as a dataset's own `id`. `reference_answer` is the
    one canonical field name written going forward; `core/experiment/
    runner.py` already tolerates the older `ground_truth` name found in
    datasets authored before this endpoint existed (e.g. handbook.v1.fast) —
    that read-side tolerance stays, this is just the write-time convention.
    `question_type`/`article_refs`/anything else stays a free passthrough,
    same open-schema policy as DatasetCreateRequest.questions — no fixed
    taxonomy is enforced here.
    """
    model_config = {"extra": "allow"}

    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    question_type: str = ""
    article_refs: list[str] = []
    # Trusted only for origin/model/corpus_id/chunk_id/preset_id, when a
    # generated draft is being saved (see services/api_gateway/routers/
    # generation.py) — created_at is always stamped server-side regardless
    # (see _fresh_provenance), never taken from the client.
    provenance: dict[str, Any] | None = None


def _fresh_provenance(client_provenance: dict[str, Any] | None) -> dict[str, Any]:
    prov = dict(client_provenance) if client_provenance else {}
    prov.setdefault("origin", "manual")
    prov["created_at"] = datetime.now(UTC).isoformat()
    return prov


async def _get_dataset_or_404(dataset_id: str, realm_id: str | None) -> dict[str, Any]:
    import adapters.mongodb as mdb
    doc = await mdb.find_one("datasets", {"id": dataset_id})
    if not doc or (realm_id and doc.get("realm_id") != realm_id):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found")
    return doc


def _dataset_detail(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        **_parse_filename(doc["filename"]),
        "id": doc.get("id"),
        "realm_id": doc.get("realm_id"),
        "source_rag_id": doc.get("source_rag_id"),
        "count": doc.get("n_questions", len(doc.get("questions", []))),
        "questions": doc.get("questions", []),
    }


def _parse_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    parts = stem.split(".")
    return {
        "name":     parts[0] if parts else stem,
        "version":  parts[1] if len(parts) > 1 else "v0",
        "speed":    parts[2] if len(parts) > 2 else "fast",
        "filename": filename,
    }


@router.get("")
async def list_datasets(realm_id: str | None = None, source_rag_id: str | None = None) -> list[dict[str, Any]]:
    # Primary: MongoDB. Filtered by realm_id when given (style scoping,
    # mirroring GET /corpus?realm_id= and GET /external-rags?realm_id=) — a
    # dataset with no realm_id (pre-scoping) is invisible under any filter,
    # same convention as corpus_ingests: there's no way to guess which Realm
    # an unscoped dataset belongs to, so it's better hidden than mis-shown to
    # every Realm (see the design notes). `source_rag_id` narrows
    # further to one RAG's own registered datasets — used by the "Resources"
    # page's per-RAG dataset list, not by the New-Run picker (which wants
    # every dataset in the Realm, platform- and RAG-authored alike).
    try:
        import adapters.mongodb as mdb
        query: dict[str, Any] = {}
        if realm_id:
            query["realm_id"] = realm_id
        if source_rag_id:
            query["source_rag_id"] = source_rag_id
        docs = await mdb.find_many("datasets", query=query, sort=[("filename", 1)])
        if docs:
            return [
                {
                    **_parse_filename(d["filename"]),
                    "id": d.get("id"),
                    "source_rag_id": d.get("source_rag_id"),
                    # `n_questions` is written by create_dataset. A dataset
                    # that arrived some other way, such as the realm import
                    # that seeds the demo's own set, carries only `questions`,
                    # and reading the counter alone showed "0 questions" beside
                    # a set the detail view opened with all fifteen. The list
                    # and the detail must not be able to disagree about a
                    # number this cheap to derive.
                    "count": d.get("n_questions", len(d.get("questions", []))),
                }
                for d in docs
            ]
        if realm_id or source_rag_id:
            # A real query that matched nothing — the Realm/RAG genuinely has
            # no datasets of its own yet. Don't fall through to the unscoped
            # file listing below, which would silently ignore the filter.
            return []
    except Exception:
        pass
    # Fallback: files. No realm/source_rag_id metadata exists at this layer,
    # so it can only serve the no-filter (global/legacy) case honestly — a
    # scoped request that reaches here (Mongo down) gets an empty list
    # rather than every file mislabeled as belonging to this Realm/RAG.
    if realm_id or source_rag_id:
        return []
    if not _GOLDEN_DIR.exists():
        return []
    return [_parse_filename(p.name) for p in sorted(_GOLDEN_DIR.glob("*.jsonl"))]


@router.get("/{filename}")
async def get_dataset(filename: str) -> dict[str, Any]:
    # Primary: MongoDB
    try:
        import adapters.mongodb as mdb
        doc = await mdb.find_one("datasets", {"filename": filename})
        if doc:
            # Found live while adding the per-question endpoints below: this
            # never returned the dataset's own `id` (or `realm_id`) at all —
            # unlike GET /datasets (list_datasets), which does. Every
            # question-write call needs `id` to address the dataset, so a
            # detail view fetched from here always sent `undefined`. Reuse
            # the one shared detail shape (_dataset_detail) instead of a
            # second, now-fixed, ad-hoc dict.
            return _dataset_detail(doc)
    except Exception:
        pass
    # Fallback: file
    path = _GOLDEN_DIR / filename
    if not path.exists() or path.suffix != ".jsonl":
        raise HTTPException(status_code=404, detail=f"Dataset {filename!r} not found")
    from eval.dataset import EvalDataset
    ds = EvalDataset.from_jsonl(path)
    return {**_parse_filename(filename), "count": len(ds), "questions": ds.questions}


@router.post("", status_code=201)
async def create_dataset(body: DatasetCreateRequest) -> dict[str, Any]:
    """First live write path this collection has ever had (see module
    docstring) — usable by a platform-side upload form or an external RAG's
    own ingest pipeline calling this directly, identically."""
    import adapters.mongodb as mdb

    existing = await mdb.find_one("datasets", {"filename": body.filename, "realm_id": body.realm_id})
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset {body.filename!r} already exists in Realm {body.realm_id!r}",
        )

    doc = {
        "id": str(uuid.uuid4())[:8],
        "filename": body.filename,
        "realm_id": body.realm_id,
        "questions": body.questions,
        "n_questions": len(body.questions),
        "source_rag_id": body.source_rag_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    await mdb.insert_one("datasets", dict(doc))
    return {**_parse_filename(doc["filename"]), **doc}


@router.delete("/by-id/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str, realm_id: str | None = None) -> None:
    # Separate path from GET/{filename} — filenames can contain dots and
    # aren't a safe/unambiguous path segment to disambiguate from the
    # generated `id` used here.
    #
    # `realm_id`, when given, must match the dataset's own — found live
    # while adding the per-question endpoints below (which all need this
    # ownership check anyway): this endpoint had no such check at all, so
    # any Realm could delete any other Realm's dataset by id. Omitted
    # `realm_id` keeps the old unscoped behavior (back-compat).
    import adapters.mongodb as mdb

    query: dict[str, Any] = {"id": dataset_id}
    if realm_id:
        query["realm_id"] = realm_id
    deleted = await mdb.delete_one("datasets", query)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found")


@router.post("/{dataset_id}/questions", status_code=201)
async def add_question(
    dataset_id: str, body: QuestionWriteRequest, realm_id: str | None = None,
) -> dict[str, Any]:
    """Append one question. Returns the updated dataset detail (same shape
    as GET /datasets/{filename}) so the frontend can re-set its local state
    from the response without a second round trip."""
    import adapters.mongodb as mdb

    doc = await _get_dataset_or_404(dataset_id, realm_id)
    question = body.model_dump()
    question["id"] = str(uuid.uuid4())[:8]
    question["provenance"] = _fresh_provenance(body.provenance)

    questions = doc.get("questions", []) + [question]
    await mdb.update_one(
        "datasets", {"id": dataset_id},
        {"$set": {"questions": questions, "n_questions": len(questions)}},
    )
    doc["questions"] = questions
    doc["n_questions"] = len(questions)
    return _dataset_detail(doc)


@router.post("/{dataset_id}/questions/batch", status_code=201)
async def add_questions_batch(
    dataset_id: str, body: list[QuestionWriteRequest], realm_id: str | None = None,
) -> dict[str, Any]:
    """Append several questions in one request — used to save a batch of
    accepted generator drafts (services/api_gateway/routers/generation.py)
    without one round trip per row."""
    import adapters.mongodb as mdb

    doc = await _get_dataset_or_404(dataset_id, realm_id)
    new_questions = []
    for item in body:
        question = item.model_dump()
        question["id"] = str(uuid.uuid4())[:8]
        question["provenance"] = _fresh_provenance(item.provenance)
        new_questions.append(question)

    questions = doc.get("questions", []) + new_questions
    await mdb.update_one(
        "datasets", {"id": dataset_id},
        {"$set": {"questions": questions, "n_questions": len(questions)}},
    )
    doc["questions"] = questions
    doc["n_questions"] = len(questions)
    return _dataset_detail(doc)


@router.put("/{dataset_id}/questions/{question_id}")
async def update_question(
    dataset_id: str, question_id: str, body: QuestionWriteRequest, realm_id: str | None = None,
) -> dict[str, Any]:
    """Replace one question's content by its own `id` (not array index —
    index shifts under concurrent add/delete). Identity (`id`) can't be
    changed through the body. `provenance` in the request body is ignored —
    provenance is derived from the question's own stored history, not from
    whatever the client happens to echo back (see below)."""
    import adapters.mongodb as mdb

    doc = await _get_dataset_or_404(dataset_id, realm_id)
    questions = doc.get("questions", [])
    idx = next((i for i, q in enumerate(questions) if q.get("id") == question_id), None)
    if idx is None:
        raise HTTPException(
            status_code=404, detail=f"Question {question_id!r} not found in dataset {dataset_id!r}",
        )

    now = datetime.now(UTC).isoformat()
    prov = dict(questions[idx].get("provenance") or {"origin": "manual", "created_at": now})
    prov["updated_at"] = now
    if prov.get("origin") == "generated":
        prov["edited_manually"] = True

    updated = body.model_dump()
    updated["id"] = question_id
    updated["provenance"] = prov
    questions[idx] = updated

    await mdb.update_one("datasets", {"id": dataset_id}, {"$set": {"questions": questions}})
    doc["questions"] = questions
    return _dataset_detail(doc)


@router.delete("/{dataset_id}/questions/{question_id}")
async def delete_question(
    dataset_id: str, question_id: str, realm_id: str | None = None,
) -> dict[str, Any]:
    import adapters.mongodb as mdb

    doc = await _get_dataset_or_404(dataset_id, realm_id)
    questions = doc.get("questions", [])
    new_questions = [q for q in questions if q.get("id") != question_id]
    if len(new_questions) == len(questions):
        raise HTTPException(
            status_code=404, detail=f"Question {question_id!r} not found in dataset {dataset_id!r}",
        )

    await mdb.update_one(
        "datasets", {"id": dataset_id},
        {"$set": {"questions": new_questions, "n_questions": len(new_questions)}},
    )
    doc["questions"] = new_questions
    doc["n_questions"] = len(new_questions)
    return _dataset_detail(doc)
