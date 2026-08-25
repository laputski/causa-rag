"""Human feedback on run answers — binary rating, per-dimension scores, free text.

Stored in its own `answer_feedback` collection, correlated with a run via
`run_id`/`question_id` rather than embedded inside the run's own
`experiment_runs` document. Run results (services/api_gateway/routers/
experiments.py#_save) are computed once and are otherwise immutable;
feedback is reviewer-authored and edited interactively, one answer at a
time — embedding it in `question_results[]` would mean rewriting the whole
run document (up to 143+ questions, each with full source_refs/stage_trace)
on every single star click, and would pull a Mongo-shaped concern into
core/experiment/runner.py, which the architecture fitness rule keeps free of
persistence deps. Same fine-grained-sub-resource choice already made for
per-question dataset edits (see datasets.py#add_question and friends).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["feedback"])


class FeedbackWriteRequest(BaseModel):
    """All fields optional — a PUT only touches the fields it sets. `scores`
    merges by key (setting `accuracy` alone must not clear an already-saved
    `completeness`) rather than replacing the whole dict. No fixed set of
    score dimensions is enforced here — the UI renders `accuracy`/
    `completeness`/`relevance` by default, but this stays an open dict so a
    new dimension can be added later without a schema migration (same
    philosophy as `question_type` in datasets.py: a free string, not an
    enum)."""
    rating: Literal["good", "bad"] | None = None
    scores: dict[str, float] = {}
    comment: str | None = None
    reviewer: str | None = None


async def _get_run_or_404(run_id: str, realm_id: str | None) -> dict[str, Any]:
    import adapters.mongodb as mdb
    doc = await mdb.find_one("experiment_runs", {"run_id": run_id})
    if not doc or (realm_id and doc.get("realm_id") != realm_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return doc


def _question_in_run(run_doc: dict[str, Any], question_id: str) -> dict[str, Any] | None:
    return next(
        (qr for qr in run_doc.get("question_results", []) if qr.get("question_id") == question_id),
        None,
    )


@router.put("/experiments/{run_id}/questions/{question_id}/feedback")
async def upsert_feedback(
    run_id: str, question_id: str, body: FeedbackWriteRequest, realm_id: str | None = None,
) -> dict[str, Any]:
    """Create-or-merge feedback for one question. Returns the merged doc."""
    import adapters.mongodb as mdb

    run_doc = await _get_run_or_404(run_id, realm_id)
    if _question_in_run(run_doc, question_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Question {question_id!r} not found in run {run_id!r}",
        )

    existing = await mdb.find_one("answer_feedback", {"run_id": run_id, "question_id": question_id})
    now = datetime.now(UTC).isoformat()

    # find_one stringifies _id for JSON-safety (adapters/mongodb.py); copying
    # that string straight into a replace_one payload makes Mongo see an
    # attempt to change the immutable _id (str != the original ObjectId) and
    # reject the write — found live, this made every *update* to an
    # existing feedback doc 500 while the first-ever create for a question
    # kept working (no `existing` doc means no `_id` to carry over).
    doc: dict[str, Any] = dict(existing) if existing else {
        "id": str(uuid.uuid4())[:8],
        "run_id": run_id,
        "question_id": question_id,
        "realm_id": run_doc.get("realm_id"),
        "rating": None,
        "scores": {},
        "comment": None,
        "reviewer": None,
        "created_at": now,
    }
    doc.pop("_id", None)

    if body.rating is not None:
        doc["rating"] = body.rating
    if body.scores:
        doc["scores"] = {**doc.get("scores", {}), **body.scores}
    if body.comment is not None:
        doc["comment"] = body.comment
    if body.reviewer is not None:
        doc["reviewer"] = body.reviewer
    doc["updated_at"] = now

    await mdb.upsert_one("answer_feedback", {"run_id": run_id, "question_id": question_id}, doc)
    return doc


@router.get("/experiments/{run_id}/feedback")
async def get_run_feedback(run_id: str, realm_id: str | None = None) -> dict[str, Any]:
    """All feedback for one run, keyed by question_id — one call powers both
    the run page and the copy-to-clipboard button (no N+1 per question)."""
    import adapters.mongodb as mdb

    await _get_run_or_404(run_id, realm_id)
    docs = await mdb.find_many("answer_feedback", query={"run_id": run_id})
    return {d["question_id"]: d for d in docs}


@router.delete("/experiments/{run_id}/questions/{question_id}/feedback", status_code=204)
async def delete_feedback(run_id: str, question_id: str, realm_id: str | None = None) -> None:
    import adapters.mongodb as mdb

    await _get_run_or_404(run_id, realm_id)
    deleted = await mdb.delete_one("answer_feedback", {"run_id": run_id, "question_id": question_id})
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"No feedback for question {question_id!r} in run {run_id!r}",
        )


class TriageDecideRequest(BaseModel):
    """Body for `POST .../feedback/triage/decide` — a human's verdict on a
    triage proposal (the improvement plan's first stage: the agent
    1": the agent proposes, a human decides). `edited_result` is required
    only for `action="edit"` — a partial dict merged over the stored
    triage_result (e.g. correcting `error_classes` the model got wrong)."""
    action: Literal["confirm", "reject", "edit"]
    edited_result: dict[str, Any] | None = None


async def _resolve_triage_generator(realm_id: str | None) -> Any:
    """The Realm's own local, self-hosted generator (triage must
    never introduce a new/cloud model dependency). Mirrors
    `services/api_gateway/main.py#_build_realm_scoped_generator`'s own
    resolution in miniature rather than importing it — `main.py` imports
    this router module, so the reverse import would be circular; falls
    back to the shared registry-resolved "ollama" instance (the same one
    chat/experiments already use without a Realm-specific override) when
    the Realm has no dedicated generator resource or resolution fails for
    any reason."""
    from core.registry import registry
    if realm_id:
        try:
            from services.api_gateway.routers.corpus import _get_realm_resource
            cfg = await _get_realm_resource(realm_id, "ollama")
            if cfg:
                from adapters.ollama_generator import OllamaGenerator
                host = cfg.get("host", "localhost")
                port = cfg.get("port", 11434)
                return OllamaGenerator(base_url=f"http://{host}:{port}", model=cfg.get("model"))
        except Exception:
            pass
    return registry.resolve("generator", "ollama")


async def _resolve_error_taxonomy(realm_id: str | None) -> list[dict[str, Any]] | None:
    """Generic, pack-agnostic lookup of an error taxonomy from whichever
    domain pack(s) are active for this Realm — mirrors
    services/api_gateway/main.py#_build_chat_pipeline's own "domain_hooks"
    resolution exactly (iterate active_packs, resolve "domain_hooks" per
    pack id, a pack listed active but not actually loaded at startup is
    logged and skipped, never a 500). This module has no hardcoded
    knowledge of any specific pack id (tests/unit/test_p1_guardian.py
    forbids a direct `from domain_packs...` import here) — `None` when no
    active pack provides one, which the caller turns into a clear "not
    configured for this Realm" error rather than guessing a default."""
    from core.registry import registry
    from services.api_gateway.routers.settings import _get_settings_doc

    settings = await _get_settings_doc(realm_id)
    for pack_id in settings.get("active_packs", []):
        try:
            hooks = registry.resolve("domain_hooks", pack_id)
        except KeyError:
            continue
        taxonomy = hooks.get("error_taxonomy")
        if taxonomy:
            return taxonomy
    return None


@router.post("/experiments/{run_id}/questions/{question_id}/feedback/triage")
async def triage_feedback(run_id: str, question_id: str, realm_id: str | None = None) -> dict[str, Any]:
    """Feedback triage: classifies the CURRENT stored
    comment for this question into structured error classes via the
    Realm's own local generator, deterministically cross-checks the
    extraction against this run's own retrieval trace (no LLM involved in
    that half), and proposes a minimal-radius lever from the taxonomy.

    Read-only recommendation only —
    first stage: this endpoint never creates a pin or changes
    anything about the run itself. It records a PROPOSAL on the feedback
    document; a human confirms, edits, or rejects it via the `/decide`
    endpoint below, and creating an actual pin from a confirmed proposal is
    a separate, explicit call to `POST /pins`.
    """
    import adapters.mongodb as mdb
    from core.eval.triage import (
        build_triage_prompt,
        cross_check,
        parse_triage_response,
        propose_lever,
    )

    run_doc = await _get_run_or_404(run_id, realm_id)
    question = _question_in_run(run_doc, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"Question {question_id!r} not found in run {run_id!r}")

    feedback_doc = await mdb.find_one("answer_feedback", {"run_id": run_id, "question_id": question_id})
    comment = (feedback_doc or {}).get("comment")
    if not comment:
        raise HTTPException(status_code=400, detail="This question has no feedback comment to triage yet")

    taxonomy = await _resolve_error_taxonomy(run_doc.get("realm_id"))
    if not taxonomy:
        # 409, not 503: nothing is down. The Realm has no domain pack active,
        # and the caller fixes that on the Domain packs page. A 503 says "come
        # back later", which is the one thing that will not help.
        raise HTTPException(
            status_code=409,
            detail="No error taxonomy is provided by any domain pack active for this Realm — "
                   "activate one on the Domain packs page first",
        )
    valid_classes = frozenset(c["id"] for c in taxonomy)

    generator = await _resolve_triage_generator(run_doc.get("realm_id"))
    prompt = build_triage_prompt(comment, taxonomy)
    try:
        raw = generator.generate(prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Triage generator call failed: {exc}") from exc

    result = parse_triage_response(raw, valid_classes)

    # Funnel verdict for cross_check — reuses the exact same helper
    # GET /experiments/{run_id} and POST /experiments/compare already rely
    # on, applied to a one-item copy of this question so it never mutates
    # the stored run document.
    from services.api_gateway.routers.experiments import _attach_funnel
    funnel_carrier = dict(question)
    _attach_funnel([funnel_carrier])
    funnel_layer = (funnel_carrier.get("funnel") or {}).get("layer")

    cross_check(
        result, source_refs=question.get("source_refs") or [],
        pre_rerank_source_refs=question.get("pre_rerank_source_refs"), funnel_layer=funnel_layer,
    )
    proposal = propose_lever(result, taxonomy)

    now = datetime.now(UTC).isoformat()
    doc = dict(feedback_doc)
    doc.pop("_id", None)
    doc["triage_result"] = {**result.to_dict(), "proposal": proposal}
    doc["triage_status"] = "proposed"
    doc["triage_created_at"] = now
    doc["updated_at"] = now
    await mdb.upsert_one("answer_feedback", {"run_id": run_id, "question_id": question_id}, doc)
    return doc


@router.post("/experiments/{run_id}/questions/{question_id}/feedback/triage/decide")
async def decide_triage(
    run_id: str, question_id: str, body: TriageDecideRequest, realm_id: str | None = None,
) -> dict[str, Any]:
    """Records a human's decision on an existing triage proposal — never
    creates a pin or any other side effect by itself (see
    `triage_feedback`'s own docstring for why)."""
    import adapters.mongodb as mdb

    await _get_run_or_404(run_id, realm_id)
    doc = await mdb.find_one("answer_feedback", {"run_id": run_id, "question_id": question_id})
    if not doc or not doc.get("triage_result"):
        raise HTTPException(status_code=404, detail="No triage proposal exists for this question yet")

    doc = dict(doc)
    doc.pop("_id", None)
    if body.action == "edit":
        if not body.edited_result:
            raise HTTPException(status_code=400, detail="edited_result is required for action='edit'")
        doc["triage_result"] = {**doc["triage_result"], **body.edited_result}
        doc["triage_status"] = "edited"
    elif body.action == "confirm":
        doc["triage_status"] = "confirmed"
    else:
        doc["triage_status"] = "rejected"
    doc["triage_decided_at"] = datetime.now(UTC).isoformat()
    await mdb.upsert_one("answer_feedback", {"run_id": run_id, "question_id": question_id}, doc)
    return doc


class PromoteToDatasetRequest(BaseModel):
    """Body for turning one question's feedback into a golden-dataset entry. `article_refs`, when omitted, is copied from the
    question's own entry in the run's source dataset — the reviewer only
    needs to supply refs when correcting them, not to re-type what's
    already the ground truth. `reference_answer`, when omitted, is copied
    from the run's own `question_results[]` entry unchanged — promoting
    does NOT regenerate or otherwise reinterpret the answer using the
    reviewer's comment (no LLM involved in this endpoint at all); if the
    comment says the ground truth itself is wrong, the reviewer corrects
    it here before submitting."""
    target_dataset_id: str
    article_refs: list[str] | None = None
    reference_answer: str | None = None


async def _source_article_refs(run_doc: dict[str, Any], question_id: str) -> list[str]:
    """Look up `article_refs` from the run's own source dataset — this
    isn't stored on a run's `question_results[]` entry (only the question
    text/answer/metrics are), so promoting a question forward means
    resolving the dataset the run was evaluated against and finding that
    question by its `id`. Falls back to an empty list (caller-supplied
    `article_refs` is then required in practice) rather than raising —
    a run whose source dataset was since deleted shouldn't block promoting
    its feedback, only lose the auto-filled convenience."""
    dataset_name = run_doc.get("dataset_name") or run_doc.get("config", {}).get("dataset_name", "")
    if not dataset_name:
        return []
    from services.api_gateway.routers.experiments import _load_dataset
    dataset = await _load_dataset(dataset_name)
    source_question = next(
        (q for q in dataset.questions if q.get("id") == question_id), None,
    )
    return source_question.get("article_refs", []) if source_question else []


def _find_existing_question(dataset_doc: dict[str, Any], question_text: str) -> dict[str, Any] | None:
    """Exact (trimmed) text match against the target dataset's own
    questions — promoting a question that's already in the target dataset
    (e.g. it was auto-generated earlier, or promoted once before) must
    update that entry in place, not create a second row for the same
    question. Matching by text rather than by id: ids are dataset-scoped
    (a question's id in the run's SOURCE dataset means nothing in a
    different TARGET dataset), so text is the only identity that survives
    the move."""
    needle = question_text.strip()
    return next(
        (q for q in dataset_doc.get("questions", []) if q.get("question", "").strip() == needle),
        None,
    )


@router.post("/experiments/{run_id}/questions/{question_id}/feedback/promote", status_code=201)
async def promote_to_dataset(
    run_id: str, question_id: str, body: PromoteToDatasetRequest, realm_id: str | None = None,
) -> dict[str, Any]:
    """Turn a reviewed question into a golden-dataset entry — the mechanical half of "feedback becomes a test": copies the
    question/reference_answer/article_refs into `target_dataset_id`, tagged
    with `provenance.origin="reviewer_feedback"` and backlinks to the run/
    question/feedback it came from, so `DatasetsPage` can show where a
    golden question actually originated instead of collapsing it into
    "manual" alongside curator-authored ones.

    If the target dataset already has this exact question (found live: the
    question generator had already added it as `origin="generated"`, and
    promoting it again created a silent 51st duplicate instead of
    confirming the existing one), the existing entry is updated in place —
    `reference_answer`/`article_refs` refreshed, `provenance.origin` becomes
    `reviewer_feedback` with `previous_origin` preserving what it was before
    — rather than appended a second time. Response carries `created: bool`
    so the caller can tell the two cases apart."""
    import adapters.mongodb as mdb

    run_doc = await _get_run_or_404(run_id, realm_id)
    qr = _question_in_run(run_doc, question_id)
    if qr is None:
        raise HTTPException(
            status_code=404, detail=f"Question {question_id!r} not found in run {run_id!r}",
        )

    question_text = qr.get("question", "")
    reference_answer = body.reference_answer if body.reference_answer is not None else qr.get("reference_answer", "")
    article_refs = body.article_refs if body.article_refs is not None else (
        await _source_article_refs(run_doc, question_id)
    )
    feedback_doc = await mdb.find_one("answer_feedback", {"run_id": run_id, "question_id": question_id})
    provenance_backlink = {
        "source_run_id": run_id,
        "source_question_id": question_id,
        "source_feedback_id": feedback_doc.get("id") if feedback_doc else None,
    }

    from services.api_gateway.routers.datasets import (
        QuestionWriteRequest,
        _dataset_detail,
        _get_dataset_or_404,
        add_question,
    )

    target_doc = await _get_dataset_or_404(body.target_dataset_id, realm_id)
    existing = _find_existing_question(target_doc, question_text)

    if existing is None:
        result = await add_question(
            body.target_dataset_id,
            QuestionWriteRequest(
                question=question_text,
                reference_answer=reference_answer,
                article_refs=article_refs,
                provenance={"origin": "reviewer_feedback", **provenance_backlink},
            ),
            realm_id,
        )
        return {**result, "created": True}

    now = datetime.now(UTC).isoformat()
    previous_provenance = existing.get("provenance") or {}
    updated = {
        **existing,
        "reference_answer": reference_answer,
        "article_refs": article_refs,
        "provenance": {
            "origin": "reviewer_feedback",
            "previous_origin": previous_provenance.get("origin"),
            "created_at": previous_provenance.get("created_at", now),
            "updated_at": now,
            **provenance_backlink,
        },
    }
    questions = [updated if q.get("id") == existing["id"] else q for q in target_doc.get("questions", [])]
    await mdb.update_one("datasets", {"id": body.target_dataset_id}, {"$set": {"questions": questions}})
    target_doc["questions"] = questions
    return {**_dataset_detail(target_doc), "created": False}


@router.get("/feedback")
async def list_feedback(
    realm_id: str | None = None, run_id: str | None = None,
    rating: str | None = None, limit: int = 200,
) -> list[dict[str, Any]]:
    """Cross-run feedback query for analysis (by developers or an agent
    looking for patterns, e.g. every 'bad'-rated answer this month) — the
    one part of this router with no existing per-run precedent. Each result
    is enriched with its parent run's question/generated_answer text so a
    caller doesn't need a second lookup per item."""
    import adapters.mongodb as mdb

    query: dict[str, Any] = {}
    if realm_id:
        query["realm_id"] = realm_id
    if run_id:
        query["run_id"] = run_id
    if rating:
        query["rating"] = rating
    docs = await mdb.find_many("answer_feedback", query=query, sort=[("updated_at", -1)], limit=limit)

    run_ids = {d["run_id"] for d in docs}
    runs_by_id: dict[str, dict[str, Any]] = {}
    for rid in run_ids:
        run_doc = await mdb.find_one("experiment_runs", {"run_id": rid})
        if run_doc:
            runs_by_id[rid] = run_doc

    enriched = []
    for d in docs:
        run_doc = runs_by_id.get(d["run_id"])
        qr = _question_in_run(run_doc, d["question_id"]) if run_doc else None
        enriched.append({
            **d,
            "question": qr.get("question") if qr else None,
            "generated_answer": qr.get("generated_answer") if qr else None,
            "run_config_name": run_doc.get("config_name") if run_doc else None,
        })
    return enriched
