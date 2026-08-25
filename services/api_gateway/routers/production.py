"""Closing the loop with production.

Every failure the platform sees today comes from a run over a golden set, so
it only ever sees questions somebody thought of in advance. A question nobody
anticipated cannot fail a test that does not exist.

Production traces close that gap. A bounded, off-by-default
trace log into the template, so any system grown from it can hand its recent
queries over; this router collects them, turns one into a golden question,
and reports where the golden set has nothing to say about real traffic.

**Pull, never push.** The platform reaches out to a served system's `/traces`
endpoint on a schedule its operator chooses. A served system that pushed
would need to know the platform's address and be up when it is — exactly the
dependency this platform refuses to create. Pulling also means a
system that has never heard of the platform is unaffected by its existence.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/production", tags=["production"])

# One collection, not one per Realm: a trace carries its Realm and corpus,
# and splitting storage by them would make "what is the whole traffic like"
# a query across an unknown number of collections.
_COLLECTION = "production_traces"


class CollectRequest(BaseModel):
    """Where to pull from, and what to file the result under.

    `url` is the served system's own trace endpoint. The platform does not
    guess it from a registration, because a system may expose traces from a
    different host than it answers queries on, and guessing would silently
    collect nothing.
    """

    realm_id: str
    corpus_id: str
    url: str
    limit: int = Field(default=200, ge=1, le=2000)


@router.post("/collect")
async def collect_traces(body: CollectRequest) -> dict[str, Any]:
    """Pull recent traces and file them as feedback candidates.

    Candidates, not feedback: a trace records what happened, and whether it
    was *wrong* is a human judgement nobody has made yet. Storing them as
    feedback would manufacture verdicts out of traffic.

    Idempotent on `trace_id`, so a collector run twice over an overlapping
    window does not double-count. A served system's log is a ring, so
    overlap is the normal case rather than an error.
    """
    import adapters.mongodb as mdb

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(body.url, params={"limit": body.limit})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"trace endpoint unreachable: {exc}") from exc

    if not payload.get("enabled", True):
        # A clear answer rather than an empty success: a deployment that has
        # not switched recording on looks identical to one with no traffic,
        # and the operator needs to tell those apart.
        return {"collected": 0, "stored": 0, "recording_enabled": False}

    traces = payload.get("traces") or []
    stored = 0
    for trace in traces:
        trace_id = str(trace.get("trace_id") or "")
        if not trace_id:
            continue
        await mdb.upsert_one(_COLLECTION, {"trace_id": trace_id}, {
            "trace_id": trace_id,
            "realm_id": body.realm_id,
            "corpus_id": body.corpus_id,
            "query": trace.get("query") or "",
            "sources": trace.get("sources") or [],
            "answer_preview": trace.get("answer_preview") or "",
            "created_at": trace.get("created_at") or "",
            "collected_at": datetime.now(UTC).isoformat(),
            # Set when somebody turns this into a golden question, so the
            # same production failure is not promoted twice by two readers.
            "promoted_question_id": None,
        })
        stored += 1

    return {"collected": len(traces), "stored": stored, "recording_enabled": True}


@router.get("")
async def list_traces(realm_id: str, corpus_id: str, limit: int = 100) -> list[dict[str, Any]]:
    import adapters.mongodb as mdb

    docs = await mdb.find_many(
        _COLLECTION, {"realm_id": realm_id, "corpus_id": corpus_id},
        sort=[("created_at", -1)],
    )
    for d in docs:
        d.pop("_id", None)
    docs = docs[:limit]

    # `promotion_live` separates "this became a question" from "that question
    # still exists". The mark on a trace records history and cannot know that
    # somebody deleted the question afterwards; without this field the page
    # would keep offering a dead reference and keep refusing a new promotion.
    #
    # One query for the whole page rather than one per trace, since a page of
    # a hundred traces would otherwise cost a hundred round trips to answer a
    # question about a handful of them.
    marked = {str(d["promoted_question_id"]) for d in docs if d.get("promoted_question_id")}
    live: set[str] = set()
    if marked:
        try:
            for dataset in await mdb.find_many("datasets", {"questions.id": {"$in": list(marked)}}):
                live |= {str(q.get("id")) for q in dataset.get("questions") or []}
        except Exception:
            # Unreachable storage answers "still there" for every mark: the
            # opposite guess would invite duplicate golden questions during an
            # outage, which is harder to notice and undo than a refusal.
            live = set(marked)
    for d in docs:
        mark = d.get("promoted_question_id")
        d["promotion_live"] = bool(mark) and str(mark) in live
    return docs


class PromoteRequest(BaseModel):
    """One action from a production failure to a golden question.

    `reference_answer` is required and never derived from the trace. What the
    system answered is precisely what a reviewer is disputing, so promoting
    it as the expected answer would enshrine the failure as the standard.
    """

    trace_id: str
    dataset_id: str
    reference_answer: str = Field(min_length=1)
    realm_id: str
    # Optional: the reviewer may already know which sources should have been
    # returned. Left empty, the question still measures generation and
    # refusal, and `answerability` classification will say what it can.
    article_refs: list[str] = Field(default_factory=list)


async def _promoted_question_exists(question_id: str) -> bool:
    """Whether the question a trace claims to have become is still there.

    Asked of the datasets collection rather than trusted from the mark on the
    trace. The mark records that a promotion happened once; it cannot know
    that somebody later deleted the question, and treating it as proof of a
    live question is what turned a deletion into a permanent refusal.

    A storage failure answers *yes*, deliberately. The alternative — assuming
    the question is gone whenever the database is unreachable — would let a
    reader create a duplicate during an outage, and a duplicate golden
    question is harder to notice and undo than a refusal is.
    """
    import adapters.mongodb as mdb

    try:
        doc = await mdb.find_one("datasets", {"questions.id": question_id})
    except Exception:
        return True
    return bool(doc)


@router.post("/promote", status_code=201)
async def promote_trace(body: PromoteRequest) -> dict[str, Any]:
    """Turns one collected trace into a golden question.

    The question text is the user's own wording, unedited. A tidied-up
    paraphrase would test a question nobody asked, and the whole value of
    production traffic is that it is not what anybody would have thought to
    write.
    """
    import adapters.mongodb as mdb
    from services.api_gateway.routers.datasets import QuestionWriteRequest, add_question

    trace = await mdb.find_one(_COLLECTION, {"trace_id": body.trace_id})
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {body.trace_id!r} not found")
    previous = trace.get("promoted_question_id")
    if previous and await _promoted_question_exists(str(previous)):
        raise HTTPException(
            status_code=409,
            detail=f"already promoted as question {previous!r}",
        )

    payload = QuestionWriteRequest(
        question=trace.get("query") or "",
        reference_answer=body.reference_answer,
        article_refs=body.article_refs,
        origin="production_trace",
        trace_id=body.trace_id,
    )
    result = await add_question(body.dataset_id, payload, realm_id=body.realm_id)

    question_id = next(
        (q.get("id") for q in reversed(result.get("questions") or [])
         if q.get("trace_id") == body.trace_id),
        str(uuid.uuid4())[:8],
    )
    await mdb.update_one(
        _COLLECTION, {"trace_id": body.trace_id},
        {"$set": {"promoted_question_id": question_id}},
    )
    return {"question_id": question_id, "dataset": result.get("name"), "count": result.get("count")}


@router.get("/coverage")
async def golden_set_coverage(
    realm_id: str, corpus_id: str, dataset_name: str, threshold: float = 0.6,
) -> dict[str, Any]:
    """Where the golden set says nothing about real traffic.

    Embedding happens here, in the services layer, because `core/eval/drift.py`
    must stay free of a model dependency. The platform's own embedder is used
    for both sides, so the two are measured on one scale — comparing vectors
    from two different models would produce a number with no meaning at all.
    """
    import adapters.mongodb as mdb
    from core.eval.drift import coverage_report, nearest_similarities, uncovered_count
    from core.registry import registry
    from services.api_gateway.routers.experiments import _load_dataset

    docs = await mdb.find_many(_COLLECTION, {"realm_id": realm_id, "corpus_id": corpus_id})
    questions = [str(d.get("query") or "") for d in docs if d.get("query")]
    dataset = await _load_dataset(dataset_name)

    # Found by re-verification: `_load_dataset` falls back to a five-question
    # stub of invented text when a name matches nothing. Coverage then
    # compares real traffic against fabricated questions and reports "100%
    # uncovered" — a confidently wrong number that reads as a finding. A
    # missing dataset is a caller mistake and is said so rather than measured.
    if dataset_name not in ("stub", "") and getattr(dataset, "name", "") == "stub":
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset {dataset_name!r} not found; refusing to measure coverage "
                "against the stub dataset it would otherwise fall back to"
            ),
        )

    golden = [str(q.get("question") or "") for q in dataset.questions if q.get("question")]

    if not questions:
        return {
            "threshold": threshold, "n_production": 0, "n_golden": len(golden),
            "uncovered_share": 0.0, "uncovered": [], "deciles": [],
        }

    embedder = registry.resolve("embedder", "bge_m3")
    nearest = nearest_similarities(
        [list(v) for v in embedder.embed(questions)],
        [list(v) for v in embedder.embed(golden)] if golden else [],
    )
    report = coverage_report(questions, nearest, n_golden=len(golden), threshold=threshold)
    data = report.to_dict()
    # The share counts every uncovered question, while the list is capped —
    # stated explicitly so the two numbers are not read as the same thing.
    data["uncovered_share"] = round(
        uncovered_count(nearest, threshold) / len(questions), 4
    )
    data["uncovered_total"] = uncovered_count(nearest, threshold)
    return data
