"""The failure catalogue, served so a person can read it.

The catalogue itself is `core/eval/atlas.py`: data the build checks, not a
document somebody keeps up to date. This router hands it to the interface and
adds the one thing the catalogue deliberately refuses to hold, which is the
reverse pointer.

Direction matters. The catalogue knows which signals evidence a failure; the
signals know nothing about the catalogue, so a detector never has to be edited
because a description of it changed. Asking "which failures does this signal
evidence" is therefore a join, and it happens here, in the layer that composes
a response, and not in either of the two things being joined.

Read-only, and that is a decision, never an omission. An entry claims a
failure is detected, and the build refuses such a claim without a bait. Let the
interface write entries and that refusal cannot hold, so the catalogue would go
back to asserting what nobody checked. Recording a failure somebody has met is
a separate path with a separate shape, and it does not put a row in here.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.eval import atlas as catalogue
from core.eval import rag_space

router = APIRouter(prefix="/atlas", tags=["atlas"])

#: Where a failure somebody met is written down before it is anything else.
#: Deliberately not the catalogue: an entry there claims a signal catches the
#: failure, and the build refuses that claim without a bait. A candidate makes
#: no such claim, so it can be written from the interface, and the refusal
#: keeps holding for everything that does.
CANDIDATES = "atlas_candidates"


def attach_failure_ids(items: list[dict[str, Any]], kind: str) -> None:
    """Name, on each finding, the catalogue entries it is evidence for.

    The join lives here because neither side may hold it. The catalogue points
    at signals; a signal that pointed back would have to be edited whenever a
    description of it changed, and a detector edited for that reason is a
    detector nobody trusts.

    A finding no entry names keeps an empty list, and that is worth seeing: it
    means the platform is saying something the catalogue has no place for.
    """
    index = catalogue.signal_index()
    for item in items:
        item["failure_ids"] = sorted(index.get(f"{kind}:{item.get('id')}", []))


def _entry(f: catalogue.FailureMode, point: rag_space.Point | None = None) -> dict[str, Any]:
    applies = f.applies_to(point) if point is not None else None
    return {
        "id": f.id,
        "title": f.title,
        "title_key": f.title_key,
        "atlas_rows": list(f.atlas_rows),
        "stage_origin": f.stage_origin,
        "stage_visible": f.stage_visible,
        "severity": {
            "quiet": f.severity.quiet,
            "cost": f.severity.cost,
            "prevalence": f.severity.prevalence,
            "total": f.severity.total,
        },
        "origin": f.origin,
        "detection": f.detection,
        "state": f.state,
        "instrument": f.instrument,
        "signals": [{"id": s.id, "kind": s.kind, "name": s.name, "side": s.side} for s in f.signals],
        "applies_when": [
            {
                "code": code,
                "values": list(values),
                "dimension": (d.name if (d := rag_space.get(code)) else None),
                "url": rag_space.dimension_url(code),
            }
            for code, values in f.applies_when
        ],
        "applies_here": applies,
        # Where else this entry can occur. A reader following a link to an entry
        # that cannot occur in the architecture on screen has to be taken
        # somewhere it can, and guessing "any other point" can send them back
        # and forth between two that both exclude it.
        "applies_to_points": sorted(
            name for name, coords in rag_space.POINTS.items()
            if f.applies_to(coords) is True
        ),
        "bait": f.bait,
        "bait_level": (
            "proving_ground" if f.bait.startswith("tests/proving_ground/")
            else "unit" if f.bait else None
        ),
        "not_detected_reason": f.not_detected_reason,
        "scope_caveat": f.scope_caveat,
        "shares_signals_with": list(f.shares_signals_with),
        "superseded_by": list(f.superseded_by),
    }


async def _declared_points() -> dict[str, rag_space.Point]:
    """Architectures somebody registered, beside the ones this platform runs.

    A registered system declares where it sits in the space, because a probe
    cannot see it. The declaration decides which entries can occur in that
    system at all, so it belongs here beside the platform's own points: a
    declaration nothing reads is the decorative field this catalogue exists
    to catch.

    An unreachable store yields nothing, and the platform's own points still
    answer. A registered system missing from the list reads as one nobody
    declared coordinates for, which is what it is until somebody does.
    """
    try:
        import adapters.mongodb as mdb
        docs = await mdb.find_many("external_rags", {})
    except Exception:  # noqa: BLE001 - the built-in points are the answer either way
        return {}
    return {
        str(doc.get("name") or doc.get("id")): dict(doc["coordinates"])
        for doc in docs
        if doc.get("coordinates") and not doc.get("deleted_at")
    }


@router.get("")
async def read_atlas(point: str | None = None) -> dict[str, Any]:
    """The catalogue, optionally narrowed to one architecture.

    Without a point the whole catalogue is returned and every entry's
    `applies_here` is null, which is honest: applicability is a question about
    an architecture and there is none to ask about.

    With a point, `applies_here` is true, false, or **null**, and the third is
    not a rounding of the second. Null means the point records nothing about a
    coordinate the entry depends on, so the failure is neither possible nor
    ruled out. Collapsing it into false would report a failure excluded when it
    was only never asked about.
    """
    points: dict[str, rag_space.Point] = {**rag_space.POINTS, **await _declared_points()}

    coordinates: rag_space.Point | None = None
    if point is not None:
        if point not in points:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown architecture {point!r}. Known: {sorted(points)}",
            )
        coordinates = points[point]

    entries = [_entry(f, coordinates) for f in catalogue.FAILURES]
    payload: dict[str, Any] = {
        "entries": entries,
        "schema": {
            "source": rag_space.SCHEMA_SOURCE,
            "release": rag_space.SCHEMA_RELEASE,
            # The two dates differ whenever the published data has been rebuilt
            # since the last release was cut, which is the state on the day
            # this was written: the copy carries two values that the release
            # named above predates. Sent so the interface can say which date it
            # means; the label it prints today names the release only.
            "verified_on": rag_space.SCHEMA_VERIFIED_ON,
            "dimensions": rag_space.SCHEMA_DIMENSION_COUNT,
        },
        "points": {
            name: {
                "coordinates": [
                    {"code": c, "value": v, "url": rag_space.dimension_url(c)}
                    for c, v in sorted(coords.items())
                ],
                "applicable": sum(
                    1 for f in catalogue.FAILURES if f.applies_to(coords) is True
                ),
                # Whose point it is. A system somebody registered is described
                # by its own declaration, and a reader has to be able to tell
                # that from a shape this platform runs itself.
                "declared": name not in rag_space.POINTS,
            }
            for name, coords in points.items()
        },
    }
    if coordinates is not None:
        payload["point"] = point
        payload["uncovered_coordinates"] = [
            {
                "code": code, "value": value,
                "dimension": (d.name if (d := rag_space.get(code)) else None),
                "url": rag_space.dimension_url(code),
            }
            for code, value in sorted(catalogue.uncovered_coordinates(coordinates).items())
        ]
    return payload


@router.get("/signals")
async def read_signals() -> dict[str, Any]:
    """Every signal named by the catalogue, and what each is evidence for.

    A signal standing for more than one entry is reported as such and never
    listed twice: where two entries name an identical set, a firing signal is
    evidence for either and for neither in particular, and a reader who is not
    told that takes it for evidence about the entry in front of them.
    """
    index = catalogue.signal_index()
    by_id = {f.id: f for f in catalogue.FAILURES}
    return {
        "signals": [
            {
                "id": signal_id,
                "side": "ui" if signal_id.startswith("ui:") else "core",
                "kind": signal_id.split(":", 1)[0],
                "failures": sorted(failure_ids),
                "singles_out": len(failure_ids) == 1,
                "bait_level": sorted(
                    {
                        (
                            "proving_ground"
                            if by_id[i].bait.startswith("tests/proving_ground/")
                            else "unit" if by_id[i].bait else "none"
                        )
                        for i in failure_ids
                    }
                ),
            }
            for signal_id, failure_ids in sorted(index.items())
        ],
    }


class CandidateRequest(BaseModel):
    """A failure somebody met, in the words of the person who met it."""

    title: str = Field(min_length=8, max_length=200)
    #: What the system looked like while it was failing. Asked for because a
    #: failure nobody could have noticed is the only kind worth cataloguing,
    #: and the answer is what a later reader needs to recognise it again.
    looked_like: str = Field(min_length=8, max_length=2000)
    #: Where it was seen: a run identifier, a corpus, a realm. Free text on
    #: purpose, because a person who has just met a failure should not have
    #: to know which of the platform's nouns applies to it.
    observed_on: str = Field(default="", max_length=500)
    #: What the reporter thinks would have caught it. Their guess, kept as
    #: theirs, and never resolved against the signal registry here: a name
    #: that does not resolve is worth reading, and rejecting the report for
    #: it would lose the report.
    suspected_signal: str = Field(default="", max_length=200)


class CandidateDecision(BaseModel):
    """The triage step: is this a failure of a served system, or of ours?"""

    status: Literal["accepted", "rejected"]
    #: Why. Required for a rejection, because "this belongs to the other
    #: catalogue" is information the reporter needs and the next reader too.
    note: str = Field(default="", max_length=2000)


class CandidatePromotion(BaseModel):
    """Which catalogue entry the candidate became."""

    failure_id: str


def _candidate(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    # Said on every candidate, on every path out of here, because a candidate
    # and an entry look alike on a screen and only one of them has been
    # proven to be caught by anything.
    out["confirmed_by_a_bait"] = False
    return out


@router.get("/candidates")
async def list_candidates(realm_id: str | None = None, status: str | None = None) -> dict[str, Any]:
    """Failures people have reported, never mixed with the catalogue itself."""
    import adapters.mongodb as mdb

    query: dict[str, Any] = {}
    if realm_id is not None:
        query["realm_id"] = realm_id
    if status is not None:
        query["status"] = status
    docs = await mdb.find_many(CANDIDATES, query, sort=[("created_at", -1)])
    return {"candidates": [_candidate(d) for d in docs]}


@router.post("/candidates")
async def create_candidate(body: CandidateRequest, realm_id: str = "") -> dict[str, Any]:
    import adapters.mongodb as mdb

    now = datetime.now(UTC).isoformat()
    doc = {
        "candidate_id": f"C{uuid.uuid4().hex[:8]}",
        "realm_id": realm_id,
        **body.model_dump(),
        "status": "proposed",
        "note": "",
        "promoted_to": "",
        "created_at": now,
        "updated_at": now,
    }
    await mdb.insert_one(CANDIDATES, dict(doc))
    return _candidate(doc)


async def _candidate_or_404(candidate_id: str) -> dict[str, Any]:
    import adapters.mongodb as mdb

    doc = await mdb.find_one(CANDIDATES, {"candidate_id": candidate_id})
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Unknown candidate {candidate_id!r}")
    return doc


@router.post("/candidates/{candidate_id}/decide")
async def decide_candidate(candidate_id: str, body: CandidateDecision) -> dict[str, Any]:
    """Accepted means a failure of a served system; rejected means ours.

    The line is the one the catalogue itself draws, and drawing it is a
    person's judgement: a report phrased without a single identifier of this
    repository can still be about this repository, and one naming a module
    can still be a universal failure somebody happened to describe in local
    words.
    """
    import adapters.mongodb as mdb

    if body.status == "rejected" and not body.note.strip():
        raise HTTPException(
            status_code=400,
            detail="A rejection needs a reason: the reporter and the next reader both need it.",
        )
    doc = dict(await _candidate_or_404(candidate_id))
    doc.pop("_id", None)
    doc["status"] = body.status
    doc["note"] = body.note
    doc["updated_at"] = datetime.now(UTC).isoformat()
    await mdb.upsert_one(CANDIDATES, {"candidate_id": candidate_id}, doc)
    return _candidate(doc)


@router.post("/candidates/{candidate_id}/promoted")
async def record_promotion(candidate_id: str, body: CandidatePromotion) -> dict[str, Any]:
    """Records which entry a candidate became, once it is one.

    Promotion itself is a change to the repository, and it has to be: an
    entry arrives with a bait, and a bait is a test. What is recorded here is
    the pointer, and only after the entry exists, so the candidate cannot
    claim to have become something nobody wrote.
    """
    import adapters.mongodb as mdb

    if catalogue.get(body.failure_id) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The catalogue holds no {body.failure_id!r}. Promotion is a change to the "
                "repository: add the entry with its bait, then record the pointer here."
            ),
        )
    doc = dict(await _candidate_or_404(candidate_id))
    doc.pop("_id", None)
    if doc.get("status") != "accepted":
        raise HTTPException(
            status_code=400,
            detail=f"Candidate {candidate_id!r} is {doc.get('status')!r}, so it was never accepted.",
        )
    doc["promoted_to"] = body.failure_id
    doc["updated_at"] = datetime.now(UTC).isoformat()
    await mdb.upsert_one(CANDIDATES, {"candidate_id": candidate_id}, doc)
    return _candidate(doc)


# Declared AFTER every literal path above. FastAPI takes the first route that
# matches, so this one would otherwise answer "the failure whose id is
# candidates" and return a 404 for the whole mechanism. The same trap caught
# DELETE /experiments/baseline once already.
@router.get("/{failure_id}")
async def read_failure(failure_id: str) -> dict[str, Any]:
    failure = catalogue.get(failure_id)
    if failure is None:
        raise HTTPException(status_code=404, detail=f"Unknown failure {failure_id!r}")
    return _entry(failure)
