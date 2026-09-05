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

from typing import Any

from fastapi import APIRouter, HTTPException

from core.eval import atlas as catalogue
from core.eval import rag_space

router = APIRouter(prefix="/atlas", tags=["atlas"])


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
    coordinates: rag_space.Point | None = None
    if point is not None:
        if point not in rag_space.POINTS:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown architecture {point!r}. Known: {sorted(rag_space.POINTS)}",
            )
        coordinates = rag_space.POINTS[point]

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
            }
            for name, coords in rag_space.POINTS.items()
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


@router.get("/{failure_id}")
async def read_failure(failure_id: str) -> dict[str, Any]:
    failure = catalogue.get(failure_id)
    if failure is None:
        raise HTTPException(status_code=404, detail=f"Unknown failure {failure_id!r}")
    return _entry(failure)
