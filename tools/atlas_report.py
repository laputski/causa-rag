"""What the platform detects, per architecture, and what it does not.

Run it and read the last column. An entry is `caught` only where a signal was
named and a bait proved it fires on the defect and stays silent without it;
`visible` where the data shows something and a person concludes; `not caught`
where nothing does, together with what is missing. A fourth state, `not
staged`, is neither of those: the failure has never been reproduced, so
nothing is known about whether anything would fire.

The report is computed for one architecture at a time, and this is not a
presentation choice. A count over the whole catalogue answers no question
anybody has: half the entries cannot occur in a dense-only system and a
different half cannot occur in a graph one. Ask it for a point and it also
prints the coordinates of that point which no entry speaks about at all,
which is how a gap becomes visible before the work begins.

    python3 -m tools.atlas_report
    python3 -m tools.atlas_report --point graph
    python3 -m tools.atlas_report --selectivity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.eval import rag_space
from core.eval.atlas import FAILURES, FailureMode, applicable_to, uncovered_coordinates
from core.eval.rag_space import POINTS

#: Where a proving-ground bait files what it observed. One file per entry,
#: written by the run itself, so this answer comes from the filesystem and not
#: from a constant. It was a constant returning False, which was true on the
#: day it was written and stayed in the file after the first entries were
#: reproduced, reporting nothing staged while six were.
EVIDENCE = Path(__file__).resolve().parent.parent / "eval" / "results" / "proving_ground"


def _staged(f: FailureMode) -> bool:
    """Whether the failure has been reproduced on the proving ground.

    Reported and not assumed, so `not staged` never reads as `not caught`: the
    first says nobody looked, the second says somebody looked and nothing
    spoke.
    """
    return (EVIDENCE / f"{f.id}.json").is_file()


def rows(point_name: str) -> list[dict[str, Any]]:
    point = POINTS[point_name]
    applicable, undetermined = applicable_to(point)
    out: list[dict[str, Any]] = []
    for f in applicable:
        out.append({
            "id": f.id,
            "stage": f.stage_visible,
            "title": f.title,
            "severity": f.severity.total,
            "state": f.state if _staged(f) else f"{f.state} (unstaged)",
            "signals": [s.id for s in f.signals],
            "instrument": f.instrument,
            "staged": _staged(f),
            "missing": f.not_detected_reason,
            "caveat": f.scope_caveat,
            "shares_with": list(f.shares_signals_with),
        })
    out.sort(key=lambda r: (-r["severity"], r["id"]))
    return out


def render(point_name: str) -> str:
    point = POINTS[point_name]
    table = rows(point_name)
    applicable, undetermined = applicable_to(point)
    gaps = uncovered_coordinates(point)

    lines = [
        f"Atlas coverage for the {point_name!r} point",
        f"  coordinates: {', '.join(f'{k}={v}' for k, v in sorted(point.items()))}",
        f"  vendored schema {rag_space.SCHEMA_RELEASE} from {rag_space.SCHEMA_SOURCE}",
        "",
        f"  {len(applicable)} of {len(FAILURES)} entries apply here"
        + (f", {len(undetermined)} undetermined" if undetermined else ""),
        "",
    ]
    width = max((len(r["title"]) for r in table), default=0)
    for r in table:
        lines.append(
            f"  {r['id']}  {r['severity']:2d}  {r['stage']:<11} {r['title']:<{width}}  "
            f"{r['state']}"
        )
    counts: dict[str, int] = {}
    for r in table:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    lines += ["", "  " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]

    staged = sum(1 for r in table if r["staged"])
    lines.append(
        f"  {staged} of {len(table)} reproduced on the proving ground: every other row's "
        "state rests on a fixture, not on a run"
    )

    shared = [r for r in table if r["shares_with"]]
    if shared:
        lines += ["", f"  {len(shared)} entries are not singled out by their own signals:"]
        for r in shared:
            lines.append(
                f"    {r['id']}  shares {', '.join(r['signals'])} with {', '.join(r['shares_with'])}"
            )
        lines.append(
            "  A signal firing there is evidence for either of them, and for neither "
            "in particular."
        )

    caveats = [r for r in table if r["caveat"]]
    if caveats:
        lines += ["", f"  {len(caveats)} entries carry a scope caveat:"]
        for r in caveats:
            lines.append(f"    {r['id']}  {r['caveat']}")

    if gaps:
        lines += [
            "",
            "  coordinates of this point that no entry speaks about:",
        ]
        for code, value in sorted(gaps.items()):
            d = rag_space.get(code)
            lines.append(
                f"    {code}={value}  ({d.name if d else '?'})  {rag_space.dimension_url(code)}"
            )
        lines.append(
            "  Nothing is known about failures peculiar to these. That is a gap in the "
            "catalogue, not a clean bill of health."
        )
    else:
        lines += ["", "  every coordinate of this point is spoken about by some entry"]
    return "\n".join(lines)


def selectivity() -> str:
    """Which entries share a signal.

    Co-firing is not forbidden, because failures do accompany one another. A
    signal standing as evidence for six entries at once is evidence for none of
    them in particular, and that has to be visible.
    """
    index: dict[str, list[str]] = {}
    for f in FAILURES:
        for s in f.signals:
            index.setdefault(s.id, []).append(f.id)
    lines = ["Signals and the entries they are evidence for", ""]
    for sid, ids in sorted(index.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"  {sid:<42} {len(ids)}  {', '.join(sorted(ids))}")
    lonely = [f.id for f in FAILURES if f.detection != "none" and not f.signals]
    if lonely:
        lines += ["", f"  claiming detection and naming no signal: {lonely}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--point", choices=sorted(POINTS), default=None,
                        help="one architecture; every one of them when omitted")
    parser.add_argument("--selectivity", action="store_true",
                        help="which entries share a signal")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    if args.selectivity:
        print(selectivity())
        return 0

    names = [args.point] if args.point else sorted(POINTS)
    if args.json:
        print(json.dumps({n: rows(n) for n in names}, ensure_ascii=False, indent=2))
        return 0
    for i, name in enumerate(names):
        if i:
            print()
        print(render(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
