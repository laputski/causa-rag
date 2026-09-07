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
    python3 -m tools.atlas_report --scaffold C1a2b3c4d
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
    evidence = EVIDENCE / f"{f.id}.json"
    if not evidence.is_file():
        return False
    # A file recording why an entry cannot be staged here is not a staging.
    # Counting one would put an entry in the reproduced column for having
    # proved that it could not be reproduced.
    try:
        return bool(json.loads(evidence.read_text(encoding="utf-8")).get("reproduced", True))
    except (OSError, ValueError):
        return True


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
    # The trailing clause is about the rows that were not reproduced, so it is
    # printed only while there are some. Left unconditional, a report of a
    # point where every row has been run said "every other row's state rests
    # on a fixture" about no rows at all.
    lines.append(
        f"  {staged} of {len(table)} reproduced on the proving ground"
        + (": every other row's state rests on a fixture, not on a run"
           if staged < len(table) else ", and none of them on a fixture alone")
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


def _next_free_id() -> str:
    taken = {int(f.id[1:]) for f in FAILURES if f.id[1:].isdigit()}
    return f"F{max(taken) + 1:02d}"


def scaffold(candidate: dict[str, Any]) -> str:
    """A draft entry and a draft bait, with every judgement left undone.

    What it fills in is what the reporter already wrote. What it refuses to
    fill in is everything a person has to decide: how loud the failure is,
    which coordinates admit it, and above all whether any signal catches it.
    A scaffold that guessed those would be a way of adding an unproven claim
    to the catalogue with less typing, which is the one thing the whole
    apparatus exists to prevent.
    """
    new_id = _next_free_id()
    title = candidate.get("title", "").replace('"', "'")
    lines = [
        f"# Candidate {candidate.get('candidate_id')}, reported on "
        f"{candidate.get('created_at', '')[:10]}, seen on "
        f"{candidate.get('observed_on') or 'nothing recorded'}.",
        "#",
        "# What the reporter saw:",
        *(f"#   {line}" for line in _wrapped(candidate.get("looked_like", ""))),
        "#",
        f"# Suspected signal, their guess and not a finding: "
        f"{candidate.get('suspected_signal') or 'none named'}",
        "",
        "# ── into core/eval/atlas.py ───────────────────────────────────────",
        "    FailureMode(",
        f'        id="{new_id}",',
        "        atlas_rows=(),  # DECIDE: the rows of the document this grew from, if any",
        f'        title="{title}",',
        '        stage_origin="",  # DECIDE: where it arises',
        '        stage_visible="",  # DECIDE: the earliest stage a signal could exist',
        "        severity=Severity(0, 0, 0),  # DECIDE: quiet, cost, prevalence",
        '        origin="",  # DECIDE: ours-N | miracl | mechanism | industry',
        '        detection="none",  # DECIDE, and a claim of anything else needs a bait below',
        '        instrument="",  # DECIDE: corpus | config | ingest | faulty_rag | platform',
        "        applies_when=(),  # DECIDE: the coordinates that admit it, empty means everywhere",
        '        not_detected_reason="",  # REQUIRED while detection stays "none"',
        "    ),",
        "",
        "# ── into tests/unit/test_atlas_baits.py, only if a signal is claimed ──",
        f"def bait_{new_id}_() -> set[str]:  # NAME IT after the defect, not after the entry",
        '    """One change to the clean baseline, and only one."""',
        "    run = clean_run()",
        "    # ... the single change that carries the defect",
        "    return _detector_ids(run)",
        "",
        f'#   BAITS["{new_id}"] = bait_{new_id}_',
        "#",
        "# Both halves or neither: the bait must fire on the payload carrying the",
        "# defect and stay silent on the clean baseline. The second half is the one",
        "# that rots, and a detector firing on everything passes the first.",
    ]
    return "\n".join(lines)


def _wrapped(text: str, width: int = 72) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]


async def _read_candidate(candidate_id: str) -> dict[str, Any] | None:
    import adapters.mongodb as mdb
    return await mdb.find_one("atlas_candidates", {"candidate_id": candidate_id})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--point", choices=sorted(POINTS), default=None,
                        help="one architecture; every one of them when omitted")
    parser.add_argument("--selectivity", action="store_true",
                        help="which entries share a signal")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument("--scaffold", metavar="CANDIDATE_ID", default=None,
                        help="draft an entry and a bait from a reported candidate")
    args = parser.parse_args(argv)

    if args.scaffold:
        import asyncio
        candidate = asyncio.run(_read_candidate(args.scaffold))
        if candidate is None:
            print(f"No candidate {args.scaffold!r}. Reported ones are listed at "
                  f"GET /atlas/candidates.")
            return 1
        print(scaffold(candidate))
        return 0

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
