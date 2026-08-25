"""Count the two prose constructions this project removes on purpose.

Both are habits rather than errors, which is why a counter beats a linter rule:
the number can be driven down deliberately and then held, and a file that
legitimately needs one can be argued about rather than silently suppressed.

**Em dashes used as connectors.** A sentence joined by an em dash usually has a
verb missing from it. Rewriting around that verb says what relates the two
halves instead of leaving the reader to infer it.

**Contrastive framing** of the "not X but Y" kind. It spends a clause on what
something is not before saying what it is, and the negated half is often the
only part the reader remembers. Stating the positive directly is shorter and
lands better.

Neither is mechanically fixable. A regex can find them; only a person can
restructure the sentence, which is the point of the exercise.

    python3 -m tools.check_style                 # the report
    python3 -m tools.check_style --json
    python3 -m tools.check_style --show core/    # the actual lines in one path
    python3 -m tools.check_style --max-em 0      # fail above a budget
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Only an em dash acting as a connector counts: one with a word on each side.
#
# Three other uses are legitimate typography and were being counted as
# violations, which made the total impossible to drive to zero and therefore
# useless as a target. A dash alone in a table cell (`| — |`) means "no value".
# A dash opening a diagram label or a list line introduces rather than joins.
# A numeric range (`12—16`) is a range.
#
# The tic this counts is the one the guidance is about: two independent clauses
# welded together where a verb is missing.
EM_DASH = re.compile(r"—")

# Three uses are legitimate typography rather than the tic, and counting them
# made the total impossible to drive to zero and therefore useless as a target.
_EM_DASH_EXEMPT = [
    re.compile(r"\|\s*—\s*\|"),        # alone in a table cell: "no value"
    re.compile(r"^\s*[-*>]?\s*—"),      # opening a line: a diagram or list label
    re.compile(r"\d\s*—\s*\d"),        # a numeric range
]


def _em_dashes(line: str) -> int:
    """Em dashes acting as connectors: the tic, not the punctuation.

    The guidance is about two independent clauses welded together where a verb
    is missing. A dash standing for an absent table value, opening a label, or
    spanning a numeric range joins nothing and is left alone.
    """
    total = line.count("—")
    for rx in _EM_DASH_EXEMPT:
        total -= len(rx.findall(line))
    return max(total, 0)

CONTRASTIVE = [
    (re.compile(r"\brather than\b", re.I), "rather than"),
    (re.compile(r"\bnot\s+\w+[^.!?]{0,40}?\bbut\b", re.I), "not X but Y"),
    (re.compile(r"\b(isn't|is not|aren't|are not)\s+\w+[^.!?]{0,30}?,\s*(it's|it is|they're|they are)\b", re.I), "isn't X, it's Y"),
    (re.compile(r"\bне\s+[\w-]+,\s*а\s+[\w-]+", re.I), "не X, а Y"),
    (re.compile(r"\bне только\b[^.!?]{0,60}?\bно и\b", re.I), "не только X, но и Y"),
]

# Directories holding generated output or third-party text. Counting model
# output would swamp the authored surface this is meant to measure.
#
# The private design directories are skipped for a different reason: they never
# publish, so counting them makes the total move whenever a design note is
# edited, and a budget that drifts for reasons unrelated to the public surface
# cannot be a ratchet. What is counted is what a reader of the public repository
# will actually read.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", "test-reports", ".deepeval",
    ".claude", ".specify", "htmlcov", "eval/results",
    "lat.md", "research", "specs", "governance", "architecture", "infra",
    # One installation's own domain packs. Gitignored, excluded from the export,
    # and 56 em dashes of it: counting them moved the total for reasons no
    # public reader could ever see.
    "packs.local",
}

# Files inside otherwise-counted directories that the export withholds.
SKIP_FILES = {"docs/ui-redesign-plan.md"}

# `.css` was missing here, so `ui/src/styles.css` was never counted: three
# thousand lines carrying the whole design rationale, invisible to the one
# instrument that measures authored prose. Widening the scope is not the same as
# raising a budget, and the budget below is re-measured against the wider scope.
SUFFIXES = {".py", ".ts", ".tsx", ".js", ".css", ".md", ".yml", ".yaml", ".sh", ".toml"}

# Its own pattern table names every construction it forbids.
SELF_EXEMPT = {"tools/check_style.py"}


def _skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & SKIP_DIRS or str(rel).startswith("eval/results"):
        return True
    if path.suffix not in SUFFIXES:
        return True
    return str(rel) in SELF_EXEMPT or str(rel) in SKIP_FILES


def scan(root: Path) -> dict:
    per_file: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lines_by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or _skip(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(root))
        for line_no, line in enumerate(text.splitlines(), start=1):
            n = _em_dashes(line)
            if n:
                per_file[rel]["em_dash"] += n
                lines_by_file[rel].append((line_no, "em_dash", line.strip()[:160]))
            for rx, label in CONTRASTIVE:
                if rx.search(line):
                    per_file[rel][label] += 1
                    lines_by_file[rel].append((line_no, label, line.strip()[:160]))

    totals: dict[str, int] = defaultdict(int)
    for counts in per_file.values():
        for key, n in counts.items():
            totals[key] += n
    return {"per_file": {k: dict(v) for k, v in per_file.items()},
            "totals": dict(totals),
            "lines": {k: v for k, v in lines_by_file.items()}}


def _render(result: dict, top: int = 20) -> str:
    totals = result["totals"]
    per_file = result["per_file"]
    if not totals:
        return "Style: clean."

    lines = ["", "  Style counts", "  " + "─" * 58]
    for key in sorted(totals, key=lambda k: -totals[k]):
        lines.append(f"  {key:<26} {totals[key]:>6}")
    lines.append("  " + "─" * 58)
    lines.append(f"  across {len(per_file)} files")
    lines.append("")
    lines.append("  Heaviest files")
    ranked = sorted(per_file.items(), key=lambda kv: -sum(kv[1].values()))
    for name, counts in ranked[:top]:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"    {sum(counts.values()):>5}  {name}  ({detail})")
    if len(ranked) > top:
        lines.append(f"    … {len(ranked) - top} more files")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show", metavar="PREFIX", help="print the matching lines under this path")
    parser.add_argument("--max-em", type=int, default=None, help="exit non-zero above this many em dashes")
    parser.add_argument("--max-contrastive", type=int, default=None)
    args = parser.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    result = scan(root)

    if args.show:
        for name, hits in sorted(result["lines"].items()):
            if not name.startswith(args.show):
                continue
            print(f"\n{name}")
            for line_no, label, text in hits:
                print(f"  {line_no:>5} [{label}] {text}")
        return 0

    if args.json:
        print(json.dumps({"totals": result["totals"], "per_file": result["per_file"]},
                         ensure_ascii=False, indent=2))
    else:
        print(_render(result))

    totals = result["totals"]
    em = totals.get("em_dash", 0)
    contrastive = sum(v for k, v in totals.items() if k != "em_dash")

    failed = False
    if args.max_em is not None and em > args.max_em:
        print(f"\nem dashes: {em} > {args.max_em}", file=sys.stderr)
        failed = True
    if args.max_contrastive is not None and contrastive > args.max_contrastive:
        print(f"contrastive constructions: {contrastive} > {args.max_contrastive}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
