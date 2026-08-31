"""Type errors under a budget, so the debt stops growing while it is repaid.

`mypy` on this repository reports dozens of errors, most of them older than
anyone remembers. Fixing them all before turning the check on would mean
never turning it on, so the step ran with `continue-on-error: true` and its
output went unread for weeks. A gate nobody can fail is not a gate.

This counts instead. The budget is per file, not a single total, because a
total hides the case that matters: a file that was clean acquiring its
first error while another file happens to lose one. A per-file ceiling
fails on the new error and says which file it is.

The baseline is measured in CI, not on a developer machine. Both run
`--ignore-missing-imports`, and that flag is silent about a package that is
absent while mypy reads the types of the same package when it is installed,
so a laptop with more packages installed legitimately sees more errors.
Local runs are therefore expected to report extras; the gate names them and
only fails on files whose count rose above the baseline.

Usage:
    python3 -m tools.check_types                 # report against the baseline
    python3 -m tools.check_types --update        # re-pin after fixing something
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = Path(__file__).with_name("mypy-baseline.json")
TARGETS = ["core/", "adapters/", "services/"]
_ERROR = re.compile(r"^([\w/.\-]+\.py):\d+: error:", re.M)


def run_mypy() -> Counter[str]:
    """Errors per file, from a real mypy run."""
    out = subprocess.run(
        [sys.executable, "-m", "mypy", *TARGETS, "--ignore-missing-imports"],
        capture_output=True, text=True,
    ).stdout
    return Counter(m.group(1) for m in _ERROR.finditer(out))


def load_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    return dict(json.loads(BASELINE.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true",
                    help="re-pin the baseline to what mypy reports now. Only ever "
                         "run this after the count went down")
    args = ap.parse_args(argv)

    current = run_mypy()
    baseline = load_baseline()

    if args.update:
        BASELINE.write_text(
            json.dumps(dict(sorted(current.items())), indent=2) + "\n", encoding="utf-8",
        )
        print(f"pinned {sum(current.values())} errors across {len(current)} files")
        return 0

    worse = sorted(
        (f, baseline.get(f, 0), n) for f, n in current.items() if n > baseline.get(f, 0)
    )
    better = sorted(
        (f, baseline[f], current.get(f, 0)) for f in baseline if current.get(f, 0) < baseline[f]
    )

    print(f"mypy: {sum(current.values())} errors in {len(current)} files "
          f"(baseline {sum(baseline.values())} in {len(baseline)})")

    if better:
        print("\nimproved, and the baseline can be tightened with --update:")
        for f, was, now in better:
            print(f"  {f}: {was} -> {now}")

    if worse:
        print("\nabove the baseline:")
        for f, was, now in worse:
            print(f"  {f}: {was} -> {now}")
        print("\nA file may also exceed its baseline because a package is installed "
              "here and absent in CI, which gives mypy more to read. Check the file "
              "before assuming the error is new.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
