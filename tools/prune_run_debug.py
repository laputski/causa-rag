"""Clear old runs' retrieval windows by hand, and see what would go first.

Dry by default: prints which runs the installation's rule would clear and why,
and changes nothing. `--write` clears them, removes the fragment texts no run
refers to any more, and prints the store's size before and after.

    python3 -m tools.prune_run_debug            # what the next pass would do
    python3 -m tools.prune_run_debug --write

The rule is the setting `debug_retention` (GET or PUT
/settings/debug-retention). The gateway runs the same pass by itself only
with CAUSA_RUN_RETENTION=on.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from services.api_gateway.run_retention import prune


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="clear, instead of showing what would be")
    args = parser.parse_args()
    report = asyncio.run(prune(write=args.write))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.write:
        print(f"\n{len(report['cleared'])} run(s) would lose their windows; nothing was changed.")


if __name__ == "__main__":
    main()
