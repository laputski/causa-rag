"""What the proving ground has filed about a catalogue entry.

A pair that stages a failure on a live stack writes one file saying what it
measured, and this reads them. Two readers need the answer, a reporting tool
and the interface, and it lived in the tool alone: the interface had no way to
know, so it showed an entry as unproven while a run had proved it and filed
the numbers.

Read from the filesystem and never from a constant. The tool's own version
began as a constant returning False, which was true on the day it was written
and stayed true in the file after six entries had been reproduced.

A file saying why an entry cannot be staged here is not a staging. Counting one
would put an entry in the reproduced column for having proved that it could
not be reproduced, which is the opposite of what it says.
"""
from __future__ import annotations

import json
from pathlib import Path

#: One file per entry, written by the pair that ran. Committed, so a fresh
#: clone reads the same answer this machine does.
EVIDENCE = Path(__file__).resolve().parents[2] / "eval" / "results" / "proving_ground"


def reproduced(failure_id: str) -> bool:
    """Whether a live run has staged this failure and filed what it saw."""
    filed = EVIDENCE / f"{failure_id}.json"
    if not filed.is_file():
        return False
    try:
        return bool(json.loads(filed.read_text(encoding="utf-8")).get("reproduced", True))
    except (OSError, ValueError):
        # A file that cannot be read is still a run that happened. The
        # alternative reads a damaged file as "nobody looked", which is the
        # one answer that is certainly wrong.
        return True
