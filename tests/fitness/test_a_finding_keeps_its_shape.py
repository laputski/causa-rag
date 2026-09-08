"""What a finding carries over the wire, pinned where something can see it.

The plan this work follows says a new field on a finding "widens the API
surface, which the snapshot in `governance/openapi.json` fixes, so the
snapshot is rebuilt". Measured: the response of `GET /experiments/{run_id}` is
declared as a bare object with `additionalProperties: true`, so the snapshot
describes nothing under it. `failure_ids` and `detail_key` were both added to
a finding and `make openapi` produced no change either time, which reads as
"the surface did not move" and meant "the surface was never described".

The snapshot guard beside this one is right about what it checks, which is
that the paths still exist. Nothing checked the shape of what they return, and
a finding is read by a second codebase in another language, so a field renamed
on one side and not the other is a blank line on a screen and no error
anywhere.

So the shape is declared here and checked against both sides. Changing it
stays possible and stops being accidental.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.eval.detectors import DiagnosticItem

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.fitness

#: Every key a finding puts on the wire, and what each is for. Written out
#: here and never read from the dataclass, because the point is that the two are
#: compared: a field added to the class and not to this list is a field nobody
#: decided to publish.
WHAT_A_FINDING_CARRIES = {
    "id": "which finding this is, and the key its title and action are looked up by",
    "severity": "how loudly to say it, and what colour it is given",
    "title": "the server's English, shown when the reader's side has no words of its own",
    "detail": "the same, for the sentence carrying what was measured",
    "action": "the same, for what to do about it",
    "failure_ids": "the catalogue entries this finding is evidence for, empty when none",
    "params": "the numbers and names the sentences are composed from",
    "detail_key": "which sentence, when one identifier has more than one",
}


def test_a_finding_publishes_exactly_what_was_declared() -> None:
    published = set(DiagnosticItem(id="x", severity="warn", title="t", detail="d").to_dict())
    assert published == set(WHAT_A_FINDING_CARRIES), (
        f"a finding now carries {sorted(published)} and this file declares "
        f"{sorted(WHAT_A_FINDING_CARRIES)}"
    )
    for name, why in WHAT_A_FINDING_CARRIES.items():
        assert len(why) > 20, f"{name} is declared with no reason worth reading"


def test_the_other_side_declares_the_same_fields() -> None:
    """A finding is rendered by TypeScript, and a name that differs by a
    letter renders as nothing at all."""
    source = (ROOT / "ui" / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    block = re.search(r"export interface DetectorItem \{(.*?)\n\}", source, re.S)
    assert block, "the interface no longer declares a finding under that name"
    declared = set(re.findall(r"^\s*(\w+)\??:", block.group(1), re.M))
    assert declared == set(WHAT_A_FINDING_CARRIES), (
        f"the interface declares {sorted(declared)} and the server sends "
        f"{sorted(WHAT_A_FINDING_CARRIES)}"
    )


def test_the_snapshot_still_says_nothing_about_this() -> None:
    """The reason this file exists, asserted so it cannot quietly stop being
    the reason. If the response ever gains a schema, this check reddens and
    whoever gave it one can decide whether the declaration above is still
    where the shape belongs."""
    snapshot = ROOT / "governance" / "openapi.json"
    if not snapshot.exists():
        pytest.skip("NOT RUN: no openapi snapshot; run `make openapi`")
    response = (json.loads(snapshot.read_text(encoding="utf-8"))
                .get("paths", {}).get("/experiments/{run_id}", {})
                .get("get", {}).get("responses", {}).get("200", {}))
    schema = response.get("content", {}).get("application/json", {}).get("schema", {})
    assert schema.get("additionalProperties") is True and "properties" not in schema, (
        "the run response now has a described shape, so the snapshot does pin a finding's "
        "fields after all and the declaration in this file is a second place to keep them"
    )
