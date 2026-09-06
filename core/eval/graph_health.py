"""What a graph index says about itself, beside the fact that it exists.

A graph is reported by its node count, its community count and a modularity.
All three arrive for a graph with no edges whatsoever: every node is its own
community, the partition is perfect, and the modularity reads as structure.
Nothing between those numbers and a reader says that the units were never
connected at all.

Found by measurement on the proving ground, and the mechanism is the
platform's own protection working exactly as designed. Units are linked by
the keywords they share; keywords are the first eight distinct words of a
unit; and a keyword occurring in more than fifty units is excluded before
bucketing, because one such keyword contributes N×(N−1)/2 edges by itself.
Paste one ordinary sentence under every heading of every document and all
eight keywords of every unit exceed the cap at once. The link step then
produces nothing, and the failure the exclusion exists to prevent is
replaced by a graph in name only.

One check and not three. Two more were written here and both were refused
by the control graph, which is why they are named here and not quietly
dropped. A check for a link rule driving the graph towards complete rested
on a ratio nobody had measured, and the healthy graph sat close enough to it
to make the number a matter of taste. A check for a community drawing on
most of the corpus was refuted outright: on a healthy proving-ground graph
the largest community holds thirty-five units from thirty-five of forty
documents, because a corpus of procedures on one subject genuinely groups
across its documents. Tuning either threshold until the control passed would
have produced a signal that fires when it is set to.

Pure arithmetic over counts the caller has already read. No driver, no
Cypher: `core/` may not reach a database, and a check that needed one could
not run where the catalogue's baits run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Below this many edges per unit a lexical graph is not connecting anything.
#: One edge per unit is already a chain and not a graph; the value is the
#: line between "sparse" and "absent", and it is deliberately low so the
#: check speaks only for the second.
_ABSENT = 0.5

@dataclass(frozen=True)
class GraphFinding:
    """One thing a graph index says about itself."""

    id: str
    severity: str
    title: str
    detail: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "severity": self.severity, "title": self.title,
                "detail": self.detail, "action": self.action}


def analyze(units: int, edges: int) -> list[GraphFinding]:
    """Findings about one loaded graph, from counts alone.

    Silent for a graph with no units: nothing was loaded, so nothing about
    the linking can be said, and reporting an absence of edges there would
    say the load failed when no load was attempted.
    """
    if units <= 0:
        return []

    per_unit = edges / units
    if per_unit >= _ABSENT:
        return []
    return [GraphFinding(
        id="graph_has_no_edges",
        severity="error",
        title="The graph has units and almost no edges",
        detail=(
            f"{units} units carry {edges} lexical edge(s) between them, {per_unit:.2f} each. "
            "A graph retrieval walks from a matched unit to its neighbours, and here there "
            "are none to walk to, so it returns what a plain search would have returned. "
            "The community count and the modularity beside it are still reported and still "
            "read as structure."
        ),
        action=(
            "Check what the link step excluded. A keyword shared by more units than the "
            "frequency cap allows is dropped before linking, and a phrase repeated across "
            "the corpus can put every unit's keywords over that cap at once."
        ),
    )]
