"""The configuration space a RAG system is a point in.

A failure mode is not a property of an architecture's *name*. "Hybrid" is not a
thing a system is; it is a coordinate a system has, and the same coordinate is
carried by systems nobody calls hybrid. Scoping a failure by name would mean
maintaining a list of names by hand and revisiting every entry whenever an
architecture is added, which is exactly the work this module exists to avoid.

So `core/eval/atlas.py` scopes each failure by the coordinates that make it
possible, and applicability is derived instead of maintained.

The coordinates are not invented here. They are the twenty-eight dimensions
published by RAG World (https://ragworld.org), a registry that places every
recorded RAG technology at a point in that space. This module is a vendored
copy of the codes and their permitted values, nothing more: no labels a person
reads, no constraints, no records.

Why a copy and not an import: this platform must build on a clean clone
with the `dev` extra alone (the `fresh-clone` job in CI), and `core/` depends
on nothing outside the platform. A cross-repository import would break both.
The price of a copy is drift, and `tests/fitness/test_atlas_registry.py` pays
it: it compares this table against the published data when the network is
reachable, and reports that the comparison **did not run** when it is not.
Saying "not checked" is the whole point; a silent pass would make a stale copy
indistinguishable from a current one.

What this module deliberately does not do:

- it does not add a dimension. A mechanism the twenty-eight do not express
  belongs to RAG World's own residual queue, under its own rule of three
  mentions, and not to a private vocabulary here. Three were found while
  scoping the failure catalogue and are written up for submission in
  `docs/rag-world-residual-submissions.md`, measured and ready to be lifted;
- it does not carry constraints between values. Validating a configuration is
  the registry's work; this platform only asks whether a point satisfies a
  predicate;
- it does not name the strata in prose. The codes are the interface.

Three things the published schema cannot express today were found while
scoping the atlas, and each is recorded on the affected entry as a
`scope_caveat` instead of being worked around silently. See `core/eval/atlas.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

# Where the coordinates come from, and which published release this copy was
# checked against. Shown beside every coordinate the interface renders, so a
# reader can reach the definition of a code they do not recognise.
SCHEMA_SOURCE = "https://ragworld.org"
SCHEMA_DATA_URL = "https://ragworld.org/data/index.json"

# The latest published release of the schema.
SCHEMA_RELEASE = "2026-08-14"

# The exact revision the values below were copied from, and its date.
#
# Two facts and not one, because they move at different rates. A release is cut
# less often than the data is rebuilt: on 2026-09-01 the published data gained
# two values while the release tag stayed at 2026-08-14, so the tag alone
# cannot say which values a copy was checked against. The revision can, and a
# test reads the schema at it and compares it with the values below, so the
# marker and the values can only move together.
#
# The date is repeated for a reader, who should not have to run a command to
# learn how old a copy is, and a test derives it from the revision so the two
# cannot drift apart.
SCHEMA_VERIFIED_AGAINST = "5bca84c142c4dc89eb56edd1601fdf28eca3f24a"
SCHEMA_VERIFIED_ON = "2026-09-01"

# The composition, fixed by a check so that a changed copy is a deliberate act
# and not a slip. RAG World holds the same count for the same reason.
SCHEMA_DIMENSION_COUNT = 28


# The section of the published article that defines the dimensions. There is no
# anchor per code: the article addresses nine sections and `schema` is the one
# holding the space. An earlier version of `dimension_url` built
# `/article#A2`, which resolves to nothing and drops the reader at the top of
# the page with no sign that the link missed. `tests/fitness/
# test_atlas_registry.py` now checks every anchor this module emits against the
# anchors the article actually declares.
SCHEMA_ARTICLE_ANCHOR = "schema"


def dimension_url(code: str) -> str:
    """Where a person reads what a code means.

    An outbound link, followed by a human on click. The interface's air-gap
    rule forbids *loading* anything from outside; it does not forbid a link.

    The code is carried as a query so the reader can see which coordinate the
    link was about, while the fragment points at an anchor that exists.
    """
    return f"{SCHEMA_SOURCE}/article?d={code}#{SCHEMA_ARTICLE_ANCHOR}"


@dataclass(frozen=True)
class Dimension:
    code: str
    name: str
    values: tuple[str, ...]
    core: bool = True
    default: str = ""

    @property
    def stratum(self) -> str:
        return self.code[0]


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("A1", "Unit of retrieval", ("passage", "proposition", "entity", "node_edge", "page_image", "table_row", "summary_node",), core=True, default="passage"),
    Dimension("A2", "Segmentation", ("fixed", "structure_aware", "late_chunking", "semantic", "none",), core=True, default="fixed"),
    Dimension("A3", "Unit enrichment", ("none", "context_prefix", "summary", "extracted_triples", "metadata",), core=True, default="none"),
    Dimension("A4", "Index topology", ("flat", "tree", "graph", "hypergraph", "community_hierarchy",), core=True, default="flat"),
    Dimension("A5", "Representation model", ("lexical", "dense_single", "dense_multi_late_interaction", "symbolic", "vision_language", "none",), core=True, default="dense_single"),
    Dimension("A6", "Temporality", ("snapshot", "append_only", "bitemporal",), core=False, default="snapshot"),
    Dimension("A7", "Modality", ("text", "image", "table", "audio", "scene_3d",), core=True, default="text"),
    Dimension("A8", "Origin of index structure", ("none", "given", "extracted", "computed", "extracted_and_computed",), core=True, default="none"),
    Dimension("B1", "Query transformation", ("identity", "hyde", "multi_reformulation", "step_back", "subquestion_decomposition", "key_extraction",), core=True, default="identity"),
    Dimension("B2", "Routing", ("static", "trained_classifier", "llm_router", "cost_aware_policy",), core=True, default="static"),
    Dimension("C1", "Search operator", ("ann", "lexical", "graph_traversal", "boolean_query", "tree_navigation", "spatial_range",), core=True, default="ann"),
    Dimension("C2", "Traversal control", ("single_shot", "multi_hop_fixed", "iterative_stopping", "agentic_open_loop",), core=True, default="single_shot"),
    Dimension("C3", "Source fusion", ("none", "rrf", "score_normalization", "learned_fusion",), core=True, default="none"),
    Dimension("C4", "Distribution", ("single_store", "multiple_local", "federation",), core=False, default="single_store"),
    Dimension("D1", "Reranking", ("none", "cross_encoder", "graph_structural", "set_cover", "path_pruning",), core=True, default="none"),
    Dimension("D2", "Selection and compression", ("top_k", "budget_aware", "abstractive_compression", "latent_compression",), core=True, default="top_k"),
    Dimension("D3", "Arrangement", ("natural_order", "reliability_ascending", "hierarchical",), core=True, default="natural_order"),
    Dimension("E1", "Generation mode", ("single_pass", "draft_verify", "ensemble_fragments", "multi_agent",), core=True, default="single_pass"),
    Dimension("E2", "Groundedness control", ("none", "pre_gen_grounding", "post_gen_check", "decoding_reflection", "decoding_trigger", "external_judge",), core=True, default="none"),
    Dimension("E3", "Attribution", ("none", "document_level", "fragment_level", "claim_level",), core=True, default="none"),
    Dimension("E5", "Coupling of generation to retrieval", ("none", "generation_seeds", "mutual_loop",), core=True, default="none"),
    Dimension("E4", "Refusal policy", ("no_refusal", "confidence_threshold", "domain_policy",), core=True, default="no_refusal"),
    Dimension("F1", "Write-back", ("none", "episodic", "consolidating",), core=False, default="none"),
    Dimension("F2", "Conflict resolution", ("none", "by_time", "by_authority", "explicit_reconciliation",), core=False, default="none"),
    Dimension("F3", "Forgetting", ("none", "by_ttl", "by_significance_decay",), core=False, default="none"),
    Dimension("G1", "Privacy", ("open", "isolated_circuit", "differential_privacy", "tee",), core=True, default="open"),
    Dimension("G2", "Execution site", ("server", "edge_device", "mixed",), core=True, default="server"),
    Dimension("G3", "Trainability of components", ("frozen", "trained_retriever", "trained_reader", "joint_training",), core=True, default="frozen"),
)


_BY_CODE: dict[str, Dimension] = {d.code: d for d in DIMENSIONS}


def get(code: str) -> Dimension | None:
    """The dimension with this code, or None. Never raises: a caller checking
    whether a code is known should not have to catch."""
    return _BY_CODE.get(code)


def is_known(code: str, value: str) -> bool:
    d = _BY_CODE.get(code)
    return d is not None and value in d.values


# A point is a partial map from code to value. Partial on purpose: a run knows
# what it applied and nothing about the rest, and completing the gaps with
# defaults would turn "not established" into "established as the base value",
# which is the confusion this whole plan exists to remove.
Point = dict[str, str]

# A predicate over points: each pair is a dimension and the values that satisfy
# it. All pairs must hold, so the tuple reads as a conjunction.
Predicate = tuple[tuple[str, tuple[str, ...]], ...]


def satisfies(predicate: Predicate, point: Point) -> bool | None:
    """Whether a point satisfies a predicate.

    Returns None, not False, when the point says nothing about a dimension the
    predicate asks about. The distinction is the same one drawn everywhere else
    in this package: a question that could not be answered is not a question
    answered "no". A caller that treats None as False will report a failure
    inapplicable to a system whose coordinates were merely never recorded.

    An empty predicate is satisfied by every point, including the empty one:
    a failure that depends on no coordinate applies everywhere.
    """
    unknown = False
    for code, allowed in predicate:
        actual = point.get(code)
        if actual is None:
            unknown = True
            continue
        if actual not in allowed:
            return False
    return None if unknown else True


# The architectures this platform can itself run, written as coordinates and
# never as names. Kept here beside the space they are points in: a reporting
# tool asked for them first and held them for a while, which put data about the
# space in a layer that only prints it.
#
# Values are taken from the published registry's records for the same shapes,
# so the platform and the registry describe a hybrid the same way, never
# each in its own words.
#
# E3 and E4 are the exception, and they are measured here instead of copied.
# The registry's records for these shapes leave both at their defaults, no
# citation and no refusal, because they describe the retrieval architecture and
# not what a prompt does on top of it. This platform does both: every answer
# carries per-fragment labels, substituted for the model's positional markers
# before it is returned (core/citation.py), and the active prompt makes the
# model say when the corpus does not cover the question, which is what
# `correct_refusal` measures at 1.0 on a healthy run. The rule of the space is
# that a coordinate says what is applied and not what was accepted, so
# leaving these at the defaults would have said the platform never refuses
# while a proving-ground pair was reproducing a badly calibrated refusal on it
# on it. That is how the divergence was found: a guard refused evidence for a
# failure that these coordinates said could not occur.
POINTS: dict[str, Point] = {
    "dense": {
        # Nothing here transforms a query: it reaches the embedder and the
        # retriever as it was written. The one rewrite in the tree belongs to
        # the retrieval-pin overlay, which is off unless a run asks for it as
        # a what-if. Recorded because it was absent, and an absent coordinate
        # is not an answer: an entry scoped on it was neither applicable nor
        # ruled out here, which is what the space says when nobody filled it
        # in.
        "B1": "identity",
        "A1": "passage", "A2": "fixed", "A4": "flat", "A5": "dense_single",
        "C1": "ann", "C3": "none", "D1": "none", "D2": "top_k",
        "E1": "single_pass", "E3": "fragment_level", "E4": "domain_policy",
    },
    "hybrid": {
        # As above: the query reaches retrieval as it was written.
        "B1": "identity",
        "A1": "passage", "A2": "fixed", "A4": "flat", "A5": "dense_single",
        "C1": "ann", "C3": "rrf", "D1": "cross_encoder", "D2": "top_k",
        "E1": "single_pass", "E3": "fragment_level", "E4": "domain_policy",
    },
    "graph": {
        # As above: the query reaches retrieval as it was written.
        "B1": "identity",
        "A1": "passage", "A2": "fixed", "A3": "extracted_triples", "A4": "graph",
        "A5": "dense_single", "A8": "extracted", "C1": "graph_traversal",
        "C3": "score_normalization", "D1": "none", "D2": "top_k",
        "E1": "single_pass", "E3": "fragment_level", "E4": "domain_policy",
    },
}
