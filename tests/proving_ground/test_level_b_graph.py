"""A graph with units and no edges, which still reports structure.

The pair was written to stage a runaway link step: units are linked by the
keywords they share, so a keyword occurring in every unit joins each to all
the others. Measurement refused it twice, and both refusals are the
platform's own protections. Keywords are the first eight distinct words of a
unit, so a repeated sentence displaces a unit's keywords and does not add
to them; and a keyword occurring in more units than a frequency cap allows
is excluded before linking, exactly because one such keyword contributes
N×(N−1)/2 edges by itself.

So the runaway cannot be staged here, and what happens instead is the
prevention misfiring: every unit's keywords go over the cap at once and the
link step produces nothing. The graph still has its nodes counted, its
communities counted, and a perfect modularity, because every node is its own
community. That is the failure this pair stages, and it needed an entry of
its own.

No index and no embedder. The graph is built from the chunker's output by
the same two calls the loader makes, which is the whole of what these
failures need, and it leaves every corpus index of the proving ground
untouched.

The graph is cleared between the halves, because Neo4j has no corpus
partitioning at all: that is why the platform gives a realm its own
container, and why this suite needs the proving ground's own instance and
refuses to touch any other.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.proving_ground.conftest import REALM, record

pytestmark = pytest.mark.proving_ground

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / "proving-ground" / "base-ru"
#: The proving ground's own instance, assigned by
#: `tools.generate_realm_neo4j_compose` and registered on the realm.
URI = os.getenv("PROVING_GROUND_NEO4J", "bolt://localhost:7478")
#: How many documents the base corpus holds, read from disk below.
DOCUMENTS = len(list(CORPUS.glob("*.md")))
#: The analyser the base corpus was indexed under, the same one every other
#: level-B suite here uses.
LANGUAGE = "ru_be"


@pytest.fixture(scope="module")
def graph() -> Any:
    """The proving ground's own graph, or a loud skip.

    Never the default instance. That one belongs to whichever realm was
    registered first, holds tens of thousands of another corpus's units, and
    a measurement of edges per unit taken across two corpora at once measures
    neither.
    """
    from adapters.neo4j_graph import Neo4jGraphRetriever

    retriever = Neo4jGraphRetriever(uri=URI, user="neo4j", password="ragplatform")
    try:
        reachable = retriever.verify()
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        pytest.skip(f"NOT RUN: {URI} unreachable ({exc}). Start it with "
                    f"`docker compose -f deploy/compose/docker-compose.yml "
                    f"-f deploy/compose/neo4j-realms.generated.yml "
                    f"--profile graph-{REALM} up -d neo4j-{REALM}`.")
    if not reachable:
        pytest.skip(f"NOT RUN: {URI} did not verify. Bring the realm's own Neo4j up first.")
    return retriever


def _chunks(corpus: dict[str, str]) -> list[Any]:
    from core.chunking.structure_aware import StructureAwareChunkingStrategy
    from core.models import Document

    strategy = StructureAwareChunkingStrategy()
    out: list[Any] = []
    for name, text in sorted(corpus.items()):
        out.extend(strategy.chunk(Document(doc_id=name, source=name, content=text, metadata={})))
    return out


def _built(graph: Any, corpus: dict[str, str]) -> dict[str, Any]:
    """Clear, add, link, and read back what the link step produced."""
    chunks = _chunks(corpus)
    graph.clear()
    graph.add_chunks(chunks)
    graph.link_related()

    with graph._connect().session() as session:
        units = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        edges = session.run(
            "MATCH ()-[r:RELATED]->() RETURN count(r) AS n").single()["n"]
    return {"units": units, "edges": edges, "edges_per_unit": edges / units if units else 0.0}


#: The largest number of units one word can join before the linker's own cap
#: removes it. The cap is what refused the first attempt at these two entries,
#: so the staging that works sits directly underneath it.
UNITS_AT_THE_CAP = 50


def _units_carrying_the_shared_word(corpus: dict[str, str],
                                    healthy: dict[str, str]) -> list[Any]:
    """The units the injected word is a keyword of.

    Asked of the chunker and of the keyword extractor, never assumed: a word
    is a keyword of a unit only if it is among the first eight distinct words
    of four characters or more, and a placement that misses that window joins
    nothing at all.
    """
    from adapters.neo4j_graph import _keywords
    from tools.corpus_mutate import _commonest_words

    word = _commonest_words(healthy, count=1)[0] * 2
    return [c for c in _chunks(corpus) if word in _keywords(c.text)]


def _edges_among(graph: Any, units: list[Any]) -> int:
    """How many of the pairs inside one set of units are linked."""
    with graph._connect().session() as session:
        return session.run(
            "MATCH (a:Chunk)-[r:RELATED]-(b:Chunk) "
            "WHERE a.chunk_id IN $ids AND b.chunk_id IN $ids AND a.chunk_id < b.chunk_id "
            "RETURN count(r) AS n", ids=[u.chunk_id for u in units]).single()["n"]


def _communities(graph: Any) -> dict[str, Any]:
    """How many distinct documents the largest community mixes.

    The number the catalogue says is reported by nothing: a modularity
    arrives beside the communities and says how cleanly they separate, and
    says nothing about whether the grouping followed meaning or a word.
    """
    result = graph.detect_communities(algorithm="louvain", edge_type="lexical")
    communities = sorted(result.get("communities") or [],
                         key=lambda c: len(c.get("doc_ids") or []), reverse=True)
    if not communities:
        return {"modularity": result.get("modularity"), "largest": 0, "documents_in_largest": 0}
    largest = communities[0]
    documents = Counter(largest.get("doc_ids") or [])
    return {
        "modularity": result.get("modularity"),
        "largest": len(largest.get("doc_ids") or []),
        "documents_in_largest": len(documents),
    }


@pytest.fixture(scope="module")
def halves(graph: Any) -> Any:
    """Both halves, built once, in order. The second overwrites the first,
    which is what clearing between them means."""
    from tools.corpus_mutate import mutate, read_corpus, share_a_word

    healthy = read_corpus(CORPUS)
    # The units a shared word reaches at the larger size, held fixed so every
    # graph below is measured over the same units and the control can be asked
    # how many of those pairs it already had.
    carrying = _units_carrying_the_shared_word(share_a_word(healthy, UNITS_AT_THE_CAP), healthy)

    control = _built(graph, healthy)
    control["communities"] = _communities(graph)
    control["edges_among_the_carriers"] = _edges_among(graph, carrying)
    broken = _built(graph, mutate(healthy, "repeat_a_phrase_in_every_document"))
    broken["communities"] = _communities(graph)

    shared: dict[int, dict[str, Any]] = {}
    for units in (UNITS_AT_THE_CAP // 2, UNITS_AT_THE_CAP):
        corpus = share_a_word(healthy, units)
        built = _built(graph, corpus)
        built["communities"] = _communities(graph)
        reached = _units_carrying_the_shared_word(corpus, healthy)
        built["carriers"] = len(reached)
        built["documents_reached"] = len({unit.doc_id for unit in reached})
        built["edges_among_the_carriers"] = _edges_among(graph, reached)
        shared[units] = built

    yield {"control": control, "broken": broken, "shared": shared}
    # Left as it was found. The proving ground is meant to be broken on
    # purpose, and a graph emptied by a suite that has finished is broken by
    # accident: the next person to open it would read the damage as the
    # realm's own.
    _built(graph, healthy)


def test_the_control_graph_is_built_and_linked(halves: dict[str, dict[str, Any]]) -> None:
    """The premise of both pairs. A control with no edges would make the
    comparison below a comparison of two absences."""
    control = halves["control"]
    assert control["units"] > 100, control
    assert control["edges"] > 0, control


def test_F41_the_guard_leaves_the_graph_with_no_edges_and_it_still_reports_structure(
    halves: dict[str, dict[str, Any]],
) -> None:
    """One sentence under every heading, and nothing is linked to anything.

    The units are all there, the communities are all there, and the
    modularity is perfect because each node is alone. A graph retrieval walks
    to neighbours that do not exist and returns what a plain search would
    have returned.
    """
    from core.eval.graph_health import analyze as analyze_graph

    control, broken = halves["control"], halves["broken"]
    assert broken["units"] == control["units"], (
        f"the two halves differ in their units as well: {control['units']} against "
        f"{broken['units']}, so this pair changed two things"
    )
    assert broken["edges"] < control["edges"], (
        f"the repeated phrase cost no edges: {broken['edges']} against {control['edges']}"
    )

    spoke = {f"health:{f.id}" for f in analyze_graph(broken["units"], broken["edges"])}
    quiet = {f"health:{f.id}" for f in analyze_graph(control["units"], control["edges"])}
    assert "health:graph_has_no_edges" in spoke, (
        f"the link step produced {broken['edges']} edges and nothing said so: {sorted(spoke)}"
    )
    assert quiet == set(), f"the control graph is reported as broken: {sorted(quiet)}"

    record("F41", "one sentence under every heading, and the link step produced nothing",
           units=control["units"], documents=DOCUMENTS,
           edges_control=control["edges"], edges_broken=broken["edges"],
           edges_per_unit_control=round(control["edges_per_unit"], 2),
           edges_per_unit_broken=round(broken["edges_per_unit"], 2),
           modularity_control=control["communities"]["modularity"],
           modularity_broken=broken["communities"]["modularity"],
           signals=sorted(spoke), signals_on_the_control=sorted(quiet))


def test_F38_one_word_joins_every_unit_it_reaches_to_every_other(
    halves: dict[str, Any],
) -> None:
    """The runaway link step, staged under the cap that refused it before.

    The first attempt put a shared sentence in every unit, and the linker
    dropped the keyword for passing its frequency cap: the graph went to
    nothing instead of to everything. The cap is fifty, so a word in fifty
    units is the most one word can join before it is removed, which is also
    where the runaway is at its worst.

    What is measured is the clique and never the total, because the injected
    word displaces one of a unit's own eight keywords and costs edges
    elsewhere: at half the size the total moved by 42 while the word itself
    contributed 300. A total is the sum of two effects and evidence for
    neither.
    """
    control, shared = halves["control"], halves["shared"]
    assert control["edges_among_the_carriers"] == 0, (
        "the units this word joins were already linked to each other in the healthy graph, "
        "so the edges below are not the word's doing"
    )
    for units, built in sorted(shared.items()):
        assert built["carriers"] == units, (
            f"the word reached {built['carriers']} units and not {units}, so the arithmetic "
            "below is about a different set"
        )
        expected = units * (units - 1) // 2
        assert built["edges_among_the_carriers"] == expected, (
            f"{units} units sharing one word are linked by {built['edges_among_the_carriers']} "
            f"edges and a complete bucket is {expected}"
        )

    small, large = sorted(shared)
    growth = shared[large]["edges_among_the_carriers"] / shared[small]["edges_among_the_carriers"]
    assert growth > 3.5, (
        f"twice the units carrying the word gave {growth:.1f} times the edges, and a count "
        "growing with the square of them would give about four"
    )


def test_F38_the_runaway_is_there_and_no_check_counts_it(halves: dict[str, Any]) -> None:
    """The other half, and the one the entry's own state rests on.

    An edge count that grows with the square of the vocabulary is a shape, and
    the only check this platform has for a graph asks whether there are any
    edges at all. So the reverse pair: the failure is in the graph, measured
    above, and everything is silent.
    """
    from core.eval.graph_health import analyze as analyze_graph

    control, shared = halves["control"], halves["shared"]
    at_the_cap = shared[UNITS_AT_THE_CAP]
    spoke = {f"health:{f.id}" for f in analyze_graph(at_the_cap["units"], at_the_cap["edges"])}
    assert spoke == set(), (
        f"something does count the edges after all, which would make this entry detectable: "
        f"{sorted(spoke)}"
    )
    record("F38",
           "one word in fifty units joins all fifty to each other, 1225 edges from a word "
           "that means nothing, and no check counts them",
           units=at_the_cap["units"], documents=DOCUMENTS,
           keyword_frequency_cap=UNITS_AT_THE_CAP, keywords_per_unit=8,
           edges_control=control["edges"], edges_at_the_cap=at_the_cap["edges"],
           edges_among_the_carriers_control=control["edges_among_the_carriers"],
           edges_among_the_carriers_at_half=shared[UNITS_AT_THE_CAP // 2][
               "edges_among_the_carriers"],
           edges_among_the_carriers_at_the_cap=at_the_cap["edges_among_the_carriers"],
           carriers_at_half=shared[UNITS_AT_THE_CAP // 2]["carriers"],
           carriers_at_the_cap=at_the_cap["carriers"],
           signals_seen=[],
           displacement=("the injected word takes one of a unit's eight keyword slots, so "
                         "the graph loses edges elsewhere while the bucket adds them: at "
                         "half the size the total rose by 42 and the word contributed 300"))


def test_F39_a_community_built_from_one_word_reads_like_structure(
    halves: dict[str, Any],
) -> None:
    """Fifty units of forty unrelated documents, joined by a word that means
    nothing, and a modularity that reads as structure.

    This is the reverse kind of pair. The grouping provably followed a word:
    the units are a complete clique, and in the healthy graph not one of those
    pairs was linked. What comes back is a count of communities, a size and a
    list of documents for each, and a modularity. None of it separates this
    grouping from one that followed meaning, which is what the entry says.

    The first attempt at this entry counted the documents a large community
    draws on, and the control refuted it: a corpus of procedures on one
    subject already groups across its documents.
    """
    control, shared = halves["control"], halves["shared"]
    at_the_cap = shared[UNITS_AT_THE_CAP]
    assert at_the_cap["documents_reached"] > DOCUMENTS * 0.8, (
        f"the word reached {at_the_cap['documents_reached']} of {DOCUMENTS} documents, so the "
        "clique it built is a group of related sections and not of unrelated ones"
    )
    modularity = at_the_cap["communities"]["modularity"]
    assert modularity is not None and modularity > 0.3, (
        f"the modularity came back at {modularity}, which reads as no structure, so a reader "
        "would not be misled by it and this entry is not staged here"
    )
    assert at_the_cap["communities"]["documents_in_largest"] >= (
        control["communities"]["documents_in_largest"] * 0.7
    ), ("the largest community narrowed sharply, which would be something a reader could see; "
        "if that holds, the entry has a signal after all")

    record("F39",
           "a clique of fifty units across forty documents, built by one meaningless word, "
           "comes back as communities with a plausible modularity and nothing that says why",
           units=at_the_cap["units"], documents=DOCUMENTS,
           documents_reached_by_the_word=at_the_cap["documents_reached"],
           edges_among_the_carriers_control=control["edges_among_the_carriers"],
           edges_among_the_carriers_at_the_cap=at_the_cap["edges_among_the_carriers"],
           modularity_control=control["communities"]["modularity"],
           modularity_at_the_cap=modularity,
           documents_in_largest_control=control["communities"]["documents_in_largest"],
           documents_in_largest_at_the_cap=at_the_cap["communities"]["documents_in_largest"],
           what_is_reported="a community count, a size and a document list for each, "
                            "and a modularity",
           signals_seen=[])


def test_this_suite_never_touches_another_realms_graph() -> None:
    """Structural, and not a matter of remembering. Neo4j has no corpus
    partitioning, so a suite that clears a graph has to be certain whose."""
    assert "7687" not in URI, (
        f"{URI} is the default instance, which belongs to the first-registered realm and "
        "holds another corpus entirely. This suite clears what it measures."
    )


# ── the graph point, run end to end ───────────────────────────────────────────

def _graph_run(embedder: Any, graph_weight: float, hops: int) -> dict[str, Any]:
    """One retrieval-only run of the graph pipeline at these two settings."""
    from core.experiment.config import ExperimentConfig
    from tests.proving_ground.conftest import retrieval_only, run_on
    from tools.seed_proving_ground import control_config

    base = control_config("base-ru").model_dump(exclude={"config_hash"})
    base["name"] = f"proving-ground-graph-w{graph_weight}-h{hops}"
    base["pipeline_id"] = "graph"
    base["graph_weight"] = graph_weight
    base["hops"] = hops
    base["chunking_strategy"] = {"kind": "chunker", "component_id": "structure_aware",
                                 "params": {}}
    return run_on(embedder, retrieval_only(ExperimentConfig(**base)), "base-ru", LANGUAGE)


def test_the_graph_point_runs_and_its_two_settings_reach_the_run(
    embedder: Any, graph: Any,
) -> None:
    """The point exists on the proving ground, and it can be varied.

    Both halves matter and the second is the one that was missing: the two
    parameters of the graph point had no fields on a configuration, so the
    pipeline could be chosen and every run of it used whatever the gateway had
    constructed. A unit test proves the field reaches the retriever; this
    proves it reaches a run against a live graph, which is where a setting
    that quietly does nothing would still look identical.
    """
    from_graph_only = _graph_run(embedder, graph_weight=1.0, hops=1)
    from_base_only = _graph_run(embedder, graph_weight=0.0, hops=1)

    assert from_graph_only["question_results"], "the graph run answered no questions at all"
    assert from_graph_only["config"]["graph_weight"] == 1.0
    assert from_graph_only["config"]["hops"] == 1

    def _ranking(run: dict[str, Any]) -> list[tuple[str, ...]]:
        return [tuple(s["chunk_id"] for s in (q.get("source_refs") or []))
                for q in run["question_results"]]

    assert _ranking(from_graph_only) != _ranking(from_base_only), (
        "the graph weight moved from one to zero and every question came back with the "
        "same fragments in the same order, so the setting reached nothing"
    )
    # Deliberately not recorded as evidence. The evidence directory holds one
    # file per catalogue entry, and a file named for something the catalogue
    # does not hold is refused by the registry's own guard. This proves the
    # point can be run and varied, which is a fact about the platform and not
    # about a failure.
    applied = from_graph_only.get("applied") or {}
    assert applied.get("index_embedder_id"), (
        "the graph run recorded no index embedder, so the check comparing the model that "
        f"indexed with the model that queries is silent on this point: {applied}"
    )
