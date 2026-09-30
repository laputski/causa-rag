"""A graph run reads the graph store of the Realm that launched it.

Found live: a graph run of one Realm took its graph half from the store the
gateway started with, which held another Realm's graph. 134 documents of that
other corpus entered the run's contexts, and 0 came from the Realm's own
store. The chat path had always bound the graph by Realm; the run path bound
the base half and left the graph alone, on the reasoning that a graph has no
corpus partitioning, which stopped being the whole story once each Realm had
its own container.
"""
from __future__ import annotations

from adapters import neo4j_graph
from adapters.neo4j_graph import Neo4jGraphRetriever
from core.experiment.runner import _for_the_corpus
from core.retrieval.graph_hybrid import GraphHybridRetriever


class _Base:
    retriever_id = "base"

    def retrieve(self, query, k, filters=None, query_vector=None):
        return []


def _hybrid() -> GraphHybridRetriever:
    return GraphHybridRetriever(Neo4jGraphRetriever(uri="bolt://localhost:7687"), _Base())


def test_the_graph_half_is_bound_to_the_realm_s_store() -> None:
    bound = _for_the_corpus(_hybrid(), "acme-corpus", "acme",
                            neo4j_cfg={"uri": "bolt://localhost:7476", "user": "neo4j", "password": "p"})
    assert bound._graph._uri == "bolt://localhost:7476"


def test_a_realm_without_its_own_store_keeps_the_default_one() -> None:
    hybrid = _hybrid()
    assert _for_the_corpus(hybrid, "c", "demo", neo4j_cfg=None) is hybrid
    assert _for_the_corpus(hybrid, "c", "demo", neo4j_cfg={"uri": "bolt://localhost:7687"}) is hybrid


def test_one_store_is_opened_once_across_runs(monkeypatch) -> None:
    """A retriever per run would open a driver per run and close none."""
    monkeypatch.setattr(neo4j_graph, "_BOUND", {})
    cfg = {"uri": "bolt://localhost:7480", "user": "neo4j", "password": "p"}
    first = _for_the_corpus(_hybrid(), "c", "acme-2", neo4j_cfg=cfg)._graph
    second = _for_the_corpus(_hybrid(), "c", "acme-2", neo4j_cfg=cfg)._graph
    assert first is second
