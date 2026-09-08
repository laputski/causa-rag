"""A retriever reading the corpus a run named, on that Realm's own instance.

Locks in the fix for a real bug: `ExperimentConfig.corpus_id` was decorative.
Registry-resolved pipelines are built exactly once at gateway start-up on the
default corpus, and the build returned that frozen instance unchanged, so a
configuration naming another corpus still searched the default one's
collection. Confirmed by direct probe before the fix.

How a retriever makes such a copy is its own knowledge now, declared as
`core.interfaces.BoundToACorpus` and answered by the retriever. It used to be
a case analysis over the names of adapter classes, written in the experiment
builder, so this file carried fakes whose `__name__` had to be rewritten to be
recognised at all, and asserted on the arguments a constructor was called
with. It builds the real adapters instead, with the two clients that open
connections replaced, and reads what came back.
"""
from __future__ import annotations

from typing import Any

import pytest

from adapters.opensearch import OpenSearchRetriever
from adapters.qdrant import QdrantRetriever
from core.experiment.runner import _for_the_corpus


@pytest.fixture(autouse=True)
def _without_their_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real adapters, minus the two clients and the two calls that reach
    a server. What is left is exactly the part under test: which collection,
    which index, and what a copy of one carries over."""
    import opensearchpy
    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kw: object())
    monkeypatch.setattr(opensearchpy, "OpenSearch", lambda **kw: object())
    monkeypatch.setattr(QdrantRetriever, "_ensure_collection", lambda self, *a: None)
    monkeypatch.setattr(OpenSearchRetriever, "_ensure_index", lambda self: None)


def _dense(corpus_id: str = "default", realm_id: str | None = None) -> QdrantRetriever:
    return QdrantRetriever(
        host="qhost", port=1111, strategy_id="structure_aware",
        embedder_id="bge_m3", corpus_id=corpus_id, realm_id=realm_id,
    )


def _sparse(corpus_id: str = "default", language: str = "ru_be") -> OpenSearchRetriever:
    return OpenSearchRetriever(
        host="ohost", port=2222, strategy_id="structure_aware",
        corpus_id=corpus_id, language=language,
    )


class _CannotBeBound:
    """A stub, or a store this platform has never met. It answers nothing
    about binding a corpus, and comes back as it was instead of raising."""


def test_a_retriever_that_cannot_be_bound_comes_back_unchanged() -> None:
    stub = _CannotBeBound()
    assert _for_the_corpus(stub, "handbook") is stub


def test_the_corpus_it_already_reads_costs_no_second_connection() -> None:
    retriever = _dense(corpus_id="handbook")
    assert _for_the_corpus(retriever, "handbook") is retriever


def test_a_dense_copy_searches_the_named_corpus_collection() -> None:
    bound = _for_the_corpus(_dense(), "handbook")
    assert bound._corpus_id == "handbook"
    assert bound._collection == "handbook__structure_aware__bge_m3", (
        "the collection is what decides which vectors are searched"
    )
    # Everything the copy has to carry, or it would search the right corpus
    # through the wrong index, on the wrong host, in another model's vectors.
    assert (bound._host, bound._port) == ("qhost", 1111)
    assert (bound._strategy_id, bound._embedder_id) == ("structure_aware", "bge_m3")


def test_a_sparse_copy_keeps_the_analyser_the_index_was_built_with() -> None:
    """`language` decides which stemmer BM25 applies for the life of the
    index. Dropped, an Arabic corpus is queried through an index this copy
    insists is Russian: the wrong stemmer or a refusal, either arriving long
    after the choice was made."""
    bound = _for_the_corpus(_sparse(language="ar"), "miracl-ar")
    assert bound._corpus_id == "miracl-ar"
    assert bound._language == "ar"
    assert (bound._host, bound._port) == ("ohost", 2222)


def test_a_hybrid_binds_both_halves_and_keeps_how_it_fuses() -> None:
    from core.retrieval.hybrid import HybridRetriever

    hybrid = HybridRetriever(dense_retriever=_dense(), sparse_retriever=_sparse(),
                             embedder=object(), merge="weighted", alpha=0.3, rrf_k=17)
    bound = _for_the_corpus(hybrid, "handbook")

    assert isinstance(bound, HybridRetriever)
    assert bound._dense._corpus_id == "handbook"
    assert bound._sparse._corpus_id == "handbook"
    assert (bound._merge, bound._alpha, bound._rrf_k) == ("weighted", 0.3, 17), (
        "binding a corpus reset how the halves are fused"
    )


def test_a_graph_hybrid_binds_its_base_and_leaves_the_graph_alone() -> None:
    """One graph regardless of corpus: it has no partitioning to bind, so
    rebuilding it would cost a connection and change nothing."""
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    graph = object()
    walker = GraphHybridRetriever(graph_retriever=graph, base_retriever=_dense(),
                                  graph_weight=0.7, hops=2)
    bound = _for_the_corpus(walker, "handbook")

    assert isinstance(bound, GraphHybridRetriever)
    assert bound._graph is graph
    assert bound._base._corpus_id == "handbook"
    assert (bound._graph_weight, bound._hops) == (0.7, 2)


# ── the Realm's own instance ─────────────────────────────────────────────────

def test_the_corpus_and_the_realm_it_already_reads_cost_nothing() -> None:
    retriever = _dense(corpus_id="handbook", realm_id="demo")
    assert _for_the_corpus(retriever, "handbook", realm_id="demo") is retriever


def test_another_realm_is_a_copy_even_when_the_corpus_matches() -> None:
    """The bug this half was for: an in-process run against another Realm kept
    querying the gateway's own instance whenever the corpus already matched,
    because only the corpus was ever compared."""
    bound = _for_the_corpus(_dense(corpus_id="handbook"), "handbook", realm_id="demo")
    assert bound._realm_id == "demo"
    assert bound._collection.startswith("demo__")


def test_a_realms_own_host_and_port_replace_the_gateways() -> None:
    bound = _for_the_corpus(
        _dense(), "handbook", realm_id="demo",
        qdrant_cfg={"host": "realm-qdrant", "port": 9999},
    )
    assert (bound._host, bound._port) == ("realm-qdrant", 9999)
    assert (bound._corpus_id, bound._realm_id) == ("handbook", "demo")


def test_a_realm_that_registered_no_instance_keeps_the_retrievers_own() -> None:
    """The older behaviour, unchanged, and never a failure."""
    bound = _for_the_corpus(_dense(), "handbook")
    assert (bound._host, bound._port) == ("qhost", 1111)
    assert bound._realm_id is None


def test_each_half_of_a_hybrid_reads_its_own_stores_resources() -> None:
    """One map of resources goes down, and each half takes the key it knows.
    A half reading the other's host would connect to a search engine speaking
    another protocol."""
    from core.retrieval.hybrid import HybridRetriever

    hybrid = HybridRetriever(dense_retriever=_dense(), sparse_retriever=_sparse(),
                             embedder=object(), merge="rrf")
    bound: Any = _for_the_corpus(
        hybrid, "handbook", realm_id="demo",
        qdrant_cfg={"host": "realm-qdrant", "port": 9999},
        opensearch_cfg={"host": "realm-opensearch", "port": 9201},
    )
    assert (bound._dense._host, bound._dense._port) == ("realm-qdrant", 9999)
    assert (bound._sparse._host, bound._sparse._port) == ("realm-opensearch", 9201)
