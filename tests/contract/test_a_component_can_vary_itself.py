"""Contract tests: the five ways a component can make a variant of itself.

A run varies what the platform built once: another corpus, another Realm's
instance, another fusion weight, another model, the same sampling every time. Which components could do
which was a case analysis over the names of adapter classes, written in the
experiment builder, so a component the analysis did not name was named by a
configuration and never varied, and a component added later arrived in that
state by default. The knowledge is the component's own now, and these are the
promises it makes when it claims one of these protocols.

Two promises are shared by all four and are the ones worth stating:

* the copy carries everything it was not asked to change, since a copy that
  reset a field would answer the run from another index, another host or
  another model without saying so;
* asking for what a component already is returns the component itself, since
  the copies here cost a connection or a loaded model.
"""
from __future__ import annotations

from typing import Any

import pytest

from adapters.ollama_generator import OllamaGenerator
from adapters.opensearch import OpenSearchRetriever
from adapters.qdrant import QdrantRetriever
from core.interfaces import (
    BoundToACorpus,
    ChoosingItsModel,
    FixingItsSampling,
    Fusing,
    WalkingAGraph,
)
from core.retrieval.graph_hybrid import GraphHybridRetriever
from core.retrieval.hybrid import HybridRetriever


@pytest.fixture(autouse=True)
def _without_their_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    import opensearchpy
    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kw: object())
    monkeypatch.setattr(opensearchpy, "OpenSearch", lambda **kw: object())
    monkeypatch.setattr(QdrantRetriever, "_ensure_collection", lambda self, *a: None)
    monkeypatch.setattr(OpenSearchRetriever, "_ensure_index", lambda self: None)


def _dense(**over: Any) -> QdrantRetriever:
    return QdrantRetriever(**{"host": "qhost", "port": 1111, "strategy_id": "fixed",
                              "embedder_id": "bge_m3", **over})


def _sparse(**over: Any) -> OpenSearchRetriever:
    return OpenSearchRetriever(**{"host": "ohost", "port": 2222, "strategy_id": "fixed",
                                  "language": "ru_be", **over})


def _hybrid(**over: Any) -> HybridRetriever:
    return HybridRetriever(**{"dense_retriever": _dense(), "sparse_retriever": _sparse(),
                              "embedder": object(), **over})


def _walker(**over: Any) -> GraphHybridRetriever:
    return GraphHybridRetriever(**{"graph_retriever": object(),
                                   "base_retriever": _dense(), **over})


# ── a retriever bound to a corpus ────────────────────────────────────────────

@pytest.mark.parametrize("build", [_dense, _sparse, _hybrid, _walker])
def test_a_retriever_declares_it_can_be_bound(build: Any) -> None:
    assert isinstance(build(), BoundToACorpus)


@pytest.mark.parametrize("build", [_dense, _sparse, _hybrid, _walker])
def test_the_corpus_it_already_reads_returns_the_same_object(build: Any) -> None:
    retriever = build()
    assert retriever.for_corpus("default", None) is retriever


@pytest.mark.parametrize("build", [_dense, _sparse, _hybrid, _walker])
def test_another_corpus_returns_something_else(build: Any) -> None:
    retriever = build()
    assert retriever.for_corpus("another", None) is not retriever


def test_a_bound_copy_keeps_what_it_was_not_asked_to_change() -> None:
    dense = _dense().for_corpus("another", "a_realm")
    assert (dense._host, dense._port) == ("qhost", 1111)
    assert (dense._strategy_id, dense._embedder_id) == ("fixed", "bge_m3")

    sparse = _sparse().for_corpus("another", "a_realm")
    assert (sparse._host, sparse._port) == ("ohost", 2222)
    assert sparse._language == "ru_be"


def test_a_realms_own_instance_replaces_the_host_and_port() -> None:
    resources = {"qdrant": {"host": "q2", "port": 3},
                 "opensearch": {"host": "o2", "port": 4}}
    dense = _dense().for_corpus("default", "a_realm", resources)
    sparse = _sparse().for_corpus("default", "a_realm", resources)
    assert (dense._host, dense._port) == ("q2", 3)
    assert (sparse._host, sparse._port) == ("o2", 4)


# ── a retriever that fuses ───────────────────────────────────────────────────

def test_only_a_fusing_retriever_declares_it_fuses() -> None:
    assert isinstance(_hybrid(), Fusing)
    assert not isinstance(_dense(), Fusing)


def test_the_fusion_it_already_does_returns_the_same_object() -> None:
    hybrid = _hybrid(merge="rrf", alpha=0.5, rrf_k=60)
    assert hybrid.with_fusion("rrf", 0.5, 60) is hybrid
    assert hybrid.with_fusion(None, None, None) is hybrid


def test_a_fusing_copy_keeps_the_halves_and_the_constant() -> None:
    """The constant is the one that was measured going missing: rebuilding
    without it reset it to sixty, so asking for another weight silently undid
    another constant set anywhere upstream."""
    hybrid = _hybrid(merge="rrf", rrf_k=10)
    changed = hybrid.with_fusion(alpha=0.9)
    assert changed._rrf_k == 10
    assert changed._dense is hybrid._dense
    assert changed._sparse is hybrid._sparse


def test_a_fusion_nobody_implements_leaves_the_retriever_as_built() -> None:
    """A whole run should not be lost to one mistyped field, and the run
    records what it used."""
    hybrid = _hybrid()
    assert hybrid.with_fusion("by telepathy") is hybrid


# ── a retriever that walks a graph ───────────────────────────────────────────

def test_only_a_walking_retriever_declares_it_walks() -> None:
    assert isinstance(_walker(), WalkingAGraph)
    assert not isinstance(_hybrid(), WalkingAGraph)


def test_the_walk_it_already_does_returns_the_same_object() -> None:
    walker = _walker(graph_weight=0.4, hops=1)
    assert walker.with_graph(0.4, 1) is walker
    assert walker.with_graph(None, None) is walker


def test_a_walking_copy_keeps_the_graph_and_the_base() -> None:
    walker = _walker()
    changed = walker.with_graph(graph_weight=0.9)
    assert changed._graph is walker._graph
    assert changed._base is walker._base
    assert (changed._graph_weight, changed._hops) == (0.9, walker._hops)


# ── a component that runs a named model ──────────────────────────────────────

def test_a_generator_declares_it_chooses_its_model() -> None:
    assert isinstance(OllamaGenerator(model="a"), ChoosingItsModel)


def test_the_model_it_already_runs_returns_the_same_object() -> None:
    generator = OllamaGenerator(model="a")
    assert generator.with_model("a") is generator
    assert generator.with_model("") is generator


def test_a_model_copy_keeps_where_it_runs_and_how_long_it_waits() -> None:
    generator = OllamaGenerator(base_url="http://elsewhere:1", model="a", timeout=7.0)
    changed = generator.with_model("b")
    assert changed._model == "b"
    assert (changed._base_url, changed._timeout) == ("http://elsewhere:1", 7.0)


def test_a_reranker_declares_it_chooses_its_model_when_it_has_one() -> None:
    from adapters.reranker import CrossEncoderRerankerLocal, CrossEncoderRerankerStub

    assert isinstance(CrossEncoderRerankerLocal(), ChoosingItsModel)
    # The stub reranks by a rule and has no model, so it makes no such promise
    # and is left alone by anything asking for one.
    assert not isinstance(CrossEncoderRerankerStub(), ChoosingItsModel)


# ── a component whose sampling can be fixed ──────────────────────────────────

def test_a_generator_declares_it_can_fix_its_sampling() -> None:
    assert isinstance(OllamaGenerator(model="a"), FixingItsSampling)


def test_the_seed_it_already_uses_returns_the_same_object() -> None:
    generator = OllamaGenerator(model="a", seed=42)
    assert generator.with_seed(42) is generator


def test_a_seeded_copy_keeps_the_model_and_where_it_runs() -> None:
    generator = OllamaGenerator(base_url="http://elsewhere:1", model="a", timeout=7.0)
    changed = generator.with_seed(42)
    assert changed._seed == 42
    assert (changed._model, changed._base_url, changed._timeout) == (
        "a", "http://elsewhere:1", 7.0)


def test_a_generator_with_no_seed_samples_as_the_server_would() -> None:
    """Absent and never zero: zero is a seed, and a caller that never asked
    for one must be left where it was."""
    assert OllamaGenerator(model="a")._seed is None
