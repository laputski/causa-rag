"""core/experiment/runner.py:_rebind_corpus_id.

Locks in the fix for a real bug: ExperimentConfig.corpus_id was decorative
— registry-resolved pipelines (naive/hybrid_rrf/hybrid_weighted/graph) are
built exactly once at gateway startup with corpus_id="default", and
_build_pipeline returned that frozen instance unchanged for any config
without reranker/grounding/etc. Confirmed by direct probe before this fix: a
config with a non-default corpus_id still resolved a retriever pointed at
the default corpus's collection.
"""
from __future__ import annotations

from unittest.mock import patch

from core.experiment.runner import _rebind_corpus_id


class _NoCorpusIdRetriever:
    """Stand-in for QdrantRetrieverStub/OpenSearchRetrieverStub — no
    _corpus_id attribute at all."""


class _FakeQdrantRetriever:
    """Duck-types adapters.qdrant.QdrantRetriever's relevant attributes
    without opening a real connection — class name matters, not identity,
    since _rebind_corpus_id dispatches on type(retriever).__name__."""

    def __init__(self, corpus_id: str) -> None:
        self._host = "qhost"
        self._port = 1111
        self._strategy_id = "structure_aware"
        self._embedder_id = "bge_m3"
        self._corpus_id = corpus_id


_FakeQdrantRetriever.__name__ = "QdrantRetriever"


class _FakeOpenSearchRetriever:
    def __init__(self, corpus_id: str) -> None:
        self._host = "ohost"
        self._port = 2222
        self._strategy_id = "structure_aware"
        self._corpus_id = corpus_id


_FakeOpenSearchRetriever.__name__ = "OpenSearchRetriever"


def test_no_op_when_retriever_has_no_corpus_id() -> None:
    """Stubs and any unrecognized retriever type pass through unchanged
    rather than raising."""
    stub = _NoCorpusIdRetriever()
    assert _rebind_corpus_id(stub, "handbook") is stub


def test_no_op_when_corpus_id_already_matches() -> None:
    retriever = _FakeQdrantRetriever(corpus_id="handbook")
    assert _rebind_corpus_id(retriever, "handbook") is retriever


def test_rebinds_qdrant_retriever_to_new_corpus_id() -> None:
    default_retriever = _FakeQdrantRetriever(corpus_id="default")
    with patch("adapters.qdrant.QdrantRetriever") as mock_cls:
        sentinel = object()
        mock_cls.return_value = sentinel
        result = _rebind_corpus_id(default_retriever, "handbook")
        assert result is sentinel
        mock_cls.assert_called_once_with(
            host="qhost", port=1111, strategy_id="structure_aware",
            embedder_id="bge_m3", corpus_id="handbook", realm_id=None,
        )


def test_rebinds_opensearch_retriever_to_new_corpus_id() -> None:
    default_retriever = _FakeOpenSearchRetriever(corpus_id="default")
    with patch("adapters.opensearch.OpenSearchRetriever") as mock_cls:
        sentinel = object()
        mock_cls.return_value = sentinel
        result = _rebind_corpus_id(default_retriever, "handbook")
        assert result is sentinel
        mock_cls.assert_called_once_with(
            host="ohost", port=2222, strategy_id="structure_aware", corpus_id="handbook", realm_id=None,
        )


def test_rebinds_hybrid_retriever_by_recursing_into_both_leaves() -> None:
    from core.retrieval.hybrid import HybridRetriever

    dense = _FakeQdrantRetriever(corpus_id="default")
    sparse = _FakeOpenSearchRetriever(corpus_id="default")
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, embedder=object(), merge="rrf")

    with patch("adapters.qdrant.QdrantRetriever") as mock_qdrant, \
         patch("adapters.opensearch.OpenSearchRetriever") as mock_os:
        mock_qdrant.return_value = object()
        mock_os.return_value = object()
        result = _rebind_corpus_id(hybrid, "handbook")

    assert isinstance(result, HybridRetriever)
    mock_qdrant.assert_called_once()
    mock_os.assert_called_once()
    assert mock_qdrant.call_args.kwargs["corpus_id"] == "handbook"
    assert mock_os.call_args.kwargs["corpus_id"] == "handbook"


def test_rebinds_graph_hybrid_retriever_leaving_graph_component_untouched() -> None:
    """The Neo4j graph has no corpus_id partitioning at all (one shared
    graph) — only the dense/sparse base retriever should be rebuilt."""
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    graph_component = object()  # never touched — identity must be preserved
    base = _FakeQdrantRetriever(corpus_id="default")
    graph_hybrid = GraphHybridRetriever(graph_retriever=graph_component, base_retriever=base)

    with patch("adapters.qdrant.QdrantRetriever") as mock_qdrant:
        mock_qdrant.return_value = object()
        result = _rebind_corpus_id(graph_hybrid, "handbook")

    assert isinstance(result, GraphHybridRetriever)
    assert result._graph is graph_component
    mock_qdrant.assert_called_once()
    assert mock_qdrant.call_args.kwargs["corpus_id"] == "handbook"


# ── realm_id / Realm-owned host:port ──────────────────────────────

def test_no_op_when_corpus_id_and_realm_id_both_already_match() -> None:
    retriever = _FakeQdrantRetriever(corpus_id="handbook")
    retriever._realm_id = "demo"
    assert _rebind_corpus_id(retriever, "handbook", realm_id="demo") is retriever


def test_rebinds_when_only_realm_id_differs_even_if_corpus_id_already_matches() -> None:
    """The actual bug: an in_process experiment run against a
    non-default Realm used to keep querying the gateway's startup-time
    env-var Qdrant even when config.corpus_id already matched, because only
    corpus_id was ever compared — realm_id had no bearing on the decision."""
    retriever = _FakeQdrantRetriever(corpus_id="handbook")
    retriever._realm_id = None
    with patch("adapters.qdrant.QdrantRetriever") as mock_cls:
        mock_cls.return_value = object()
        _rebind_corpus_id(retriever, "handbook", realm_id="demo")
    mock_cls.assert_called_once_with(
        host="qhost", port=1111, strategy_id="structure_aware",
        embedder_id="bge_m3", corpus_id="handbook", realm_id="demo",
    )


def test_qdrant_cfg_overrides_host_and_port() -> None:
    default_retriever = _FakeQdrantRetriever(corpus_id="default")
    with patch("adapters.qdrant.QdrantRetriever") as mock_cls:
        mock_cls.return_value = object()
        _rebind_corpus_id(
            default_retriever, "handbook", realm_id="demo",
            qdrant_cfg={"host": "realm-qdrant", "port": 9999},
        )
    mock_cls.assert_called_once_with(
        host="realm-qdrant", port=9999, strategy_id="structure_aware",
        embedder_id="bge_m3", corpus_id="handbook", realm_id="demo",
    )


def test_missing_qdrant_cfg_falls_back_to_retrievers_own_host_and_port() -> None:
    """No Realm resource registered (or realm_id="") ⇒ the older behaviour
    unchanged, not a hard failure."""
    default_retriever = _FakeQdrantRetriever(corpus_id="default")
    with patch("adapters.qdrant.QdrantRetriever") as mock_cls:
        mock_cls.return_value = object()
        _rebind_corpus_id(default_retriever, "handbook")
    mock_cls.assert_called_once_with(
        host="qhost", port=1111, strategy_id="structure_aware",
        embedder_id="bge_m3", corpus_id="handbook", realm_id=None,
    )
