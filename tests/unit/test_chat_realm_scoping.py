"""Chat's in-process fallback used to always resolve the registry's
startup-time retriever/generator — built once from env-var Qdrant/Ollama, on
corpus_id="default", identically regardless of which Realm was active. A
Realm with its own `resources[]` (or its own corpus_id) was unreachable from
chat without registering an external RAG (see the design notes "Chat routing
via Realm"). These tests pin the fix: _build_realm_scoped_retriever/
_generator actually read the Realm's own resources when present.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.qdrant import QdrantRetriever
from services.api_gateway.main import _build_realm_scoped_generator, _build_realm_scoped_retriever


class _FakeEmbedder:
    embedder_id = "bge_m3"


def _mock_qdrant_client():
    """QdrantRetriever.__init__ makes a real collection_exists() call to
    ensure its collection exists (not get_collections(), which
    doesn't resolve aliases; see adapters/qdrant.py#_ensure_collection) —
    these tests are about which host/port/corpus_id it gets constructed
    with, not about a live Qdrant, so the network client itself is mocked
    out."""
    client = MagicMock()
    client.collection_exists.return_value = True
    return patch("qdrant_client.QdrantClient", return_value=client)


async def test_retriever_uses_the_realms_own_qdrant_resource() -> None:
    acme_qdrant = {"type": "qdrant", "host": "acme-qdrant.example.com", "port": 6400}

    async def fake_find_one(collection, query):
        assert query == {"id": "acme"}
        return {"id": "acme", "resources": [acme_qdrant]}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), _mock_qdrant_client():
        retriever = await _build_realm_scoped_retriever("acme", "acme-corpus", _FakeEmbedder())

    assert isinstance(retriever, QdrantRetriever)
    assert retriever._host == "acme-qdrant.example.com"
    assert retriever._port == 6400
    assert retriever._corpus_id == "acme-corpus"


async def test_retriever_falls_back_to_env_var_qdrant_when_realm_has_no_resource() -> None:
    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}  # no qdrant resource registered

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), _mock_qdrant_client():
        retriever = await _build_realm_scoped_retriever("acme", "default", _FakeEmbedder())

    # Falls through to corpus.py's own env-var default (localhost) — same
    # degrade-to-shared-infra behavior already documented for Content/Health/
    # Graph, not a new gap introduced here.
    assert isinstance(retriever, QdrantRetriever)
    assert retriever._host == "localhost"


async def test_retriever_with_no_realm_id_uses_env_var_qdrant_directly() -> None:
    """No Realm active at all (shouldn't normally happen — every page
    requires one) still degrades to the old global behavior rather than
    crashing on a None realm_id."""
    with patch("adapters.mongodb.find_one") as mock_find_one, _mock_qdrant_client():
        retriever = await _build_realm_scoped_retriever(None, "default", _FakeEmbedder())
    mock_find_one.assert_not_called()
    assert isinstance(retriever, QdrantRetriever)


async def test_generator_uses_the_realms_own_ollama_resource() -> None:
    acme_ollama = {"type": "ollama", "host": "acme-ollama.example.com", "port": 11500, "model": "qwen-acme"}

    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": [acme_ollama]}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one):
        generator = await _build_realm_scoped_generator("acme")

    assert generator._base_url == "http://acme-ollama.example.com:11500"
    assert generator._model == "qwen-acme"


async def test_generator_falls_back_to_shared_instance_when_realm_has_no_ollama_resource() -> None:
    import services.api_gateway.main as gateway_main
    from adapters.generator_stub import GeneratorStub
    from core.registry import ComponentRegistry

    reg = ComponentRegistry()
    shared = GeneratorStub()
    reg.register("generator", "ollama", shared)

    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), \
         patch.object(gateway_main, "registry", reg):
        generator = await _build_realm_scoped_generator("acme")

    assert generator is shared


# ── pipeline_id (chat was permanently dense-only before this) ────

def _mock_opensearch_client():
    client = MagicMock()
    client.indices.exists.return_value = True
    return patch("opensearchpy.OpenSearch", return_value=client)


async def test_naive_pipeline_id_is_unchanged_dense_only_behavior() -> None:
    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), _mock_qdrant_client():
        retriever = await _build_realm_scoped_retriever("acme", "default", _FakeEmbedder(), "naive")

    assert isinstance(retriever, QdrantRetriever)


async def test_hybrid_rrf_wraps_dense_and_realm_scoped_sparse() -> None:
    from adapters.opensearch import OpenSearchRetriever
    from core.retrieval.hybrid import HybridRetriever

    acme_opensearch = {"type": "opensearch", "host": "acme-os.example.com", "port": 9250}

    async def fake_find_one(collection, query):
        if query == {"id": "acme"}:
            return {"id": "acme", "resources": [acme_opensearch]}
        return None

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), \
         _mock_qdrant_client(), _mock_opensearch_client():
        retriever = await _build_realm_scoped_retriever("acme", "acme-corpus", _FakeEmbedder(), "hybrid_rrf")

    assert isinstance(retriever, HybridRetriever)
    assert isinstance(retriever._sparse, OpenSearchRetriever)
    assert retriever._sparse._host == "acme-os.example.com"
    assert retriever._sparse._port == 9250
    assert retriever._merge == "rrf"


async def test_hybrid_weighted_uses_weighted_merge() -> None:
    from core.retrieval.hybrid import HybridRetriever

    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), \
         _mock_qdrant_client(), _mock_opensearch_client():
        retriever = await _build_realm_scoped_retriever("acme", "default", _FakeEmbedder(), "hybrid_weighted")

    assert isinstance(retriever, HybridRetriever)
    assert retriever._merge == "weighted"


async def test_hybrid_falls_back_to_dense_when_opensearch_unavailable() -> None:
    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), _mock_qdrant_client(), \
         patch("opensearchpy.OpenSearch", side_effect=RuntimeError("down")):
        retriever = await _build_realm_scoped_retriever("acme", "default", _FakeEmbedder(), "hybrid_rrf")

    assert isinstance(retriever, QdrantRetriever)


async def test_graph_wraps_dense_when_neo4j_available() -> None:
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    fake_graph = MagicMock()
    fake_graph.is_available.return_value = True
    fake_graph.verify.return_value = True

    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), _mock_qdrant_client(), \
         patch("services.api_gateway.routers.corpus._resolve_neo4j", return_value=fake_graph):
        retriever = await _build_realm_scoped_retriever("acme", "default", _FakeEmbedder(), "graph")

    assert isinstance(retriever, GraphHybridRetriever)
    assert retriever._graph is fake_graph


async def test_graph_falls_back_to_dense_when_neo4j_unreachable() -> None:
    """Mirrors _register_graph_pipeline's own availability gate at startup —
    chat must degrade gracefully, not 500 every graph-mode message just
    because this Realm's Neo4j (or the shared one) happens to be down."""
    fake_graph = MagicMock()
    fake_graph.is_available.return_value = False

    async def fake_find_one(collection, query):
        return {"id": "acme", "resources": []}

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one), _mock_qdrant_client(), \
         patch("services.api_gateway.routers.corpus._resolve_neo4j", return_value=fake_graph):
        retriever = await _build_realm_scoped_retriever("acme", "default", _FakeEmbedder(), "graph")

    assert isinstance(retriever, QdrantRetriever)
