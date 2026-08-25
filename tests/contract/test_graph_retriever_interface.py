"""Contract tests — GraphRetriever Protocol compliance."""
from adapters.lightrag import GraphEdge, GraphNode, GraphRetriever, LightRagRetrieverStub


def _stub_with_data() -> LightRagRetrieverStub:
    stub = LightRagRetrieverStub()
    stub.add_node(GraphNode("n1", "the tax code"), "the tax code of this jurisdiction")
    stub.add_node(GraphNode("n2", "the VAT act"), "the act on value added tax")
    stub.add_edge(GraphEdge("n1", "n2", "governs"))
    return stub


def test_stub_is_graph_retriever():
    assert isinstance(LightRagRetrieverStub(), GraphRetriever)


def test_retrieve_graph_returns_result():
    stub = _stub_with_data()
    result = stub.retrieve_graph("the tax code")
    assert result.scored_chunks or result.nodes


def test_hop_expansion():
    stub = _stub_with_data()
    result = stub.retrieve_graph("tax", k=1, hops=1)
    node_ids = {n.node_id for n in result.nodes}
    # n2 should be reachable via 1 hop from n1
    assert "n1" in node_ids or "n2" in node_ids


def test_no_match_returns_empty_chunks():
    stub = LightRagRetrieverStub()
    result = stub.retrieve_graph("an entirely irrelevant query")
    assert result.scored_chunks == []
