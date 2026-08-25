"""Neo4jGraphRetriever.project_graph/detect_communities/community_subgraph —
mocking the driver/session (no real Neo4j needed for these).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.neo4j_graph import Neo4jGraphRetriever


def _retriever_with_mock_session():
    retriever = Neo4jGraphRetriever()
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    retriever._driver = mock_driver
    return retriever, mock_session


def test_detect_communities_rejects_unsupported_algorithm() -> None:
    retriever, _ = _retriever_with_mock_session()
    with pytest.raises(ValueError, match="bogus"):
        retriever.detect_communities(algorithm="bogus")


def test_community_subgraph_rejects_unsupported_algorithm() -> None:
    retriever, _ = _retriever_with_mock_session()
    with pytest.raises(ValueError, match="bogus"):
        retriever.community_subgraph(1, algorithm="bogus")


def test_detect_communities_rejects_unsupported_edge_type() -> None:
    retriever, _ = _retriever_with_mock_session()
    with pytest.raises(ValueError, match="bogus"):
        retriever.detect_communities(edge_type="bogus")


def test_community_subgraph_rejects_unsupported_edge_type() -> None:
    retriever, _ = _retriever_with_mock_session()
    with pytest.raises(ValueError, match="bogus"):
        retriever.community_subgraph(1, edge_type="bogus")


def test_project_graph_rejects_unsupported_edge_type() -> None:
    retriever, _ = _retriever_with_mock_session()
    with pytest.raises(ValueError, match="bogus"):
        retriever.project_graph(edge_type="bogus")


def test_detect_communities_uses_per_algorithm_write_property() -> None:
    """leiden and louvain must write to distinct properties — running one
    must not silently overwrite/invalidate the other's last result."""
    retriever, session = _retriever_with_mock_session()
    # `project_graph` now yields the projection's size, and `detect_communities`
    # reads it before running the algorithm — a projection with no edges is
    # answered with an empty result rather than by letting GDS throw. So the
    # mock has to report edges, or nothing downstream is exercised at all.
    session.run.return_value.single.return_value = {
        "nodeCount": 4, "relationshipCount": 6, "communityCount": 0, "modularity": 0.0,
    }
    session.run.return_value.__iter__ = lambda self: iter([])

    retriever.detect_communities(algorithm="leiden")
    calls = [c for c in session.run.call_args_list if "writeProperty" in str(c)]
    assert any("_leidenCommunity" in str(c) for c in calls)

    session.run.reset_mock()
    retriever.detect_communities(algorithm="louvain")
    calls = [c for c in session.run.call_args_list if "writeProperty" in str(c)]
    assert any("_louvainCommunity" in str(c) for c in calls)


def test_detect_communities_returns_aggregate_shape() -> None:
    retriever, session = _retriever_with_mock_session()

    def run_side_effect(query, **kwargs):
        result = MagicMock()
        if "YIELD communityCount" in query:
            result.single.return_value = {"communityCount": 2, "modularity": 0.42}
        elif "collect(DISTINCT n.doc_id)" in query:
            result.__iter__ = lambda self: iter([
                {"community_id": 0, "size": 3, "doc_ids": ["d1", "d2"]},
                {"community_id": 1, "size": 1, "doc_ids": ["d3"]},
            ])
        elif "count(*) AS weight" in query:
            result.__iter__ = lambda self: iter([{"community_a": 0, "community_b": 1, "weight": 2}])
        else:
            result.__iter__ = lambda self: iter([])
        return result

    session.run.side_effect = run_side_effect

    with patch.object(retriever, "project_graph", return_value=(9, 12)) as mock_project:
        out = retriever.detect_communities(algorithm="leiden")
        mock_project.assert_called_once()

    assert out["algorithm"] == "leiden"
    assert out["community_count"] == 2
    assert out["modularity"] == 0.42
    assert out["communities"] == [
        {"community_id": 0, "size": 3, "doc_ids": ["d1", "d2"]},
        {"community_id": 1, "size": 1, "doc_ids": ["d3"]},
    ]
    assert out["inter_community_edges"] == [{"community_a": 0, "community_b": 1, "weight": 2}]
    assert out["edge_type"] == "lexical"


def test_detect_communities_uses_related_semantic_when_edge_type_is_semantic() -> None:
    """edge_type="semantic" must project/query RELATED_SEMANTIC, not the
    default lexical RELATED — and write to a distinct property suffix so
    running both signals back-to-back doesn't overwrite either result
    (lexical vs semantic kept independently comparable)."""
    retriever, session = _retriever_with_mock_session()
    session.run.return_value.single.return_value = {
        "nodeCount": 4, "relationshipCount": 6, "communityCount": 0, "modularity": 0.0,
    }
    session.run.return_value.__iter__ = lambda self: iter([])

    out = retriever.detect_communities(algorithm="leiden", edge_type="semantic")

    assert out["edge_type"] == "semantic"
    project_calls = [c for c in session.run.call_args_list if "gds.graph.project" in str(c)]
    assert any("RELATED_SEMANTIC" in str(c) for c in project_calls)
    write_property_calls = [c for c in session.run.call_args_list if "writeProperty" in str(c)]
    assert any("_leidenCommunity_semantic" in str(c) for c in write_property_calls)


def test_detect_communities_on_an_edgeless_projection_returns_an_empty_result() -> None:
    """A realm whose chunks were never linked projects nodes and no edges, and
    `gds.leiden.write` throws a NullPointerException on that — which reached the
    screen as a bare 500. The size of the projection is cheaper to ask for than
    someone else's stack trace is to read, and an empty answer is the honest
    one: a graph without links has no communities."""
    retriever, session = _retriever_with_mock_session()

    with patch.object(retriever, "project_graph", return_value=(120, 0)):
        out = retriever.detect_communities(algorithm="leiden")

    assert out["community_count"] == 0
    assert out["communities"] == []
    assert out["inter_community_edges"] == []
    assert out["algorithm"] == "leiden"
    # The algorithm is never asked to run — that is the whole point.
    assert not [c for c in session.run.call_args_list if "writeProperty" in str(c)]


def test_community_subgraph_returns_nodes_and_edges() -> None:
    retriever, session = _retriever_with_mock_session()

    def run_side_effect(query, **kwargs):
        result = MagicMock()
        if "RETURN n.chunk_id" in query:
            result.__iter__ = lambda self: iter([
                {"chunk_id": "c1", "doc_id": "d1", "path": "document/article[Article 1]"},
            ])
        else:
            result.__iter__ = lambda self: iter([{"source": "c1", "target": "c2"}])
        return result

    session.run.side_effect = run_side_effect

    out = retriever.community_subgraph(5, algorithm="leiden")
    assert out["community_id"] == 5
    assert out["nodes"] == [{"chunk_id": "c1", "doc_id": "d1", "path": "document/article[Article 1]"}]
    assert out["edges"] == [{"source": "c1", "target": "c2"}]
