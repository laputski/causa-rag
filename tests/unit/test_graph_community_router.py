"""GET /corpus/{corpus_id}/graph/communities[/{community_id}] — community
detection diagnostics (GDS Leiden/Louvain over the Neo4j graph, labeled with
each community's source_code mix from Qdrant).

corpus_id only selects which Qdrant collection supplies the source_code
labels — the Neo4j graph itself has no corpus_id partitioning, so
detect_communities() always runs over the whole graph (see
adapters/neo4j_graph.py).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.models import Chunk
from services.api_gateway.routers.corpus import _doc_to_source_code_map, _label_communities, router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _chunk(doc_id: str, source_code: str | None) -> Chunk:
    return Chunk(doc_id=doc_id, text="x", metadata={"source_code": source_code} if source_code else {})


class _FakeQdrant:
    """Simulates QdrantRetriever.scroll() pagination across multiple pages,
    the exact thing _doc_to_source_code_map must page past (real corpus has
    ~35k chunks, scroll_all()'s hard cap is 5000)."""

    def __init__(self, pages: list[list[Chunk]]) -> None:
        self._pages = pages

    def scroll(self, offset=None, limit=1000):
        idx = int(offset) if offset else 0
        if idx >= len(self._pages):
            return [], None
        page = self._pages[idx]
        next_offset = str(idx + 1) if idx + 1 < len(self._pages) else None
        return page, next_offset


def test_doc_to_source_code_map_paginates_across_pages() -> None:
    pages = [
        [_chunk("d1", "SRC004"), _chunk("d2", "SRC001")],
        [_chunk("d3", "SRC001"), _chunk("d4", None)],  # no metadata -> skipped
    ]
    with patch(
        "services.api_gateway.routers.corpus._resolve_qdrant",
        return_value=_FakeQdrant(pages),
    ):
        mapping = _doc_to_source_code_map("handbook", "structure_aware", "bge_m3")
    assert mapping == {"d1": "SRC004", "d2": "SRC001", "d3": "SRC001"}


def test_label_communities_computes_source_distribution_and_drops_doc_ids() -> None:
    communities = [
        {"community_id": 1, "size": 4, "doc_ids": ["d1", "d2", "d3", "d4"]},
    ]
    doc_to_code = {"d1": "SRC004", "d2": "SRC004", "d3": "SRC001", "d4": "SRC001"}
    _label_communities(communities, doc_to_code)
    c = communities[0]
    assert "doc_ids" not in c
    assert c["distinct_source_count"] == 2
    assert c["dominant_source_share"] == 0.5
    assert {d["source_code"] for d in c["source_code_distribution"]} == {"SRC004", "SRC001"}


def test_label_communities_handles_no_matched_docs() -> None:
    """A community whose doc_ids never made it into the Qdrant map (e.g.
    stale graph data from a since-deleted corpus) must not crash with a
    division by zero."""
    communities = [{"community_id": 2, "size": 3, "doc_ids": ["missing1", "missing2"]}]
    _label_communities(communities, doc_to_code={})
    c = communities[0]
    assert c["dominant_source_share"] == 0.0
    assert c["distinct_source_count"] == 0
    assert c["source_code_distribution"] == []


def test_graph_communities_endpoint_returns_labeled_aggregate(client) -> None:
    fake_result = {
        "algorithm": "leiden",
        "community_count": 1,
        "modularity": 0.5,
        "communities": [{"community_id": 1, "size": 2, "doc_ids": ["d1", "d2"]}],
        "inter_community_edges": [],
    }
    with patch("adapters.neo4j_graph.Neo4jGraphRetriever.detect_communities", return_value=fake_result), \
         patch(
             "services.api_gateway.routers.corpus._resolve_qdrant",
             return_value=_FakeQdrant([[_chunk("d1", "SRC004"), _chunk("d2", "SRC004")]]),
         ):
        resp = client.get("/corpus/handbook/graph/communities?algorithm=leiden")
    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_id"] == "handbook"
    assert body["communities"][0]["dominant_source_share"] == 1.0
    assert "doc_ids" not in body["communities"][0]


def test_graph_communities_endpoint_503_when_graph_unavailable(client) -> None:
    from adapters.neo4j_graph import GraphUnavailable

    with patch(
        "adapters.neo4j_graph.Neo4jGraphRetriever.detect_communities",
        side_effect=GraphUnavailable("no connection"),
    ):
        resp = client.get("/corpus/handbook/graph/communities")
    assert resp.status_code == 503


def test_graph_communities_endpoint_400_on_bad_algorithm(client) -> None:
    with patch(
        "adapters.neo4j_graph.Neo4jGraphRetriever.detect_communities",
        side_effect=ValueError("unsupported algorithm 'bogus'"),
    ):
        resp = client.get("/corpus/handbook/graph/communities?algorithm=bogus")
    assert resp.status_code == 400


def test_graph_community_detail_endpoint(client) -> None:
    fake_subgraph = {
        "community_id": 1,
        "nodes": [{"chunk_id": "c1", "doc_id": "d1", "path": "document/article[Article 1]"}],
        "edges": [],
    }
    with patch("adapters.neo4j_graph.Neo4jGraphRetriever.community_subgraph", return_value=fake_subgraph):
        resp = client.get("/corpus/handbook/graph/communities/1?algorithm=leiden")
    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_id"] == "handbook"
    assert body["community_id"] == 1
    assert len(body["nodes"]) == 1
