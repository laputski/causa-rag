"""GraphRetriever adapter — LightRAG (production) + in-memory stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.models import Chunk, ScoredChunk


@dataclass
class GraphNode:
    node_id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class GraphResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    scored_chunks: list[ScoredChunk]


@runtime_checkable
class GraphRetriever(Protocol):
    retriever_id: str

    def retrieve_graph(
        self,
        query: str,
        k: int = 5,
        hops: int = 1,
        filters: dict[str, Any] | None = None,
    ) -> GraphResult:
        ...


class LightRagRetrieverStub:
    """In-memory graph stub using adjacency list for unit tests."""

    retriever_id = "lightrag_stub"

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._node_text: dict[str, str] = {}

    def add_node(self, node: GraphNode, text: str = "") -> None:
        self._nodes[node.node_id] = node
        self._node_text[node.node_id] = text or node.label

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges.append(edge)

    def _neighbors(self, node_id: str, hops: int) -> set[str]:
        visited = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for e in self._edges:
                if e.source_id in frontier and e.target_id not in visited:
                    next_frontier.add(e.target_id)
                if e.target_id in frontier and e.source_id not in visited:
                    next_frontier.add(e.source_id)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return visited - {node_id}

    def retrieve_graph(
        self,
        query: str,
        k: int = 5,
        hops: int = 1,
        filters: dict[str, Any] | None = None,
    ) -> GraphResult:
        query_tokens = set(query.lower().split())

        # Score nodes by token overlap with their label/text
        scored: list[tuple[float, str]] = []
        for nid in self._nodes:
            text = self._node_text[nid].lower()
            overlap = sum(1 for t in query_tokens if t in text)
            if overlap > 0:
                scored.append((overlap, nid))

        scored.sort(reverse=True)
        seed_ids = [nid for _, nid in scored[:k]]

        # Expand via hops
        expanded: set[str] = set(seed_ids)
        for nid in seed_ids:
            expanded |= self._neighbors(nid, hops)

        result_nodes = [self._nodes[nid] for nid in expanded if nid in self._nodes]
        result_edges = [
            e for e in self._edges
            if e.source_id in expanded and e.target_id in expanded
        ]

        chunks = []
        for i, nid in enumerate(seed_ids[:k]):
            if nid not in self._nodes:
                continue
            node = self._nodes[nid]
            score = scored[i][0] / max(len(query_tokens), 1)
            chunk = Chunk(
                doc_id=nid,
                text=self._node_text[nid],
                structural_path=node.label,
                strategy_id="graph",
            )
            chunks.append(ScoredChunk(chunk=chunk, score=score, retriever_id=self.retriever_id))

        return GraphResult(nodes=result_nodes, edges=result_edges, scored_chunks=chunks)


class LightRagRetriever:
    """Production LightRAG retriever (requires running LightRAG service)."""

    retriever_id = "lightrag"

    def __init__(self, base_url: str = "http://localhost:9621") -> None:
        self._base_url = base_url

    def retrieve_graph(
        self,
        query: str,
        k: int = 5,
        hops: int = 1,
        filters: dict[str, Any] | None = None,
    ) -> GraphResult:
        import httpx
        payload = {"query": query, "k": k, "hops": hops, "filters": filters or {}}
        resp = httpx.post(f"{self._base_url}/query/graph", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        nodes = [GraphNode(**n) for n in data.get("nodes", [])]
        edges = [GraphEdge(**e) for e in data.get("edges", [])]
        chunks_raw = data.get("chunks", [])
        chunks = [
            ScoredChunk(
                chunk=Chunk(**c["chunk"]),
                score=c["score"],
                retriever_id=self.retriever_id,
            )
            for c in chunks_raw
        ]
        return GraphResult(nodes=nodes, edges=edges, scored_chunks=chunks)
