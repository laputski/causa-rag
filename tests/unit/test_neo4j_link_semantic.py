"""Neo4jGraphRetriever.link_semantic — mocking the driver/session and a
Qdrant client (no real Neo4j/Qdrant needed for these).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from adapters.neo4j_graph import Neo4jGraphRetriever


def _retriever_with_mock_session():
    retriever = Neo4jGraphRetriever()
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    retriever._driver = mock_driver
    return retriever, mock_session


def _hit(point_id: str, score: float) -> MagicMock:
    h = MagicMock()
    h.id = point_id
    h.score = score
    return h


def _fake_qdrant(hits_by_chunk: dict[str, list[MagicMock]]) -> MagicMock:
    qdrant = MagicMock()
    qdrant._collection = "coll"

    def query_points(collection_name, query, limit, with_payload):
        result = MagicMock()
        result.points = hits_by_chunk.get(query, [])
        return result

    qdrant._client.query_points.side_effect = query_points
    return qdrant


def _run_side_effect(chunk_ids: list[str], write_counts: list[int]):
    """First call (MATCH (n:Chunk) RETURN n.chunk_id) is iterated for the
    scan; each subsequent call is a write batch whose .single()['c'] is the
    real MERGE-matched count (see _write_semantic_batch docstring — this is
    what makes `written` trustworthy instead of an optimistic len(edges))."""
    counts = iter(write_counts)

    def run(query, **kwargs):
        result = MagicMock()
        if "UNWIND $edges" in query:
            result.single.return_value = {"c": next(counts)}
        else:
            result.__iter__ = lambda self: iter([{"cid": cid} for cid in chunk_ids])
        return result

    return run


def test_link_semantic_writes_edges_above_threshold() -> None:
    retriever, session = _retriever_with_mock_session()
    session.run.side_effect = _run_side_effect(["c1", "c2"], write_counts=[1])

    qdrant = _fake_qdrant({
        "c1": [_hit("c1", 1.0), _hit("c2", 0.9)],
        "c2": [_hit("c2", 1.0), _hit("c1", 0.9)],
    })

    written = retriever.link_semantic(qdrant, threshold=0.75, top_k=5)

    assert written == 1  # symmetric pair deduped to one edge
    unwind_calls = [c for c in session.run.call_args_list if "UNWIND $edges" in str(c)]
    assert len(unwind_calls) == 1
    edges = unwind_calls[0].kwargs["edges"]
    assert edges == [{"a": "c1", "b": "c2", "score": 0.9}]


def test_link_semantic_excludes_self_and_below_threshold() -> None:
    retriever, session = _retriever_with_mock_session()
    session.run.side_effect = _run_side_effect(["c1"], write_counts=[])

    qdrant = _fake_qdrant({
        "c1": [_hit("c1", 1.0), _hit("c2", 0.5)],  # self-hit + below threshold
    })

    written = retriever.link_semantic(qdrant, threshold=0.75, top_k=5)

    assert written == 0


def test_link_semantic_tolerates_missing_qdrant_point() -> None:
    """A chunk_id absent from this Qdrant collection (e.g. different
    corpus_id) must be skipped, not abort the whole scan."""
    retriever, session = _retriever_with_mock_session()
    session.run.side_effect = _run_side_effect(["missing", "c1", "c2"], write_counts=[1])

    qdrant = MagicMock()
    qdrant._collection = "coll"

    def query_points(collection_name, query, limit, with_payload):
        if query == "missing":
            raise Exception("point not found")
        result = MagicMock()
        result.points = [_hit("c1", 1.0), _hit("c2", 0.8)]
        return result

    qdrant._client.query_points.side_effect = query_points

    written = retriever.link_semantic(qdrant, threshold=0.75, top_k=5)

    assert written == 1


def test_link_semantic_resolves_qdrant_hyphenated_id_to_literal_neo4j_value() -> None:
    """Real bug found live: Qdrant always returns point IDs in canonical
    hyphenated UUID form, but Neo4j's chunk_id can be the same UUID without
    hyphens (core/models.py:_chunk_id's sha256-derived id, used by
    structure_aware chunking) — an exact-string MERGE on the literal
    Qdrant-returned ID silently wrote nothing despite `written` optimistically
    counting len(edges). The fix resolves Qdrant's hyphenated neighbor ID
    back to the literal Neo4j chunk_id (via a normalized→literal lookup built
    from the chunk_ids already scanned) *before* writing — keeping the MERGE
    on an exact, index-backed match rather than a Cypher-side `replace()`
    (which would lose the chunk_id index and turn every batch into a full
    label scan — measured live: ~14 min projected to ~2.5h on 35k chunks)."""
    retriever, session = _retriever_with_mock_session()
    # Neo4j stores chunk_id WITHOUT hyphens (c2's literal form is "c2id");
    # Qdrant returns the SAME point's neighbor id WITH hyphens.
    session.run.side_effect = _run_side_effect(["c1id", "c2id"], write_counts=[1])

    qdrant = _fake_qdrant({
        "c1id": [_hit("c2-id", 0.9)],  # hyphenated form of "c2id"
        "c2id": [],
    })

    written = retriever.link_semantic(qdrant, threshold=0.75, top_k=5)

    assert written == 1
    unwind_calls = [c for c in session.run.call_args_list if "UNWIND $edges" in str(c)]
    assert len(unwind_calls) == 1
    edges = unwind_calls[0].kwargs["edges"]
    # edge must carry the literal Neo4j value ("c2id"), not Qdrant's raw
    # hyphenated response ("c2-id") — that's what keeps MERGE index-backed.
    assert edges == [{"a": "c1id", "b": "c2id", "score": 0.9}]
    query_text = unwind_calls[0].args[0]
    assert "{chunk_id: e.a}" in query_text
    assert "{chunk_id: e.b}" in query_text


def test_link_semantic_skips_neighbor_id_unresolvable_to_any_known_chunk() -> None:
    """A neighbor ID that doesn't normalize-match any chunk_id actually in
    this Neo4j graph (e.g. stale Qdrant point) must be skipped, not crash."""
    retriever, session = _retriever_with_mock_session()
    session.run.side_effect = _run_side_effect(["c1id"], write_counts=[])

    qdrant = _fake_qdrant({
        "c1id": [_hit("ghost-id", 0.95)],
    })

    written = retriever.link_semantic(qdrant, threshold=0.75, top_k=5)

    assert written == 0
