"""Integration test fixtures.

All fixtures skip automatically if the required service is not reachable.
Run with: pytest tests/integration/ -v -m integration
Services: docker compose -f deploy/compose/docker-compose.yml up -d
"""
from __future__ import annotations

import os
import uuid

import pytest

_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
_OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ragplatform")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: mark test as requiring live services")


# ── Qdrant ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=3)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def qdrant_client(qdrant_available: bool):  # type: ignore[return]
    if not qdrant_available:
        pytest.skip("Qdrant not reachable — start with docker compose up -d")
    from qdrant_client import QdrantClient
    return QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)


@pytest.fixture
def qdrant_collection(qdrant_client):
    """Temporary Qdrant collection, cleaned up after test."""
    from qdrant_client.models import Distance, VectorParams

    name = f"test_{uuid.uuid4().hex[:8]}"
    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    yield name, qdrant_client
    # Everything with this prefix is removed, rather than one exact name.
    #
    # The tests pass `name` as `strategy_id`, and adapters/qdrant.py's
    # `_collection_name` builds a SECOND collection from it, `{name}__{embedder}`.
    # The old cleanup removed only the one it had created itself, so derived
    # collections accumulated: a working install had 200 of them, every one
    # suffixed `__bge_m3` and not one without. Cleanup is now a property of the
    # name rather than a match between the name and what gets deleted.
    try:
        for c in qdrant_client.get_collections().collections:
            if c.name == name or c.name.startswith(f"{name}__"):
                qdrant_client.delete_collection(c.name)
    except Exception:
        pass


# ── OpenSearch ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def opensearch_available() -> bool:
    try:
        from opensearchpy import OpenSearch
        client = OpenSearch(
            hosts=[{"host": _OPENSEARCH_HOST, "port": _OPENSEARCH_PORT}],
            http_compress=True,
            timeout=3,
        )
        client.cluster.health()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def opensearch_client(opensearch_available: bool):  # type: ignore[return]
    if not opensearch_available:
        pytest.skip("OpenSearch not reachable — start with docker compose up -d")
    from opensearchpy import OpenSearch
    return OpenSearch(
        hosts=[{"host": _OPENSEARCH_HOST, "port": _OPENSEARCH_PORT}],
        http_compress=True,
    )


@pytest.fixture
def opensearch_index(opensearch_client):
    """Temporary OpenSearch index with Russian analyzer, cleaned up after test."""
    index = f"test_{uuid.uuid4().hex[:8]}"
    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "ru_be_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    }
                }
            },
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "ru_be_analyzer"},
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
            }
        },
    }
    opensearch_client.indices.create(index=index, body=body)
    opensearch_client.indices.refresh(index=index)
    yield index, opensearch_client
    # The same as Qdrant above: adapters/opensearch.py's `_index_name` builds
    # `rag__{name}` from the name it is given, so deleting one exact name left
    # the second index behind forever. A pattern rather than a name.
    try:
        opensearch_client.indices.delete(index=f"*{index}*", ignore_unavailable=True)
    except Exception:
        pass


# ── Neo4j ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def neo4j_available() -> bool:
    try:
        from adapters.neo4j_graph import Neo4jGraphRetriever
        retriever = Neo4jGraphRetriever(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
        return retriever.is_available() and retriever.verify()
    except Exception:
        return False


@pytest.fixture
def neo4j_graph(neo4j_available: bool):  # type: ignore[return]
    """Real Neo4jGraphRetriever, with its test-created nodes cleaned up after
    (deleted by chunk_id prefix, not a full graph wipe — other tests/real
    corpora may share this Neo4j instance)."""
    if not neo4j_available:
        pytest.skip("Neo4j not reachable — start with docker compose --profile graph up -d")
    from adapters.neo4j_graph import Neo4jGraphRetriever

    retriever = Neo4jGraphRetriever(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    retriever.ensure_indexes()
    prefix = f"itest_{uuid.uuid4().hex[:8]}_"
    yield retriever, prefix
    driver = retriever._connect()
    with driver.session() as session:
        session.run(
            "MATCH (n:Chunk) WHERE n.doc_id STARTS WITH $prefix DETACH DELETE n",
            prefix=prefix,
        ).consume()


# ── Sample numbered-unit chunks ───────────────────────────────────────────────────────

@pytest.fixture
def sample_article_chunks():
    """Ten sample chunks shaped like a numbered-unit corpus, for integration tests."""
    from core.models import Chunk

    # Synthetic articles in the shape a numbered-unit corpus has: an id, a
    # heading and a body, each distinct enough for a vector search to tell
    # apart. The content is invented. It used to be ten articles of real
    # legislation, which is somebody else's text and was never needed: nothing
    # here asserts on what an article says, only that the right one comes back.
    # The numbers are unchanged, because a consumer looks up "44" by path.
    articles = [
        ("1", "Scope of these rules", "These rules apply to every employee, contractor and contingent worker of Northwind Robotics."),
        ("2", "Who the rules bind", "A person is bound by these rules from their first working day, whether or not they have acknowledged them in writing."),
        ("3", "Where the rules come from", "These rules consist of this handbook and any policy the board has adopted under it."),
        ("5", "Filling a gap", "Where no rule covers a situation, the closest applicable rule governs, and the department head decides."),
        ("7", "Raising a dispute", "A dispute about the application of these rules is settled by the people operations team."),
        ("10", "Established practice", "A practice followed openly and consistently for a year carries the weight of a rule."),
        ("44", "What a team may decide", "A team may decide anything within its own budget and headcount without a further approval."),
        ("46", "How a team is named", "A team's name states the function it performs and the division it reports into."),
        ("48", "Where a team sits", "A team is located where its lead is located, which determines its public holidays."),
        ("154", "What counts as an agreement", "An agreement exists once both sides have stated the same terms in writing, whatever form the writing takes."),
    ]

    chunks = []
    for num, title, text in articles:
        chunks.append(Chunk(
            text=f"Article {num}. {title}\n\n{text}",
            doc_id="SRC001",
            structural_path=f"handbook/{num}",
            metadata={
                "act_kind": "handbook",
                "act_number": "HB-001",
                "status": "active",
                "article_num": num,
            },
        ))
    return chunks
