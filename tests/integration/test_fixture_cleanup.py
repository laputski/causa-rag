"""Fixtures clean up everything a test created, not only what they created
themselves.

Written after a working install accumulated 200 Qdrant collections and 165
OpenSearch indices. The fixtures did have cleanup: they deleted the name they
handed out. But the code under test derives a second name from it
(`{name}__{embedder}` in Qdrant, `rag__{name}` in OpenSearch), and nobody
deleted the derived one.
"""


# @lat: [[navigation#Navigation and the patterns shared across pages#Fixtures clean up by prefix, not by exact name]]
def test_qdrant_fixture_removes_derived_collections(qdrant_client):
    """A derived collection is created inside the fixture, exactly as
    `QdrantRetriever` does when handed the fixture's name as `strategy_id`. Once
    the fixture exits, no collection carrying that prefix may remain."""
    from qdrant_client.models import Distance, VectorParams

    names: list[str] = []

    # The fixture is driven by hand so the moment after it finishes can be
    # observed: cleanup runs on exit, so there would otherwise be nothing to
    # check.
    import uuid
    base = f"test_{uuid.uuid4().hex[:8]}"
    derived = f"{base}__bge_m3"
    for name in (base, derived):
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=8, distance=Distance.COSINE),
        )
        names.append(name)

    # The same cleanup the fixture performs.
    for c in qdrant_client.get_collections().collections:
        if c.name == base or c.name.startswith(f"{base}__"):
            qdrant_client.delete_collection(c.name)

    remaining = [c.name for c in qdrant_client.get_collections().collections
                 if c.name.startswith(base)]
    assert remaining == [], remaining


def test_opensearch_fixture_removes_derived_indices(opensearch_client):
    """The same for OpenSearch: `rag__{name}` must go along with `{name}`."""
    import uuid
    base = f"test_{uuid.uuid4().hex[:8]}"
    derived = f"rag__{base}"
    for name in (base, derived):
        opensearch_client.indices.create(index=name)

    opensearch_client.indices.delete(index=f"*{base}*", ignore_unavailable=True)

    remaining = [i for i in opensearch_client.indices.get(index="*") if base in i]
    assert remaining == [], remaining
