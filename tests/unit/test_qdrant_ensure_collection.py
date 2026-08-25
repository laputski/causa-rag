"""adapters/qdrant.py#QdrantRetriever._ensure_collection.

Found live (verified against a real Qdrant while checking the
Content/Health/Graph fix): a corpus_id that resolves to a Qdrant
*alias* (tools/migrate_corpus_aliases.py points a realm-scoped canonical
name at a pre-existing physical collection without a reindex) made every
browse/health/graph call 503 — get_collections() lists only real physical
collections, never aliases, so the old check always concluded the alias
"didn't exist" and tried to create_collection() a name Qdrant already owns
as an alias, which Qdrant rejects with a 400.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.qdrant import QdrantRetriever


def _client(collection_exists: bool) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = collection_exists
    return client


def test_does_not_attempt_to_create_when_collection_exists() -> None:
    client = _client(collection_exists=True)
    with patch("qdrant_client.QdrantClient", return_value=client):
        QdrantRetriever(strategy_id="structure_aware", embedder_id="bge_m3", corpus_id="handbook")
    client.create_collection.assert_not_called()


def test_does_not_attempt_to_create_when_name_is_an_alias():
    """The actual bug: an alias-backed name must be treated identically to
    an ordinary existing collection — collection_exists() (not a raw
    get_collections() name scan) is what makes that true, since Qdrant
    resolves aliases through collection_exists() the same way it does for
    a real query."""
    client = _client(collection_exists=True)  # aliases report True here too
    with patch("qdrant_client.QdrantClient", return_value=client):
        QdrantRetriever(
            strategy_id="structure_aware", embedder_id="bge_m3",
            corpus_id="handbook", realm_id="demo",
        )
    client.create_collection.assert_not_called()


def test_creates_collection_when_it_genuinely_does_not_exist() -> None:
    client = _client(collection_exists=False)
    with patch("qdrant_client.QdrantClient", return_value=client):
        QdrantRetriever(strategy_id="structure_aware", embedder_id="bge_m3", corpus_id="brand_new_corpus")
    client.create_collection.assert_called_once()
    assert client.create_collection.call_args.kwargs["collection_name"] == "brand_new_corpus__structure_aware__bge_m3"
