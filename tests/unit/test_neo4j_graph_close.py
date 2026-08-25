"""Neo4jGraphRetriever.close() — connection lifecycle, added alongside the
Neo4j dump-ingestion "replace" option (services/api_gateway/routers/
corpus.py#ingest_neo4j_cypher), which opens a short-lived retriever instance
just to clear the graph and must release it afterward.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from adapters.neo4j_graph import Neo4jGraphRetriever


def test_close_is_a_noop_when_never_connected():
    retriever = Neo4jGraphRetriever()
    retriever.close()  # must not raise
    assert retriever._driver is None


def test_close_releases_the_driver_and_is_idempotent():
    retriever = Neo4jGraphRetriever()
    mock_driver = MagicMock()
    retriever._driver = mock_driver

    retriever.close()

    mock_driver.close.assert_called_once()
    assert retriever._driver is None

    retriever.close()  # calling again must not re-close or raise
    mock_driver.close.assert_called_once()
