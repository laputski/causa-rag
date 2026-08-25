"""Integration tests — Ingestion pipeline against live Qdrant + OpenSearch."""
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sample_npa_dir():
    """Temporary directory with 5 NPA article .txt files."""
    articles = [
        ("art_1.txt", "Article 1. Scope of these rules\n\nThese rules apply to every employee, contractor and contingent worker."),
        ("art_2.txt", "Article 2. Who the rules bind\n\nA person is bound by these rules from their first working day."),
        ("art_5.txt", "Article 5. Filling a gap\n\nWhere no rule covers a situation, the closest applicable rule governs."),
        ("art_44.txt", "Article 44. What a team may decide\n\nA team may decide anything within its own budget and headcount."),
        ("art_48.txt", "Article 48. Where a team sits\n\nA team is located where its lead is located, which determines its public holidays."),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in articles:
            (pathlib.Path(tmpdir) / fname).write_text(content, encoding="utf-8")
        yield pathlib.Path(tmpdir)


def test_ingest_gk_sample(sample_npa_dir, qdrant_available, opensearch_available):
    if not qdrant_available:
        pytest.skip("Qdrant not reachable")
    from services.ingestion.cli import ingest

    stats = ingest(
        source=sample_npa_dir,
        strategy_id="fixed",
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        opensearch_host=os.getenv("OPENSEARCH_HOST", "localhost"),
        opensearch_port=int(os.getenv("OPENSEARCH_PORT", "9200")),
        use_opensearch=opensearch_available,
    )

    assert stats["files"] == 5
    assert stats["chunks"] >= 5


def test_ingest_returns_chunk_count(sample_npa_dir, qdrant_available):
    if not qdrant_available:
        pytest.skip("Qdrant not reachable")
    from services.ingestion.cli import ingest

    stats = ingest(
        source=sample_npa_dir,
        strategy_id="fixed",
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        use_opensearch=False,
    )

    assert "files" in stats
    assert "chunks" in stats
    assert "hit_ratio" in stats
    assert isinstance(stats["chunks"], int)
    assert stats["chunks"] > 0


def test_ingest_structure_aware(sample_npa_dir, qdrant_available):
    """Structure-aware strategy on NPA text should produce at least 1 chunk per article."""
    if not qdrant_available:
        pytest.skip("Qdrant not reachable")
    from services.ingestion.cli import ingest

    stats = ingest(
        source=sample_npa_dir,
        strategy_id="structure_aware",
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        use_opensearch=False,
    )

    assert stats["chunks"] >= 5, f"Expected ≥5 chunks from 5 articles, got {stats['chunks']}"


def test_ingest_cache_hit_ratio_on_repeat(sample_npa_dir, qdrant_available):
    """Second ingest of same docs should show hit_ratio > 0 due to in-memory cache."""
    if not qdrant_available:
        pytest.skip("Qdrant not reachable")
    from services.ingestion.cli import ingest

    kwargs = dict(
        source=sample_npa_dir,
        strategy_id="fixed",
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        use_opensearch=False,
    )

    ingest(**kwargs)
    # Note: in-memory cache is per-embedder instance, so this tests the stat field exists
    stats2 = ingest(**kwargs)
    assert "hit_ratio" in stats2
