"""Confusion test — retrieval exact-match stability under corpus growth.

Tests that adding irrelevant documents does not displace relevant ones from top-k.
Uses only in-memory stubs — no live services needed.
"""
from __future__ import annotations

from adapters.opensearch import OpenSearchRetrieverStub
from core.models import Chunk


def _make_stub_with_docs(core_docs: list[str], noise_docs: list[str]) -> OpenSearchRetrieverStub:
    stub = OpenSearchRetrieverStub()
    for i, text in enumerate(core_docs):
        stub.index(Chunk(chunk_id=f"core_{i}", doc_id=f"d_{i}", text=text))
    for i, text in enumerate(noise_docs):
        stub.index(Chunk(chunk_id=f"noise_{i}", doc_id=f"dn_{i}", text=text))
    return stub


CORE_DOCS = [
    "article 5 of the handbook governs a lease agreement",
    "article 48 of the handbook sets out the registration procedure",
    "article 656 of the handbook covers liability under a works contract",
]

NOISE_50 = [f"irrelevant document number {i} about other matters" for i in range(50)]
NOISE_150 = [f"irrelevant document number {i} about other matters" for i in range(150)]

QUERY = "handbook article agreement"


def _exact_match_in_top_k(stub: OpenSearchRetrieverStub, k: int = 5) -> int:
    results = stub.retrieve(QUERY, k=k)
    ids = {sc.chunk.chunk_id for sc in results}
    return sum(1 for i in range(len(CORE_DOCS)) if f"core_{i}" in ids)


def test_exact_match_baseline():
    stub = _make_stub_with_docs(CORE_DOCS, [])
    assert _exact_match_in_top_k(stub) == len(CORE_DOCS)


def test_exact_match_stable_50pct_noise():
    stub = _make_stub_with_docs(CORE_DOCS, NOISE_50)
    # All core docs should still be in top-5 despite 50 noise docs
    assert _exact_match_in_top_k(stub, k=5) == len(CORE_DOCS)


def test_exact_match_stable_full_corpus():
    stub = _make_stub_with_docs(CORE_DOCS, NOISE_150)
    # Core docs must survive in top-10 even with 150 noise docs
    assert _exact_match_in_top_k(stub, k=10) >= len(CORE_DOCS)
