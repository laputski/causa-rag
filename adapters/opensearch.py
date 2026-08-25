"""OpenSearch sparse (BM25) retriever adapter.

Each (strategy_id) maps to its own OpenSearch index.
Includes Russian and Belarusian morphology analyzers.
"""
from __future__ import annotations

from typing import Any

from core.models import Chunk, ScoredChunk

# Analyzer settings for Russian and Belarusian morphology.
# Requires the `analysis-icu` and `opensearch-analysis-morfologik` plugins in prod.
# The settings are defined here; enabling them requires the plugins to be installed.
RU_BE_ANALYZER = {
    "analyzer": {
        "ru_be_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "ru_stop", "ru_stemmer"],
        }
    },
    "filter": {
        "ru_stop": {
            "type": "stop",
            "stopwords": "_russian_",
        },
        "ru_stemmer": {
            "type": "stemmer",
            "language": "russian",
        },
    },
}

_INDEX_SETTINGS = {
    "settings": {
        "analysis": RU_BE_ANALYZER,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "ru_be_analyzer",
                "search_analyzer": "ru_be_analyzer",
            },
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "strategy_id": {"type": "keyword"},
            "structural_path": {"type": "keyword"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
        }
    },
}


def _index_name(strategy_id: str, corpus_id: str = "default", realm_id: str | None = None) -> str:
    """Namespace-per-strategy, extended with corpus_id and
    realm_id (see adapters/qdrant.py#_collection_name, same reasoning).

    The "default" corpus keeps the original index name for backward
    compatibility. `realm_id` is omitted entirely when the caller
    doesn't pass one, so older callers are unaffected.
    """
    base = f"rag__{strategy_id}"
    name = base if corpus_id == "default" else f"rag__{corpus_id}__{strategy_id}"
    return name if not realm_id else f"rag__{realm_id}__{name[len('rag__'):]}"


class OpenSearchRetriever:
    """BM25 sparse retriever backed by OpenSearch."""

    retriever_id = "opensearch_bm25"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        strategy_id: str = "fixed",
        http_auth: tuple[str, str] | None = None,
        corpus_id: str = "default",
        realm_id: str | None = None,
    ) -> None:
        from opensearchpy import OpenSearch

        # Stored so core/experiment/runner.py can rebuild a corpus_id-bound
        # copy of a registry-resolved (corpus_id="default") instance without
        # re-deriving connection params from env vars a second time.
        self._host = host
        self._port = port
        self._strategy_id = strategy_id
        self._corpus_id = corpus_id
        self._realm_id = realm_id
        self._index = _index_name(strategy_id, corpus_id, realm_id)
        self._client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=False,
        )
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(index=self._index, body=_INDEX_SETTINGS)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        from opensearchpy.helpers import bulk

        actions = [
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "strategy_id": chunk.strategy_id,
                    "structural_path": chunk.structural_path,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    # Eval Measurement Trustworthiness, Phase 0 — without this,
                    # any chunk surfaced only via the sparse/BM25 side of a
                    # hybrid merge (core/retrieval/hybrid.py) loses
                    # source_code/article_no, making retrieval_recall_at_k/
                    # retrieval_precision_at_k silently undercount whenever
                    # sparse uniquely contributes a chunk dense didn't surface.
                    "metadata": chunk.metadata,
                },
            }
            for chunk in chunks
        ]
        bulk(self._client, actions, refresh=True)

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        must: list[dict[str, Any]] = [{"match": {"text": {"query": query}}}]
        if filters:
            for field, value in filters.items():
                must.append({"term": {field: value}})

        body = {
            "query": {"bool": {"must": must}},
            "size": k,
        }
        response = self._client.search(index=self._index, body=body)
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=hit["_id"],
                    doc_id=hit["_source"]["doc_id"],
                    text=hit["_source"]["text"],
                    structural_path=hit["_source"].get("structural_path", ""),
                    strategy_id=hit["_source"].get("strategy_id", ""),
                    start_char=hit["_source"].get("start_char", 0),
                    end_char=hit["_source"].get("end_char", 0),
                    metadata=hit["_source"].get("metadata", {}),
                ),
                score=hit["_score"],
                retriever_id=self.retriever_id,
            )
            for hit in response["hits"]["hits"]
        ]


class OpenSearchRetrieverStub:
    """In-memory BM25 stub — no OpenSearch server needed for unit tests."""

    retriever_id = "opensearch_bm25_stub"

    def __init__(self) -> None:
        self._store: list[Chunk] = []

    def index_chunks(self, chunks: list[Chunk]) -> None:
        self._store.extend(chunks)

    def index(self, chunk: Chunk) -> None:
        self._store.append(chunk)

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        query_terms = query.lower().split()
        scored: list[ScoredChunk] = []
        for chunk in self._store:
            chunk_terms = chunk.text.lower().split()
            # BM25-like: count total query-term occurrences in chunk (with repetition)
            overlap = sum(chunk_terms.count(t) for t in query_terms)
            if overlap > 0:
                scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=float(overlap),
                        retriever_id=self.retriever_id,
                    )
                )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]
