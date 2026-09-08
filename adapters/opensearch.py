"""OpenSearch sparse (BM25) retriever adapter.

Each (strategy_id) maps to its own OpenSearch index.

**The analyser is chosen per corpus.** It used to be fixed: every index, in
every language, was created with the Russian/Belarusian one, so an Arabic or
English corpus was stemmed by Russian rules and filtered through Russian stop
words. BM25 still returned something, which is what made it hard to notice.
The numbers were not low, they were meaningless, and any hybrid merge
inherited that.

The language is only needed when the index is created: the analyser lives in
the field mapping, so queries against an existing index inherit it. That leaves
one hazard, guarded below: an index created before ingestion, by a query that
arrived first, freezes the wrong analyser and nothing says so.
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

#: What each language name resolves to. `ru_be` is the custom analyser above and
#: stays the default, so an existing corpus keeps the behaviour it was indexed
#: with. The rest are OpenSearch's own language analysers: each normalises and
#: stems by that language's rules, which is the whole point of choosing one.
#:
#: `standard` is the honest answer for a language not listed: it tokenises and
#: lowercases and claims nothing more. A wrong stemmer is worse than none.
# Every language analyser Lucene ships with OpenSearch, addressable by its
# own name, plus ISO 639-1 codes for the ones this project uses by hand.
# The set was nine entries and rejected italian, portuguese, hindi and the
# rest with a ValueError, while the README promises "real language
# analysers, which matters once documents stop being English". A limit
# nobody chose is worse than one that is argued for.
_BUILT_IN = (
    "arabic", "armenian", "basque", "bengali", "brazilian", "bulgarian",
    "catalan", "cjk", "czech", "danish", "dutch", "english", "estonian",
    "finnish", "french", "galician", "german", "greek", "hindi", "hungarian",
    "indonesian", "irish", "italian", "latvian", "lithuanian", "norwegian",
    "persian", "portuguese", "romanian", "russian", "sorani", "spanish",
    "swedish", "thai", "turkish",
)

# ISO 639-1 for the built-ins, so a corpus can be named the way its
# documents are tagged, not the way Lucene spells the language.
_ISO_639_1 = {
    "ar": "arabic", "bg": "bulgarian", "bn": "bengali", "ca": "catalan",
    "cs": "czech", "da": "danish", "de": "german", "el": "greek",
    "en": "english", "es": "spanish", "et": "estonian", "eu": "basque",
    "fa": "persian", "fi": "finnish", "fr": "french", "ga": "irish",
    "gl": "galician", "hi": "hindi", "hu": "hungarian", "hy": "armenian",
    "id": "indonesian", "it": "italian", "lt": "lithuanian", "lv": "latvian",
    "nl": "dutch", "no": "norwegian", "pt": "portuguese", "ro": "romanian",
    "ru": "russian", "sv": "swedish", "th": "thai", "tr": "turkish",
}

LANGUAGE_ANALYZERS: dict[str, str] = {
    # The one analyser this project defines itself: Russian stemming with
    # Belarusian stop words, for corpora where that morphology matters.
    "ru_be": "ru_be_analyzer",
    # Tokenises and lowercases without claiming to stem, which is the
    # honest choice for a language nothing here has an analyser for.
    "standard": "standard",
    **{name: name for name in _BUILT_IN},
    **_ISO_639_1,
}

DEFAULT_LANGUAGE = "ru_be"


class AnalyzerMismatch(RuntimeError):
    """The index already exists and carries a different language's analyser.

    A type of its own, not a bare RuntimeError, because the call site in
    services/ingestion/cli.py wraps construction in a try/except that logs
    "opensearch_unavailable" and carries on without the sparse index. A loud
    refusal would arrive there as a quiet one, which is the failure this whole
    guard exists to prevent. Named, it can be let through.
    """


def index_settings(language: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """Index settings with the analyser this corpus's language calls for."""
    if language not in LANGUAGE_ANALYZERS:
        raise ValueError(
            f"unknown analyser language {language!r}; "
            f"known: {', '.join(sorted(LANGUAGE_ANALYZERS))}"
        )
    analyzer = LANGUAGE_ANALYZERS[language]
    settings: dict[str, Any] = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
    # The custom one has to be declared; a built-in is referred to by name.
    if analyzer == "ru_be_analyzer":
        settings["analysis"] = RU_BE_ANALYZER
    return {
        "settings": settings,
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": analyzer,
                    "search_analyzer": analyzer,
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


#: Kept so existing callers that imported the constant keep working; it is the
#: settings for the default language and nothing more.
_INDEX_SETTINGS = index_settings()


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
        language: str = DEFAULT_LANGUAGE,
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
        self._language = language
        self._index = _index_name(strategy_id, corpus_id, realm_id)
        self._client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=False,
        )
        self._ensure_index()

    def for_corpus(
        self,
        corpus_id: str,
        realm_id: str | None = None,
        resources: dict[str, dict[str, Any] | None] | None = None,
    ) -> Any:
        """A copy of this reading `corpus_id` in `realm_id`, on that Realm's
        own instance where it keeps one. See `core.interfaces.BoundToACorpus`.

        The language travels with the copy. Dropped, it reverts to the default
        analyser, so an Arabic corpus ingested as Arabic would be queried
        through an index this copy insists is Russian: either the wrong
        stemmer or a refusal, and both arrive long after the choice was made.
        """
        instance = (resources or {}).get("opensearch") or {}
        if corpus_id == self._corpus_id and realm_id == self._realm_id and not instance:
            return self
        return OpenSearchRetriever(
            host=instance.get("host", self._host),
            port=int(instance.get("port", self._port)),
            strategy_id=self._strategy_id,
            corpus_id=corpus_id,
            realm_id=realm_id,
            language=self._language,
        )

    def _ensure_index(self) -> None:
        """Create the index with this corpus's analyser, or refuse a mismatch.

        The refusal is the point. An index carries its analyser for life: a
        query that arrives before ingestion creates the index with whatever the
        caller happened to ask for, and ingestion then finds it existing and
        leaves it alone. Arabic text lands in an index that stems by Russian
        rules, BM25 keeps returning results, and nothing anywhere says so.

        That failure is exactly the kind this platform exists to catch, so it
        is caught here, and never reported as a low score later.
        """
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(
                index=self._index, body=index_settings(self._language)
            )
            return

        wanted = LANGUAGE_ANALYZERS[self._language]
        found = self._existing_analyzer()
        if found is not None and found != wanted:
            raise AnalyzerMismatch(
                f"index {self._index!r} was built with the {found!r} analyser "
                f"and this corpus asks for {wanted!r} ({self._language}). "
                "An index keeps its analyser for life, so the text would be "
                "stemmed by the wrong language's rules and BM25 would return "
                "results that mean nothing. Delete the index and ingest again."
            )

    def _existing_analyzer(self) -> str | None:
        """The analyser the index actually carries, or None if it cannot be read.

        None, never a guess: an unreadable mapping is not evidence of a
        mismatch, and refusing to start over one would be worse than the
        problem. An index with no analyser named uses OpenSearch's default.
        """
        try:
            mapping = self._client.indices.get_mapping(index=self._index)
        except Exception:
            return None
        properties = (
            mapping.get(self._index, {}).get("mappings", {}).get("properties", {})
        )
        text = properties.get("text")
        if not isinstance(text, dict):
            return None
        return text.get("analyzer", "standard")

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
    """In-memory BM25 stub: no OpenSearch server needed for unit tests."""

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
