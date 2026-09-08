"""Qdrant dense retriever adapter.

Each (strategy_id, embedder_id) pair maps to its own Qdrant collection.
This implements the namespace-per-strategy principle.
"""
from __future__ import annotations

from typing import Any

from core.models import Chunk, ScoredChunk


def _collection_name(
    strategy_id: str, embedder_id: str, corpus_id: str = "default", realm_id: str | None = None,
) -> str:
    """Namespace-per-strategy, extended with corpus_id
    and realm_id (two Realms sharing one physical Qdrant instance
    used to collide on identical collection names; see the design notes).

    The "default" corpus keeps the original collection name for backward
    compatibility — only a non-default corpus_id adds a prefix.
    `realm_id` is omitted entirely (not even for the "default" corpus) when
    the caller doesn't pass one, so older callers (tests, the stub
    fitness harness) are byte-for-byte unaffected.
    """
    base = f"{strategy_id}__{embedder_id}"
    name = base if corpus_id == "default" else f"{corpus_id}__{base}"
    return name if not realm_id else f"{realm_id}__{name}"


def _as_filter(filters: dict[str, Any] | None) -> Any:
    """A payload filter for the store, or nothing when nothing was asked for.

    This parameter was accepted and never used. The signature took it, the
    protocol declared it, the lexical half applied its own, and this half
    passed a query with no filter at all: a question asked with a filter that
    matches nothing came back with a full page of results from here and an
    empty one from there, so a filtered query returned unfiltered fragments
    and nothing said so. Found while trying to stage the catalogue's entry for
    a filter that excludes everything, which could not be staged because on
    this half a filter excluded nothing.

    Keys name payload fields, which is what the lexical half's own filter
    does, so one dictionary means the same thing on both sides.
    """
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key=field, match=MatchValue(value=value))
                        for field, value in filters.items()])


def _matches(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    """The same decision the store makes, for the stub that has no store.

    A stub that ignores a filter the real adapter applies is a unit test
    agreeing with itself, so this is here for the same reason the stub is.
    """
    if not filters:
        return True
    for field, value in filters.items():
        found = getattr(chunk, field, None)
        if found is None:
            found = (chunk.metadata or {}).get(field)
        if found != value:
            return False
    return True


class QdrantRetriever:
    """Dense retriever backed by Qdrant.

    Requires a running Qdrant instance.  In unit tests use QdrantRetrieverStub.
    """

    retriever_id = "qdrant_dense"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        strategy_id: str = "fixed",
        embedder_id: str = "bge_m3",
        vector_size: int = 1024,
        corpus_id: str = "default",
        realm_id: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # Stored so core/experiment/runner.py can rebuild a corpus_id-bound
        # copy of a registry-resolved (corpus_id="default") instance without
        # re-deriving connection params from env vars a second time.
        self._host = host
        self._port = port
        self._strategy_id = strategy_id
        self._embedder_id = embedder_id
        self._corpus_id = corpus_id
        self._realm_id = realm_id
        self._collection = _collection_name(strategy_id, embedder_id, corpus_id, realm_id)
        self._client = QdrantClient(host=host, port=port, timeout=60, check_compatibility=False)
        self._vector_size = vector_size
        self._ensure_collection(Distance, VectorParams)

    def for_corpus(
        self,
        corpus_id: str,
        realm_id: str | None = None,
        resources: dict[str, dict[str, Any] | None] | None = None,
    ) -> Any:
        """A copy of this reading `corpus_id` in `realm_id`, on that Realm's
        own instance where it keeps one.

        See `core.interfaces.BoundToACorpus`. The collection name carries all
        three, so a run that named another corpus and got this object was
        searching the one the gateway happened to start against.
        """
        instance = (resources or {}).get("qdrant") or {}
        if (corpus_id == self._corpus_id and realm_id == self._realm_id and not instance):
            return self
        return QdrantRetriever(
            host=instance.get("host", self._host),
            port=int(instance.get("port", self._port)),
            strategy_id=self._strategy_id,
            embedder_id=self._embedder_id,
            vector_size=self._vector_size,
            corpus_id=corpus_id,
            realm_id=realm_id,
        )

    def _ensure_collection(self, Distance: Any, VectorParams: Any) -> None:
        from qdrant_client.models import Distance as D
        from qdrant_client.models import VectorParams as VP

        # collection_exists() (not a manual get_collections() scan)
        # because it resolves Qdrant aliases the same way a query would.
        # self._collection is frequently an alias now (tools/migrate_corpus_
        # aliases.py points a realm-scoped canonical name at a pre-existing
        # physical collection without a reindex) — a manual name-list scan
        # doesn't see aliases at all, so it always looked "missing" and this
        # tried to create_collection() a name Qdrant already owns as an
        # alias, which Qdrant rejects outright (found live: every browse/
        # health/graph call against an aliased corpus_id 503'd).
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VP(size=self._vector_size, distance=D.COSINE),
            )

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(chunk.chunk_id),
                vector=vec,
                payload={
                    # Written into the payload as well as used as the point id.
                    # Qdrant requires a point id to be a UUID or an integer, so
                    # it normalises a thirty-two character hexadecimal chunk id
                    # into dashed UUID form and hands that form back on read.
                    # The sparse index returns the same id undashed, so the two
                    # halves of a hybrid merge never recognised one chunk as one
                    # chunk: measured on this proving ground, twenty dense and
                    # nineteen sparse candidates shared none, and a context of
                    # five carried four distinct texts. The sparse adapter
                    # already carries the id in its own document body, and this
                    # is the same measure on this side.
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "structural_path": chunk.structural_path,
                    "strategy_id": chunk.strategy_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + batch_size],
            )

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        query_vector: list[float] | None = None,
    ) -> list[ScoredChunk]:
        if query_vector is None:
            raise ValueError("query_vector required for QdrantRetriever")

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            query_filter=_as_filter(filters),
        ).points
        scored: list[ScoredChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunk = Chunk(
                # From the payload, falling back to the point id for anything
                # written before the payload carried it. The fallback is the
                # dashed form, so a collection loaded by an older build keeps
                # today's behaviour until it is loaded again.
                chunk_id=str(payload.get("chunk_id") or hit.id),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                structural_path=payload.get("structural_path", ""),
                strategy_id=payload.get("strategy_id", ""),
                start_char=payload.get("start_char", 0),
                end_char=payload.get("end_char", 0),
                metadata=payload.get("metadata", {}),
            )
            scored.append(ScoredChunk(chunk=chunk, score=hit.score, retriever_id=self.retriever_id))
        return scored

    def scroll(
        self, offset: str | None = None, limit: int = 50, text_filter: str = ""
    ) -> tuple[list[Chunk], str | None]:
        """Paginate raw chunks from the collection — corpus browser.

        Returns (chunks, next_offset). next_offset is None when exhausted.
        text_filter is applied client-side (case-insensitive substring) since
        Qdrant payload text search needs a full-text index we don't assume here.
        """
        chunks: list[Chunk] = []
        next_offset = offset
        # Over-fetch a bit when filtering so a page of `limit` is still likely.
        batch = limit * 4 if text_filter else limit
        points, next_offset = self._client.scroll(
            collection_name=self._collection, offset=next_offset, limit=batch
        )
        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "")
            if text_filter and text_filter.lower() not in text.lower():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str((p.payload or {}).get("chunk_id") or p.id),
                    doc_id=payload.get("doc_id", ""),
                    text=text,
                    structural_path=payload.get("structural_path", ""),
                    strategy_id=payload.get("strategy_id", ""),
                    start_char=payload.get("start_char", 0),
                    end_char=payload.get("end_char", 0),
                    metadata=payload.get("metadata", {}),
                )
            )
            if len(chunks) >= limit:
                break
        return chunks, next_offset

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Point lookup by ``chunk_id`` — for a caller that already knows
        exactly which chunks it wants (snapshotting a
        ``retrieval_pins`` pin's chunk text/structural_path once, at pin
        creation time), not a vector/text search. A ``chunk_id`` that no
        longer exists (e.g. the corpus was re-indexed) is silently omitted
        from the result rather than raising — the caller decides how to
        react to fewer chunks coming back than ids requested."""
        if not chunk_ids:
            return []
        points = self._client.retrieve(collection_name=self._collection, ids=chunk_ids, with_payload=True)
        chunks: list[Chunk] = []
        for p in points:
            payload = p.payload or {}
            chunks.append(Chunk(
                chunk_id=str(payload.get("chunk_id") or p.id),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                structural_path=payload.get("structural_path", ""),
                strategy_id=payload.get("strategy_id", ""),
                start_char=payload.get("start_char", 0),
                end_char=payload.get("end_char", 0),
                metadata=payload.get("metadata", {}),
            ))
        return chunks

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count

    def scroll_all(self, page_size: int = 200) -> list[Chunk]:
        """Fetch every chunk in the collection — used by corpus health.

        Bounded by a hard cap so a misconfigured/huge collection can't hang the
        health endpoint; health is a sample-based signal, not an audit.
        """
        out: list[Chunk] = []
        offset = None
        cap = 5000
        while len(out) < cap:
            batch, offset = self.scroll(offset=offset, limit=page_size)
            if not batch:
                break
            out.extend(batch)
            if offset is None:
                break
        return out


class QdrantRetrieverStub:
    """In-memory stub for unit tests — no Qdrant server needed."""

    retriever_id = "qdrant_dense_stub"

    def __init__(self) -> None:
        self._store: list[tuple[Chunk, list[float]]] = []

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._store.extend(zip(chunks, vectors, strict=True))

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        query_vector: list[float] | None = None,
    ) -> list[ScoredChunk]:
        if not self._store or query_vector is None:
            return []

        def dot(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=True))

        scored = [
            ScoredChunk(chunk=chunk, score=dot(query_vector, vec), retriever_id=self.retriever_id)
            for chunk, vec in self._store
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return [s for s in scored if _matches(s.chunk, filters)][:k]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        wanted = set(chunk_ids)
        return [chunk for chunk, _vec in self._store if chunk.chunk_id in wanted]
