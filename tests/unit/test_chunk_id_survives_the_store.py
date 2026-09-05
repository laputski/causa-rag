"""A chunk has one identity, and both indexes have to agree on it.

Qdrant requires a point identifier to be a UUID or an integer. A chunk id here
is thirty-two hexadecimal characters, which Qdrant accepts and normalises into
dashed UUID form, and hands back in that form on read. The sparse index stores
the same id verbatim and returns it verbatim.

The hybrid merge is keyed on the chunk id, so the two halves never recognised
one chunk as one chunk. Measured on the proving ground before the fix: twenty
dense candidates and nineteen sparse ones shared none at all, and a context of
five carried four distinct texts, one of them twice under two spellings. Rank
fusion could therefore never add the two halves' evidence, and the fusion
constant could not change the order of a merge of two disjoint lists: it moved
nothing on any of fifteen questions. After the fix the same measurement gives
ten shared candidates and an order that moves on ten questions of the fifteen.

The user-visible context was clean throughout, because the pipeline drops
repeated text before answering. That is what kept this quiet.
"""
from __future__ import annotations

from typing import Any

from core.models import Chunk


class _Hit:
    def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
        self.id = point_id
        self.payload = payload
        self.score = 0.5
        self.vector = None


class _Response:
    def __init__(self, points: list[_Hit]) -> None:
        self.points = points


class _Client:
    """Records what was written and replays it the way Qdrant would."""

    def __init__(self) -> None:
        self.written: list[Any] = []

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.written.extend(points)

    def query_points(self, **kwargs: Any) -> _Response:
        return _Response([_Hit(_as_uuid(p.id), p.payload) for p in self.written])

    def collection_exists(self, collection_name: str) -> bool:
        return True


def _as_uuid(value: str) -> str:
    """What Qdrant does to a thirty-two character hexadecimal id."""
    raw = value.replace("-", "")
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def _retriever(client: _Client) -> Any:
    from adapters.qdrant import QdrantRetriever

    retriever = QdrantRetriever.__new__(QdrantRetriever)
    retriever._client = client
    retriever._collection = "c"
    retriever._embedder_id = "bge_m3"
    retriever._strategy_id = "structure_aware"
    return retriever


def _chunk() -> Chunk:
    return Chunk(chunk_id="e51cd72f2dd9503b5afcb86a774ed129", doc_id="base-ru/02",
                 text="Verification is carried out every twelve months.",
                 structural_path="document/section[2]", strategy_id="structure_aware")


def test_the_written_point_carries_the_chunk_id_in_its_payload() -> None:
    """The point identifier alone cannot answer this, because the store rewrites
    it. The sparse index already carries the id in its document body."""
    client = _Client()
    _retriever(client).upsert([_chunk()], [[0.1] * 8])
    assert client.written[0].payload["chunk_id"] == "e51cd72f2dd9503b5afcb86a774ed129"


def test_what_is_read_back_is_the_identity_that_was_written() -> None:
    client = _Client()
    retriever = _retriever(client)
    retriever.upsert([_chunk()], [[0.1] * 8])
    got = retriever.retrieve(query="q", k=5, query_vector=[0.1] * 8)
    assert got[0].chunk.chunk_id == "e51cd72f2dd9503b5afcb86a774ed129"


def test_the_store_really_does_rewrite_the_identifier() -> None:
    """The premise of the two tests above, asserted and never assumed. Were
    the store to stop rewriting, they would pass for a different reason and
    the payload field would look like decoration."""
    client = _Client()
    retriever = _retriever(client)
    retriever.upsert([_chunk()], [[0.1] * 8])
    assert client.query_points().points[0].id == "e51cd72f-2dd9-503b-5afc-b86a774ed129"
    assert client.query_points().points[0].id != _chunk().chunk_id


def test_a_point_written_before_the_payload_carried_the_id_still_reads() -> None:
    """A collection loaded by an older build has no such field. Falling back to
    the point identifier keeps it working, in the spelling it has always had,
    until it is loaded again."""
    client = _Client()
    retriever = _retriever(client)
    retriever.upsert([_chunk()], [[0.1] * 8])
    del client.written[0].payload["chunk_id"]
    got = retriever.retrieve(query="q", k=5, query_vector=[0.1] * 8)
    assert got[0].chunk.chunk_id == "e51cd72f-2dd9-503b-5afc-b86a774ed129"


def test_two_halves_returning_one_chunk_merge_it_once() -> None:
    """The consequence, stated where it bites.

    Rank fusion adds a document's contribution from each list it appears in.
    With the two halves disagreeing about the spelling of an identity, no
    document ever appeared in both, so every score had exactly one term and the
    merge was an interleaving of two disjoint lists.
    """
    from core.models import ScoredChunk
    from core.retrieval.hybrid import HybridRetriever

    chunk = _chunk()

    class _Half:
        def __init__(self, chunk_id: str) -> None:
            self._out = [ScoredChunk(
                chunk=Chunk(**{**chunk.__dict__, "chunk_id": chunk_id}),
                score=1.0, retriever_id="r")]

        def retrieve(self, **kwargs: Any) -> list:
            return self._out

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1]] * len(texts)

    agreeing = HybridRetriever(dense_retriever=_Half(chunk.chunk_id),
                               sparse_retriever=_Half(chunk.chunk_id),
                               embedder=_Half(chunk.chunk_id), merge="rrf")
    assert len(agreeing.retrieve("q", k=5)) == 1

    disagreeing = HybridRetriever(dense_retriever=_Half(_as_uuid(chunk.chunk_id)),
                                  sparse_retriever=_Half(chunk.chunk_id),
                                  embedder=_Half(chunk.chunk_id), merge="rrf")
    assert len(disagreeing.retrieve("q", k=5)) == 2, (
        "one chunk under two spellings no longer occupies two places, so this "
        "test no longer describes the failure it was written for"
    )
