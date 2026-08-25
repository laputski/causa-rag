from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.models import Answer, Chunk, Document, QueryRequest, ScoredChunk


@runtime_checkable
class ChunkingStrategy(Protocol):
    strategy_id: str

    def chunk(self, doc: Document) -> list[Chunk]:
        ...


@runtime_checkable
class Embedder(Protocol):
    embedder_id: str
    version: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class Retriever(Protocol):
    retriever_id: str

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        ...


@runtime_checkable
class Reranker(Protocol):
    reranker_id: str

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        ...


@runtime_checkable
class Generator(Protocol):
    generator_id: str

    def generate(self, prompt: str, **params: Any) -> str:
        ...


@runtime_checkable
class Pipeline(Protocol):
    pipeline_id: str

    def run(self, request: QueryRequest) -> Answer:
        ...
