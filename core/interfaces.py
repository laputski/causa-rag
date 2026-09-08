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
class BoundToACorpus(Protocol):
    """A retriever that can make a copy of itself reading another corpus.

    A run names the corpus it queries and the Realm whose instance holds it,
    and the retriever the platform built at start-up reads neither. Which
    retrievers could be rebound was written in the experiment builder, as a
    case analysis over the names of adapter classes, so a retriever the
    analysis did not name came back unchanged and the field asking for the
    change did nothing. A retriever added tomorrow would arrive in that state
    by default, which is what a case analysis over known names always
    promises.

    `resources` maps a store's name to the `{"host", "port"}` a Realm keeps
    its own instance at, the shape the services layer already resolves. Each
    retriever reads the key it recognises and passes the whole map to
    whatever it is made of, so a store nobody has written yet needs no change
    here.

    Returning `self` when nothing would differ is part of the contract: a run
    on the corpus the retriever already reads must not pay for a second
    connection.
    """

    def for_corpus(
        self,
        corpus_id: str,
        realm_id: str | None = None,
        resources: dict[str, dict[str, Any] | None] | None = None,
    ) -> Any:
        ...


@runtime_checkable
class Fusing(Protocol):
    """A retriever that fuses lists, and can make a copy fusing differently.

    Which strategy and which weight were two registry entries built at
    start-up with their fusion baked in, so a configuration could name any
    strategy and any weight and the run used what the entry was constructed
    with.
    """

    def with_fusion(
        self, merge: str | None = None, alpha: float | None = None, rrf_k: int | None = None,
    ) -> Any:
        ...


@runtime_checkable
class WalkingAGraph(Protocol):
    """A retriever that walks a graph, and can make a copy walking it
    differently. Its two parameters are the only two the graph point has."""

    def with_graph(self, graph_weight: float | None = None, hops: int | None = None) -> Any:
        ...


@runtime_checkable
class ChoosingItsModel(Protocol):
    """A component that runs a named model and can make a copy running
    another. What decides which languages a rerank step understands, and
    which model wrote an answer."""

    def with_model(self, model: str) -> Any:
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
