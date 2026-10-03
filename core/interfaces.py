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
class ReadingAnEmbeddersIndex(Protocol):
    """A retriever that reads vectors, and can make a copy reading the index
    another embedder built.

    The collection a corpus lives in is named after the embedder that filled
    it. A run could name another embedder, its queries were embedded with it,
    and the retriever went on reading the collection of the embedder the
    gateway started with: one model's vectors searched with another's. A
    retriever made of others passes the embedder on to the half that reads
    vectors and returns `self` when nothing would differ.
    """

    def for_embedder(self, embedder: Any) -> Any:
        ...


@runtime_checkable
class SayingWhereItReads(Protocol):
    """A retriever that can say which stores it reads and where they are.

    Returns the store's kind mapped to `{"address", "remote"}`, with nothing
    that opens it. A retriever made of others returns what its halves say
    together. A run records the answer, because a store that is not on this
    machine receives every query of the run.
    """

    def reads_from(self) -> dict[str, dict[str, Any]]:
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
class ServedByARealm(Protocol):
    """A component that calls a model server, and can make a copy calling the
    Realm's own.

    A pipeline is built once, when the gateway starts, around the server the
    gateway was started with. A Realm that keeps its own was answered by the
    gateway's all the same: the chat asked the Realm's server and a run did
    not. `resources` maps a resource type to what the Realm declares for it,
    with its secrets opened; a component answers for the type it speaks and
    comes back unchanged when the Realm declares none.

    `served_from` says where its answers come from, in words that carry no
    key, so a run can record it.
    """

    def for_realm(self, resources: dict[str, dict[str, Any] | None] | None) -> Any:
        ...

    def served_from(self) -> dict[str, Any]:
        ...


@runtime_checkable
class FixingItsSampling(Protocol):
    """A component that samples, and can make a copy sampling the same way
    every time.

    A configuration has carried a seed since it was first written, with a
    sentence on the run page in two languages saying that the same
    configuration and the same seed give identical answers on a repeat run.
    Measured on the model this platform runs: identical requests can give
    different answers and a seed removes it, occasionally and depending on
    the question. The numbers are in
    `tests/unit/test_a_run_repeats_itself.py`.

    The generator is the only part of an in-process run that samples at all,
    so this is the whole of what a seed can fix here. A system answering over
    HTTP samples on its own side, where a seed of ours does not reach.
    """

    def with_seed(self, seed: int) -> Any:
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
