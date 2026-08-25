"""Contract tests: verify that concrete implementations satisfy Protocol shapes."""
from typing import Any

from core.interfaces import ChunkingStrategy, Embedder, Generator, Pipeline, Retriever
from core.models import Answer, Chunk, Document, QueryRequest, ScoredChunk


class _StubChunker:
    strategy_id = "stub"

    def chunk(self, doc: Document) -> list[Chunk]:
        return [Chunk(doc_id=doc.doc_id, text=doc.content)]


class _StubEmbedder:
    embedder_id = "stub"
    version = "0.1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class _StubRetriever:
    retriever_id = "stub"

    def retrieve(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        return []


class _StubGenerator:
    generator_id = "stub"

    def generate(self, prompt: str, **params: Any) -> str:
        return "stub answer"


class _StubPipeline:
    pipeline_id = "stub"

    def run(self, request: QueryRequest) -> Answer:
        return Answer(text="stub")


def test_chunker_satisfies_protocol():
    assert isinstance(_StubChunker(), ChunkingStrategy)


def test_embedder_satisfies_protocol():
    assert isinstance(_StubEmbedder(), Embedder)


def test_retriever_satisfies_protocol():
    assert isinstance(_StubRetriever(), Retriever)


def test_generator_satisfies_protocol():
    assert isinstance(_StubGenerator(), Generator)


def test_pipeline_satisfies_protocol():
    assert isinstance(_StubPipeline(), Pipeline)


def test_chunker_returns_chunks():
    doc = Document(source="f.txt", content="hello world")
    result = _StubChunker().chunk(doc)
    assert isinstance(result, list)
    assert all(isinstance(c, Chunk) for c in result)


def test_embedder_shape():
    vecs = _StubEmbedder().embed(["a", "b"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 4


def test_generator_returns_str():
    out = _StubGenerator().generate("prompt")
    assert isinstance(out, str)


def test_pipeline_returns_answer():
    ans = _StubPipeline().run(QueryRequest(text="q"))
    assert isinstance(ans, Answer)
