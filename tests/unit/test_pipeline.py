
from adapters.bge_m3 import BgeM3Embedder
from adapters.generator_stub import GeneratorStub
from adapters.qdrant import QdrantRetrieverStub
from core.models import Chunk, QueryRequest
from core.pipeline import NaivePipeline


def _make_pipeline() -> NaivePipeline:
    return NaivePipeline(
        retriever=QdrantRetrieverStub(),
        embedder=BgeM3Embedder(),
        generator=GeneratorStub(),
    )


def test_pipeline_id():
    assert _make_pipeline().pipeline_id == "naive"


def test_returns_answer_with_text():
    p = _make_pipeline()
    ans = p.run(QueryRequest(text="what is the procedure?"))
    assert ans.text
    assert "[STUB]" in ans.text


def test_empty_retrieval_still_answers():
    p = _make_pipeline()
    ans = p.run(QueryRequest(text="a question"))
    assert ans.source_refs == []
    assert ans.text


def test_source_refs_populated():
    stub = QdrantRetrieverStub()
    chunk = Chunk(doc_id="d1", text="the document text")
    emb = BgeM3Embedder()
    vec = emb.embed([chunk.text])[0]
    stub.upsert([chunk], [vec])

    p = NaivePipeline(retriever=stub, embedder=emb, generator=GeneratorStub())
    ans = p.run(QueryRequest(text="the document text"))
    assert len(ans.source_refs) == 1
    assert ans.source_refs[0].doc_id == "d1"


def test_trace_id_in_metadata():
    p = _make_pipeline()
    req = QueryRequest(text="q")
    ans = p.run(req)
    assert ans.metadata.get("trace_id") == req.trace_id


class TestRetrievalPinsOverlay:
    """core/pipeline.py's wiring of core/pins/overlay.py into
    NaivePipeline.run. Uses the pin's own query embedded against ITSELF as
    query_vec (cosine similarity 1.0 with the matching query, near-zero
    with an unrelated one under BgeM3Embedder's deterministic stub vectors)
    so these tests don't depend on knowing the stub's exact vector space."""

    def test_no_pins_leaves_output_byte_identical(self) -> None:
        emb = BgeM3Embedder()
        req = QueryRequest(text="what is the procedure?")
        with_none = NaivePipeline(
            retriever=QdrantRetrieverStub(), embedder=emb, generator=GeneratorStub(), retrieval_pins=None,
        )
        with_empty = NaivePipeline(
            retriever=QdrantRetrieverStub(), embedder=emb, generator=GeneratorStub(), retrieval_pins=[],
        )
        ans_none = with_none.run(req)
        ans_empty = with_empty.run(req)
        assert ans_none.text == ans_empty.text == _make_pipeline().run(req).text
        assert ans_none.source_refs == ans_empty.source_refs == []

    def test_matching_pin_injects_a_chunk_tagged_pinned(self) -> None:
        from core.pins.overlay import Pin

        emb = BgeM3Embedder()
        query_text = "what is the procedure?"
        pin = Pin(
            id="pin1", query_vec=emb.embed([query_text])[0], threshold=0.99,
            pin_chunks=[Chunk(chunk_id="pinned-1", doc_id="d1", text="pinned content", structural_path="art-1")],
        )
        p = NaivePipeline(
            retriever=QdrantRetrieverStub(), embedder=emb, generator=GeneratorStub(), retrieval_pins=[pin],
        )
        ans = p.run(QueryRequest(text=query_text))
        pinned_refs = [sr for sr in ans.source_refs if sr.chunk_id == "pinned-1"]
        assert len(pinned_refs) == 1
        assert pinned_refs[0].pinned is True

    def test_non_matching_pin_does_not_apply(self) -> None:
        from core.pins.overlay import Pin

        emb = BgeM3Embedder()
        pin = Pin(
            id="pin1", query_vec=emb.embed(["a completely different question about animals"])[0], threshold=0.9,
            pin_chunks=[Chunk(chunk_id="pinned-1", doc_id="d1", text="pinned content")],
        )
        p = NaivePipeline(
            retriever=QdrantRetrieverStub(), embedder=emb, generator=GeneratorStub(), retrieval_pins=[pin],
        )
        ans = p.run(QueryRequest(text="what is the procedure?"))
        assert not any(sr.chunk_id == "pinned-1" for sr in ans.source_refs)

    def test_demote_removes_a_retrieved_chunk_from_the_answer(self) -> None:
        from core.pins.overlay import Pin

        stub = QdrantRetrieverStub()
        emb = BgeM3Embedder()
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="the document text")
        stub.upsert([chunk], [emb.embed([chunk.text])[0]])
        query_text = "the document text"
        pin = Pin(id="pin1", query_vec=emb.embed([query_text])[0], threshold=0.99, demote_chunk_ids=["c1"])

        p = NaivePipeline(retriever=stub, embedder=emb, generator=GeneratorStub(), retrieval_pins=[pin])
        ans = p.run(QueryRequest(text=query_text))
        assert not any(sr.chunk_id == "c1" for sr in ans.source_refs)

    def test_rewrite_pulls_in_an_additional_retrieval_result(self) -> None:
        from core.pins.overlay import Pin

        stub = QdrantRetrieverStub()
        emb = BgeM3Embedder()
        rewrite_text = "a rewritten query"
        rewrite_chunk = Chunk(chunk_id="from-rewrite", doc_id="d2", text=rewrite_text)
        stub.upsert([rewrite_chunk], [emb.embed([rewrite_text])[0]])

        query_text = "what is the procedure?"
        pin = Pin(id="pin1", query_vec=emb.embed([query_text])[0], threshold=0.99, rewrite=rewrite_text)
        p = NaivePipeline(retriever=stub, embedder=emb, generator=GeneratorStub(), retrieval_pins=[pin])
        ans = p.run(QueryRequest(text=query_text))
        assert any(sr.chunk_id == "from-rewrite" for sr in ans.source_refs)


# ── The candidate window, separate from the context ──────────

class _WideRetriever:
    """Returns as many chunks as asked for, so a test can see exactly how
    many the pipeline requested."""

    def __init__(self) -> None:
        self.asked_for: list[int] = []

    def retrieve(self, query, k, filters=None, query_vector=None):
        from core.models import Chunk, ScoredChunk
        self.asked_for.append(k)
        return [
            ScoredChunk(chunk=Chunk(doc_id=f"d{i}", text=f"t{i}", structural_path=f"p{i}"),
                        score=1.0 - i / 100, retriever_id="stub")
            for i in range(k)
        ]


def _pipeline(**kwargs) -> NaivePipeline:
    # Same stub embedder/generator the tests above use — only the retriever
    # is replaced, since what is under test is how many candidates the
    # pipeline asks it for.
    return NaivePipeline(
        retriever=_WideRetriever(), embedder=BgeM3Embedder(), generator=GeneratorStub(), **kwargs,
    )


def test_widening_the_candidate_window_does_not_widen_the_answer_context() -> None:
    """The whole risk of fetch_k: fetching more must search wider, never
    silently put more into the answer."""
    from core.models import QueryRequest

    narrow = _pipeline(top_k=3)
    wide = _pipeline(top_k=3, fetch_k=20)
    a_narrow = narrow.run(QueryRequest(text="q", top_k=3, trace_id="t"))
    a_wide = wide.run(QueryRequest(text="q", top_k=3, trace_id="t"))

    assert wide._retriever.asked_for == [20]
    assert narrow._retriever.asked_for == [3]
    assert len(a_wide.source_refs) == len(a_narrow.source_refs) == 3


def test_the_wide_window_is_recorded_for_the_counterfactual() -> None:
    from core.models import QueryRequest

    answer = _pipeline(top_k=3, fetch_k=20).run(QueryRequest(text="q", top_k=3, trace_id="t"))
    assert len(answer.candidate_source_refs) == 20


def test_no_window_is_recorded_when_it_would_duplicate_the_final_list() -> None:
    """An identical copy answers no counterfactual and doubles what every
    run stores."""
    from core.models import QueryRequest

    answer = _pipeline(top_k=3).run(QueryRequest(text="q", top_k=3, trace_id="t"))
    assert answer.candidate_source_refs == []


def test_a_window_narrower_than_the_context_is_ignored() -> None:
    """Asking for a narrower search would shrink the answer instead, which
    is the opposite of what requesting a candidate window means."""
    from core.models import QueryRequest

    pipeline = _pipeline(top_k=5, fetch_k=2)
    pipeline.run(QueryRequest(text="q", top_k=5, trace_id="t"))
    assert pipeline._retriever.asked_for == [5]
