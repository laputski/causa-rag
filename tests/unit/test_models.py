from core.models import (
    Answer,
    AuditEvent,
    AuditEventType,
    Chunk,
    Document,
    DocumentNode,
    GroundingResult,
    QueryRequest,
    ScoredChunk,
)


def test_document_defaults():
    doc = Document(source="test.txt", content="hello")
    assert doc.doc_id
    assert doc.content_hash == ""
    assert doc.structure is None


def test_document_node_children():
    child = DocumentNode(node_id="c1", node_type="paragraph", content="text")
    parent = DocumentNode(node_id="p1", node_type="section", children=[child])
    assert parent.children[0].node_id == "c1"


def test_chunk_defaults():
    chunk = Chunk(doc_id="d1", text="some text")
    assert chunk.chunk_id
    assert chunk.strategy_id == ""


def test_scored_chunk():
    chunk = Chunk(doc_id="d1", text="t")
    sc = ScoredChunk(chunk=chunk, score=0.9)
    assert sc.score == 0.9


def test_query_request_trace_id():
    q = QueryRequest(text="a question")
    assert q.trace_id
    assert q.query_id != q.trace_id


def test_answer_defaults():
    a = Answer(text="an answer")
    assert not a.refused
    assert a.source_refs == []


def test_grounding_result():
    gr = GroundingResult(is_grounded=False, unsupported_claims=["claim1"])
    assert len(gr.unsupported_claims) == 1


def test_audit_event():
    ev = AuditEvent(trace_id="t1", event_type=AuditEventType.QUERY)
    assert ev.event_type == "query"
    assert ev.event_id
