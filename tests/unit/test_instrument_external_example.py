"""Proves the example adapter actually works end to end:
a system with zero awareness of this platform, wrapped in <=20 lines via
core.sdk, produces a full pipeline-trace through the standard Pipeline
interface.
"""
from __future__ import annotations

from core.models import QueryRequest
from examples.instrument_external.adapter import build_demo_pipeline


def test_demo_pipeline_satisfies_pipeline_protocol():
    from core.interfaces import Pipeline
    pipeline = build_demo_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_demo_pipeline_yields_nonempty_stage_trace():
    pipeline = build_demo_pipeline()
    answer = pipeline.run(QueryRequest(text="How long do cats sleep on average", top_k=3))

    assert answer.text
    assert answer.source_refs
    assert all("cats" in sr.chunk_text.lower() or "whiskers" in sr.chunk_text.lower() for sr in answer.source_refs)

    trace = answer.stage_trace
    assert trace is not None
    assert trace.n_dense > 0
    assert trace.dense_retrieve_ms >= 0
    assert trace.generate_ms >= 0
    assert trace.total_ms >= 0
