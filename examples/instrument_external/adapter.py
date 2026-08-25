"""The <=20-line glue wiring demo_rag.py into this platform via core.sdk
(model B). This is the entire amount of code someone needs to
write to get full pipeline-trace diagnostics on a system that was never
written against this platform's interfaces.
"""
from core.sdk import build_pipeline, wrap_generator, wrap_retriever
from examples.instrument_external import demo_rag


def build_demo_pipeline():
    retriever = wrap_retriever(lambda query, k, filters=None: demo_rag.search(query, k))
    generator = wrap_generator(demo_rag.answer)
    return build_pipeline(retriever, generator, pipeline_id="demo_external")
