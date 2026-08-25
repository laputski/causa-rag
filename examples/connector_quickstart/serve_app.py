"""Run this to expose toy_rag.py over the platform's tier-1 contract — the
entire amount of code an external RAG author writes to become testable,
via causa_rag_client.serve().

    python3 -m examples.connector_quickstart.serve_app

Requires the `serve` extra: pip install -e clients/python/[serve]
"""
from causa_rag_client import run_server, serve

from examples.connector_quickstart import toy_rag

app = serve(toy_rag.retrieve, toy_rag.generate)

if __name__ == "__main__":
    run_server(app, port=8800)
