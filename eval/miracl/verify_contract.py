"""Measure the same thing twice: through the pipeline, and through the contract.

Every row of the Configuration Report is produced by the in-process
pipeline, and what the platform is for is measuring somebody else's
retrieval over docs/external-rag-contract.md. A report that never exercises
the contract does not demonstrate the thing it is offered as evidence for.

So this serves the platform's own retriever behind that contract and runs
the identical questions both ways. The two paths should return the same
sources, in the same order, with the same text, for every question. Where
they differ, the contract is losing something on the way out or on the way
back, and that loss would be invisible to anyone whose RAG only ever spoke
to it that way.

The text is compared as well as the identifiers, and that is why this is
worth running on Arabic and Russian, not on English alone. Retrieval
metrics match on source_code and article_no, so a passage mangled in a JSON
round trip would reach the reranker and the answer while every number in
the report stayed identical.

The app is served over real HTTP on a loopback port, because the platform
speaks to an external RAG with a synchronous client and an in-memory ASGI
shortcut would skip the transport it actually uses. What is exercised is
request shaping, JSON round-tripping, response parsing and SourceRef
construction, which is where a loss would happen.

Usage:
    python3 -m eval.miracl.verify_contract --lang en --limit 50
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from core.models import QueryRequest

_CLIENT_HINT = (
    "this needs the client package, which is a separate distribution in this "
    "repository: pip install -e 'clients/python[serve]'"
)

_TOP_K = 10


def _sources_from(scored: list[Any]) -> list[dict[str, Any]]:
    """A retrieved chunk as the contract carries it.

    source_code and article_no travel explicitly. They are what retrieval
    metrics match on, and a RAG that omits them can be scored for nothing
    but latency, which the contract's own documentation says out loud.
    """
    return [
        {
            "doc_id": sc.chunk.doc_id,
            "chunk_id": sc.chunk.chunk_id,
            "structural_path": sc.chunk.structural_path,
            "chunk_text": sc.chunk.text,
            "score": float(sc.score),
            "source_code": sc.chunk.metadata.get("source_code"),
            "article_no": sc.chunk.metadata.get("article_no"),
        }
        for sc in scored
    ]


def _serve() -> Any:
    """The client package's serve(), or a sentence saying how to get it.

    It is a separate distribution under clients/python and deliberately not
    a dependency of the platform, so it is absent from an ordinary install.
    This used to reach it by putting that directory on sys.path, which
    worked only because this file happens to live in the same checkout and
    left no way to run against an installed copy.
    """
    try:
        from causa_rag_client.serve import serve
    except ImportError as exc:
        raise SystemExit(f"{exc}. {_CLIENT_HINT}") from exc
    return serve


def build_app(retriever: Any, embedder: Any) -> Any:
    """The platform's own retriever, behind the customer-facing contract.

    It embeds the query itself, the way any real external RAG would: the
    contract carries text, and what a RAG does with that text is its own
    business. Handing it a vector would test a shortcut nobody has.
    """
    serve = _serve()

    def retrieve_fn(query: str, top_k: int, corpus_id: str | None) -> list[dict[str, Any]]:
        vector = embedder.embed([query])[0]
        return _sources_from(retriever.retrieve(query, k=top_k, query_vector=vector))

    return serve(
        retrieve_fn,
        capabilities={
            "supports_trace": False,
            "supports_retrieval_only": True,
            "retrieve_endpoint": "/retrieve",
        },
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def serving(app: Any, log: Callable[[str], None] = print) -> Iterator[str]:
    """Run the contract app on loopback for the length of the comparison."""
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        raise RuntimeError("the contract server did not come up")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def compare(lang: str, limit: int, log: Callable[[str], None] = print) -> int:
    from adapters.bge_m3 import BgeM3Embedder
    from adapters.http_pipeline import HttpPipeline
    from eval.dataset import EvalDataset
    from eval.miracl.report import _build_registry, _RefusingGenerator

    embedder = BgeM3Embedder(use_real_model=True)
    registry = _build_registry(lang, embedder, _RefusingGenerator())
    inprocess = registry.resolve("pipeline", "naive")

    dataset = EvalDataset.from_jsonl(Path(f"eval/golden/miracl-{lang}.v1.fast.jsonl"))
    questions = dataset.questions[:limit]

    order_differs = []
    text_differs = []
    with serving(build_app(inprocess._retriever, embedder), log) as base:
        external = HttpPipeline(url=base + "/", retrieve_endpoint=base + "/retrieve")
        for q in questions:
            mine = inprocess.retrieve(QueryRequest(text=q["question"], top_k=_TOP_K))
            theirs = external.retrieve(QueryRequest(text=q["question"], top_k=_TOP_K))

            a = [(s.source_code, s.article_no) for s in mine.source_refs]
            b = [(s.source_code, s.article_no) for s in theirs.source_refs]
            if a != b:
                order_differs.append((q["id"], a, b))
                continue

            for x, y in zip(mine.source_refs, theirs.source_refs, strict=True):
                if x.chunk_text != y.chunk_text:
                    text_differs.append((q["id"], x.source_code, x.article_no,
                                         x.chunk_text[:60], y.chunk_text[:60]))
                    break

    log(f"{len(questions)} questions: {len(order_differs)} disagree on which sources "
        f"and in what order, {len(text_differs)} on the text of a source")
    for qid, a, b in order_differs[:3]:
        log(f"  {qid} order\n    pipeline: {a}\n    contract: {b}")
    for qid, sc, an, mine_text, theirs_text in text_differs[:3]:
        log(f"  {qid} text of {sc}/{an}\n    pipeline: {mine_text!r}"
            f"\n    contract: {theirs_text!r}")
    return 0 if not (order_differs or text_differs) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args(argv)
    return compare(args.lang, args.limit, lambda m: print(m, file=sys.stderr, flush=True))


if __name__ == "__main__":
    raise SystemExit(main())
