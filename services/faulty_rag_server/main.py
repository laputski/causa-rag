"""A RAG server that answers badly on purpose, one named way at a time.

The proving ground breaks documents with `tools/corpus_mutate.py`, settings
with `tools/config_distort.py`, and indexes with `tools/ingest_distort.py`.
Some failures live in none of those places: a system that never refuses, that
cites the wrong fragment, that answers from its own knowledge. Those belong to
whoever generates the answer, and the platform sees that system only through
the external-RAG contract. This is such a system, and it is wrong deliberately.

**It wraps the reference server and never modifies it.** That server exists
so the contract is checked by an honest implementation of it, and putting code
into it that lies on purpose destroys exactly what it is for. The pipeline is
resolved by the reference server's own function, the answer is produced by the
reference server's own pipeline, and only then is it spoiled. Everything the
contract promises stays where it was written.

Five measures keep the two apart, and each is structural, resting on nothing
anybody has to remember:

- a module of its own, so no line that lies lives in the honest server;
- the reference server's `_resolve_pipeline` is imported, so the contract is
  never copied and cannot drift;
- a port of its own, 8092 against the reference server's 8091, so nobody
  reaches the wrong one by mistyping;
- `/health` and `/capabilities` say that answers are distorted and name the
  mode in force, so a system that asked cannot be misled;
- a fitness test asserts the reference server never mentions this one.

The mode is chosen by the environment and holds for the life of the process.
A per-request field was the first design and is worse: a run is a hundred
requests, and a mode that can change between them describes no system anybody
operates.

    RAG_REALM=proving-ground RAG_FAULT=none python3 -m services.faulty_rag_server.main
    RAG_REALM=proving-ground RAG_FAULT=F19_never_refuses python3 -m services.faulty_rag_server.main
"""
from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from core.models import QueryRequest
from services.reference_rag_server.main import ExternalRagRequest, _resolve_pipeline

PORT = 8092
REFERENCE_PORT = 8091

#: Which realm's corpora this server answers from, empty for the unscoped ones.
#:
#: The reference server has no realm in its contract, and that is correct: a
#: realm is this platform's own way of keeping one installation's data apart,
#: and an external system has no reason to know of it. But the proving ground's
#: corpora live in a realm, so a server that is a fixture of that proving
#: ground has to reach them. It rebinds what the reference server resolved,
#: and resolves nothing itself, so the contract stays in one place.
REALM = os.getenv("RAG_REALM", "")

_bound: dict[tuple[str, str, str], Any] = {}

#: What the platform's own refusal pattern recognises. Written here and never
#: imported, because this server plays the part of somebody else's system and
#: has no business reading the platform's internals; and a wrong guess here
#: would show up as a mode that stages nothing, which the tests would catch.
REFUSAL = "The context does not contain the answer to this question."
CONFIDENT = "The regulation sets the interval at twelve months for every instrument class."


@dataclass(frozen=True)
class Fault:
    """One named way to answer badly, and what it is meant to make observable."""

    name: str
    #: The catalogue entries this mode is meant to stage.
    provokes: tuple[str, ...]
    #: What the answer looks like afterwards, in the words a reader would use.
    describes: str
    #: Rewrites the response the honest pipeline produced. Never mutates it.
    apply: Callable[[dict[str, Any], ExternalRagRequest], dict[str, Any]]


def _never_refuses(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Answer every question, including the ones with no answer in the corpus.

    A system with no confidence threshold returns its best guess whatever the
    score, so a question the corpus does not cover comes back answered rather
    than refused.
    """
    out = dict(response)
    if not out["answer"].strip() or _reads_as_refusal(out["answer"]):
        out["answer"] = CONFIDENT
    return out


def _always_refuses(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Refuse everything, including what the corpus answers plainly.

    The other direction of the same miscalibration, and the pair of it: a
    threshold set so high that nothing clears it.
    """
    return {**response, "answer": REFUSAL}


def _returns_nothing(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Return an empty answer while retrieval succeeded.

    What a reasoning model does when it spends its whole budget thinking: the
    sources are there, the answer is not, and nothing in the response says the
    budget ran out.
    """
    return {**response, "answer": ""}


def _cites_the_wrong_fragment(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Move every citation marker one place along.

    The fragment that answers the question was found and is in the context;
    the number printed beside the sentence belongs to a different one.
    """
    sources = response.get("sources") or []
    # Guarded against nothing to point at, and nothing else. With one source
    # the arithmetic below already leaves the number where it was, which is
    # the right answer: there is no other fragment to name. The guard was
    # written for two and covered a case that needed no covering, which a bait
    # found by leaving the test green with the guard removed.
    if not sources:
        return dict(response)

    def shift(match: re.Match[str]) -> str:
        n = int(match.group(1))
        return f"(Fragment {n % len(sources) + 1})"

    return {**response, "answer": re.sub(r"\(Fragment (\d+)\)", shift, response["answer"])}


def _ignores_the_context(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Answer from somewhere other than the sources returned.

    The sources are correct and the answer does not come from them, which is
    the failure that a retrieval metric cannot see: recall is one and the
    answer is invented.
    """
    return {
        **response,
        "answer": "Instruments of this class are generally serviced once a year, "
                  "and most operators follow the manufacturer's own schedule.",
    }


def _buries_the_middle(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Fill the context far past what was asked for, and answer from its edges.

    A context stuffed with everything retrieved leaves the middle unread. The
    answer here quotes the first source and the last, and the sources between
    them are returned and unused.
    """
    sources = response.get("sources") or []
    if len(sources) < 3:
        return dict(response)
    return {
        **response,
        "sources": sources,
        "answer": f"(Fragment 1) and (Fragment {len(sources)}) together give the answer.",
    }


def _cuts_sources_mid_sentence(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """Return sources cut at a fixed width, wherever that falls.

    What a system chunking by character count returns: a fragment that begins
    or ends inside a thought, so the grounds for the answer are split between
    two of them and neither carries them.
    """
    cut = 60
    sources = []
    for source in response.get("sources") or []:
        text = source.get("chunk_text") or ""
        sources.append({**source, "chunk_text": text[cut : cut * 2]})
    return {**response, "sources": sources}


def _unchanged(response: dict[str, Any], request: ExternalRagRequest) -> dict[str, Any]:
    """The control. Every pair needs one, and it has to be the same server."""
    return dict(response)


FAULTS: tuple[Fault, ...] = (
    Fault("none", (), "answers honestly: the control half of every pair", _unchanged),
    Fault("F19_never_refuses", ("F19",),
          "answers every question, including those the corpus does not cover", _never_refuses),
    Fault("F32_always_refuses", ("F32",),
          "refuses every question, including those the corpus answers plainly", _always_refuses),
    Fault("F28_returns_nothing", ("F28",),
          "returns an empty answer while retrieval succeeded", _returns_nothing),
    Fault("F29_cites_the_wrong_fragment", ("F29",),
          "prints a citation number belonging to another fragment", _cites_the_wrong_fragment),
    Fault("F31_ignores_the_context", ("F31",),
          "answers from somewhere other than the sources it returned", _ignores_the_context),
    Fault("F30_buries_the_middle", ("F30",),
          "returns everything retrieved and answers from the edges of it", _buries_the_middle),
    Fault("F05_cuts_sources_mid_sentence", ("F05",),
          "returns fragments cut at a fixed width, wherever that falls", _cuts_sources_mid_sentence),
)

_BY_NAME = {f.name: f for f in FAULTS}


def _reads_as_refusal(answer: str) -> bool:
    text = answer.lower()
    return not text.strip() or "does not contain" in text or "не содержит" in text


def active_fault() -> Fault:
    """The mode this process runs in.

    An unknown name is refused at import and never ignored: a server started
    with a misspelled mode would answer honestly while its operator believed it
    was staging a failure, and every pair built on it would prove the opposite
    of what it claimed.
    """
    name = os.getenv("RAG_FAULT", "none")
    if name not in _BY_NAME:
        raise SystemExit(
            f"Unknown RAG_FAULT {name!r}. Known: {', '.join(sorted(_BY_NAME))}"
        )
    return _BY_NAME[name]


class _Answer(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []
    trace: dict[str, Any] | None = None


app = FastAPI(title="Faulty RAG Server (a system that answers badly on purpose)",
              version="1.0.0")


def _in_realm(body: ExternalRagRequest) -> Any:
    """The reference server's pipeline, bound to this server's realm.

    Rebound and not rebuilt: `_rebind_corpus_id` walks down to the leaf
    retrievers and rebuilds them around the same embedder and generator, which
    is exactly what the platform's own runner does for an in-process run.
    Without it the collection name carries no realm and the query reaches an
    index that is not there, which comes back as an answer with no sources and
    reads like a system that found nothing.
    """
    pipeline = _resolve_pipeline(body.pipeline_id, body.corpus_id, body.reranker_id)
    if not REALM:
        return pipeline
    key = (body.pipeline_id, body.corpus_id, body.reranker_id or "")
    if key in _bound:
        return _bound[key]
    from core.experiment.runner import _rebind_corpus_id

    _bound[key] = type(pipeline)(
        retriever=_rebind_corpus_id(pipeline._retriever, body.corpus_id, REALM),
        embedder=pipeline._embedder, generator=pipeline._generator,
        pipeline_id=pipeline.pipeline_id, reranker=pipeline._reranker,
    )
    return _bound[key]


@app.get("/health")
async def health() -> dict[str, Any]:
    """Says what this is before anybody asks it a question."""
    fault = active_fault()
    return {
        "status": "ok",
        "distorts_answers": True,
        "fault": fault.name,
        "describes": fault.describes,
        "warning": "This server answers badly on purpose. The honest one is on "
                   f"port {REFERENCE_PORT}.",
    }


@app.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    """The contract's own declaration, with the distortion named in it.

    A system that asked what this one can do and was told only about pipelines
    could register it as an ordinary RAG and measure it as one.
    """
    from services.reference_rag_server.main import capabilities as honest

    declared = dict(await honest())
    fault = active_fault()
    declared["distorts_answers"] = True
    declared["fault"] = fault.name
    declared["fault_describes"] = fault.describes
    declared["provokes"] = list(fault.provokes)
    return declared


@app.post("/")
async def query(body: ExternalRagRequest) -> dict[str, Any]:
    """The honest pipeline answers, and then the answer is spoiled.

    In that order, and never the other way: a distortion applied to the request
    would change what was retrieved, and then the pair would differ in its
    retrieval as well as in its answer, which is two changes and proves
    neither.
    """
    pipeline = _in_realm(body)
    request = QueryRequest(
        text=body.query, top_k=body.top_k, filters=body.filters,
        trace_id=body.trace_id or str(uuid.uuid4()),
    )
    answer = pipeline.run(request)
    response: dict[str, Any] = {
        "answer": answer.text,
        "sources": [source.model_dump() for source in answer.source_refs],
    }
    if answer.stage_trace is not None:
        trace: dict[str, Any] = {
            "stage_trace": answer.stage_trace.model_dump(),
            "rendered_prompt": answer.rendered_prompt_preview,
        }
        if pipeline._reranker is not None:
            trace["pre_rerank_source_refs"] = [
                source.model_dump() for source in answer.pre_rerank_source_refs
            ]
        response["trace"] = trace
    return active_fault().apply(response, body)


@app.post("/retrieve")
async def retrieve(body: ExternalRagRequest) -> dict[str, Any]:
    """Retrieval only, and every mode here spoils the answer and not the search.

    Forwarded to the honest implementation unchanged, apart from the one mode
    that describes a system whose fragments are cut at a fixed width, which is
    a property of what it retrieves.
    """
    pipeline = _in_realm(body)
    k = body.top_k or pipeline._top_k
    vector = pipeline._embedder.embed([body.query])[0]
    chunks = pipeline._retriever.retrieve(body.query, k=k, filters=body.filters, query_vector=vector)
    from core.pipeline import _to_source_refs

    response = {"sources": [source.model_dump() for source in _to_source_refs(chunks)]}
    if active_fault().name.startswith("F05"):
        return _cuts_sources_mid_sentence(response, body)
    return response


def main() -> None:
    import uvicorn

    fault = active_fault()
    print(f"Faulty RAG server on port {PORT}, mode {fault.name!r}: {fault.describes}")
    print(f"The honest one is `make reference-rag`, on port {REFERENCE_PORT}.")
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # noqa: S104


if __name__ == "__main__":
    main()
