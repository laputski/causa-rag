# The native HTTP contract for an external RAG (tier 1)

This is the full field-by-field reference for implementing the platform's
contract **directly**, in your own FastAPI, Flask, Express or anything else,
without `causa-rag-client`.

If your RAG is written in Python and you do not mind one dependency, read
[connector-guide.md](connector-guide.md) first: `serve()` wraps a pair of
functions in this same contract for you and most of this file becomes
unnecessary.

Read this one when:

- your RAG is not in Python; the contract is ordinary JSON over HTTP and is
  language-independent;
- you already have an HTTP layer and want to add two endpoints to it directly;
- you are hardening an existing integration and need the exact register of
  fields rather than a happy-path example.

The typical shape of a direct integration is an application with its own
pipeline exposing `/platform/query` and `/platform/retrieve` on top of logic
that already exists. The examples below come from a real integration that has
been through a hardening pass, and each "found live" note records a failure
that actually happened.

> Русская версия: [docs/ru/external-rag-contract.md](ru/external-rag-contract.md).

## Contract version 1, and what that promises

This contract is **version 1**, and it is versioned separately from the platform.

The reason is asymmetric cost. The platform can change freely: it is one
codebase and its own tests catch the breakage. This contract is implemented in
other people's services, by people who will not be watching this repository, and
a change here breaks software the platform cannot see and cannot fix.

**The promise, for as long as this is version 1:**

- No field listed below is removed or renamed.
- No optional field becomes required.
- The meaning of an existing field does not change.
- New fields are additive and optional, and an implementation that ignores them
  keeps working.

A change that cannot be made under those rules becomes version 2, served
alongside version 1 rather than in place of it.

**The client tracks this contract, not the platform.** `causa-rag-client` is at
`1.x` because the contract is at version 1, and it will stay on `1.x` while this
document does. The platform's own version moves independently and says nothing
about your integration.

## Two endpoints

The platform calls exactly two URLs, both given at registration.

| Endpoint | Method | What it does |
|---|---|---|
| `url` (full URL, required) | `POST` | Retrieval plus answer generation |
| `retrieve_endpoint` (optional) | `POST` | Retrieval only, with no LLM call, which is faster and cheaper for pure retrieval metrics |

With no `retrieve_endpoint` the platform falls back to the full `url` and
discards the answer text, so you can start with one endpoint and add the second
later as an optimisation.

## Request: what the platform sends

```json
{
  "query": "What penalty applies to a late delivery?",
  "top_k": 5,
  "filters": {},
  "trace_id": "407b8318-d2f0-4196-83ff-4a44c0c4cd75",
  "corpus_id": "handbook",
  "realm_id": "demo",
  "pipeline_id": null,
  "reranker_id": null,
  "params": {"prompt_version": "v2"}
}
```

| Field | Type | Must you read it | Meaning |
|---|---|---|---|
| `query` | `str` | yes | The question text |
| `top_k` | `int` | yes | How many sources to return |
| `filters` | `dict` | no | Arbitrary filters, ignorable if you support none |
| `trace_id` | `str` | not required, but useful | An end-to-end id; carry it into your own logs so a platform request can be found there |
| `corpus_id` | `str \| null` | **yes, if you serve more than one collection** | Which corpus to search. See the section below; this is the commonly missed one |
| `realm_id` | `str \| null` | no | Which project the run belongs to. `corpus_id` alone is not guaranteed unique across realms, so both together identify a corpus |
| `pipeline_id` / `reranker_id` | `str \| null` | no | Only meaningful for the platform's own reference server; a third-party RAG ignores them |
| `params` | `dict` | no | Extensible knobs such as temperature or prompt version. Read only the keys you declared in `supported_params` at registration |

## corpus_id is the commonly missed detail

**If your service holds more than one collection, `corpus_id` is the only
channel telling you which one to search.** Declaring the field is not enough;
it has to be read and used when choosing the collection.

Found live: one integration declared `corpus_id` on its request model and read
it nowhere, so the vector store was built once at startup against a single
hardcoded collection. Every run answered from that same collection whatever
`corpus_id` said. Nothing raised. The symptom looked like poor recall rather
than like a routing defect, which is the expensive kind of failure.

Wrong, and this is the code as it was found:

```python
class PlatformRequest(BaseModel):
    corpus_id: str | None = None  # declared, never read

@router.post("/query")
async def platform_query(body: PlatformRequest, ...):
    # vector_store always points at ONE hardcoded collection from settings
    results = vector_store.dense_search(embedding, top_k=body.top_k)
```

Right, resolving the collection per request, with a per-corpus cache when
rebuilding a client is expensive:

```python
_collection_cache: dict[str, VectorStoreService] = {}

def _resolve_collection(corpus_id: str | None) -> VectorStoreService:
    name = corpus_id or settings.default_collection  # None means the default
    if name not in _collection_cache:
        _collection_cache[name] = VectorStoreService(
            url=settings.qdrant_url, collection=name, dim=embedding_service.dim,
        )
    return _collection_cache[name]

@router.post("/query")
async def platform_query(body: PlatformRequest, ...):
    vector_store = _resolve_collection(body.corpus_id)
    results = vector_store.dense_search(embedding, top_k=body.top_k)
```

With a single corpus you may honestly ignore the field, but then check that the
run on the platform side also leaves `corpus_id` unset or set to your default.
Otherwise the same silent mismatch returns from the other direction.

## Finding out which embedder indexed a corpus

Found live: `corpus_id` resolved to the right collection and the request still
failed with a dimension error from the vector database. The RAG embedded each
query with its own model at 768 dimensions while the corpus had been indexed
with the platform's BGE-M3 at 1024. A collection name says nothing about what
filled it.

`GET /corpus/collections?realm_id=<realm_id>`, the same endpoint that lists
corpora for choosing `corpus_id`, answers this too. Each record carries
`backends`, and each physical store inside it may have its own `embedder_id`:

```json
[
  {
    "realm_id": "demo",
    "corpus_id": "handbook",
    "backends": {
      "qdrant": {"collection": "demo__handbook__structure_aware__bge_m3", "embedder_id": "bge_m3"},
      "opensearch": {"index": "rag__demo__handbook__structure_aware"}
    }
  }
]
```

`embedder_id` sits on the individual backend rather than on the corpus, because
one corpus can genuinely carry several embedders at once. A GraphRAG-style
corpus has a separate embedder for community embeddings in Neo4j, distinct from
the dense chunk embedder in Qdrant. OpenSearch has no embedder at all: BM25 is a
lexical signal rather than a vector one.

In Python the same thing is one line:

```python
embedders = client.get_corpus_embedders(realm_id="demo", corpus_id="handbook")
# ["bge_m3"]
```

Both `realm_id` and `corpus_id` are required, because `corpus_id` alone is not
guaranteed unique across realms. Both arrive in the request the platform sends
you, so no extra lookup is needed.

## Response: what your endpoint must return

```json
{
  "answer": "The generated answer text",
  "sources": [
    {
      "doc_id": "demo_handbook",
      "chunk_id": "demo_handbook#12",
      "structural_path": "Purchase approval/Approval thresholds",
      "score": 0.9123,
      "chunk_text": "The text of the chunk the answer rests on",
      "source_code": "demo_handbook",
      "article_no": "01"
    }
  ],
  "trace": {
    "stage_trace": {
      "embed_ms": 12.4,
      "dense_retrieve_ms": 45.1,
      "rerank_ms": 8.3,
      "generate_ms": 1230.5,
      "total_ms": 1296.3,
      "n_dense": 20,
      "n_sparse": 10,
      "n_merged": 15,
      "n_reranked": 5
    },
    "rendered_prompt": "the first 500 characters of the assembled prompt, for debugging",
    "pre_rerank_source_refs": [],
    "embedders": ["bge_m3"]
  },
  "metadata": {
    "model": "qwen3:8b",
    "request_id": "407b8318-d2f0-4196-83ff-4a44c0c4cd75"
  }
}
```

### `sources[]`, required and optional fields

| Field | Required | Why |
|---|---|---|
| `doc_id` | **yes** | The one required field. Without it `recall@k` and `precision@k` cannot be computed at all, and the platform will not invent an id you did not return |
| `chunk_id` | no, defaults to `""` | Chunk granularity, if you have it |
| `chunk_text` | no, but strongly recommended | Without it `faithfulness`, `context_support` and `grounded_in_correct_source` go uncomputed and you get only `answer_similarity` |
| `source_code` / `article_no` | no | What golden `article_refs` are matched against: a structural reference rather than your internal `doc_id`. Without them `recall@k` and `precision@k` stay uncomputed even when `doc_id` is present |
| `score` | no | Informational; it changes nothing |

### `trace` is optional in its entirety

Omitting `trace` breaks nothing. That run simply gets **outcome metrics only**,
comparing answer text, with no per-layer attribution across retrieval, rerank
and generation, and no latency breakdown on the run page.

Return at least `stage_trace` if you can. It is cheap, being a few
`time.perf_counter()` calls around steps you already have, and it immediately
gives the platform a chart showing which stage of your pipeline is slow.

`pre_rerank_source_refs` is the source list **before** the reranker, when you
have one. Without it the platform cannot tell "retrieval never found the chunk"
from "retrieval found it and the reranker discarded it". Both look like zero
recall, and they call for opposite fixes.

### `embedders` tells the platform what you answered with

A list of strings naming the embedder or embedders used for **this** request.

Found live, the same incident as in the section above: without this field the
platform learns about an embedding-space mismatch only as a hard vector
database error during a real run, rather than beforehand.

When you do return it, the next "Test" on the Resources page compares it
against what actually indexed the corpus (`GET /corpus/collections`) and warns
on a mismatch. **The platform never downloads or tries to instantiate an
embedder from this value.** The comparison is informational, for diagnosis
alone. Omitting the field is not an error; it downgrades a diagnosis into a
soft hint.

A list rather than a single string, because one corpus can genuinely involve
more than one embedder. If you cannot describe the set honestly, omit the field
rather than guessing.

## Timeouts: never let a request hang

The platform waits a bounded time for you (`HttpPipeline`, 30 seconds by
default). Exceed it and the platform sees a network `Operation timed out` with
no diagnosis of what stalled on your side.

**The rule:** your own timeout, on the LLM call, the vector store round trip and
anything else over a network, must be **strictly shorter** than the platform's,
so your service returns a legible error before the platform gives up blind.

Taken from a real fix, where a stalled model hung requests for 35 seconds
without a line in any log:

```python
import asyncio

LLM_TIMEOUT = 25.0  # strictly below the platform's timeout, 30s by default

async def generate_answer(messages) -> str:
    try:
        return await asyncio.wait_for(llm_service.ainvoke(messages), timeout=LLM_TIMEOUT)
    except TimeoutError as e:
        # A legible error becomes a 500 with a message, rather than an
        # unbounded wait on the platform's side.
        raise TimeoutError(f"LLM generation exceeded {LLM_TIMEOUT}s") from e
```

The same applies to your vector store, cache and reranker whenever they are
reached over a network.

## Do not swallow errors

Where a step can degrade quietly, say so. A lexical index built once at startup
from a collection that was empty at the time stays empty for the life of the
process, and half your retrieval signal disappears with nothing reporting it.

Log it explicitly, and better still attempt one rebuild on the first real
request:

```python
if not bm25_index.is_ready:
    log.warning("bm25_index_empty, rebuilding from the vector store now")
    bm25_index.rebuild_from(vector_store.get_all_documents())
```

An error should reach the platform as a legible HTTP 5xx carrying its reason.
The platform isolates one failed question and continues the rest of the run
rather than aborting it, but only when it receives **a response**. A hang
gives it nothing to isolate.

## Logs: carry the trace_id through

Put the request's `trace_id` on every log line for that request. It is the only
way to find, in your logs, what happened to one specific question of one
specific run:

```python
log.info("request query=%r top_k=%s trace_id=%s", body.query[:80], body.top_k, trace_id)
# ... all processing ...
log.info("response n_sources=%d total_ms=%.1f trace_id=%s", len(sources), total_ms, trace_id)
```

## `GET /health` and `GET /capabilities`, optional but useful

Neither is part of the required contract, and the platform uses both at
registration and for debugging when they exist:

```python
@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@router.get("/capabilities")
async def capabilities() -> dict:
    return {
        "supports_trace": True,
        "supports_retrieval_only": True,
        "retrieve_endpoint": "/platform/retrieve",
        "supported_params": ["prompt_version"],
    }
```

## Registering over the raw HTTP API

Without `causa-rag-client`, registration is an ordinary POST:

```bash
curl -X POST http://localhost:8081/external-rags \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_rag",
    "url": "http://localhost:8002/platform/query",
    "retrieve_endpoint": "http://localhost:8002/platform/retrieve",
    "supported_params": ["prompt_version"],
    "coordinates": {"A5": "dense_single", "C3": "rrf", "D1": "cross_encoder"}
  }'
```

`coordinates` says where your system sits in the space of architectures, as
`{dimension code: value}`. It decides which entries of the failure atlas can
occur in your system at all, so a run against it is diagnosed against the
failures its own shape admits and never against every failure known. Like
`supported_params`, it is declared and never probed: no request's answer says
whether the thing on the other end fuses two sources.

The codes and values come from the schema published at
[ragworld.org](https://ragworld.org). The platform keeps a verified copy and
answers 400 for a code or value that does not resolve in it: a coordinate
stored unchecked matches no entry, and the system would read as one in which
no failure can happen. Omit the field to declare nothing, which is honest and
leaves the atlas saying only what it says about every system.

Then upload golden questions (`POST /datasets`) and start a run
(`POST /experiments`, with `external_rag_id` set to the id the registration
returned, and `corpus_id` when your service is multi-corpus), either from the
interface or through the same raw calls.

## Registering a corpus prepared outside the platform

When your own ingest writes straight into Qdrant, OpenSearch or Neo4j, bypassing
the platform's upload, the corpus exists physically and the platform knows
nothing about it: it does not appear in the list for choosing `corpus_id`, and
its embedders are not discoverable through `GET /corpus/collections`.

One public endpoint covers this for any owner, the platform's own form or your
ingestor:

```bash
curl -X POST http://localhost:8081/corpus/collections \
  -H "Content-Type: application/json" \
  -d '{
    "realm_id": "demo",
    "corpus_id": "external_medical",
    "storage_type": "dense_only",
    "backends": {
      "qdrant": {"collection": "demo__external_medical", "embedder_id": "bge_m3"}
    },
    "owner": "external-ingestor",
    "description": "Indexed by the external RAG's own ingestor"
  }'
```

`backends` uses the same shape you see in `GET /corpus/collections`: one key per
physical store, with `embedder_id` inside each where it applies.

**What this registration does not do:** by itself it grants the Content, Health
and Graph tabs no access to the corpus contents. That still requires the store
to be attached to the realm's own resources (see `uses_realm_resources` on the
RAG registration form). This registration is only about the corpus being
*known*: listed, selectable as a `corpus_id`, and with discoverable embedders.

## A complete minimal example

```python
from fastapi import APIRouter
from pydantic import BaseModel
import asyncio
import time

router = APIRouter(prefix="/platform")

class PlatformRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict = {}
    trace_id: str | None = None
    corpus_id: str | None = None
    realm_id: str | None = None
    params: dict = {}

@router.post("/query")
async def platform_query(body: PlatformRequest):
    t0 = time.perf_counter()

    # 1. corpus_id decides the collection, see the section above
    store = resolve_collection(body.corpus_id)

    # 2. retrieval, timed
    t1 = time.perf_counter()
    chunks = store.search(body.query, top_k=body.top_k)
    retrieve_ms = (time.perf_counter() - t1) * 1000

    # 3. generation, bounded
    t2 = time.perf_counter()
    try:
        answer = await asyncio.wait_for(generate(body.query, chunks), timeout=25.0)
    except TimeoutError:
        raise  # becomes a 500, so the platform sees an explicit error
    generate_ms = (time.perf_counter() - t2) * 1000

    return {
        "answer": answer,
        "sources": [
            {"doc_id": c.doc_id, "chunk_text": c.text, "score": c.score}
            for c in chunks
        ],
        "trace": {
            "stage_trace": {
                "dense_retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": round(generate_ms, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "n_dense": len(chunks),
            },
        },
    }
```

## Checklist for hardening an existing RAG

The order comes from doing this to a service that already worked.

1. **Timeouts on every network call**, LLM, vector store and cache, strictly
   below the platform's 30 seconds. Without them one stalled backend hangs the
   platform's request indefinitely.
2. **`corpus_id` genuinely read and mapped to a collection**, when the service
   is multi-corpus. Otherwise a run against the wrong corpus looks like a
   quality regression rather than a routing defect.
3. **An empty or half-built index must not be silent.** Log it, and rebuild once
   on the first request where you can, rather than staying broken for the life
   of the process.
4. **`stage_trace` filled with real numbers** rather than zeros. The platform
   already draws the latency chart; it needs the data.
5. **Errors as a legible 5xx carrying the reason**, never a silent hang. The
   platform isolates one failed question and keeps the rest of the run, but only
   when it gets a response.
6. **`trace_id` on every log line.** Without it, reproducing one failing
   question from a platform run inside your own logs is close to impossible.

## Testing the integration

Start small: one `curl` against `/platform/query`, and check the response shape
by hand.

Then `/platform/retrieve` on its own, which separates retrieval problems from
generation problems before you run a dataset at all.

`GET /external-rag-spec` on the platform returns the current contract version,
which is the authority whenever this guide and the code disagree.

Run a small dataset of five to ten questions before the full golden set. A hang
or a routing mistake then shows in seconds rather than after thirty minutes over
hundreds of questions.
