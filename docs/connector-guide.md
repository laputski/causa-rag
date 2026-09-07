# Connecting your RAG with causa-rag-client

For someone building their own RAG, elsewhere, who wants it measured by this
platform without rewriting its internals.

> Русская версия: [docs/ru/connector-guide.md](ru/connector-guide.md).

## What this is for

The platform computes the same metrics for any RAG that implements its HTTP
contract as it does for its own reference pipeline: recall and precision,
semantic similarity, and per-layer attribution across retrieval, reranking and
generation. The contract itself is small, an `answer` plus `sources[]`, with an
optional `trace`.

`causa-rag-client` is a thin Python package (`clients/python/`) that makes the
connection minimal: a couple of client calls, plus optionally a couple of your
own functions instead of a hand-written HTTP server. Evaluation always happens
**on the platform**. The client drives an existing REST API and computes
nothing itself.

If your RAG is not in Python, or you would rather add two endpoints to a
service you already run without taking a dependency, read
[external-rag-contract.md](external-rag-contract.md) instead. It is the full
language-independent reference: every request and response field, `corpus_id`
routing, timeouts, and a hardening checklist.

## Two roles, two dependency sets

```bash
pip install causa-rag-client                 # the client alone (httpx)
pip install causa-rag-client[serve]          # plus the serve() helper (fastapi, uvicorn)
```

Until the first PyPI release, install it from a checkout instead:

```bash
pip install -e 'path/to/causa-rag/clients/python[serve]'
```

**The client** (`RagPlatformClient`) calls the platform's own endpoints:
`/external-rags`, `/experiments`, `/external-rag-spec`. Use it even when your
RAG is already exposed over HTTP by hand.

**`serve()`** matters only when you do not yet have an HTTP endpoint and would
rather not write one.

Use the client half whenever your RAG is in Python, whichever of the two applies.
Registering once by hand with `curl` leaves nothing in your own repository and
cannot be repeated by whoever inherits it; `register_rag()` and
`check_contract_version()` make registration part of your service's code.

Calling `register_rag()` from your service's startup needs one guard: the method
**does not deduplicate**, so a call on every restart inserts a fresh record
every time. Check `client.list_rags(realm_id=...)` for a matching `url` first,
which is what the real integrations do.

## Two phases, kept separate

Connecting a RAG is always two distinct phases, and conflating them is the
common mistake.

**One, expose the RAG.** Either you already have an HTTP endpoint answering
`{"answer": "...", "sources": [...]}`, in which case there is nothing to do
beyond passing its URL, or you have only `retrieve(query, top_k)` and
`generate(query, sources)` functions, in which case `serve()` wraps them in the
contract for you.

**Two, register and run.** `RagPlatformClient` registers the URL, uploads
control questions, starts a run and collects the result.

The platform must be able to reach your RAG over HTTP. Locally that just means
both are on localhost. A production RAG needs a production platform with
outbound access, governed by `RAG_HTTP_ALLOWLIST`.

## Phase 1a: you already have an endpoint

Nothing from `serve()` is needed. Check that it answers `POST /` with a body of

```json
{"query": "the user's question", "top_k": 5}
```

and returns

```json
{"answer": "the answer text", "sources": [{"doc_id": "...", "chunk_text": "..."}]}
```

`doc_id` is the one field each source must carry for retrieval metrics. Without
it recall and precision go uncomputed, because the platform will not invent an
identifier the RAG did not return.

## Phase 1b: you have functions, not an endpoint

```python
# my_rag/serve_app.py
from causa_rag_client import serve, run_server

def retrieve(query: str, top_k: int) -> list[dict]:
    results = my_index.search(query, top_k)
    return [{"doc_id": r.id, "chunk_text": r.text} for r in results]

def generate(query: str, sources: list[dict]) -> str:
    return my_llm.answer(query, [s["chunk_text"] for s in sources])

app = serve(retrieve, generate)

if __name__ == "__main__":
    run_server(app, port=8800)
```

Serving several corpora from one service? Add `corpus_id` as a third parameter.
`serve()` inspects the signature of `retrieve` to decide whether to pass it, so
a two-argument function keeps working unchanged:

```python
def retrieve(query: str, top_k: int, corpus_id: str | None) -> list[dict]:
    # Map corpus_id onto YOUR OWN index name explicitly. Do not assume the
    # platform's corpus_id matches it literally: when the two happen to be
    # equal that is a coincidence rather than a guarantee. This caused a real
    # defect, where a platform corpus_id matched no collection on the RAG side
    # and retrieval failed silently on every question.
    collection = CORPUS_TO_COLLECTION.get(corpus_id, DEFAULT_COLLECTION)
    results = my_index.search(query, top_k, collection=collection)
    return [{"doc_id": r.id, "chunk_text": r.text} for r in results]

app = serve(retrieve, generate)
```

`corpus_id` may arrive as `None` when the platform did not send one. Treat that
as "no opinion" rather than as an error.

When your RAG has a separate reranking step, pass
`rerank_fn(query, sources, top_k) -> sources`. The platform can then tell
"retrieval never found the chunk" from "retrieval found it and the reranker
discarded it" rather than attributing both to retrieval:

```python
app = serve(retrieve, generate, rerank_fn=my_rerank)
```

## Phase 2: register and run

```python
from causa_rag_client import RagPlatformClient

client = RagPlatformClient("http://localhost:8081")  # the platform gateway

rag = client.register_rag(name="my_rag", url="http://localhost:8800/", realm_id="my-realm")

client.upload_dataset(
    rag["id"],
    filename="golden.jsonl",
    realm_id="my-realm",
    questions=[
        {"id": "q1", "question": "...", "article_refs": ["CODE/1"], "answerability": "answerable"},
    ],
)

run = client.run_experiment(name="my_rag_run", dataset_name="golden.jsonl", rag_id=rag["id"])
result = client.wait_for_completion(run["run_id"], timeout=300)
print(result["aggregate_metrics"])
```

`run_experiment` accepts either `rag_id`, a registered URL, or `http_endpoint`,
a URL used directly without registration. Exactly one of the two.

### Run properties

| Parameter | What it does |
|---|---|
| `rag_id` / `http_endpoint` | Where the platform sends requests. Exactly one of the two |
| `corpus_id` | An opaque platform string. It need not match your own collection name; mapping it onto your storage is your `retrieve_fn`'s job, as in the example above |
| `pipeline_id` / `reranker_id` | Which built-in strategy to dog-food. Applies only to the platform's reference server |
| `top_k` | How many sources to request |
| `metric_embedder_id` | The platform's **measuring** embedder for semantic metrics, which is not the retrieval embedder inside your RAG. See below |
| `params` | Arbitrary extra knobs such as fetch depth or temperature. See below |

### The measuring embedder is not your retrieval embedder

`metric_embedder_id` selects the embedder the platform uses to compute
`answer_similarity`, `context_support` and `grounded_in_correct_source`, by
comparing your answer and sources against the reference. It is a measuring
instrument, held fixed so that different RAGs stay comparable.

The platform neither needs nor sees whatever embedder your RAG uses internally
for retrieval. It only ever sees the `sources` and `answer` you returned.

### Extensible params and declared capabilities

Knobs you want to vary from the platform's run form, and that have no dedicated
field, travel through `params={...}` on `run_experiment`. Declare which keys you
actually read at registration:

```python
client.register_rag(name="my_rag", url="...", supported_params=["fetch_k", "temperature"])
```

### Where your system sits

The platform diagnoses a run against the failures your architecture admits,
and half its catalogue cannot occur in a dense-only system while a different
half cannot occur in a graph one. Which half is yours is decided by
coordinates, and no probe can see them: no request's answer says whether the
thing on the other end fuses two sources. So you declare them, the same way
you declare the knobs you read.

```python
client.register_rag(
    name="my_rag", url="...",
    coordinates={"A5": "dense_single", "C3": "rrf", "D1": "cross_encoder"},
)
```

The codes and their values come from the published schema at
[ragworld.org](https://ragworld.org); the platform keeps a verified copy and
refuses a code or value that does not resolve in it, because a coordinate
stored unchecked matches no entry and would make your system read as one in
which no failure can happen. Declaring nothing is allowed and means what it
says: the atlas can then tell you only what it tells every system.

The platform assumes nothing about an undeclared key. It passes `params` through
as given, and the interpretation is entirely yours.

## Which embedder indexed the corpus you are tested against

Found live: a RAG resolved `corpus_id` to the right collection and still failed,
because it embedded each query with its own model while the corpus had been
indexed with the platform's BGE-M3. A collection name is no guarantee of what
filled it.

```python
embedders = client.get_corpus_embedders(realm_id="demo", corpus_id="handbook")
# ["bge_m3"]
```

Both parameters are required, because `corpus_id` alone is not guaranteed unique
across realms. A list rather than a string, because one corpus can carry more
than one embedder at once, for instance a dense one for chunk search and a
separate one for graph community embeddings.

Then return `embedders` inside `trace` on your own responses. The platform
compares it against what actually indexed the corpus and warns on a mismatch on
the Resources page, with no further code on your side. See the `embedders`
section of [external-rag-contract.md](external-rag-contract.md).

## The library version is the contract version

On creation, unless `check_version=False`, `RagPlatformClient` compares its own
version against what the platform publishes at `GET /external-rag-spec`.

A major mismatch raises `ContractVersionMismatch` with instructions to upgrade,
rather than breaking quietly somewhere mid-run. A minor or patch difference,
meaning an additive contract growth, produces a warning instead: the platform
does not force an SDK upgrade for every backward-compatible addition, it just
mentions that one is available.

## Reading the result

`result["aggregate_metrics"]` holds the same metrics the run page shows for the
built-in pipeline. `correct_refusal` covers every question;
`retrieval_recall_at_k`, `retrieval_precision_at_k`, `answer_similarity`,
`context_support`, `grounded_in_correct_source` and `citation_number_coverage`
cover the `answerable` ones.

When you passed `rerank_fn` to `serve()`, the full result from
`GET /experiments/{run_id}` or `client.get_results` carries a funnel verdict per
question, naming which layer failed when one did.

## Testing your integration

Before a full run, call `client.test_rag(rag["id"])`. It sends one real request
and reports the connection tier plus a preview of the parsed sources. A wrong
`doc_id` or a broken mapping shows immediately, rather than masquerading as poor
retrieval quality after a hundred-question run.

## A complete working example

`examples/connector_quickstart/` walks a toy RAG through `serve()`,
`run_experiment()` and the results, with no `curl` and no hand-written FastAPI.
