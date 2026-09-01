# causa-rag-client

The Python client for [Causa RAG](https://github.com/laputski/causa-rag),
a diagnostic bench for retrieval-augmented generation.

Point the bench at a RAG system you already run, and it will score every
question in a golden set separately, name the pipeline stage that failed, and
compare two runs question by question — so the effect of a configuration change
is visible instead of assumed.

This package is the thin end: it drives an existing REST API and computes
nothing itself. **Evaluation always happens on the platform.**

## Install

```bash
pip install causa-rag-client            # the client alone (httpx)
pip install causa-rag-client[serve]     # plus serve(), which needs fastapi and uvicorn
```

## Two roles

**Connecting your RAG.** `serve()` wraps your own functions in the platform's
HTTP contract and returns an app; `run_server()` runs it, so you do not write
an HTTP server by hand. Retrieval alone is enough to be measured — pass a
generator too and the bench can score the answers as well:

```python
from causa_rag_client import serve, run_server

def retrieve(question: str, top_k: int) -> list[dict]:
    ...   # your retrieval
    return [{"doc_id": d, "chunk_text": t, "score": s} for d, t, s in hits]

def generate(question: str, sources: list[dict]) -> str:
    ...   # your generation, optional
    return answer

run_server(serve(retrieve, generate), port=8000)
```

A third parameter, `corpus_id`, is passed to `retrieve` only if its signature
asks for it: a two-argument function keeps working untouched.

**Driving the bench.** `RagPlatformClient` registers that endpoint, uploads a
dataset, runs an experiment and fetches the result:

```python
from causa_rag_client import RagPlatformClient

client = RagPlatformClient("http://localhost:8081")
rag = client.register_rag(name="my-rag", url="http://localhost:8000", realm_id="demo")
run = client.run_experiment(
    name="first run", dataset_name="golden.v1", rag_id=rag["id"], top_k=10,
)
print(client.get_results(run["run_id"]))
```

## Two version numbers, and which one to read

The package has its own version, and the **HTTP contract** it speaks has
another. They are not the same number, and an earlier release of this README
said they were. `CONTRACT_VERSION` inside the package is the contract's; the
version on the package page is the library's.

The contract is at version 1 and holds while all of this holds:

- no field is removed or renamed;
- no optional field becomes required;
- the meaning of an existing field does not change;
- new fields are additive and optional.

The client checks the platform's reported contract version before sending
anything. A different **major** version raises `ContractVersionMismatch`
instead of sending a request the platform cannot parse; a different minor
version only warns, because the platform bumps the minor part for additions
that older clients can ignore.

## Not in Python?

The contract is ordinary JSON over HTTP and needs no client at all. The
field-by-field reference is
[docs/external-rag-contract.md](https://github.com/laputski/causa-rag/blob/main/docs/external-rag-contract.md);
adding two endpoints to a service you already run is the whole integration.

## Licence

Apache-2.0. Copyright 2026 Alexander Laputski.
