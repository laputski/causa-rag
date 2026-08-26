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

**Connecting your RAG.** `serve()` wraps a pair of your own functions in the
platform's HTTP contract, so you do not write an HTTP server by hand:

```python
from causa_rag_client import serve

def answer(question: str, corpus_id: str) -> dict:
    ...   # your pipeline
    return {"answer": text, "sources": [{"doc_id": d, "text": t} for d, t in hits]}

serve(answer=answer, port=8000)
```

**Driving the bench.** `RagPlatformClient` registers that endpoint, uploads a
dataset, runs an experiment and fetches the result:

```python
from causa_rag_client import RagPlatformClient

client = RagPlatformClient("http://localhost:8081")
client.register_rag(name="my-rag", url="http://localhost:8000")
run = client.run_experiment(rag="my-rag", dataset="golden.v1")
print(client.get_results(run))
```

## The version number is the contract version

`causa-rag-client` is versioned by the **HTTP contract** it speaks, not by the
platform's own version. The contract is at version 1, so the client is at `1.x`
and will stay there while the contract holds:

- no field is removed or renamed;
- no optional field becomes required;
- the meaning of an existing field does not change;
- new fields are additive and optional.

`register_rag()` and `run_experiment()` check the platform's reported contract
version before sending anything and raise `ContractVersionMismatch` on a
mismatch, rather than sending a request the platform cannot parse.

## Not in Python?

The contract is ordinary JSON over HTTP and needs no client at all. The
field-by-field reference is
[docs/external-rag-contract.md](https://github.com/laputski/causa-rag/blob/main/docs/external-rag-contract.md);
adding two endpoints to a service you already run is the whole integration.

## Licence

Apache-2.0. Copyright 2026 Alexander Laputski.
