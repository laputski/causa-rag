# connector_quickstart

End-to-end example of `causa-rag-client` — register an external RAG, run an experiment, read results, without curl or hand-written FastAPI.

`toy_rag.py` is a fake RAG (two plain functions, no platform imports) standing in for your real retrieval/generation code.

## Run it

1. Start the platform (from the repo root): `make up` or `make bootstrap`.
2. Install the client with the `serve` extra: `pip install -e clients/python/[serve]`
3. In one terminal, expose `toy_rag.py` over the platform's contract:
   ```bash
   python3 -m examples.connector_quickstart.serve_app
   ```
4. In another terminal, register it and run an experiment:
   ```bash
   python3 -m examples.connector_quickstart.run_experiment
   ```

You should see the run start, poll to completion, and print aggregate metrics — the same metrics the platform computes for its own built-in pipeline (see the design notes "Golden-parity E2E").

## What each file maps to

- `toy_rag.py` — your RAG's `retrieve(query, top_k)` / `generate(query, sources)`.
- `serve_app.py` — `causa_rag_client.serve()`: turns those two functions into the platform's tier-1 HTTP contract.
- `run_experiment.py` — `causa_rag_client.RagPlatformClient`: `register_rag` → `upload_dataset` → `run_experiment` → `wait_for_completion`.
