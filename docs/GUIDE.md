# RAG Platform Developer & Agent Guide

## Purpose & Who This Is For

This is a **tool for iterative debugging and comparison of RAG configurations** on your own data, not a finished product with one hardcoded pipeline.

**Primary user**: an engineer/researcher configuring and validating a RAG system before handing it to end users. Not the chatbot's reader, but the person running experiments, inspecting traces and metrics, swapping one component (chunker, reranker, prompt) at a time, and comparing the result against the previous run.

**The pain this solves**: RAG systems are black boxes where silent degradation is easy to miss: stub embeddings instead of real ones, duplicate chunks in the index, an LLM's thinking mode eating the whole generation budget into an empty answer, BM25 ranking on stop-words. Without tracing and comparison tools, the only way to catch this is digging through logs after a user complains. Every feature in this platform (Pipeline Trace, MongoDB experiment history, run diagnostics) grew out of a real debugging incident; see [DEBUGGING.md](DEBUGGING.md).

This is why the architecture centers on a registry pattern (swappable chunker/embedder/retriever/reranker) and `ExperimentConfig` with a deterministic `config_hash`, so any combination of components is reproducible and comparable. `domain_packs/manuals/` is the worked consumer of that registry, and the demo realm runs on it; the platform's goal is debugging **any** RAG on **any** data.

---

## Architecture

```
Ingest:
  Corpus (.txt/.md)
      ↓ Chunker (fixed / structure_aware / sentence / paragraph)
      ↓ Embedder (BGE-M3 1024-dim, L2-normalized)
      ↓ Qdrant (dense) always; OpenSearch (BM25 sparse) and Neo4j (graph)
        if that pipeline is used / USE_GRAPH=true at ingest time

Query — `pipeline_id` selects ONE of these (not combinable, since graph does not
also run sparse/BM25):

  naive            Qdrant dense only
  hybrid_rrf       Qdrant dense + OpenSearch sparse  → Reciprocal Rank Fusion
  hybrid_weighted  Qdrant dense + OpenSearch sparse  → score = α·dense + (1-α)·sparse
  graph            Neo4j graph + Qdrant dense        → score = β·graph + (1-β)·dense
                     (β = graph_weight, hardcoded 0.4, NOT the merge_strategy
                     RRF/weighted knob, which only applies to dense+sparse)
  configurable     wraps any of the above + optional Reranker/Grounding/Routing
                     (set when reranker/grounding/route_policy is configured)

  → [Reranker (opt.)] → Dedup → Build Context → LLM (Ollama) → [Grounding (opt.)]
  → Answer + StageTrace + SourceRef[]

Query — external pipeline (same diagnostics where the external
system supports it), bypasses all of the above entirely:
  Question → core.sdk-wrapped Python pipeline (model B)
           | HTTP POST to a configured RAG service (model C, adapters/http_pipeline.py)
```

Which path runs is decided per-experiment by `ExperimentConfig`: `pipeline_id`
picks one of the five in-process pipelines above, or `pipeline_source="http"` +
`http_endpoint` bypasses the registry entirely for an external service (see
"External / Injectable RAG" below). GraphRAG is its own pipeline (`graph`),
not a variant of hybrid: it never touches OpenSearch.

**Components:**
- **Qdrant**: the vector store for semantic search (cosine similarity on 1024-dim BGE-M3 embeddings). Namespaced per `(strategy_id, embedder_id, corpus_id)`.
- **OpenSearch**: BM25 for keyword search (Russian/Belarusian morphological analyzer). Same namespacing.
- **Neo4j**: an optional graph store (`docker compose --profile graph`, extra `[graph]`); chunk-adjacency (`NEXT`) + two independent similarity signals built at ingest when `USE_GRAPH=true`: shared-keyword (`RELATED`, lexical) and embedding-cosine (`RELATED_SEMANTIC`, reuses the chunk's existing Qdrant vector via k-NN, no new computation), both traversed together by `Neo4jGraphRetriever.retrieve_graph()` for multi-hop retrieval (`pipeline_id=graph`). Kept as two separate relationship types deliberately, not merged, so lexical vs. semantic clustering can be compared.
- **BGE-M3**: the multilingual embedder, 1024 dimensions, L2-normalized. Must load real weights (`USE_REAL_BGE_M3=true`) or it silently returns random stub vectors (see DEBUGGING.md #1).
- **Ollama**: the local LLM server. Active model managed via `PUT /settings/model`.
- **MongoDB**: stores prompts, experiment runs, datasets, corpus ingest history, platform settings, saved external RAG endpoints.
- **Langfuse**: request tracing (trace_id per query, viewable at localhost:3001).

---

## Quick Start

`./install.sh` does all of the below and checks the result. Run the steps by
hand when you want to change one of them.

```bash
# 1. Start infrastructure
docker compose -f deploy/compose/docker-compose.yml up -d
ollama pull qwen3:8b

# 2. Ingest a corpus (path is positional, "ingest" is the subcommand).
#    USE_REAL_BGE_M3=true is not optional: without it the embedder returns
#    hash vectors and every similarity number downstream is noise (DEBUGGING.md #1).
USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ingest corpus/demo_handbook \
  --strategy structure_aware \
  --chunk-size 512 --overlap 64 \
  --corpus-id handbook --realm-id demo

# `--corpus-id handbook --realm-id demo` are what the demo realm expects; a
# corpus of your own takes its own pair. `--structure <pack>/<parser>` resolves
# a domain pack's structure parser through the registry when the corpus needs
# one (`manuals/manual_section` is the one shipped here); the demo handbook is
# markdown with headings, so the structure_aware chunker derives the path from
# those and no parser is needed. `--exclude full.txt` is the default already.
# USE_GRAPH=true additionally builds the Neo4j graph, which requires
# `docker compose --profile graph up -d neo4j` and the `[graph]` extra.

# 3. Start API Gateway
USE_REAL_BGE_M3=true uvicorn services.api_gateway.main:app --port 8081 --reload

# 4. Start UI
cd ui && npm run dev
```

---

## Chunking Strategies

| Strategy | When to use | Notes |
|----------|-------------|-------|
| `fixed` | Short docs, quick start | Predictable size; may split mid-sentence |
| `structure_aware` | Statutes and other hierarchical docs | Preserves section/article structure in `structural_path` |
| `sentence` | Natural prose, contracts | Semantically coherent chunks; depends on punctuation |
| `paragraph` | Documents with blank-line separators | Good granularity for article-per-paragraph docs |

**Structure parsers.** `structure_aware` derives `structural_path` from the
document's own headings, which is enough for markdown like the demo handbook.
A corpus whose units are marked some other way (numbered articles, a manual's
section codes) needs a parser that knows that convention, supplied by a domain
pack and selected with `--structure <pack_id>/<parser_id>`;
`domain_packs/manuals/structure_parser.py` is the worked example. When neither
applies, `doc.structure` stays unset and the chunker falls back to flat 512-char
windows with `structural_path="root"` regardless of the strategy name, which
looks like the strategy did nothing; see DEBUGGING.md #8.

**chunk_size = 512**: sized for BGE-M3, whose limit is 512 tokens.

---

## Configurable Pipeline

`ExperimentConfig` drives the pipeline: each optional step is resolved from the
registry and actually runs when configured. A/B "reranker off vs on" yields
different traces and metrics.

| Step | Components | Effect | Trace fields |
|------|-----------|--------|--------------|
| Reranker | `cross_encoder_local` (in-process), `cross_encoder` (HTTP), `cross_encoder_stub` | Re-orders top-k after merge | `rerank_ms`, `n_reranked` |
| Grounding | `token_overlap` | Flags unsupported claims vs context | `grounding_ms`, `n_unsupported` |
| Routing | `naive` | Classifies mode/question type (recorded in run metadata) | — |
| GraphRAG | retriever `graph_hybrid` → pipeline `graph` | Graph built at ingest (Neo4j), multi-hop Cypher traversal | `graph_ms`, `n_graph` |

- **Backward-compatible `config_hash`**: unset optional fields are excluded from
  the hash, so historical runs stay comparable; enabling a step changes the hash.
- **Optional extras**: `pip install '.[reranker]'` (sentence-transformers),
  `pip install '.[graph]'` (neo4j). Missing → component absent from the registry,
  platform still starts.
- **GraphRAG end-to-end**: `docker compose --profile graph up -d neo4j`, then ingest
  with `USE_GRAPH=true`. Inspect the graph via **Panels → Neo4j Browser** (`localhost:7474`).
- Form "New run" builds its reranker/grounding/routing/graph selects from `/registry`,
  so a newly registered adapter appears without frontend changes.

---

## Diagnostic Intelligence & Regression Guard

Every run is auto-compared against a pinned **baseline** so silent degradation
is visible in the UI without going through CI.

- **Pin a baseline:** `PUT /experiments/{id}/baseline`, or "Set as baseline" on
  the run page. Stored in MongoDB `settings.baseline_run_id`.
- **Auto-compare:** `GET /experiments/{id}` includes `regression`, a pass or fail
  per metric vs baseline with a 5% threshold (`core/eval/regression.py`, shared
  with `eval/gate.py` so CI and the gateway use one implementation).
- **5 silent-degradation detectors** (`core/eval/detectors.py`, deterministic,
  no LLM): stub embedder (near-zero/random dense scores), duplicate chunks in
  top-k, header-only chunks, BM25 dominance over dense, empty/"not found"
  answers. Surfaced as `diagnostics` in the run detail and rendered by
  `RunDiagnostics.tsx`.
- **Charts** on the run page (`RunCharts.tsx`, recharts): per-question metric
  bars, baseline-vs-current paired bars, and answer-length/source-count per
  question, built specifically to surface clusters of identical-length
  refusal boilerplate at a glance instead of grepping the JSON by hand.

---

## Corpus Observability & Multi-Corpus

`corpus_id` is a first-class namespace: `_collection_name`/`_index_name` in
`adapters/qdrant.py`/`adapters/opensearch.py` accept it, and `"default"` keeps
the original collection/index name unchanged (no migration needed for
existing data). `ExperimentConfig.corpus_id` participates in `config_hash`
only when non-default, so historical runs stay comparable.

- **Content browser:** Data → Corpus → "Content": paginated real chunks
  via `GET /corpus/{id}/chunks` (`adapters/qdrant.py` `scroll`/`scroll_all`),
  with `structural_path`, length, and header-only/duplicate flags. Full chunk
  text, no truncation.
- **Health check:** Data → Corpus → "Health": `GET /corpus/{id}/health`
  runs `core/eval/corpus_health.py` (pure, deterministic, no LLM) over the
  fetched chunks:
  | Detector | Catches |
  |---|---|
  | `duplicates` | Exact-text duplicate chunks (non-deterministic `chunk_id`) |
  | `header_only` | >30% of chunks are near-empty (just a structural path) |
  | `too_short` | Average chunk length <50 chars |
  | `duplicate_structural_numbers` | Same numbered unit (e.g. an article 47) used by two different nodes: a source-export collision, *or* normal if multiple documents share one `corpus_id` |
  | `missing_structural_numbers` | Gap in sequential numbering: an article whose heading the parser could not't recognize, or a file dropped during export (see DEBUGGING.md #10) |
- Ingest multiple corpora side by side: pass `--corpus-id` to the CLI or set
  "Corpus ID" in the upload form; pick which one to query via the corpus
  selector on Chat/New Run pages.

---

## External / Injectable RAG

The platform can diagnose a RAG it didn't build, two ways:

**Model B, in-process, via `core/sdk.py`.** If you have the external
system's `retrieve()`/`generate()` as plain Python functions (even with a
loose, non-standard return shape), wrap them:

```python
from core.sdk import wrap_retriever, wrap_generator, build_pipeline

retriever = wrap_retriever(lambda query, k, filters=None: my_rag.search(query, k))
generator = wrap_generator(my_rag.answer)
pipeline = build_pipeline(retriever, generator)
```

`wrap_retriever` accepts `ScoredChunk` objects, `(text, score)` tuples, or
dicts with `text`/`score`/`doc_id`/`structural_path` keys, whichever shape
needs the least change to the wrapped system. The resulting `Pipeline`
produces a full `StageTrace`, same as an in-process pipeline. See
`examples/instrument_external/` for a complete, runnable example (a
standalone demo RAG with zero platform imports + a 3-line adapter).

**Model C, by URL, via `adapters/http_pipeline.HttpPipeline`.** For a
system running as its own service. Manage saved endpoints on **External
RAG** (`/external-rags`): add a name and URL once, then pick it from a
dropdown on **New run** instead of retyping the URL every time. That
page also documents the API contract in full; the short version:

| Direction | Shape |
|---|---|
| Platform → external (POST body) | `{"query": str, "top_k": int, "filters": dict, "trace_id": str}` (+ header `X-Trace-Id`) |
| External → platform (response, HTTP 200) | `{"answer": str, "trace": {...}}`, where `trace` and everything inside it is optional |

If the response includes `trace.stage_trace`/`trace.sources` (an
`ExternalTrace`, `core/models.py`), the run gets the same per-stage
diagnostics as an in-process pipeline. If not, diagnostics degrade to
outcome-only (eval + regression-guard on the text alone) rather than
erroring, and the run page shows a **white-box / black-box** badge so it's
always visible which depth was actually achieved. Egress is restricted to
`RAG_HTTP_ALLOWLIST` (comma-separated URL prefixes), checked before any
network call.

Both models participate in `config_hash` via `ExperimentConfig.pipeline_source`
(`"in_process"` | `"http"`) + `http_endpoint`, following the same
backward-compatible pattern as `corpus_id`: configs
created before those fields existed keep their original hash.

---

## Understanding Metrics

As of Eval Measurement Trustworthiness (Phase 0), runs are scored by
`services/api_gateway/routers/experiments.py:_CompositeEvaluator`. The
earlier Jaccard token-overlap metrics (`faithfulness`/`answer_relevancy`/
`reference_overlap`) are **legacy**, kept only so historical runs still
render, never computed for new runs. They were structurally incapable of
producing a meaningful score: `faithfulness` was capped near ~0.05
regardless of answer quality because the denominator was the entire
retrieved context (thousands of tokens) against a ~50-token answer. If you
see those three keys on a run, it predates Phase 0 and isn't comparable to
anything scored since.

| Metric | Measures | Computed on |
|--------|----------|-------------|
| `correct_refusal` | Did the system correctly decide to answer vs. refuse | All questions |
| `retrieval_recall_at_k` | Fraction of ground-truth `article_refs` found in top-k | `answerable` questions only |
| `retrieval_precision_at_k` | Fraction of top-k that are relevant (deduped by article) | `answerable` questions only |
| `answer_similarity` | Cosine similarity, answer vs. reference embedding | `answerable` questions only |
| `context_support` | Max (not mean) cosine similarity, answer vs. any retrieved chunk | `answerable` questions only |
| `grounded_in_correct_source` | Cosine similarity, answer vs. specifically the chunk(s) matching `article_refs` (not the max over all chunks) | `answerable`, only when recall_at_k found a matching chunk |
| `pre_rerank_recall_at_k` | Recall@k before the reranker cut the list | `answerable`, only when a reranker is configured |
| `retrieval_average_precision` | Ranking-aware retrieval precision (position matters, not just presence) | `answerable` questions only |
| `answer_relevance` | Cosine similarity, answer vs. the question itself (not the reference), answering the *right* question, independent of correctness | `answerable` questions only |
| `citation_number_coverage` | Fraction of structural numbers (from retrieved chunks' labels) that literally appear in the answer text | `answerable`, only when retrieved chunks carry a `structural_path` label |

`citation_number_coverage` divides by **all** distinct numbers among the
top-k retrieved chunks' labels, not just the one the question actually
needed: at `top_k=10` this caps a perfectly-correct single-citation answer
near `1/8`, since ~8 distinct article numbers are typically present in the
retrieved set. A low value alone doesn't mean the citation is wrong; check
RunPage's retrieval panel against the actual cited text (see "Citation
generation" below) before concluding generation regressed.

A question is `answerable` only if every one of its `article_refs` resolves
to a real file in the ingested corpus (`core/eval/answerability.py`):
otherwise it's `uncovered` (a real corpus gap) or `out_of_scope` (no
article_refs at all, e.g. negative-rejection probes); both feed
`correct_refusal` but not the retrieval/answer metrics, since there's no
ground truth to score them against.

### Why no LLM-as-judge by default

Most RAG eval tooling (RAGAS, DeepEval's default metrics, TruLens) scores
faithfulness/relevancy by asking an LLM to judge the answer, which is
non-deterministic and adds a second model's failure modes on top of the
one being evaluated. This platform's default path uses **no LLM judge**:
`retrieval_recall_at_k`/`retrieval_precision_at_k` are exact-match against
ground-truth `article_refs`, and `answer_similarity`/`context_support` are
plain embedding cosine, both fully deterministic given a fixed embedder.
`eval/deepeval_runner.py` (LLM-judge via Ollama) exists as an **optional**,
separately-invoked deeper check (`make test-eval`), not the default gate:
treat its output as a second opinion to spot-check against, never as the
metric a regression-guard decision should hinge on.

### Citation generation: position rather than transcription

The model never writes the real article/section number into its answer.
The active prompt only allows it to reference a fragment by position, as
`(Fragment N)`, N being the fragment's index in the context list shown in the
prompt. The demo realm's prompt (`tools/seed_demo.py`) is written this way, and
`tests/unit/test_demo_realm.py` checks it stays that way. After generation,
`core/citation.py:substitute_fragment_markers()` (called from every
`pipeline.run()`) deterministically replaces that marker with the real
`structural_path` label of `source_refs[N-1]` before the answer is returned
so users never see the literal "Fragment N" text. This exists because three
earlier prompt iterations asking the model to transcribe the number itself
all failed at scale (see `docs/DEBUGGING.md` #11): a content-to-citation
*binding* problem, not a knowledge problem, that wording alone can't fix.

`Answer.computed_citations` (also populated by `core/pipeline.py` on every
run, surfaced in RunPage under each generated answer) is the same
deterministic mapping, useful for a quick sanity check independent of
whatever marker the model actually wrote.

---

## Pipeline Trace (StageTrace)

Each query response includes `stage_trace`:

```json
{
  "embed_ms": 12.3,
  "dense_retrieve_ms": 45.1,
  "sparse_retrieve_ms": 38.7,
  "merge_ms": 2.1,
  "generate_ms": 2100.0,
  "total_ms": 2198.2,
  "n_dense": 10,
  "n_sparse": 10,
  "n_merged": 7,
  "n_deduped": 7,
  "input_tokens": 1842,
  "output_tokens": 187,
  "context_chars": 3210
}
```

The configurable pipeline added `rerank_ms`/`n_reranked`, `grounding_ms`/`n_unsupported`,
`graph_ms`/`n_graph` (all `0`/`0.0` when that step is disabled; see
`core/models.py` `StageTrace` for the full field list). Visible in Chat UI
under "⏱ Pipeline trace" and available via `GET /trace/latest`.

---

## API Reference (new endpoints)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/models` | List available Ollama models |
| GET | `/settings` | Get active model, embedder mode, retriever mode |
| PUT | `/settings/model` | `{"model": "qwen3:8b"}`, switching the active model |
| GET | `/corpus` | List corpus ingest history |
| POST | `/corpus/ingest` | Multipart upload + ingest (strategy, chunk_size, overlap, corpus_id) |
| DELETE | `/corpus/{job_id}` | Remove ingest record |
| WS | `/corpus/progress/{job_id}` | WebSocket progress stream |
| GET | `/corpus/{corpus_id}/chunks` | Paginated real indexed chunks |
| GET | `/corpus/{corpus_id}/health` | Deterministic corpus health detectors |
| GET | `/trace/latest` | Latest query stage trace |
| GET | `/trace/{trace_id}` | Specific trace by ID |
| PUT | `/experiments/{id}/baseline` | Pin this run as the regression baseline |
| GET | `/external-rags` | List saved external RAG endpoints |
| POST | `/external-rags` | Save a new endpoint `{name, url, description, headers}` |
| DELETE | `/external-rags/{id}` | Remove a saved endpoint |
| POST | `/external-rags/{id}/test` | Probe it with one real query; reports `has_trace`/`n_sources` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_REAL_BGE_M3` | `false` | Load real BGE-M3 weights. Without it, embeddings are random stub vectors; see DEBUGGING.md #1 |
| `USE_REAL_QDRANT` | `true` | Gateway connects to real Qdrant; falls back to `QdrantRetrieverStub` if unreachable |
| `USE_GRAPH` | `false` | CLI ingest also builds the Neo4j chunk graph (needs `[graph]` extra + reachable Neo4j) |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant connection (not a combined URL) |
| `OPENSEARCH_HOST` / `OPENSEARCH_PORT` | `localhost` / `9200` | OpenSearch connection |
| `QDRANT_STRATEGY` | `structure_aware` | Chunking strategy the gateway's dense retriever is bound to at startup |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `ragplatform` | Neo4j credentials |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (not `OLLAMA_URL`) |
| `OLLAMA_MODEL` | `qwen3:8b` | Default generation model |
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | `ragplatform` | MongoDB database name |
| `RAG_HTTP_ALLOWLIST` | unset (= unrestricted) | Comma-separated URL prefixes `HttpPipeline` may call; empty means no restriction |
| `LANGFUSE_ENABLED` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_HOST` | — | Langfuse tracing (optional) |
| `DEEPEVAL_URL` | — | DeepEval service for `make test-eval` (LLM-judge semantic eval) |
