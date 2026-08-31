"""Run the configuration grid over MIRACL and write the report's numbers.

Retrieval only, on every row. MIRACL ships relevance judgements and no
reference answers, so there is nothing for a generated answer to be scored
against, and the generator registered here raises if anything calls it.
That is not a safety net around the run so much as the run's own proof:
a report that quietly generated 46 000 answers would have cost a night and
measured nothing extra.

The axes are top-k, retrieval mode, and the reranker. Chunk size is fixed
and deliberately absent from the grid: a MIRACL passage is a few hundred
characters, a chunk larger than the passage cuts nothing, and the axis only
exists below that length.

Usage:
    python3 -m eval.miracl.report --limit 200          # smoke run first
    python3 -m eval.miracl.report                      # the real thing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.registry import ComponentRegistry
from eval.dataset import EvalDataset

LANGUAGES = ("ar", "ru", "en")
TOP_K = (5, 10, 20)
MODES = ("naive", "hybrid_rrf")
RERANKER = "cross_encoder_local"

# The platform's default local cross-encoder is ms-marco-MiniLM, which is
# trained on English only. Running it over Arabic and Russian would score
# those rows on noise, and the report would read "the reranker hurts
# outside English" when the finding is "this reranker only speaks English".
# bge-reranker-v2-m3 is multilingual and pairs with the BGE-M3 embeddings
# already in use. Overridable, because that is a claim the report should be
# able to re-test rather than assume.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# The candidate window, held constant across the whole grid so that
# "reranker on" and "reranker off" see the same fifty passages and differ
# only in the order they put them in. Letting fetch_k follow top_k would
# move two things at once and leave the reranker's own effect unreadable.
FETCH_K = 50

# Above the 99th percentile of passage length in all three languages
# (ar 2553, ru 2375, en 2249 characters), so nearly every passage is one
# chunk. The long tail still splits, which is harmless: chunks of one
# passage carry one ref-id and the metrics collapse them.
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 0

_RESULTS = Path("eval/results/miracl")
# Derived, large, and rebuildable from the corpus. Kept out of the
# repository while the rows themselves stay in it.
_CACHE = _RESULTS / "cache"


class _RefusingGenerator:
    """Registered where the generator goes, and raises if reached.

    The report's claim is that it never generated. An unused stub would
    make that claim unfalsifiable; this makes the run fail loudly the first
    time it stops being true.
    """

    generator_id = "never_called"

    def generate(self, prompt: str) -> str:
        raise AssertionError(
            "the generator was called during a retrieval-only report run"
        )


# Written after this many freshly scored candidate sets. Small enough that
# an interrupted run loses minutes, large enough that the writing itself is
# not part of the measurement.
_CACHE_EVERY = 50


class _MemoisingReranker:
    """The reranker, with its answers remembered — within a run and across runs.

    Two savings, one mechanism.

    Within a run: the pipeline hands the reranker the whole candidate window
    and cuts the result itself (`rerank(...)[:k]` in core/pipeline.py), so
    top-k 5, 10 and 20 ask it the identical question and differ only in where
    they slice the identical answer. Scoring the same fifty passages three
    times is two thirds of the grid's running time for no extra measurement.

    Across runs: measured on real passages, one Arabic rerank takes about
    four seconds, so a language is hours of work concentrated in the first
    reranked configuration — the other two read that one's answers. Without
    a cache on disk, "resume" means losing every hour spent inside whichever
    configuration was running, which is not resuming.

    Not a shortcut around what is measured. The pipeline still calls rerank()
    on every question, and what comes back is what the wrapped reranker
    returned for that exact query and candidate set, scores included; a
    cross-encoder in eval mode is a pure function of its inputs. The key
    carries the model name, so changing the reranker invalidates every entry
    rather than silently replaying the old model's ranking.
    """

    def __init__(self, inner: Any, model_name: str = "", cache_path: Path | None = None) -> None:
        self._inner = inner
        self._model_name = model_name or getattr(inner, "reranker_id", "")
        self.reranker_id = getattr(inner, "reranker_id", "reranker")
        self._cache: dict[str, list[tuple[str, float]]] = {}
        self._path = cache_path
        self._unsaved = 0
        self.hits = 0
        self.misses = 0
        self.loaded = 0
        if cache_path is not None and cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            self._cache = {k: [(i, float(sc)) for i, sc in v] for k, v in raw.items()}
            self.loaded = len(self._cache)

    def _key(self, query: str, candidates: list[Any]) -> str:
        h = hashlib.sha1()
        h.update(self._model_name.encode("utf-8"))
        h.update(b"\0")
        h.update(query.encode("utf-8"))
        for c in candidates:
            h.update(b"\0")
            h.update(str(c.chunk.chunk_id).encode("utf-8"))
        return h.hexdigest()

    def save(self) -> None:
        if self._path is None or not self._unsaved:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self._path)
        self._unsaved = 0

    def rerank(self, query: str, candidates: list[Any]) -> list[Any]:
        if not candidates:
            return []
        key = self._key(query, candidates)
        scored = self._cache.get(key)
        if scored is None:
            self.misses += 1
            ranked = self._inner.rerank(query, candidates)
            scored = [(c.chunk.chunk_id, c.score) for c in ranked]
            self._cache[key] = scored
            self._unsaved += 1
            if self._unsaved >= _CACHE_EVERY:
                self.save()
        else:
            self.hits += 1

        from core.models import ScoredChunk
        by_id = {c.chunk.chunk_id: c.chunk for c in candidates}
        return [
            ScoredChunk(chunk=by_id[cid], score=score, retriever_id=self.reranker_id)
            for cid, score in scored
            if cid in by_id
        ]


@dataclass
class Row:
    language: str
    mode: str
    top_k: int
    reranker: bool
    reranker_model: str
    n_questions: int
    # Questions the run could not answer at all. Averages are taken over
    # what succeeded, so a row with failures is a mean of a subset wearing
    # the whole set's name unless the count travels with it.
    n_failed: int
    seconds: float
    metrics: dict[str, float]


def _build_registry(
    lang: str, embedder: Any, generator: Any, reranker_model: str = RERANKER_MODEL,
    use_cache: bool = True,
) -> ComponentRegistry:
    """One registry per language, because the analyser is per language.

    An OpenSearch index carries its analyser for life. Building the sparse
    retriever with the language of the corpus it is about to query is what
    keeps Arabic from being stemmed by Russian rules, and the runner's
    corpus rebind carries the choice through.
    """
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.chunking.fixed import FixedChunkingStrategy
    from core.pipeline import NaivePipeline
    from core.retrieval.hybrid import HybridRetriever

    corpus_id = f"miracl-{lang}"
    dense = QdrantRetriever(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        strategy_id="fixed",
        embedder_id=embedder.embedder_id,
        corpus_id=corpus_id,
    )
    sparse = OpenSearchRetriever(
        host=os.getenv("OPENSEARCH_HOST", "localhost"),
        port=int(os.getenv("OPENSEARCH_PORT", "9200")),
        strategy_id="fixed",
        corpus_id=corpus_id,
        language=lang,
    )
    hybrid = HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, embedder=embedder, merge="rrf",
    )

    registry = ComponentRegistry()
    registry.register("embedder", embedder.embedder_id, embedder)
    registry.register("generator", generator.generator_id, generator)
    registry.register("chunker", FixedChunkingStrategy.strategy_id, FixedChunkingStrategy())
    registry.register("pipeline", "naive", NaivePipeline(
        retriever=dense, embedder=embedder, generator=generator, pipeline_id="naive",
    ))
    registry.register("pipeline", "hybrid_rrf", NaivePipeline(
        retriever=hybrid, embedder=embedder, generator=generator, pipeline_id="hybrid_rrf",
    ))

    from adapters.reranker import CrossEncoderRerankerLocal
    registry.register(
        "reranker", RERANKER,
        _MemoisingReranker(
            CrossEncoderRerankerLocal(model_name=reranker_model),
            model_name=reranker_model,
            cache_path=(_CACHE / f"rerank-{lang}.json") if use_cache else None,
        ),
    )
    return registry


def indexed_count(lang: str, embedder_id: str) -> int | None:
    """How many chunks the dense index holds for this language, or None.

    Checked before the grid starts. A corpus that was never ingested does
    not fail: every query returns nothing, every metric averages to zero,
    and twelve rows of zeros land in the table looking like a finding about
    retrieval rather than about an empty collection.

    Three states, not two. None means the server could not be reached at
    all, which is a different problem from an empty collection and needs a
    different sentence said to the person running this. Returning 0 for it
    would send them to re-run an ingest that was never the trouble.
    """
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name

    name = _collection_name("fixed", embedder_id, f"miracl-{lang}", None)
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=10,
            check_compatibility=False,
        )
        if not client.collection_exists(name):
            return 0
        return int(client.count(collection_name=name).count)
    except Exception:
        return None


def _config(lang: str, mode: str, k: int, reranker: bool, embedder_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"miracl-{lang}-{mode}-k{k}" + ("-rerank" if reranker else ""),
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id=embedder_id),
        generator=ComponentRef(kind="generator", component_id="never_called"),
        pipeline_id=mode,
        corpus_id=f"miracl-{lang}",
        top_k=k,
        fetch_k=FETCH_K,
        retrieval_only=True,
        reranker=ComponentRef(kind="reranker", component_id=RERANKER) if reranker else None,
    )


_LOOP: Any = None


def _await(coro: Any) -> Any:
    """One event loop for the whole process.

    The Mongo client binds to the loop it is first used in, so a fresh
    asyncio.run() per save works once and then fails for every run after
    it. That failure is invisible from the outside: the gateway's _save
    keeps a file copy and swallows the database error on purpose, so the
    log says the run was filed while the UI never sees it.
    """
    import asyncio

    global _LOOP
    if _LOOP is None:
        _LOOP = asyncio.new_event_loop()
    return _LOOP.run_until_complete(coro)


DEMO_REALM = "demo"


def _persist(result: Any, log: Callable[[str], None], realm_id: str = DEMO_REALM) -> None:
    """File a finished run where the UI can show it, under the demo Realm.

    Two deliberate choices.

    The Realm is set on the result after the run rather than passed into it.
    Handing a realm_id to the runner rebinds the retriever, and the index
    names would change to a Realm that holds none of this data. What is
    wanted here is where the run is filed, not where it read from.

    The configuration's name gains the question count, because a screenshot
    of a fifty-question slice must not be mistakable for the published row
    it is named after. The published rows cover the whole dev split, and
    nothing else in the interface says which of the two this is.
    """
    import adapters.mongodb as mdb
    from services.api_gateway.routers.experiments import _save

    result.realm_id = realm_id
    result.config.name = f"{result.config.name}-demo{result.n_questions}"
    result.run_id = f"{result.config.name}_{result.config.config_hash}"

    _await(_save(result))
    # Read it back rather than trusting the write. _save swallows database
    # errors by design, having kept a file copy, so "saved" on its own says
    # only that nothing crashed.
    try:
        stored = _await(mdb.find_one("experiment_runs", {"run_id": result.run_id}))
    except Exception as exc:
        # Most often MongoDB is simply not running, which arrives as a
        # thirty-second server-selection timeout and then a traceback. The
        # measurement itself is already on disk under eval/results/runs, so
        # this is one line about the copy the UI reads, not a failure of
        # the run.
        log(f"    {result.run_id} could not be filed: {type(exc).__name__}. "
            "The run itself is written under eval/results/runs. Is MongoDB up?")
        return
    if stored is None:
        log(f"    {result.run_id} did not reach MongoDB; the UI will not show it")
    else:
        log(f"    filed as {result.run_id} in Realm {realm_id}")


def run_language(
    lang: str, limit: int | None, use_reranker: bool, log: Callable[[str], None],
    reranker_model: str = RERANKER_MODEL, resume: bool = True,
    persist: bool = False, realm_id: str = DEMO_REALM,
) -> list[Row]:
    # Imported rather than reimplemented. A second copy of the metric
    # definitions would drift from the platform's own, and the report would
    # then describe a scoring nobody else uses.
    from adapters.bge_m3 import BgeM3Embedder
    from services.api_gateway.routers.experiments import _CompositeEvaluator

    embedder = BgeM3Embedder(use_real_model=True)
    generator = _RefusingGenerator()
    registry = _build_registry(lang, embedder, generator, reranker_model, use_cache=resume)
    runner = ExperimentRunner(registry=registry)

    dataset = EvalDataset.from_jsonl(Path(f"eval/golden/miracl-{lang}.v1.fast.jsonl"))
    if limit is not None:
        dataset.questions = dataset.questions[:limit]

    # Rows this language already has, from an earlier run that was stopped.
    # Matched on the axes rather than on the name, so a renamed
    # configuration is recomputed instead of silently reused.
    done: dict[tuple[str, int, bool], Row] = {}
    out = _RESULTS / f"{lang}.json"
    # A persisting run is producing screenshots over a slice of the
    # questions, so it must neither skip a configuration because the
    # published grid already has it nor overwrite that grid with rows
    # measured over fifty questions.
    if resume and not persist and out.exists():
        for raw in json.loads(out.read_text(encoding="utf-8")):
            row = Row(**raw)
            done[(row.mode, row.top_k, row.reranker)] = row
        if done:
            log(f"  {len(done)} configurations already measured, keeping them")

    rows: list[Row] = []
    rerankers = (False, True) if use_reranker else (False,)
    for mode in MODES:
        for k in TOP_K:
            for rr in rerankers:
                if (mode, k, rr) in done:
                    rows.append(done[(mode, k, rr)])
                    continue
                config = _config(lang, mode, k, rr, embedder.embedder_id)
                log(f"  {config.name}: {len(dataset.questions)} questions")
                t0 = time.perf_counter()
                result = runner.run(config, dataset, evaluator=_CompositeEvaluator(embedder, top_k=k))
                elapsed = round(time.perf_counter() - t0, 1)

                if persist:
                    _persist(result, log, realm_id)

                errors = [q for q in result.question_results if q.error]
                if errors:
                    log(f"    {len(errors)} questions failed, first: {errors[0].error}")
                rows.append(Row(
                    language=lang, mode=mode, top_k=k, reranker=rr,
                    reranker_model=reranker_model if rr else "",
                    n_questions=len(result.question_results), n_failed=len(errors),
                    seconds=elapsed,
                    metrics={k2: round(v, 4) for k2, v in result.aggregate_metrics.items()},
                ))
                log(f"    recall@k {rows[-1].metrics.get('retrieval_recall_at_k', float('nan')):.3f}"
                    f"  in {elapsed}s")

                if not persist:
                    # After every configuration, not at the end of the
                    # language. A stopped run keeps what it finished.
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    memo = registry.resolve("reranker", RERANKER)
    if isinstance(memo, _MemoisingReranker):
        memo.save()
        if memo.hits or memo.loaded:
            log(f"  reranker scored {memo.misses} candidate sets, reused {memo.hits}, "
                f"{memo.loaded} came from the previous run")
    return rows


def to_markdown(rows: list[Row]) -> str:
    out = [
        "| Language | Mode | top-k | Reranker | Recall@k | Precision@k | MAP | Questions | Failed |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    def cell(metrics: dict[str, float], key: str) -> str:
        return f"{metrics[key]:.3f}" if key in metrics else "not measured"

    for r in rows:
        m = r.metrics
        out.append(
            f"| {r.language} | {r.mode} | {r.top_k} | {'on' if r.reranker else 'off'} "
            f"| {cell(m, 'retrieval_recall_at_k')} | {cell(m, 'retrieval_precision_at_k')} "
            f"| {cell(m, 'retrieval_average_precision')} | {r.n_questions} | {r.n_failed} |"
        )
    return "\n".join(out)


DOCUMENT = Path("docs/reports/miracl-configuration-report.md")
# The one part of the report a person writes. Kept in its own file because
# the document is regenerated by every run, and prose that lives inside a
# generated file survives exactly until the next one.
READING = Path("docs/reports/miracl-reading.md")


def _absent(rows: list[Row]) -> str:
    """Which cells of the intended grid this document does not contain.

    A row that is present but wrong is guarded everywhere in here. A row
    that is simply missing is invisible: the table looks complete, and
    "Arabic was never reranked" reads exactly like "Arabic has no reranked
    row because it did not help".
    """
    have = {(r.language, r.mode, r.top_k, r.reranker) for r in rows}
    langs = sorted({r.language for r in rows})
    missing = [
        f"{lang} {mode} k={k} reranker {'on' if rr else 'off'}"
        for lang in langs for mode in MODES for k in TOP_K for rr in (False, True)
        if (lang, mode, k, rr) not in have
    ]
    if not missing:
        return ""
    return (
        "\n**Not measured yet**, and absent rather than zero: "
        + ", ".join(missing)
        + ". Running the same command again fills them in.\n"
    )


def _reading() -> str:
    """What a person concluded from the table, if anyone has written it yet.

    Absent by default and absent in the document when absent on disk, rather
    than standing in with a placeholder. A heading with nothing under it
    reads as a finished thought that says nothing.
    """
    if not READING.exists():
        return ""
    text = READING.read_text(encoding="utf-8").strip()
    return f"## Reading\n\n{text}\n\n" if text else ""


def to_document(rows: list[Row], limit: int | None) -> str:
    """The report itself, assembled from the run rather than written around it.

    The three caveats lead. They are what separates a measurement from a
    claim, and a reader who meets them in a footnote has already formed the
    wrong impression of the table above it.
    """
    langs = sorted({r.language for r in rows})
    n = {r.language: r.n_questions for r in rows}
    rerank_models = sorted({r.reranker_model for r in rows if r.reranker_model})
    scope = (f"the first {limit} questions of each language"
             if limit is not None else "every question in the dev split")

    return f"""# Configuration Report: retrieval on MIRACL

What changes when you change a RAG's retrieval, measured on a public
multilingual benchmark rather than on a demo corpus. Languages: {', '.join(langs)}.
Questions: {', '.join(f'{lang} {n[lang]}' for lang in langs)}, {scope}.

## Why this replaced the demo corpus

Every number this platform had published came from a synthetic handbook of
eight documents and fifteen questions. As evidence it fell apart the moment
anyone opened the file.

Half of it was also wrong, and that surfaced only while preparing this
report. **The demo corpus is English, and its sparse index was built with a
Russian analyser** — the platform hardcoded one analyser for every corpus,
whatever language it held. Checked against the running index:
`approving purchases requires evidence` tokenises to
`approving · purchases · requires · evidence`, where an English analyser
gives `approv · purchas · requir · evid`. A query for "approve purchase"
did not match a document saying "approving purchases".

Dense retrieval was unaffected, since the embedder does not consult an
analyser. BM25 and every hybrid configuration on that corpus were measuring
something broken. It is recorded here rather than quietly fixed because a
measurement platform that cannot say where its own numbers came from has
nothing to offer anyone else's.

## Read these three things first

**The pool is the judged passages, not the whole corpus.** Arabic MIRACL
holds 2.06 million passages; the roughly 26 thousand a native speaker
judged for the dev split are what is indexed here, positives and
rejections together. This raises every absolute number and narrows the gaps
between configurations. Two rows that tie here are not shown to tie on the
full corpus.

**Only the dev split.** MIRACL keeps the judgements for test-a and test-b
private, so there is nothing to score against there.

**No generation is measured, in any language.** MIRACL ships relevance
judgements and no reference answers. Every row below stopped before the
model was asked anything, enforced by a generator component that raises if
called. Answer quality needs a benchmark that carries reference answers,
which is a separate piece of work.

## Conditions

Identical for every row unless the row says otherwise.

- Embeddings: BGE-M3, run locally
- Chunking: fixed, {CHUNK_SIZE} characters, no overlap. Above the 99th
  percentile of passage length in all three languages, so nearly every
  passage is a single chunk and chunk size is not an axis
- Candidate window: {FETCH_K} for every row, so the reranked and unreranked
  halves see the same passages and differ only in ordering
- Sparse index: one OpenSearch analyser per language, not one for all
- Reranker, where on: {', '.join(rerank_models) or 'none run'}
- Everything ran on one machine, by hand. None of it runs in CI

## Results

{to_markdown(rows)}

{_absent(rows)}
Numbers are means over the questions that produced them. A cell reading
"not measured" is a metric this run had no basis to compute, left empty
rather than filled with a zero. The last column counts questions the run
could not answer at all; a row with failures is an average over a subset,
and the count is printed rather than folded into the mean.

{_reading()}## Reproducing it

```
make miracl-fetch
make miracl-ingest
make miracl-report
```

Data: MIRACL, Zhang et al., TACL 2023, arXiv:2210.09984, Apache-2.0.
Passage text is Wikipedia under CC BY-SA 3.0 and is not redistributed;
the answer key travels with this repository under `eval/golden/`.
Provenance and the fetcher's own caveats: `eval/miracl/README.md`.
"""


def _write_outputs(rows: list[Row], limit: int | None, log: Callable[[str], None]) -> None:
    rows = sorted(rows, key=lambda r: (r.language, r.mode, r.top_k, r.reranker))
    table = _RESULTS / "table.md"
    table.write_text(to_markdown(rows) + "\n", encoding="utf-8")
    log(f"wrote {table}")

    DOCUMENT.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT.write_text(to_document(rows, limit), encoding="utf-8")
    log(f"wrote {DOCUMENT}")
    if not READING.exists():
        log(f"the reading of these numbers goes in {READING}, which does not "
            "exist yet; it is kept apart so this command cannot overwrite it")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="*", default=list(LANGUAGES))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N questions per language — use this for a smoke run")
    ap.add_argument("--no-reranker", action="store_true",
                    help="skip the reranked half of the grid; it is the slow half")
    ap.add_argument("--persist", action="store_true",
                    help="file runs in MongoDB under Realm demo so the UI can show them, "
                         "and publish nothing. Requires --limit: one run of a full dev "
                         "split is 62 to 225 MB depending on the language, against "
                         "MongoDB's 16 MB cap on a single document")
    ap.add_argument("--realm", default=DEMO_REALM,
                    help=f"Realm the persisted runs are filed under (default: {DEMO_REALM})")
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild the table and document from the measurements already "
                         "on disk, measuring nothing")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore rows and reranker scores kept from an earlier run")
    ap.add_argument("--reranker-model", default=RERANKER_MODEL,
                    help=f"cross-encoder to rerank with (default: {RERANKER_MODEL}, "
                         "multilingual; the platform's own default is English-only)")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    _RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows: list[Row] = []

    if args.persist:
        if args.limit is None:
            log("--persist needs --limit. A run over a full dev split is 62 to 225 MB "
                "and MongoDB caps a document at 16 MB, so it would fail on the way in.")
            return 2
        for lang in args.languages:
            log(f"{lang}:")
            run_language(lang, args.limit, not args.no_reranker, log,
                         args.reranker_model, resume=True, persist=True,
                         realm_id=args.realm)
        log(f"filed in Realm {args.realm}. Nothing published: the grid and the document are "
            "measured over the whole split and these runs are not.")
        return 0

    if args.regenerate:
        # For after the reading is written, and for the hours a long run
        # leaves the document behind its own measurements: the repository
        # should not sit in a state its own consistency check rejects.
        for path in sorted(_RESULTS.glob("*.json")):
            all_rows.extend(Row(**raw) for raw in json.loads(path.read_text(encoding="utf-8")))
        if not all_rows:
            log("nothing measured yet, so there is nothing to rebuild")
            return 2
        _write_outputs(all_rows, args.limit, log)
        return 0

    for lang in args.languages:
        golden = Path(f"eval/golden/miracl-{lang}.v1.fast.jsonl")
        if not golden.exists():
            log(f"{lang}: no golden set. Run: python3 -m eval.miracl.fetch --lang {lang}")
            return 2
        log(f"{lang}:")
        indexed = indexed_count(lang, "bge_m3")
        if indexed is None:
            log(f"{lang}: Qdrant is not answering on "
                f"{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}. "
                "Bring the infrastructure up first: make up")
            return 2
        if indexed == 0:
            log(f"{lang}: nothing indexed. Run: make miracl-ingest")
            return 2
        expected = sum(1 for _ in Path(f"corpus/miracl-{lang}").rglob("*.txt"))
        if indexed < expected:
            # Partial is worse than absent: the run completes, the numbers
            # look ordinary, and nothing in the table says a fifth of the
            # pool was missing from it.
            log(f"{lang}: {indexed} chunks indexed but {expected} passages on disk. "
                "Finish the ingest before measuring, or the pool is not the one "
                "the report describes.")
            return 2
        rows = run_language(
            lang, args.limit, not args.no_reranker, log, args.reranker_model,
            resume=not args.fresh, persist=args.persist,
        )
        all_rows.extend(rows)
        log(f"  wrote {_RESULTS / f'{lang}.json'}")

    _write_outputs(all_rows, args.limit, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
