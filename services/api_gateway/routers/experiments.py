"""Experiments REST router + WebSocket progress stream."""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# One definition of the refusal-phrase pattern, in core/eval/detectors.py. This
# file used to keep its own copy, and the two had drifted: the copy here matched
# four phrasings where the original matched eight, so the same answer counted as
# a refusal in the diagnostics panel and not in the funnel diagnosis beside it.
from core.eval.detectors import NOT_FOUND_RE as _NOT_FOUND
from core.eval.regression import PairedDiffReport
from core.experiment.compare import CompatWarning, check_comparability
from core.experiment.compare import compare as do_compare
from core.experiment.config import ExperimentConfig
from core.experiment.runner import ExperimentResult, ExperimentRunner
from core.registry import registry

router = APIRouter(prefix="/experiments", tags=["experiments"])

_STORE_DIR = Path(__file__).parents[3] / "eval" / "results" / "runs"
_STORE_DIR.mkdir(parents=True, exist_ok=True)

_progress: dict[str, list[dict[str, Any]]] = {}
# Async job-model: a run_id appears here the instant
# create_experiment returns (before any question has run) and is removed
# once the background task finishes successfully (the result is then
# findable via _get_results instead) or fails (moved to _errors). Lets
# GET /{run_id} answer "still running" vs "never existed" vs "404" without
# a new queue/store — reuses the same dict-based pattern as _progress.
_running: set[str] = set()
_errors: dict[str, str] = {}
# Found live: a run picking a slow model (or hitting a stuck external RAG)
# had no way to be interrupted — POST /{run_id}/stop adds run_id here;
# _run_experiment_background's should_stop closure (read from the worker
# thread, same GIL-atomic set-membership-check safety as _progress's
# list.append) makes ExperimentRunner.run() notice it between questions.
_stop_requested: set[str] = set()
#: What a run costs in the database, per question, at its worst.
#:
#: Measured over every run stored in this repository and never estimated, and
#: named here because two places state it in prose and both had gone stale: it
#: stood at 215 KB, which was the worst rate when it was written and put the
#: ceiling at seventy-eight questions. A guard now reads the stored runs and
#: reddens when one of them costs more than this says, so the number moves
#: when the measurement does and the ceiling below moves with it.
#:
#: The rate is driven by `candidate_source_refs`, which carries the whole text
#: of every candidate: a wide fetch window with a reranker is what reaches it.
WORST_BYTES_PER_QUESTION = 533_911
#: How many questions a run of that shape can hold before the engine refuses
#: the document. The engine's own limit is 16 MiB.
QUESTIONS_A_RUN_CAN_HOLD = 16 * 1024 * 1024 // WORST_BYTES_PER_QUESTION
# Minimal metadata stored at run start so list_experiments can show running
# runs before they complete and land in _get_results().
_running_meta: dict[str, dict[str, Any]] = {}


# ── Persistence ───────────────────────────────────────────────────────────────

async def _save(result: ExperimentResult) -> None:
    """Write the run to the database and to a file beside it.

    The database write is allowed to fail: a run document larger than the
    engine's own limit is rejected, and a run with a wide candidate window
    costs `WORST_BYTES_PER_QUESTION` of document per question, which puts that
    limit where `QUESTIONS_A_RUN_CAN_HOLD` says. The file copy is what
    survives it.

    The failure is logged now. It used to be swallowed. With `_get_results` preferring the database whenever it returned anything
    at all, a rejected run existed on disk and appeared nowhere: not in the
    list, not on its own page. A run that cost hours to produce vanished
    without a line anywhere saying so.
    """
    data = result.to_dict()
    try:
        import adapters.mongodb as mdb
        await mdb.upsert_one("experiment_runs", {"run_id": result.run_id}, data)
    except Exception as exc:
        import structlog
        structlog.get_logger().warning(
            "experiment.save.database_rejected",
            run_id=result.run_id,
            n_questions=len(result.question_results),
            size_bytes=len(json.dumps(data, ensure_ascii=False)),
            error=str(exc),
            hint="the run is on disk and is still readable; a document over the "
                 "engine's size limit is the usual cause",
        )
    # keep file copy as backup
    p = _STORE_DIR / f"{result.run_id}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_result(data: dict[str, Any], stem: str = "") -> ExperimentResult | None:
    from core.experiment.runner import QuestionResult
    try:
        cfg_data = data.get("config") or {}
        if not cfg_data:
            cfg_data = {
                "name": data.get("config_name", stem),
                "chunking_strategy": {"kind": "chunker", "component_id": "fixed"},
                "embedder": {"kind": "embedder", "component_id": "bge_m3"},
                "generator": {"kind": "generator", "component_id": "ollama"},
            }
        cfg = ExperimentConfig(**cfg_data)
        r = ExperimentResult(config=cfg, run_id=data.get("run_id", stem))
        r.aggregate_metrics = data.get("aggregate_metrics", {})
        r.started_at = data.get("started_at", "")
        r.finished_at = data.get("finished_at", "")
        r.n_questions = data.get("n_questions", len(data.get("question_results", [])))
        r.dataset_name = data.get("dataset_name", data.get("config", {}).get("dataset_name", ""))
        r.prompt_id = data.get("prompt_id", "")
        r.prompt_version = data.get("prompt_version", 0)
        r.realm_id = data.get("realm_id", "")
        r.stopped = data.get("stopped", False)
        r.generator_model = data.get("generator_model", "")
        r.coverage_check = data.get("coverage_check") or {}
        # What ran, and what the corpus was built from. Both were written by
        # the run and dropped here, which is the failure the catalogue calls
        # "the read path loses data" happening to the two fields added to
        # catch three others: every check consulting the load record went
        # silent the moment a run was read back from the store,
        # and a silent check is indistinguishable from a passing one.
        r.applied = data.get("applied") or {}
        r.corpus_manifest = data.get("corpus_manifest") or {}
        r.unavailable_components = data.get("unavailable_components") or []
        for qd in data.get("question_results", []):
            r.question_results.append(QuestionResult(
                question_id=qd.get("question_id", ""),
                question=qd.get("question", ""),
                reference_answer=qd.get("reference_answer", ""),
                generated_answer=qd.get("generated_answer", ""),
                source_refs=qd.get("source_refs", []),
                pre_rerank_source_refs=qd.get("pre_rerank_source_refs", []),
                candidate_source_refs=qd.get("candidate_source_refs", []),
                expected_refs=qd.get("expected_refs", []),
                metrics=qd.get("metrics", {}),
                computed_citations=qd.get("computed_citations", []),
                stage_trace=qd.get("stage_trace"),
                error=qd.get("error"),
                answerability=qd.get("answerability"),
                root_cause=qd.get("root_cause"),
            ))
        return r
    except Exception:
        return None


async def _get_results() -> dict[str, ExperimentResult]:
    """Every run, from both places it can live.

    The two are merged, and no longer tried in order. The file store used to be a
    fallback reached only when the database returned nothing at all, so a run
    the database had rejected was invisible for as long as any other run
    existed, which is always. A run present in one place and absent from the
    other is exactly the case the store has to survive, and it is the case the
    ordering could not express.

    The database wins on a run present in both: it is written first and a file
    copy is only ever as new as the write that produced it.
    """
    results: dict[str, ExperimentResult] = {}
    try:
        import adapters.mongodb as mdb
        docs = await mdb.find_many("experiment_runs", sort=[("started_at", -1)])
        for data in docs:
            r = _parse_result(data, data.get("run_id", ""))
            if r:
                results[r.run_id] = r
    except Exception:
        pass

    # Only the runs the database does not have. Which those are is decided by
    # the file name, since `_save` writes `{run_id}.json`, so a run present in
    # both costs a directory entry and not a parse. The first version of
    # this merge parsed every file on every request: with the runs on this
    # machine that is a quarter of a gigabyte of JSON for one page of a list.
    for path in _STORE_DIR.glob("*.json"):
        if path.stem in results:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            r = _parse_result(data, path.stem)
            if r:
                results[r.run_id] = r
        except Exception:
            pass

    return results


async def _get_one_result(run_id: str) -> ExperimentResult | None:
    """One run by id, without reading every other run to find it.

    `_get_results` above loads and parses the whole store, which is the right
    trade for the list and for a comparison that is about to resample. The
    pre-flight check runs on every change of either selector, and a run
    document carries all of its question results, so paying for the entire
    store there would make choosing a pair slower than comparing one.
    """
    try:
        import adapters.mongodb as mdb
        doc = await mdb.find_one("experiment_runs", {"run_id": run_id})
        if doc:
            return _parse_result(doc, doc.get("run_id", run_id))
    except Exception:
        pass
    path = _STORE_DIR / f"{run_id}.json"
    if path.exists():
        try:
            return _parse_result(json.loads(path.read_text(encoding="utf-8")), run_id)
        except Exception:
            return None
    return None


# ── Dataset loader ────────────────────────────────────────────────────────────

async def _load_dataset(dataset_name: str, external_rag_id: str | None = None) -> Any:
    from eval.dataset import EvalDataset, make_stub_dataset

    if dataset_name and dataset_name not in ("stub", ""):
        # per-rag datasets live in their own collection,
        # scoped by external_rag_id so two different external RAGs can each
        # register a dataset with the same filename without colliding.
        if external_rag_id:
            try:
                import adapters.mongodb as mdb
                doc = await mdb.find_one(
                    "external_rag_datasets",
                    {"external_rag_id": external_rag_id, "filename": dataset_name},
                )
                if doc and doc.get("questions"):
                    ds = EvalDataset.__new__(EvalDataset)
                    ds.name = doc.get("name", dataset_name)
                    ds.version = doc.get("version", "v0")
                    ds.speed = doc.get("speed", "fast")
                    ds.questions = doc["questions"]
                    return ds
            except Exception:
                pass
        # Primary: MongoDB datasets collection
        try:
            import adapters.mongodb as mdb
            doc = await mdb.find_one("datasets", {"filename": dataset_name}) \
                  or await mdb.find_one("datasets", {"name": dataset_name.split(".")[0]})
            if doc and doc.get("questions"):
                ds = EvalDataset.__new__(EvalDataset)
                ds.name = doc.get("name", dataset_name)
                ds.version = doc.get("version", "v0")
                ds.speed = doc.get("speed", "fast")
                ds.questions = doc["questions"]
                return ds
        except Exception:
            pass
        # Fallback: files
        golden_dir = Path(__file__).parents[3] / "eval" / "golden"
        for p in sorted(golden_dir.glob("*.jsonl")):
            if p.name == dataset_name or p.stem == dataset_name:
                return EvalDataset.from_jsonl(p)
        for p in sorted(golden_dir.glob("*.jsonl"), reverse=True):
            if p.stem.split(".")[0] == dataset_name:
                return EvalDataset.from_jsonl(p)
    return make_stub_dataset(n=5)


# ── Composite evaluator (Eval Measurement Trustworthiness, Phase 0) ───────────
# Replaces the old Jaccard token-overlap evaluator, which was structurally
# incapable of producing a meaningful score (faithfulness was capped near
# ~0.05 by construction regardless of answer quality — see the Phase 0 plan
# for the full audit). See core/eval/{answerability,retrieval_metrics,
# semantic_metrics}.py for the metric primitives this composes.

def _display_answerability(qr: dict[str, Any], metrics: dict[str, Any]) -> str:
    """The answerability class GET /experiments/{run_id} should show in a
    question's funnel diagnosis.

    Found live: a question with NO article_refs at all (out_of_scope — e.g.
    one the LLM-based question generator produced without populating refs)
    displayed the exact same "the reference source is absent from the corpus
    (uncovered)" wording as a question whose refs genuinely don't resolve.
    Root cause: this used to re-guess "answerable vs not" purely from
    whether `retrieval_recall_at_k` was present in `metrics` — a signal
    that can tell "answerable" apart from "everything else", but can never
    recover which non-answerable sub-class (uncovered vs out_of_scope) the
    evaluator actually resolved, since both leave the exact same metrics
    shape (`{"correct_refusal": ...}` only).

    `qr["answerability"]` (persisted at eval time — see
    core/experiment/runner.py#QuestionResult.answerability) is preferred
    now, since it's the real class. Falls back to the old metric-key
    inference only for runs stored before that field existed."""
    return qr.get("answerability") or (
        "answerable" if "retrieval_recall_at_k" in metrics else "not_applicable"
    )


def _attach_refusal_verdict(question_results: list[dict[str, Any]]) -> None:
    """Decide once, here, whether each answer refuses.

    The decision existed in three places. `core/eval/detectors.py` holds the
    canonical pattern, this router used to keep a weaker copy of it and now
    imports the original, and the interface keeps a third: eleven fixed
    substrings against the pattern's fifteen alternatives, several of which
    match on a word stem. The interface does not know `не указан`, `нет
    информац`, the stem of `отсутству`, `не содержит` in any form but one,
    `could not find`, `does not say/include/mention`, `cannot find` or `no
    relevant information`.

    So the two halves of the platform disagreed about the same answer on the
    same screen, and the comment beside the canonical pattern records that this
    exact drift had already been found and removed once, between two copies on
    this side. Deciding server-side and sending the verdict leaves one rule in
    one place.
    """
    for qr in question_results:
        answer = qr.get("generated_answer") or ""
        qr["is_refusal"] = not answer.strip() or bool(_NOT_FOUND.search(answer))


def _attach_funnel(question_results: list[dict[str, Any]]) -> None:
    """Mutates each ``question_results`` dict in place, adding a ``funnel``
    verdict (``core/eval/funnel.py``). Shared by ``GET /experiments/{run_id}``
    and ``POST /experiments/compare`` so the two paths
    never disagree on how a question's layer is diagnosed."""
    from core.eval.funnel import diagnose_question
    for qr in question_results:
        metrics = qr.get("metrics") or {}
        answerability = _display_answerability(qr, metrics)
        qr["funnel"] = diagnose_question(answerability, metrics, metrics.get("pre_rerank_recall_at_k")).to_dict()


class _CompositeEvaluator:
    """Per-question metrics, gated by answerability class so retrieval/answer
    quality is only scored where there is a real ground truth to score it
    against (FR from the Phase 0 plan): correct_refusal is computed for
    every question; retrieval_recall_at_k/retrieval_precision_at_k/
    answer_similarity/context_support are computed only for "answerable"
    questions (their keys are simply absent otherwise, so the runner's
    per-key mean aggregation in core/experiment/runner.py naturally averages
    them over answerable questions only).
    """

    def __init__(self, embedder: Any, top_k: int, ref_resolver: Any = None) -> None:
        self._embedder = embedder
        self._top_k = top_k
        # Coverage is resolved against what is INDEXED, via a
        # core/eval/ref_resolution.py RefResolver, not against a hardcoded
        # corpus path on the platform's own disk (see that module's
        # docstring for the four incidents that coupling caused). None is a
        # legitimate value: an external RAG's dataset carries its own
        # explicit per-row `answerability` field, and a run whose index is
        # unreachable degrades to "unknown" rather than to a false
        # "uncovered".
        self._ref_resolver = ref_resolver

    def resolve_answerability(self, question: dict[str, Any]) -> str:
        """Exposed so the runner can persist the actual answerability class
        onto QuestionResult (core/experiment/runner.py) — same resolution
        `evaluate()` uses internally, just made independently callable so
        GET /experiments/{run_id} doesn't have to re-guess "answerable" vs
        "not" from metric-key presence alone, which can't tell "uncovered"
        apart from "out_of_scope" (see services/api_gateway/routers/
        experiments.py's funnel-diagnosis loop)."""
        from core.eval.answerability import resolve_answerability
        return resolve_answerability(question, self._ref_resolver)

    def evaluate(self, question: dict[str, Any], answer: Any) -> dict[str, float]:
        from core.citation import citation_number_coverage
        from core.eval.answerability import resolve_answerability
        from core.eval.retrieval_metrics import (
            average_precision,
            extracted_refs,
            precision_at_k,
            recall_at_k,
        )
        from core.eval.semantic_metrics import (
            answer_relevance,
            answer_similarity,
            context_support,
            grounded_in_correct_source,
        )

        article_refs = question.get("article_refs") or []
        answerability = resolve_answerability(question, self._ref_resolver)

        answer_text = answer.text or ""
        is_refusal = not answer_text.strip() or bool(_NOT_FOUND.search(answer_text))

        # A retrieval-only run stops before the generator and returns an
        # empty text on purpose (core/pipeline.py, adapters/http_pipeline.py,
        # both marking it in metadata). Every metric derived from that text
        # then scores it: correct_refusal reads the emptiness as a refusal
        # and returns 0.0 for every answerable question, answer_relevance
        # and context_support return 0.0 against an empty string. Averaged
        # over a whole dataset those are three columns of confident zeros
        # about something nobody measured, which is the exact failure this
        # platform exists to catch.
        retrieval_only = bool((getattr(answer, "metadata", None) or {}).get("retrieval_only"))

        metrics: dict[str, float] = {}
        if not retrieval_only:
            metrics["correct_refusal"] = (
                (0.0 if is_refusal else 1.0)
                if answerability == "answerable"
                else (1.0 if is_refusal else 0.0)
            )

        if answerability != "answerable":
            return metrics

        source_refs = [sr.model_dump() for sr in answer.source_refs]
        retrieved = extracted_refs(source_refs)
        metrics["retrieval_recall_at_k"] = recall_at_k(article_refs, retrieved, k=self._top_k)
        metrics["retrieval_precision_at_k"] = precision_at_k(article_refs, retrieved, k=self._top_k)
        # Ranking-aware precision (deterministic counterpart to deepeval's
        # ContextualPrecisionMetric) — precision_at_k is blind to WHERE
        # within top-k the relevant article landed; this isn't.
        metrics["retrieval_average_precision"] = average_precision(article_refs, retrieved)

        # Funnel diagnosis, Phase 1 — only present when a reranker actually
        # ran (core/pipeline.py only snapshots pre_rerank_source_refs in
        # that case); lets core/eval/funnel.py tell "retrieval never found
        # it" apart from "retrieval found it, the reranker demoted it".
        pre_rerank_refs = getattr(answer, "pre_rerank_source_refs", None)
        if pre_rerank_refs:
            pre_rerank_retrieved = extracted_refs([sr.model_dump() for sr in pre_rerank_refs])
            metrics["pre_rerank_recall_at_k"] = recall_at_k(article_refs, pre_rerank_retrieved, k=self._top_k)

        if retrieval_only:
            return metrics

        reference = question.get("ground_truth") or question.get("reference_answer", "")
        if reference:
            metrics["answer_similarity"] = answer_similarity(answer_text, reference, self._embedder)

        # Deterministic counterpart to deepeval's AnswerRelevancyMetric — is
        # the answer on-topic for the QUESTION, independent of correctness
        # (answer_similarity only compares against ground_truth).
        question_text = question.get("question", "")
        if question_text:
            metrics["answer_relevance"] = answer_relevance(answer_text, question_text, self._embedder)

        context_chunks = [sr.get("chunk_text", "") for sr in source_refs if sr.get("chunk_text")]
        if context_chunks:
            metrics["context_support"] = context_support(answer_text, context_chunks, self._embedder)

        # grounded_in_correct_source returns None (not 0.0) when no
        # retrieved chunk matches article_refs — omit the key rather than
        # record a misleading zero (core/eval/semantic_metrics.py docstring).
        grounded = grounded_in_correct_source(answer_text, source_refs, article_refs, self._embedder)
        if grounded is not None:
            metrics["grounded_in_correct_source"] = grounded

        # Catches a different failure mode than grounded_in_correct_source
        # (semantic alignment) or retrieval_recall_at_k (right chunk found
        # at all): does the answer TEXT contain the actual structural
        # number retrieval found, or did the model mistranscribe/hallucinate
        # a different one while citing? See core/citation.py module
        # docstring — found live: article 1 written regardless of the
        # correctly-retrieved article 210.5. None (not 0.0) when the
        # corpus has no extractable structural label at all.
        coverage = citation_number_coverage(answer_text, answer.source_refs, article_refs)
        if coverage is not None:
            metrics["citation_number_coverage"] = coverage

        return metrics


# ── Models ────────────────────────────────────────────────────────────────────

class ExperimentListItem(BaseModel):
    run_id: str
    name: str
    config_hash: str
    aggregate_metrics: dict[str, float]
    started_at: str = ""
    finished_at: str = ""
    n_questions: int = 0
    dataset_name: str = ""
    prompt_id: str = ""
    prompt_version: int = 0
    # the async job-model — "running" while backgrounded, "done" once saved.
    status: str = "done"
    progress_processed: int = 0
    progress_total: int = 0
    # Which Realm this run belongs to. "" = no Realm (legacy/global).
    realm_id: str = ""
    # Found live: the list page had no way to tell a run stopped early
    # (ExperimentRunner.run#should_stop, see core/experiment/runner.py) apart
    # from a normally-completed one — it showed the full planned question
    # count and no indication anything was cut short. Mirrors what
    # RunPage.tsx already does for the single-run detail view.
    stopped: bool = False
    # Which run is the baseline was answerable only from GET /experiments/
    # {run_id}, so the list could neither mark it nor compare against it —
    # the runs table had to render every row's metrics with no reference, and
    # the client could not tell a regression from an ordinary finish. Same
    # single lookup the detail endpoint already does, hoisted out of the loop.
    is_baseline: bool = False


class CompareRequest(BaseModel):
    # Exactly two. The endpoint reads ids[0] and ids[1], so a single id used to
    # raise IndexError and surface as a 500, and a third id was dropped in
    # silence. Both now fail validation with a 422 that says which.
    ids: list[str] = Field(..., min_length=2, max_length=2)
    # The comparison can spend minutes resampling flipped questions
    # (up to _MAX_RESAMPLE_FLIPS × _RESAMPLE_N pipeline runs) with nothing on
    # screen but a spinner. The client picks an id, listens on the existing
    # /experiments/{id}/progress socket, and gets "N of M" as they finish.
    progress_id: str | None = None


class NewExperimentRequest(BaseModel):
    config: dict[str, Any]
    dataset_name: str = "handbook.v1.fast.jsonl"
    # Realm this run is scoped to. "" = no Realm (legacy/global).
    realm_id: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ExperimentListItem])
async def list_experiments(
    sort_by: str = "run_id",
    dataset: str | None = None,
    # Scope the list to one Realm. None = no filter (all runs,
    # back-compat for callers that don't pass a Realm context).
    realm_id: str | None = None,
) -> list[ExperimentListItem]:
    results = await _get_results()
    baseline_id = await _get_baseline_run_id()
    items = [
        ExperimentListItem(
            run_id=r.run_id,
            name=r.config.name,
            config_hash=r.config.config_hash,
            aggregate_metrics=r.aggregate_metrics,
            started_at=r.started_at,
            finished_at=r.finished_at,
            # n_questions is the dataset's planned total; a stopped run's
            # question_results is legitimately shorter — show the actual
            # answered count, same distinction RunPage.tsx already makes.
            n_questions=len(r.question_results) if r.stopped else r.n_questions,
            dataset_name=r.dataset_name,
            prompt_id=r.prompt_id,
            prompt_version=r.prompt_version,
            status="done",
            realm_id=r.realm_id,
            stopped=r.stopped,
            is_baseline=r.run_id == baseline_id,
        )
        for r in results.values()
        if (dataset is None or r.config.dataset_name == dataset)
        and (realm_id is None or r.realm_id == realm_id)
    ]
    # Prepend in-flight runs so they appear at the top regardless of sort.
    for run_id in list(_running):
        meta = _running_meta.get(run_id, {})
        if dataset is not None and meta.get("dataset_name") != dataset:
            continue
        if realm_id is not None and meta.get("realm_id", "") != realm_id:
            continue
        latest = (_progress.get(run_id) or [{}])[-1]
        items.insert(0, ExperimentListItem(
            run_id=run_id,
            name=meta.get("name", run_id),
            config_hash=meta.get("config_hash", ""),
            aggregate_metrics={},
            started_at=meta.get("started_at", ""),
            dataset_name=meta.get("dataset_name", ""),
            status="running",
            progress_processed=latest.get("processed", 0),
            progress_total=latest.get("total", 0),
            realm_id=meta.get("realm_id", ""),
        ))
    if sort_by in ("run_id", "name", "config_hash"):
        items.sort(key=lambda x: getattr(x, sort_by))
    return items


# Declared BEFORE `/{run_id}`: FastAPI takes the first route that matches, so a
# parameterised path swallows a literal one. `DELETE /experiments/baseline` was
# going to "delete the run whose id is baseline" and answering 404. Found by a
# live check immediately after the route was added.
@router.delete("/baseline")
async def unset_baseline() -> dict[str, Any]:
    """Clears the baseline mark.

    Setting a baseline was possible and clearing it was not: the run page simply
    drew no button on a marked run, so getting back to "there is no baseline"
    required marking a different one. A state you cannot return to is a trap
    rather than a state.
    """
    try:
        import adapters.mongodb as mdb
        await mdb.update_one(
            "settings", {"_id": "global"},
            {"$set": {"baseline_run_id": None}}, upsert=True,
        )
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Cannot persist baseline: {exc}") from exc
    return {"baseline_run_id": None, "status": "unpinned"}


@router.delete("/{run_id}", status_code=204)
async def delete_experiment(run_id: str) -> None:
    from fastapi import HTTPException
    deleted = False
    try:
        import adapters.mongodb as mdb
        n = await mdb.delete_one("experiment_runs", {"run_id": run_id})
        deleted = n > 0
    except Exception:
        pass
    p = _STORE_DIR / f"{run_id}.json"
    if p.exists():
        p.unlink()
        deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


async def _get_baseline_run_id() -> str | None:
    try:
        import adapters.mongodb as mdb
        doc = await mdb.find_one("settings", {"_id": "global"})
        return (doc or {}).get("baseline_run_id")
    except Exception:
        return None


@router.put("/{run_id}/baseline")
async def set_baseline(run_id: str) -> dict[str, Any]:
    """Pin a run as the regression baseline."""
    results = await _get_results()
    if run_id not in results:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    try:
        import adapters.mongodb as mdb
        await mdb.update_one(
            "settings", {"_id": "global"},
            {"$set": {"baseline_run_id": run_id}}, upsert=True,
        )
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Cannot persist baseline: {exc}") from exc
    return {"baseline_run_id": run_id, "status": "pinned"}


class AcceptanceRequest(BaseModel):
    """The two runs to judge, and the questions to judge over.

    `question_ids` comes from a prescription's own verification lists, which
    is what makes the acceptance set fixed at the moment the prescription was
    written. A criterion the recipient could still widen or narrow afterwards
    is not one.
    """

    before_run_id: str
    after_run_id: str
    question_ids: list[str]


@router.post("/acceptance")
async def judge_acceptance_endpoint(body: AcceptanceRequest) -> dict[str, Any]:
    """Whether a prescription was carried out.

    Both runs get their funnel verdicts attached first, because acceptance is
    judged by the same "is this question fine" definition the regression
    machinery uses, and computing it here rather than reusing that definition
    would let the two disagree about the same pair of runs.
    """
    from fastapi import HTTPException

    from core.eval.prescription import judge_acceptance

    results = await _get_results()
    missing = [r for r in (body.before_run_id, body.after_run_id) if r not in results]
    if missing:
        raise HTTPException(status_code=404, detail=f"Run(s) not found: {', '.join(missing)}")

    before = results[body.before_run_id].to_dict()["question_results"]
    after = results[body.after_run_id].to_dict()["question_results"]
    _attach_funnel(before)
    _attach_funnel(after)
    return judge_acceptance(body.question_ids, before, after).to_dict()


@router.get("/frontier/{realm_id}")
async def get_frontier(realm_id: str, quality_metric: str = "retrieval_recall_at_k") -> dict[str, Any]:
    """Which of this Realm's finished runs are worth choosing between.

    Built over runs that already happened rather than by launching a sweep.
    A configuration search that starts runs would hide its own cost at the
    call site, and every run here was paid for once already; what was missing
    was only the comparison.
    """
    from core.eval.frontier import frontier_by_source, point_from_run, tokens_comparable

    results = await _get_results()
    points = []
    for run_id, result in results.items():
        if realm_id and result.realm_id != realm_id:
            continue
        payload = result.to_dict()
        payload["run_id"] = run_id
        point = point_from_run(payload, quality_metric=quality_metric)
        if point is not None:
            points.append(point)

    by_source = frontier_by_source(points)
    frontier_ids = {p.run_id for group in by_source.values() for p in group}
    # Whether the token axis took part, said per group rather than
    # once. It participates only where every run of that group carries counts,
    # and a reader who believes three axes were compared when two were would
    # draw a conclusion the computation never supported.
    grouped_raw: dict[str, list[Any]] = {}
    for point in points:
        grouped_raw.setdefault(point.pipeline_source, []).append(point)
    return {
        "quality_metric": quality_metric,
        "considered": len(points),
        # Grouped, and labelled as such: latency means the whole pipeline for
        # an in-process run and only the handover for an external one, so a
        # single ranking across both would compare a measurement artefact.
        "latency_comparable_within_source_only": True,
        "tokens_included": {k: tokens_comparable(v) for k, v in grouped_raw.items()},
        "frontier_by_source": {k: [p.to_dict() for p in v] for k, v in by_source.items()},
        # Both lists, because "these are the choices" reads very differently
        # depending on how many configurations were beaten outright.
        "dominated": [p.to_dict() for p in points if p.run_id not in frontier_ids],
    }


@router.get("/{run_id}/calibration")
async def get_calibration(
    run_id: str, metric: str = "answer_similarity", threshold: float = 0.6,
) -> dict[str, Any]:
    """Where this run's metric disagrees with its reviewers.

    Metric trustworthiness travels with it: both ask whether a metric can be trusted, and
    both are answered from the same run, so returning them separately would
    make a reader fetch twice to reach one conclusion.
    """
    from core.eval.counterfactual import first_hit_rank
    from core.eval.frontier import calibrate_metric, measure_position_bias

    payload = await get_experiment(run_id)
    question_results = payload.get("question_results") or []

    feedback: dict[str, Any] = {}
    try:
        import adapters.mongodb as mdb
        for doc in await mdb.find_many("answer_feedback", {"run_id": run_id}):
            feedback[str(doc.get("question_id") or "")] = doc
    except Exception:
        feedback = {}

    # Only questions whose expected source actually reached the context: the
    # measurement is about placement, so a question that never had the source
    # at all would confuse absence with position.
    top_k = int((payload.get("config") or {}).get("top_k") or 0)
    observations = []
    for qr in question_results:
        refs = qr.get("expected_refs") or []
        candidates = qr.get("candidate_source_refs") or qr.get("source_refs") or []
        quality = (qr.get("metrics") or {}).get(metric)
        rank = first_hit_rank(list(refs), candidates) if refs and candidates else None
        if rank is not None and quality is not None and (not top_k or rank <= top_k):
            observations.append((rank, float(quality)))

    return {
        "calibration": calibrate_metric(question_results, feedback, metric, threshold).to_dict(),
        "position_bias": measure_position_bias(observations).to_dict(),
    }


@router.get("/{run_id}/prescription")
async def get_prescription(run_id: str) -> dict[str, Any]:
    """Phase 4 — the document an external system's owner works from.

    Built from this endpoint's own `GET /{run_id}` payload rather than from
    stored results directly, so the document and the page cannot disagree
    about a single number.
    """
    from core.eval.prescription import build_prescription, render_markdown

    payload = await get_experiment(run_id)
    prescription = build_prescription(payload)
    return {**prescription.to_dict(), "markdown": render_markdown(prescription)}


@router.get("/{run_id}")
async def get_experiment(run_id: str) -> dict[str, Any]:
    from fastapi import HTTPException

    results = await _get_results()
    if run_id not in results:
        # A run_id that's mid-flight (or just failed) in
        # the background task isn't in the result store yet; report status
        # instead of a bare 404 so a polling client can tell "still running"
        # apart from "never existed".
        if run_id in _errors:
            raise HTTPException(status_code=500, detail=_errors[run_id])
        if run_id in _running:
            events = _progress.get(run_id, [])
            return {
                "run_id": run_id,
                "status": "running",
                "progress": events[-1] if events else None,
            }
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    payload = results[run_id].to_dict()
    payload["status"] = "done"

    # How much searching this number is the best of. Computed here because it
    # is a property of the run store and not of the run, and attached to the
    # payload so the detectors stay pure: a check that went to the database
    # itself would be a check nobody could run on a saved run.
    payload["tuning_provenance"] = _tuning_provenance(run_id, results)

    # silent-degradation detectors, each carrying the catalogue entries it is
    # evidence for so a reader can tell a known failure from a bare sentence.
    from core.eval.detectors import run_detectors
    from services.api_gateway.routers.atlas import attach_failure_ids
    payload["diagnostics"] = [d.to_dict() for d in run_detectors(payload)]
    attach_failure_ids(payload["diagnostics"], "detector")

    _attach_funnel(payload["question_results"])
    _attach_refusal_verdict(payload["question_results"])

    # What this run made impossible to check, beside what it did check. The two
    # used to live on different tabs, so a reader saw the findings and had no
    # sign that a whole class of them could not have been produced at all. A
    # check that could not run and a check that ran and found nothing are the
    # same absence on a screen unless one of them is named.
    from core.eval.trace_completeness import assess_trace_completeness, diagnosis_depth
    payload["trace_gaps"] = [g.to_dict() for g in assess_trace_completeness(payload["question_results"])]
    payload["diagnosis_depth"] = diagnosis_depth(payload["question_results"])

    # How many questions each root cause accounts for. Derived at
    # read time from the stored per-question verdicts rather than stored
    # separately, so the two can never disagree. This is the number phase 2's
    # prioritisation is built on: it turns a list of failures into a
    # comparison between kinds of work.
    from core.eval.root_cause import count_causes, lever_for
    payload["root_cause_counts"] = count_causes(payload["question_results"])

    # A run from a window of older versions carries a cause but
    # no lever. Derived here rather than in the interface so the cause-to-lever
    # mapping stays in one place, beside the reasoning that produced the cause.
    # Found live: without this the page fell back to "verify the index" for a
    # cause it already knew, which reads as ignorance where there is none.
    for qr in payload["question_results"]:
        cause = qr.get("root_cause") or {}
        if cause.get("cause") and not cause.get("lever"):
            cause["lever"] = lever_for(cause["cause"])

    # Phase 2 — the ordered list of work, and the share of failures that
    # reduce to a common cause. Derived here from the same stored verdicts
    # the counts come from, so a task list can never disagree with the causes
    # a reader sees on the questions themselves.
    from core.eval.prioritization import clusterable_share, group_failures
    fix_tasks = group_failures(payload["question_results"])
    payload["fix_tasks"] = [t.to_dict() for t in fix_tasks]
    payload["clusterable_share"] = clusterable_share(fix_tasks)

    # Phase 3 — what the run would have shown at a different context size,
    # answered from what it already recorded rather than by running again.
    payload["context_size_advice"] = _context_size_advice(payload)

    # auto-compare against the pinned baseline.
    baseline_id = await _get_baseline_run_id()
    payload["is_baseline"] = baseline_id == run_id
    if baseline_id and baseline_id != run_id and baseline_id in results:
        from core.eval.regression import compare as regression_compare
        report = regression_compare(
            payload.get("aggregate_metrics", {}),
            results[baseline_id].aggregate_metrics,
        )
        payload["regression"] = {**report.to_dict(), "baseline_run_id": baseline_id}

    # Optional deepeval LLM-judge report for this same run_id (eval/deepeval_runner.py,
    # run separately/optionally — see GuidePage.tsx's quality-metrics section). MongoDB-only
    # lookup, no file fallback: deepeval reports are large and rare enough that
    # requiring the migration step (tools/migrate_to_mongodb.py) is acceptable.
    try:
        import adapters.mongodb as mdb
        deepeval_doc = await mdb.find_one("deepeval_results", {"run_id": run_id})
        payload["deepeval_report"] = (
            {"metrics": deepeval_doc.get("metrics", {}), "timestamp": deepeval_doc.get("timestamp")}
            if deepeval_doc else None
        )
    except Exception:
        payload["deepeval_report"] = None
    return payload


_MAX_MISS_DIAGNOSIS_WIDENED_K = 200


@router.post("/{run_id}/questions/{question_id}/diagnose-miss")
async def diagnose_retrieval_miss_endpoint(
    run_id: str, question_id: str, widened_k: int = 50,
) -> dict[str, Any]:
    """On-demand re-query of THIS question with a wider k, to tell apart
    "the right chunk ranked just outside top_k" (tuning problem) from "it's
    nowhere near the top even widened" (indexing/embedding problem) — see
    core/eval/miss_diagnosis.py module docstring. Not run automatically for
    every miss (re-querying retrieval per question is not free); triggered
    explicitly from RunPage per question, same on-demand pattern as the
    corpus deep-diagnostics endpoints.
    """
    from fastapi import HTTPException

    from core.eval.miss_diagnosis import diagnose_retrieval_miss

    widened_k = min(max(widened_k, 1), _MAX_MISS_DIAGNOSIS_WIDENED_K)

    results = await _get_results()
    if run_id not in results:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    result = results[run_id]

    question = next((qr for qr in result.question_results if qr.question_id == question_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail=f"Question {question_id!r} not found in run {run_id!r}")

    dataset = await _load_dataset(result.dataset_name)
    dataset_question = next((q for q in dataset.questions if q.get("id") == question_id), None)
    article_refs = (dataset_question or {}).get("article_refs") or []
    if not article_refs:
        raise HTTPException(
            status_code=400,
            detail=f"Question {question_id!r} has no article_refs ground truth to diagnose against",
        )

    try:
        runner = ExperimentRunner(registry=registry)
        # Rebuild against the SAME Realm the original run used
        # (result.realm_id, set post-run in _run_experiment_background),
        # not the gateway's own env-var instance — otherwise a miss-
        # diagnosis on a non-default-Realm run would query the wrong corpus
        # entirely and its "miss" verdict would be meaningless.
        qdrant_cfg = opensearch_cfg = None
        if result.realm_id and result.config.pipeline_source != "http":
            from services.api_gateway.routers.corpus import _get_realm_resource
            qdrant_cfg = await _get_realm_resource(result.realm_id, "qdrant")
            opensearch_cfg = await _get_realm_resource(result.realm_id, "opensearch")
        pipeline = runner._build_pipeline(result.config, result.realm_id, qdrant_cfg, opensearch_cfg)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not rebuild pipeline: {e}") from e

    diagnosis = diagnose_retrieval_miss(pipeline, question.question, article_refs, widened_k=widened_k)
    return {"run_id": run_id, "question_id": question_id, **diagnosis}


_RESAMPLE_N = 2  # additional samples beyond the flip's own original verdict
_MAX_RESAMPLE_FLIPS = 20  # a fix with more flips than this is already bad enough — save the compute


async def _resample_flip_verdicts(
    after: ExperimentResult, flip_ids: list[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[bool]] | None:
    """Re-runs each flipped question ``_RESAMPLE_N`` more times against the
    SAME (after) config, to tell a genuine per-question regression apart
    from generation-metric noise (the criterion left open when ``paired_diff``
    first shipped: "a noisy flip is extinguished on repeat run"). ``paired_diff``/``confirm_flips`` themselves stay pure and
    never re-run anything (see their own docstrings in
    ``core/eval/regression.py``) — this is the one caller that does.

    Mirrors ``diagnose_retrieval_miss_endpoint``'s own rebuild-a-pipeline-
    from-a-stored-config pattern (the only other place in this router that
    re-executes a single question against a freshly rebuilt pipeline instead
    of a whole dataset via ``ExperimentRunner.run``).

    Returns ``None`` (not an exception) when the pipeline or dataset can't be
    rebuilt at all (e.g. missing Realm resources, a since-deleted dataset) —
    the caller then degrades to trusting the original single-sample
    verdicts, exactly as if resampling had never been attempted.
    """
    try:
        from core.eval.funnel import diagnose_question
        from core.eval.retrieval_metrics import extracted_refs, recall_at_k
        from core.models import QueryRequest

        runner = ExperimentRunner(registry=registry)
        qdrant_cfg = opensearch_cfg = None
        retrieval_pins: list[Any] = []
        if after.realm_id and after.config.pipeline_source != "http":
            from services.api_gateway.routers.corpus import _get_realm_resource
            qdrant_cfg = await _get_realm_resource(after.realm_id, "qdrant")
            opensearch_cfg = await _get_realm_resource(after.realm_id, "opensearch")
            # Mirrors the run being resampled. Resampling exists to
            # tell a real flip from generation noise, so it has to retrieve
            # under the same conditions the run itself used; loading pins here
            # for a run that had them off would compare two different systems.
            if after.config.retrieval_pins_enabled:
                retrieval_pins = await _load_active_pins(after.realm_id, after.config.corpus_id)
        pipeline = runner._build_pipeline(after.config, after.realm_id, qdrant_cfg, opensearch_cfg, retrieval_pins)

        dataset = await _load_dataset(after.dataset_name, external_rag_id=after.config.external_rag_id)
        dataset_by_id = {q.get("id"): q for q in dataset.questions}

        embedder = registry.resolve("embedder", after.config.embedder.component_id)
        evaluator = _CompositeEvaluator(embedder=embedder, top_k=after.config.top_k)
    except Exception:
        return None

    # Only the questions still in the dataset are resampled, and the total
    # reported to the reader counts those — not the flips, some of which may
    # have no question left to re-ask.
    live_ids = [qid for qid in flip_ids if dataset_by_id.get(qid) is not None]

    def _sample_all() -> dict[str, list[bool]]:
        resampled: dict[str, list[bool]] = {}
        for done, qid in enumerate(live_ids, start=1):
            q = dataset_by_id[qid]
            samples: list[bool] = []
            for _ in range(_RESAMPLE_N):
                try:
                    answer = pipeline.run(QueryRequest(text=q.get("question", ""), top_k=after.config.top_k))
                    metrics = evaluator.evaluate(q, answer)
                    answerability = evaluator.resolve_answerability(q)
                    pre_rerank_recall = None
                    pre_refs = getattr(answer, "pre_rerank_source_refs", None)
                    if pre_refs:
                        pre_retrieved = extracted_refs([sr.model_dump() for sr in pre_refs])
                        pre_rerank_recall = recall_at_k(q.get("article_refs") or [], pre_retrieved, k=after.config.top_k)
                    verdict = diagnose_question(answerability, metrics, pre_rerank_recall)
                    samples.append(verdict.layer == "ok")
                except Exception:
                    # A hard failure to even answer counts as "not ok" for this
                    # sample — conservative, matches how a pipeline exception
                    # during a real run is already recorded as QuestionResult.error
                    # rather than silently dropped.
                    samples.append(False)
            resampled[qid] = samples
            if on_progress is not None:
                on_progress(done, len(live_ids))
        return resampled

    if on_progress is not None:
        on_progress(0, len(live_ids))
    # `pipeline.run` is plain synchronous Python and this loop is the whole
    # cost of a comparison. Left on the event loop it blocks every other
    # request — including the progress socket that is meant to be reporting
    # on it, which would then deliver all its events at once, at the end.
    # Same asyncio.to_thread reasoning as _run_experiment_job.
    return await asyncio.to_thread(_sample_all)


def _run_facts(result: ExperimentResult) -> dict[str, Any]:
    """What a reader needs to see which two runs are on the table.

    `n_questions` here is how many questions the run actually answered. The
    stored field is the dataset's planned total, which a stopped run never
    reaches, and comparing a planned total against an answered one would show
    a short run as a different dataset.
    """
    return {
        "run_id": result.run_id,
        "dataset_name": result.config.dataset_name,
        "n_questions": len(result.question_results),
        "realm_id": result.realm_id,
        "corpus_id": result.config.corpus_id,
        "stopped": result.stopped,
    }


def _paired_between(
    before: ExperimentResult, after: ExperimentResult,
) -> tuple[PairedDiffReport, dict[str, Any], dict[str, Any]]:
    """The per-question pairing plus both payloads it was computed from.

    Pure and free of I/O, so the pre-flight check can run it as readily as the
    full comparison does. Shared by both so the two can never disagree on how
    many questions a pair has in common.
    """
    from core.eval.regression import paired_diff
    before_payload = before.to_dict()
    after_payload = after.to_dict()
    _attach_funnel(before_payload["question_results"])
    _attach_funnel(after_payload["question_results"])
    paired = paired_diff(before_payload["question_results"], after_payload["question_results"])
    return paired, before_payload, after_payload


def _comparability(
    before: ExperimentResult, after: ExperimentResult, paired: PairedDiffReport,
    warnings: list[CompatWarning],
) -> dict[str, Any]:
    return {
        "comparable": all(w.severity != "error" for w in warnings),
        "matched": len(paired.fixed) + len(paired.flips) + len(paired.unchanged),
        "only_in_before": len(paired.only_in_before),
        "only_in_after": len(paired.only_in_after),
        "before": _run_facts(before),
        "after": _run_facts(after),
        "warnings": [w.to_dict() for w in warnings],
    }


@router.get("/compare/preflight")
async def compare_preflight(a: str, b: str) -> dict[str, Any]:
    """Whether these two runs can be compared, before anyone compares them.

    Answers the same question POST /experiments/compare answers on its way
    past, through the same `check_comparability`, so the picker and the report
    can never state different things about one pair. No resampling and no LLM
    here: this reads two stored runs and intersects their question ids.
    """
    from fastapi import HTTPException
    before = await _get_one_result(a)
    after = await _get_one_result(b)
    missing = [i for i, r in ((a, before), (b, after)) if r is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"Runs not found: {missing}")
    assert before is not None and after is not None
    paired, _, _ = _paired_between(before, after)
    return _comparability(before, after, paired, check_comparability(before, after, paired))


@router.post("/compare")
async def compare_experiments(body: CompareRequest) -> dict[str, Any]:
    # Everything the comparison does is arithmetic over two stored runs except
    # the resampling below, which re-answers questions and takes as long as a
    # small run. `progress_id` opts into reporting it: the same event shape and
    # the same socket the run page already uses, so the client side is the
    # `useProgress` hook it already has.
    pid = body.progress_id
    if pid:
        _progress[pid] = []

    def _publish(processed: int, total: int) -> None:
        _progress.setdefault(pid, []).append(
            {"type": "progress", "processed": processed, "total": total}
        )

    try:
        return await _compare_experiments(body, _publish if pid else None)
    finally:
        if pid:
            # The socket closes on `done`; without it a comparison with nothing
            # to resample would leave the reader watching a live connection
            # that has already said everything it will ever say.
            _progress.setdefault(pid, []).append({"type": "done"})


async def _compare_experiments(
    body: CompareRequest, on_progress: Callable[[int, int], None] | None,
) -> dict[str, Any]:
    results = await _get_results()
    missing = [i for i in body.ids if i not in results]
    if missing:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Runs not found: {missing}")
    before = results[body.ids[0]]
    after = results[body.ids[1]]
    report = do_compare(before, after)

    # per-question paired diff alongside the existing
    # aggregate-only comparison above: an aggregate metric_delta can stay
    # within threshold while a fix that helped some questions broke others
    # (averaged away), which is exactly the regression this is meant to catch.
    from core.eval.regression import confirm_flips
    paired, before_payload, after_payload = _paired_between(before, after)

    # A flip reported from a single before/after sample can be
    # generation-metric noise rather than a real regression; resample each
    # flipped question a couple more times before trusting it. Capped and
    # best-effort: a fix with dozens of flips is already bad regardless of
    # noise (skip resampling entirely — see _MAX_RESAMPLE_FLIPS), and a
    # pipeline/dataset that can't be rebuilt degrades to trusting the
    # original verdicts rather than failing the whole comparison.
    resample_attempted = False
    noise_filtered: list[str] = []
    if paired.flips and len(paired.flips) <= _MAX_RESAMPLE_FLIPS:
        resampled = await _resample_flip_verdicts(after, paired.flips, on_progress)
        if resampled is not None:
            resample_attempted = True
            confirmable_ids = [qid for qid in paired.flips if qid in resampled]
            full_samples = {qid: [False] + resampled[qid] for qid in confirmable_ids}
            confirmed, noise_filtered = confirm_flips(full_samples)
            unconfirmable_ids = [qid for qid in paired.flips if qid not in resampled]
            paired.flips = sorted(confirmed + unconfirmable_ids)
            paired.unchanged = sorted(paired.unchanged + noise_filtered)

    # `paired_diff` itself stays pure (ids only, see core/eval/regression.py's
    # own docstring) — the question text + funnel transition a reviewer
    # actually needs to tell one id apart from another is assembled here,
    # only for the ids that matter (fixed/flips; `unchanged` can be dozens
    # of questions and only needs a count, already in `paired`).
    before_by_id = {qr["question_id"]: qr for qr in before_payload["question_results"]}
    after_by_id = {qr["question_id"]: qr for qr in after_payload["question_results"]}
    questions_lookup: dict[str, dict[str, str]] = {}
    for qid in paired.fixed + paired.flips:
        qr = after_by_id.get(qid) or before_by_id.get(qid)
        if qr is None:
            continue
        questions_lookup[qid] = {
            "question": qr.get("question", ""),
            "funnel_before": ((before_by_id.get(qid) or {}).get("funnel") or {}).get("layer", ""),
            "funnel_after": ((after_by_id.get(qid) or {}).get("funnel") or {}).get("layer", ""),
        }

    # Computed after the resample pass, so the counts here and the counts on
    # screen come from one object. Resampling moves ids between flips and
    # unchanged and leaves the overlap itself alone, so no rule above changes
    # its mind because of it.
    report.compatibility = check_comparability(before, after, paired)
    compatibility = _comparability(before, after, paired, report.compatibility)

    return {
        "config_diff": report.config_diff,
        "compatibility": compatibility,
        "metric_deltas": [
            {
                "metric": d.metric, "before": d.before, "after": d.after, "delta": d.delta,
                "delta_pct": None if d.delta_pct == float("inf") else d.delta_pct,
            }
            for d in report.metric_deltas
        ],
        "summary": report.summary(),
        "paired_diff": {
            **paired.to_dict(), "questions": questions_lookup,
            "resample_attempted": resample_attempted, "noise_filtered": noise_filtered,
        },
    }


# Upper bound on how many chunks are read to build a coverage index. Reading
# is O(corpus) and happens once per run, so the bound exists only to keep a
# pathologically large collection from exhausting memory — NOT to sample.
# Hitting it means the index could not be read in full, which degrades to
# "unknown" (see _build_ref_resolver): a partial read would make every ref
# past the cut look absent, which is precisely the false-negative this whole
# work removes. This is why adapters/qdrant.py#scroll_all is deliberately NOT
# reused here — its 5000-chunk cap is a sampling cap for health checks, and
# silently truncating a coverage check is exactly the wrong trade.
_REF_INDEX_MAX_CHUNKS = 200_000
_REF_INDEX_PAGE = 500


async def _build_ref_resolver(
    realm_id: str | None, corpus_id: str, qdrant_cfg: dict[str, Any] | None = None,
) -> Any:
    """Build a `core/eval/ref_resolution.py` resolver over this run's index.

    Replaces the former hardcoded single-corpus disk lookup: coverage is now decided against
    what the served system actually searches, for whichever Realm and corpus
    this run targets.

    Every failure path returns `UnknownRefResolver`, never a partial
    `IndexRefResolver`. An unreachable index, a retriever that cannot
    enumerate, or a collection too large to read in full all mean "coverage
    was not verified" — they must not mean "not covered", because that
    silently drops questions out of every retrieval metric with no error
    surfaced anywhere (the exact failure this work exists to remove).
    """
    from core.eval.ref_resolution import UnknownRefResolver, resolver_from_chunks

    try:
        from core.experiment.runner import _rebind_corpus_id
        from core.registry import registry

        base = registry.resolve("pipeline", "naive")
        retriever = _rebind_corpus_id(base._retriever, corpus_id, realm_id, qdrant_cfg, None)
        scroll = getattr(retriever, "scroll", None)
        if scroll is None:
            return UnknownRefResolver(reason_id="the_retriever_cannot_enumerate",
                                       reason="retriever cannot enumerate chunks")

        chunks: list[Any] = []
        offset = None
        while len(chunks) < _REF_INDEX_MAX_CHUNKS:
            batch, offset = scroll(offset=offset, limit=_REF_INDEX_PAGE)
            if not batch:
                break
            chunks.extend(batch)
            if offset is None:
                break
        else:
            return UnknownRefResolver(reason_id="the_index_is_too_large",
                                       reason="index too large to verify coverage in full")
        if offset is not None:
            return UnknownRefResolver(reason_id="the_index_is_too_large",
                                       reason="index too large to verify coverage in full")
    except Exception as exc:
        return UnknownRefResolver(reason_id="the_index_is_unreachable",
                                   reason=f"index unreachable: {exc}", note=str(exc))

    # A zero-chunk read is NOT evidence of an empty corpus. Found live:
    # QdrantRetriever.scroll against a collection that does not exist returns
    # an empty page and raises nothing, so a wrong/missing corpus_id would
    # build an empty ref index, mark every ref "absent", and classify every
    # question "uncovered" — the exact silent-measurement-loss this work
    # removed, reproduced with a new cause. Treated as "could not verify".
    #
    # Nothing is lost by refusing to verify here: a genuinely empty corpus
    # answers no question at all, and core/eval/corpus_health.py already
    # reports that case loudly and separately as an `empty_corpus` error.
    if not chunks:
        return UnknownRefResolver(reason_id="the_index_is_empty_or_absent",
                                   reason="index empty or unavailable for this corpus")

    return resolver_from_chunks(chunks)


# A judgment carries no threshold of its own — the retrieval-pin mechanism's
# per-record threshold was part of what the objective review rejected, since
# it let one record decide how far its own effect should spread. When a run
# opts into the overlay it opts into one stated similarity for all of them,
# so the reach of the what-if is a property of the run and is visible in its
# config rather than buried per record.
_JUDGMENT_OVERLAY_THRESHOLD = 0.93


async def _load_active_pins(realm_id: str, corpus_id: str) -> list[Any]:
    """Builds the optional overlay from the Realm's judgments file.

    This used to read MongoDB. It now reads
    `eval/judgments/<realm>__<corpus>.jsonl`, which is the whole point:
    a served system must not depend on the platform's database at query time,
    and a file is an artefact that can be
    handed over instead. Only called when a run sets
    `ExperimentConfig.retrieval_pins_enabled`, and never on a chat request.

    A judgment's `relevant` chunks become the pinned ones and its
    `irrelevant` chunks the demoted ones, which is the honest reading of the
    reviewer's statement rather than a separate authored instruction. The
    query vector is embedded here, once per run, because the file stores the
    question's text and not a vector: a vector belongs to one embedding
    model, and pinning it into the artefact would silently expire the file
    the day the model changes.

    Returns an empty list (never raises). Losing the overlay degrades a
    what-if, not a measurement, so failing the run over it would trade
    something cheap for something expensive.
    """
    try:
        from core.judgments.store import judgments_path, load_judgments
        from core.models import Chunk
        from core.pins.overlay import Pin
        from core.registry import registry
        from services.api_gateway.routers.judgments import _JUDGMENTS_DIR

        judgments = [
            j for j in load_judgments(judgments_path(_JUDGMENTS_DIR, realm_id, corpus_id))
            if j.is_active and not j.is_empty
        ]
        if not judgments:
            return []
        embedder = registry.resolve("embedder", "bge_m3")
        vectors = embedder.embed([j.question for j in judgments])
    except Exception:
        return []

    pins: list[Pin] = []
    for judgment, vector in zip(judgments, vectors, strict=True):
        try:
            pins.append(Pin(
                id=judgment.id,
                query_vec=list(vector),
                threshold=_JUDGMENT_OVERLAY_THRESHOLD,
                pin_chunks=[
                    Chunk(
                        chunk_id=c.chunk_id, doc_id=c.doc_id,
                        text=c.text, structural_path=c.structural_path,
                    )
                    for c in judgment.relevant
                ],
                demote_chunk_ids=[c.chunk_id for c in judgment.irrelevant],
            ))
        except Exception:
            continue  # one malformed record shouldn't cost every other judgment
    return pins


# How far past the run's own top_k the diagnostic re-query looks. Large
# enough that "not found here either" is a real statement about the query
# and the text rather than about the cut-off, small enough that one extra
# retrieval per failed question stays cheap.
_ROOT_CAUSE_WIDENED_K = 50


def _observed_ranks(payload: dict[str, Any], expected_by_question: dict[str, list[str]]) -> list[int | None]:
    """Where each failed question's expected source actually sat.

    Two sources, in order of precision. The candidate window recorded by a
    run with `fetch_k` set is the direct evidence: it is the ranked list the
    run itself produced. Where that is absent, as in every run made before
    `fetch_k` existed,
    the `ranking` cause's own evidence carries the rank the widened re-query
    found, which measures the same thing with a different query budget.

    A failed question with neither contributes None, meaning no context size
    would have helped. That is deliberately distinct from being absent from
    the list: one says the cut-off is not the problem, the other says nothing
    at all.
    """
    from core.eval.counterfactual import first_hit_rank

    ranks: list[int | None] = []
    for qr in payload.get("question_results") or []:
        candidates = qr.get("candidate_source_refs") or []
        refs = qr.get("expected_refs") or expected_by_question.get(str(qr.get("question_id") or "")) or []
        if candidates and refs:
            # Every question with a window contributes, whatever its funnel
            # verdict. Found live: restricting this to questions carrying a
            # root cause meant only the `retrieval` verdict counted, since
            # the funnel checks `suspected_ungrounded_answer` first and stops
            # there — on a real 50-question run that reported a payoff of 2
            # where the recorded windows held 23, an understatement bad
            # enough to argue against a change worth making.
            ranks.append(first_hit_rank(list(refs), candidates))
            continue

        # Without a window, the only measured position available is the one
        # the `ranking` cause recorded from its widened re-query. Other
        # causes found nothing to rank, which is exactly what None means.
        cause = qr.get("root_cause") or {}
        if cause.get("cause"):
            ranks.append((cause.get("evidence") or {}).get("rank"))
    return ranks


def _context_size_advice(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Phase 3 — the smallest context size that captures the whole payoff.

    Returns None when nothing would be gained, so the interface shows advice
    only where there is advice to give rather than proposing the status quo.
    """
    from core.eval.counterfactual import recommend_context_size

    top_k = int((payload.get("config") or {}).get("top_k") or 0)
    if top_k <= 0:
        return None
    # Falls back to whatever refs a root cause happens to carry, for runs
    # stored before the result recorded what it was measured against.
    expected_by_question = {
        str(qr.get("question_id") or ""): ((qr.get("root_cause") or {}).get("evidence") or {}).get("refs") or []
        for qr in payload.get("question_results") or []
    }
    ranks = _observed_ranks(payload, expected_by_question)
    if not ranks:
        return None
    # Bounded by the widest window anything was actually observed in: a
    # recommendation past that would be extrapolation, and the whole point of
    # this answer is that it is measured rather than guessed.
    max_k = max((r for r in ranks if r is not None), default=0)
    advice = recommend_context_size(ranks, top_k, max_k)
    return advice.to_dict() if advice else None


def _expected_refs_index(dataset: Any) -> dict[str, list[str]]:
    """Maps a question to its expected refs by id *and* by text.

    Found live: a dataset authored before per-question ids existed stores
    every question with `id: None`, so an id-keyed lookup collapsed all of
    them onto one entry and matched none. Every retrieval failure then
    reported "this question has no expected refs" — a confident, wrong
    verdict, and the worst kind, because it looks like an answer.

    Question text is the fallback key rather than the primary one because a
    dataset may legitimately contain two identically worded questions with
    different expected sources, whereas ids, when present, are unique.
    """
    index: dict[str, list[str]] = {}
    for q in getattr(dataset, "questions", []) or []:
        refs = list(q.get("article_refs") or [])
        if not refs:
            continue
        text = (q.get("question") or "").strip()
        if text:
            index.setdefault(text, refs)
        qid = q.get("id")
        if qid:
            index[str(qid)] = refs
    return index


def _diagnose_root_causes(
    result: Any, pipeline: Any, ref_resolver: Any, expected_refs: dict[str, list[str]],
) -> None:
    """Fills `root_cause` on every question that failed on the
    retrieval layer. Mutates `result` in place.

    Runs for the whole run rather than on demand for one question, which is
    the point: one verdict tells an engineer what went wrong once, the counts
    over a run tell them which kind of work would pay off most.

    Cost is one extra widened retrieval per *failed* question, not per
    question. A run where retrieval works pays nothing.

    Never raises. A diagnosis that cannot be produced leaves `root_cause`
    None, which the reader can tell apart from an analysis that ran and
    reached no conclusion (cause "unknown").
    """
    from core.eval.funnel import diagnose_question
    from core.eval.miss_diagnosis import diagnose_retrieval_miss
    from core.eval.root_cause import classify_retrieval_cause

    top_k = getattr(result.config, "top_k", 0) or 0
    for qr in result.question_results:
        try:
            verdict = diagnose_question(
                qr.answerability or "answerable", qr.metrics,
                qr.metrics.get("pre_rerank_recall_at_k"),
            )
            # Only the `retrieval` verdict is ambiguous about cause. `rerank`
            # already names one, `generation` is a different layer entirely,
            # and `not_applicable` has no retrieval failure to explain.
            if verdict.layer != "retrieval":
                continue

            refs = expected_refs.get(qr.question_id) or expected_refs.get((qr.question or "").strip()) or []
            presences = {
                ref: (ref_resolver.presence(ref) if ref_resolver is not None else "unknown")
                for ref in refs
            }

            miss = None
            # Skipped when a ref is already known absent: re-querying to
            # locate something the index does not contain wastes a retrieval
            # to confirm what the resolver just said.
            if pipeline is not None and refs and not any(v == "absent" for v in presences.values()):
                try:
                    miss = diagnose_retrieval_miss(
                        pipeline, qr.question, refs, widened_k=_ROOT_CAUSE_WIDENED_K,
                    )
                except Exception:
                    miss = None

            # How many chunks each expected unit occupies, which
            # separates a splitting problem from a wording one. A resolver
            # built without counts reports None, and the split reads that as
            # "cannot tell" rather than as "one chunk".
            counter = getattr(ref_resolver, "chunk_count", None)
            chunk_counts = {ref: (counter(ref) if counter else None) for ref in refs}

            qr.root_cause = classify_retrieval_cause(
                presences, miss, top_k, chunk_counts,
            ).to_dict()
        except Exception:
            continue  # one undiagnosable question must not cost the others


def _tuning_provenance(run_id: str, results: dict[str, Any]) -> dict[str, Any]:
    """What else was tried on these same questions before this run.

    A number reported after a search over configurations is the best of that
    search, and the more was tried the more of it is the search and the less
    of it is the system. Nothing tied the two together, so a figure arrived with no
    way of knowing whether it was the first thing anybody ran or the best of
    forty.

    Counted over the runs stored before this one on the same question set,
    which is the same store the list endpoint already reads.
    """
    this = results.get(run_id)
    if this is None:
        return {}
    dataset = this.dataset_name or this.config.dataset_name
    if not dataset:
        return {}
    earlier = [
        r for r in results.values()
        if (r.dataset_name or r.config.dataset_name) == dataset
        and r.started_at and this.started_at and r.started_at < this.started_at
    ]
    fusion_constants = {
        getattr(r.config, "rrf_k", None) for r in [*earlier, this]
        if getattr(r.config, "merge_strategy", "") == "rrf"
    }
    return {
        "dataset_name": dataset,
        "runs_before": len(earlier),
        "configurations_before": len({r.config.config_hash for r in earlier}),
        "fusion_constants_tried": sorted(c for c in fusion_constants if c is not None),
        "merge_strategy": getattr(this.config, "merge_strategy", ""),
    }


async def _corpus_manifest(realm_id: str, corpus_id: str) -> dict[str, Any]:
    """The registry's record of how this corpus was loaded.

    Empty when there is none, and empty is the honest answer: a corpus
    loaded before manifests existed, or by something that does not write
    one, has no record, and inventing one would let a check compare a run
    against a guess.
    """
    try:
        import adapters.mongodb as mdb
        doc = await mdb.find_one("corpora", {"realm_id": realm_id, "corpus_id": corpus_id})
    except Exception as exc:
        log = __import__("structlog").get_logger()
        log.warning("experiment.manifest.unreachable", corpus_id=corpus_id, error=str(exc))
        return {}
    return dict((doc or {}).get("manifest") or {})


async def _run_experiment_background(
    run_id: str, cfg: ExperimentConfig, dataset: Any, evaluator: Any, runner: ExperimentRunner,
    realm_id: str = "",
    qdrant_cfg: dict[str, Any] | None = None, opensearch_cfg: dict[str, Any] | None = None,
    retrieval_pins: list[Any] | None = None, ref_resolver: Any = None,
) -> None:
    """Runs the (potentially long, N-question) experiment
    off the request/response cycle, so create_experiment can return run_id
    immediately and a large or parallel run can't time out or block the
    gateway's event loop. `runner.run()` is plain synchronous Python (no
    asyncio dependency of its own) — asyncio.to_thread() runs it on a worker
    thread while this coroutine awaits it, freeing the event loop for other
    requests (including this run's own progress WebSocket) in the meantime.
    """
    def _on_progress(processed: int, total: int) -> None:
        # Called from the worker thread; list.append is GIL-atomic, so this
        # is safe to read concurrently from the WebSocket handler on the
        # event loop thread without a lock.
        _progress.setdefault(run_id, []).append(
            {"type": "progress", "processed": processed, "total": total}
        )

    try:
        result = await asyncio.to_thread(
            runner.run, cfg, dataset, evaluator=evaluator, on_progress=_on_progress,
            should_stop=lambda: run_id in _stop_requested,
            realm_id=realm_id, qdrant_cfg=qdrant_cfg, opensearch_cfg=opensearch_cfg,
            retrieval_pins=retrieval_pins,
        )
        # Found live: POST /{run_id}/stop kept accepting stop requests (and
        # replying {"status": "stopping"}) for a run whose question loop had
        # already exited — should_stop is only ever consulted from inside
        # that loop, so a stop landing during the `await _save` below would
        # have no effect, silently. Discarding from `_running` the moment
        # runner.run() returns (rather than only in `finally`, after the
        # Mongo/file write) closes that window — POST /{run_id}/stop now
        # correctly 404s once the loop has actually finished, instead of
        # accepting a stop it can never honor.
        _running.discard(run_id)
        result.run_id = run_id
        result.realm_id = realm_id
        # What the corpus this run queried was built from and built by. Read
        # here because core/ may not reach the registry, and recorded on the
        # run because a detector reads a run and nothing else: a check that
        # went to the database at read time would answer differently every
        # time the corpus was reloaded, about a run that had already happened.
        result.corpus_manifest = await _corpus_manifest(realm_id, cfg.corpus_id)
        # Record whether answerability was verified against the
        # index or merely trusted from the dataset's refs, so
        # core/eval/detectors.py#detect_unverified_coverage can say so
        # instead of leaving an unverified run indistinguishable from a
        # verified one.
        if ref_resolver is not None:
            result.coverage_check = {
                "checked": bool(getattr(ref_resolver, "checked", False)),
                "reason": str(getattr(ref_resolver, "reason", "")),
                # Beside the sentence and never instead of it: a run stored
                # before the identifier existed carries only the sentence, and
                # the finding falls back to it.
                "reason_id": str(getattr(ref_resolver, "reason_id", "")),
                "note": str(getattr(ref_resolver, "note", "")),
            }
        # Split every retrieval failure into its actual cause
        # before the result is stored, so the run carries the answer rather
        # than requiring a per-question button press later. Skipped for an
        # external RAG: the widened re-query needs this platform's own
        # retriever and embedder, which a black-box HTTP pipeline has not
        # got, and guessing on its behalf would be worse than saying nothing.
        if cfg.pipeline_source != "http":
            expected_refs = _expected_refs_index(dataset)
            try:
                diag_pipeline = await asyncio.to_thread(
                    runner._build_pipeline, cfg, realm_id, qdrant_cfg, opensearch_cfg, retrieval_pins,
                )
            except Exception:
                diag_pipeline = None
            await asyncio.to_thread(
                _diagnose_root_causes, result, diag_pipeline, ref_resolver, expected_refs,
            )
        await _save(result)
    except Exception as exc:
        log = __import__("structlog").get_logger()
        log.error("experiment.background.failed", run_id=run_id, error=str(exc))
        _errors[run_id] = str(exc)
    finally:
        _running.discard(run_id)
        _running_meta.pop(run_id, None)
        _stop_requested.discard(run_id)


@router.post("")
async def create_experiment(body: NewExperimentRequest) -> dict[str, Any]:
    cfg_data = dict(body.config)
    cfg_data["dataset_name"] = body.dataset_name
    cfg = ExperimentConfig(**cfg_data)
    run_id = str(uuid.uuid4())[:8]
    _progress[run_id] = []

    # Resolve external_rag_id -> a dict of the fields
    # HttpPipeline needs (url, headers, retrieve_endpoint, and the tier-2
    # request_template/response_mapping from) from the registered
    # ExternalRag record. Fetched up-front with await (Mongo is async; the
    # runner itself is sync and can't re-enter the event loop), then handed
    # to the runner as a pre-resolved closure.
    _external_rag_resolver = None
    if cfg.external_rag_id:
        import adapters.mongodb as mdb
        rag_doc = await mdb.find_one("external_rags", {"id": cfg.external_rag_id})
        if not rag_doc:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404, detail=f"external_rag_id {cfg.external_rag_id!r} not found"
            )
        _resolved = {
            "url": rag_doc["url"],
            "headers": rag_doc.get("headers") or {},
            "retrieve_endpoint": rag_doc.get("retrieve_endpoint"),
            "request_template": rag_doc.get("request_template"),
            "response_mapping": rag_doc.get("response_mapping"),
            # the follow-up — what this RAG declared it actually reads,
            # so `fetch_k` can be forwarded only where it will be applied.
            # The platform's own rule is never to send a knob absent from
            # this list, since a silently ignored parameter is exactly how a
            # configuration change looks applied while doing nothing.
            "supported_params": (rag_doc.get("capabilities") or {}).get("supported_params") or [],
        }
        _external_rag_resolver = lambda _rag_id: _resolved  # noqa: E731
        # Found live: a run's stored config left http_endpoint null whenever
        # external_rag_id was used (the URL only ever lived in the
        # ExternalRag record, resolved fresh right here) — the run detail
        # page could only show "External (—)", with no way to tell which
        # external service a failure (e.g. Connection refused) was even
        # against without manually looking up external_rag_id in
        # GET /external-rags. Backfilled here, AFTER cfg's config_hash was
        # already computed at construction (ExperimentConfig#_compute_hash
        # runs once, on validation) — display-only, doesn't shift the hash
        # if this RAG's registered URL changes later.
        cfg.external_rag_name = rag_doc.get("name")
        cfg.http_endpoint = _resolved["url"]

    runner = ExperimentRunner(
        registry=registry, external_rag_resolver=_external_rag_resolver,
    )
    # Resolved up-front (same reasoning as _external_rag_resolver
    # above: Mongo is async, ExperimentRunner.run() is sync) so an in_process
    # run against a non-default Realm hits THAT Realm's own Qdrant/OpenSearch
    # host:port instead of always the gateway's startup-time env-var instance
    # (see core/experiment/runner.py#_rebind_corpus_id). No-op (None, None)
    # for the pre-Realm/backward-compat "" realm_id case.
    _qdrant_cfg = None
    _opensearch_cfg = None
    # The pin lookup is opt-in and OFF by default. A run that did
    # not ask for the overlay never pays the store round-trip, so "disabled"
    # costs nothing rather than loading pins and ignoring them.
    #
    # Still never fetched for pipeline_source="http": an external RAG owns
    # its own retrieval and has no chunk_id space this platform's pins could
    # refer to.
    _retrieval_pins: list[Any] = []
    if body.realm_id and cfg.pipeline_source != "http":
        from services.api_gateway.routers.corpus import _get_realm_resource
        _qdrant_cfg = await _get_realm_resource(body.realm_id, "qdrant")
        _opensearch_cfg = await _get_realm_resource(body.realm_id, "opensearch")
        if cfg.retrieval_pins_enabled:
            _retrieval_pins = await _load_active_pins(body.realm_id, cfg.corpus_id)
    dataset = await _load_dataset(body.dataset_name, external_rag_id=cfg.external_rag_id)
    embedder = registry.resolve("embedder", cfg.embedder.component_id)
    # Coverage is resolved against the INDEX this run actually
    # searches, built once per run. External RAG keeps relying on each row's
    # explicit `answerability` field: the platform never
    # sees that RAG's corpus, so there is nothing here it could check.
    ref_resolver = None
    if not cfg.external_rag_id:
        ref_resolver = await _build_ref_resolver(
            realm_id=body.realm_id, corpus_id=cfg.corpus_id, qdrant_cfg=_qdrant_cfg,
        )
    evaluator = _CompositeEvaluator(embedder=embedder, top_k=cfg.top_k, ref_resolver=ref_resolver)

    import datetime
    _running.add(run_id)
    _running_meta[run_id] = {
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "dataset_name": body.dataset_name,
        "started_at": datetime.datetime.utcnow().isoformat(),
        "realm_id": body.realm_id,
    }
    asyncio.create_task(_run_experiment_background(
        run_id, cfg, dataset, evaluator, runner, realm_id=body.realm_id,
        qdrant_cfg=_qdrant_cfg, opensearch_cfg=_opensearch_cfg,
        retrieval_pins=_retrieval_pins, ref_resolver=ref_resolver,
    ))

    return {"run_id": run_id, "config_hash": cfg.config_hash, "status": "running"}


@router.post("/{run_id}/stop")
async def stop_experiment(run_id: str) -> dict[str, Any]:
    """Found live: no way to interrupt a run picking a slow model (or hitting
    a stuck external RAG) short of waiting out every remaining question.
    Cooperative, not immediate — ExperimentRunner.run() only checks
    should_stop between questions (see core/experiment/runner.py), so a
    single question already in flight still finishes; already-answered
    questions are kept, not discarded (same "partial results over nothing"
    posture as per-question error isolation)."""
    from fastapi import HTTPException
    if run_id not in _running:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} is not currently running")
    _stop_requested.add(run_id)
    return {"run_id": run_id, "status": "stopping"}


@router.websocket("/{run_id}/progress")
async def experiment_progress(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    try:
        sent = 0
        closed = False
        while True:
            events = _progress.get(run_id, [])
            for event in events[sent:]:
                await websocket.send_text(json.dumps(event))
                sent += 1
                # A publisher that says `done` itself ends the stream. Runs
                # never do — they finish by appearing in the results — but a
                # comparison does, and without this the loop below would poll
                # for a run id that will never become a result, until the
                # client disconnected.
                closed = closed or event.get("type") == "done"
            if closed:
                break
            if run_id in _errors:
                await websocket.send_text(json.dumps({"type": "error", "run_id": run_id, "detail": _errors[run_id]}))
                break
            if run_id not in _running and run_id in await _get_results() and sent >= len(events):
                await websocket.send_text(json.dumps({"type": "done", "run_id": run_id}))
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
