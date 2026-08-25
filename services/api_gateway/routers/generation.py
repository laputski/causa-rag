"""Question generator — LLM-drafted control questions grounded in real
indexed chunks of a chosen corpus, using a chosen Ollama model.

Two resource families live here, not in datasets.py: this module talks to
Qdrant (corpus sampling) and Ollama (generation) — datasets.py stays a pure
Mongo-CRUD layer over the `datasets` collection (layer isolation, see
the design notes).

Generated questions are never written to Mongo by this module — this
endpoint only returns drafts. `OllamaGenerator`'s own docstring
(adapters/ollama_generator.py) records a real incident (qwen3:8b looping on
some generations even with mitigation) — LLM output here is not
unconditionally trustworthy, so a human reviews/edits drafts in the same UI
used for manually-added questions before anything is saved via
datasets.py's per-question endpoints (POST /datasets/{id}/questions[/batch]).

`article_refs` are never asked of the LLM — they're derived deterministically
from the sampled chunk(s)' own `source_code`/`article_no` metadata, the same
"{source_code}/{article_no}" shape core/eval/retrieval_metrics.py matches
golden refs against. The LLM only ever writes `question`/`reference_answer`.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, datetime
from random import Random
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

log = structlog.get_logger()

router = APIRouter(tags=["generation"])

_PRESETS_COLLECTION = "generation_presets"

# job_id -> ordered progress events, same in-memory poll-buffer pattern as
# corpus.py's ingestion _progress / experiments.py's run _progress — a
# generation batch is a single one-shot job with no "list all jobs" need, so
# (unlike experiments.py) there's no separate _running/_errors set: the
# background task's own "done"/"error" event, embedding the final drafts, is
# the only signal the WebSocket handler needs.
_progress: dict[str, list[dict[str, Any]]] = {}

# ── Generation presets — a separate, lighter concept than prompts.py's ───────
# PromptTemplate: that model is tied to the file-backed core.prompt_store
# singleton (dual-write, numbered `version`, one process-wide `is_active`
# template for the chat pipeline) — reusing it here would need either an
# `activate()` that means nothing for a generation preset, or bypassing
# prompt_store entirely for "generation-purpose" docs. Presets here are
# simpler on purpose: multiple simultaneously-usable, freely named templates,
# picked one-per-generation-call, no version/active concept at all.
#
# Presets are entirely optional — a Realm with zero presets can still
# generate; picking one is a way to customize wording, not a prerequisite.
# An earlier version auto-seeded three default presets into every new Realm on
# first use, all of them phrased for one subject area, including realms that had
# nothing to do with it (found live: a realm holding equipment documentation was
# offered a preset written entirely in another field's vocabulary). Removed:
# presets are now purely
# opt-in, managed through a real CRUD UI (ui/src/pages/PresetsPage.tsx) rather
# than force-seeded. _BUILTIN_TEMPLATES below replaces the seed's other job, a
# domain-neutral fallback so generation still works with no preset chosen.
#
# The templates themselves are written in Russian, which makes them
# domain-neutral but not language-neutral: generating against an English corpus
# with no preset produces Russian questions about English text.
#
# That is deliberate and settled. The realms these serve hold Russian corpora,
# and translating the templates would change what their generator emits for no
# gain. A realm working in another language creates a preset, which is exactly
# what presets are for and costs one form.

_BUILTIN_TEMPLATES: dict[str, str] = {
    "single": (
        "Ты — эксперт по составлению контрольных вопросов для оценки RAG-систем.\n"
        "Тебе дан текстовый фрагмент из корпуса документов. Составь ОДИН вопрос и эталонный "
        "ответ на него, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном тексте — не добавляй "
        "фактов, которых там нет.\n\n"
        "Тип вопроса: {question_type}\n{question_type_hint}\n\n"
        "Текст фрагмента:\n{chunk_text}\n\n"
        "Верни строго один плоский JSON-объект без пояснений и без вложенных объектов:\n"
        '{"question": "...", "reference_answer": "..."}\n'
        "reference_answer — обычный текст ответа, а не JSON и не структура с полями "
        "вроде answer/reasoning."
    ),
    "pair": (
        "Тебе даны тексты ДВУХ фрагментов одного документа. Составь ОДИН вопрос, "
        "требующий сравнения или сопоставления этих фрагментов (чем отличаются, как "
        "соотносятся, что приоритетнее), и эталонный ответ, основанный исключительно на "
        "этих двух текстах.\n\n"
        "Фрагмент A: {chunk_text_a}\n\nФрагмент B: {chunk_text_b}\n\n"
        "Верни строго один плоский JSON-объект без вложенных объектов:\n"
        '{"question": "...", "reference_answer": "..."}\n'
        "reference_answer — обычный текст ответа, а не JSON и не структура с полями "
        "вроде answer/reasoning."
    ),
    "range": (
        "Тебе даны тексты нескольких соседних фрагментов одного документа, идущих подряд. "
        "Составь ОДИН обзорный вопрос об их содержании в целом и эталонный ответ, кратко "
        "суммирующий положения всех предоставленных фрагментов.\n\n"
        "Фрагменты: {chunks_text_joined}\n\n"
        "Верни строго один плоский JSON-объект без вложенных объектов:\n"
        '{"question": "...", "reference_answer": "..."}\n'
        "reference_answer — обычный текст ответа, а не JSON и не структура с полями "
        "вроде answer/reasoning."
    ),
}

_QUESTION_TYPE_HINTS = {
    "closed": "Вопрос должен предполагать ответ да/нет с обоснованием.",
    "open": "Вопрос должен требовать развёрнутого ответа, не предполагающего простого да/нет.",
    "clarifying": "Вопрос должен уточнять значение термина или понятия, встречающегося в тексте.",
}


class GenerationPresetCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    template: str = Field(min_length=1)
    realm_id: str | None = None


class GenerationPresetUpdateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    template: str = Field(min_length=1)


@router.get("/generation-presets")
async def list_generation_presets(realm_id: str | None = None) -> list[dict[str, Any]]:
    import adapters.mongodb as mdb
    query = {"realm_id": realm_id} if realm_id else {}
    docs = await mdb.find_many(_PRESETS_COLLECTION, query=query, sort=[("created_at", 1)])
    return [{k: v for k, v in d.items() if k != "_id"} for d in docs]


@router.post("/generation-presets", status_code=201)
async def create_generation_preset(body: GenerationPresetCreateRequest) -> dict[str, Any]:
    import adapters.mongodb as mdb
    doc = {
        "id": str(uuid.uuid4())[:8],
        "name": body.name,
        "description": body.description,
        "template": body.template,
        "realm_id": body.realm_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    await mdb.insert_one(_PRESETS_COLLECTION, dict(doc))
    return doc


@router.put("/generation-presets/{preset_id}")
async def update_generation_preset(
    preset_id: str, body: GenerationPresetUpdateRequest, realm_id: str | None = None,
) -> dict[str, Any]:
    import adapters.mongodb as mdb
    query: dict[str, Any] = {"id": preset_id}
    if realm_id:
        query["realm_id"] = realm_id
    existing = await mdb.find_one(_PRESETS_COLLECTION, query)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id!r} not found")
    updates = {"name": body.name, "description": body.description, "template": body.template}
    await mdb.update_one(_PRESETS_COLLECTION, query, {"$set": updates})
    return {**{k: v for k, v in existing.items() if k != "_id"}, **updates}


@router.delete("/generation-presets/{preset_id}", status_code=204)
async def delete_generation_preset(preset_id: str, realm_id: str | None = None) -> None:
    import adapters.mongodb as mdb
    query: dict[str, Any] = {"id": preset_id}
    if realm_id:
        query["realm_id"] = realm_id
    deleted = await mdb.delete_one(_PRESETS_COLLECTION, query)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id!r} not found")


# ── Chunk sampling — strategy depends on question_type ───────────────────────
# closed/open/clarifying/unknown: 1 chunk. comparative: 2 chunks sharing a
# source_code. navigational: 3-8 chunks sharing a source_code, contiguous by
# article_no. QdrantRetriever.scroll()'s offset is an opaque Qdrant cursor,
# not a number — no cheap "random offset" exists, so this reuses the same
# scroll_all()+random.sample() pattern corpus_health already established
# (services/api_gateway/routers/corpus.py) rather than inventing a new one.

_ARTICLE_NO_SPLIT_RE = re.compile(r"[.\-]")


def _article_no_sort_key(article_no: str) -> tuple[float, ...]:
    try:
        return tuple(int(p) for p in _ARTICLE_NO_SPLIT_RE.split(article_no))
    except ValueError:
        return (float("inf"),)


def _sample_single(chunks: list[Any], n: int, rng: Random) -> list[list[Any]]:
    picked = rng.sample(chunks, min(n, len(chunks)))
    return [[c] for c in picked]


def _group_by_source_code(chunks: list[Any]) -> dict[str, list[Any]]:
    by_source: dict[str, list[Any]] = {}
    for c in chunks:
        source_code = c.metadata.get("source_code")
        article_no = c.metadata.get("article_no")
        if source_code and article_no:
            by_source.setdefault(source_code, []).append(c)
    return by_source


def _sample_pairs(chunks: list[Any], n: int, rng: Random) -> list[list[Any]]:
    by_source = {sc: cs for sc, cs in _group_by_source_code(chunks).items() if len(cs) >= 2}
    candidates = list(by_source)
    results: list[list[Any]] = []
    attempts = 0
    while len(results) < n and candidates and attempts < n * 5:
        attempts += 1
        source_code = rng.choice(candidates)
        results.append(rng.sample(by_source[source_code], 2))
    return results


def _sample_ranges(chunks: list[Any], n: int, rng: Random, size: tuple[int, int] = (3, 8)) -> list[list[Any]]:
    by_source = {sc: cs for sc, cs in _group_by_source_code(chunks).items() if len(cs) >= size[0]}
    candidates = list(by_source)
    results: list[list[Any]] = []
    attempts = 0
    while len(results) < n and candidates and attempts < n * 5:
        attempts += 1
        source_code = rng.choice(candidates)
        ordered = sorted(by_source[source_code], key=lambda c: _article_no_sort_key(c.metadata.get("article_no", "")))
        k = min(rng.randint(*size), len(ordered))
        start = rng.randint(0, len(ordered) - k)
        results.append(ordered[start : start + k])
    return results


def _sample_chunk_groups(chunks: list[Any], question_type: str, n: int, rng: Random | None = None) -> list[list[Any]]:
    rng = rng or Random()
    if question_type == "comparative":
        groups = _sample_pairs(chunks, n, rng)
    elif question_type == "navigational":
        groups = _sample_ranges(chunks, n, rng)
    else:
        groups = []
    if len(groups) < n:
        # Honest degradation, not a hard failure: no matching pair/range
        # found (e.g. a corpus with no multi-article documents) falls back
        # to single-chunk questions for the remainder rather than dropping
        # the whole batch.
        groups += _sample_single(chunks, n - len(groups), rng)
    return groups


def _article_refs(group: list[Any]) -> list[str]:
    """Realm-agnostic by design — see core/eval/retrieval_metrics.py#extract_ref_id
    for the matching fallback used on the retrieval side (must stay in sync:
    a ref this side produces has to be reproducible from a SourceRef's own
    fields on the retrieval side, or recall_at_k/precision_at_k can never
    match it). `article_no` needs a one-numbered-file-per-unit corpus layout
    (a legal-code convention); `structural_path` (the chunking-derived
    heading breadcrumb) is populated regardless of file layout, so a corpus
    that isn't organized that way (e.g. a single large manual) still gets a
    usable ref instead of none.

    Found live: a general document corpus (technical manuals, no
    legal-document-code system at all) has no `source_code` metadata at
    all — every chunk used to be skipped outright here, so every question
    generated against such a corpus got a completely empty article_refs,
    universally classified "out_of_scope" downstream regardless of whether
    the corpus actually covers the content. Falls back to `doc_id` (always
    present on a Chunk) the same way extract_ref_id now does."""
    refs = []
    for c in group:
        source_code = c.metadata.get("source_code")
        if source_code:
            article_no = c.metadata.get("article_no")
            if article_no:
                refs.append(f"{source_code}/{article_no}")
            elif c.structural_path:
                refs.append(f"{source_code}#{c.structural_path}")
            continue
        if c.structural_path:
            refs.append(f"{c.doc_id}#{c.structural_path}")
        else:
            refs.append(c.doc_id)
    return refs


def _render_prompt(template: str, question_type: str, group: list[Any]) -> str:
    substitutions = {
        "{question_type}": question_type,
        "{question_type_hint}": _QUESTION_TYPE_HINTS.get(question_type, ""),
    }
    if len(group) == 1:
        substitutions["{chunk_text}"] = group[0].text
    elif len(group) == 2:
        substitutions["{chunk_text_a}"] = group[0].text
        substitutions["{chunk_text_b}"] = group[1].text
    else:
        substitutions["{chunks_text_joined}"] = "\n\n---\n\n".join(c.text for c in group)

    rendered = template
    for key, value in substitutions.items():
        rendered = rendered.replace(key, value)
    return rendered


def _parse_llm_json(
    raw: str, required_keys: tuple[str, ...] = ("question", "reference_answer"),
) -> dict[str, Any] | None:
    """Tolerant JSON extraction — the model may still wrap the object in
    prose or a code fence even with response_format="json" requested.

    `required_keys` defaults to the question-generator's own shape so every
    existing call site is unaffected; other callers with a different
    expected JSON shape (e.g. [[services/api_gateway/routers/prompts.py#generate_prompt_draft]]'s
    `{"template": ...}`) pass their own."""
    import json

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not all(obj.get(k) for k in required_keys):
        return None
    return obj


def _clean_reference_answer(value: str) -> str:
    """Defends against the model wrapping `reference_answer` in its own
    nested JSON despite the prompt explicitly forbidding it — a real,
    observed qwen3:8b quirk (found live: `{"answer": "Да", "reasoning":
    "..."}` as the value of reference_answer). Unwraps that one recognizable
    {"answer": ..., "reasoning"?: ...} shape; anything else is returned
    unchanged rather than guessed at."""
    import json

    stripped = value.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return value
    try:
        inner = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value
    if isinstance(inner, dict) and isinstance(inner.get("answer"), str):
        reasoning = inner.get("reasoning")
        return f"{inner['answer']}. {reasoning}" if isinstance(reasoning, str) and reasoning else inner["answer"]
    return value


class QuestionTypeCount(BaseModel):
    question_type: str
    n_questions: int = Field(ge=1, le=50)


class GenerateQuestionsRequest(BaseModel):
    realm_id: str
    corpus_id: str
    model: str
    # Optional — no preset picked falls back to _BUILTIN_TEMPLATES,
    # keyed by chunk-group shape (see _resolve_template) rather than one
    # domain-specific default. See the "Presets are entirely optional" note
    # above _BUILTIN_TEMPLATES for why this isn't required.
    preset_id: str | None = None
    # A batch can mix question types in one go (e.g. 5 closed + 3 open + 2
    # clarifying) instead of one uniform type for the whole request — each
    # entry is sampled/generated independently via _sample_chunk_groups, then
    # concatenated (see generate_questions below). Defaults to the old
    # single-type shape's defaults (10 "open") so a caller that doesn't care
    # about the mix still gets sensible behavior with no fields set.
    type_counts: list[QuestionTypeCount] = Field(
        default_factory=lambda: [QuestionTypeCount(question_type="open", n_questions=10)], min_length=1,
    )
    strategy: str = "structure_aware"
    embedder: str = "bge_m3"


def _resolve_template(preset: dict[str, Any] | None, group: list[Any]) -> str:
    """The chosen preset's template if one was picked; otherwise a built-in,
    domain-neutral template selected by the group's own chunk-shape (single/
    pair/range) — same shape dispatch _sample_chunk_groups already uses to
    decide HOW to sample, applied here to decide which placeholder set the
    fallback template must offer."""
    if preset is not None:
        return preset["template"]
    if len(group) == 1:
        return _BUILTIN_TEMPLATES["single"]
    if len(group) == 2:
        return _BUILTIN_TEMPLATES["pair"]
    return _BUILTIN_TEMPLATES["range"]


def _generate_one_sync(generator: Any, preset: dict[str, Any] | None, question_type: str, group: list[Any]) -> str:
    """The actual blocking HTTP call to Ollama — run via asyncio.to_thread so
    the event loop stays free to serve this same job's progress WebSocket
    while a batch (one blocking call per group) is in flight."""
    prompt = _render_prompt(_resolve_template(preset, group), question_type, group)
    return generator.generate(prompt, response_format="json", temperature=0.4)


async def _run_generation_background(
    job_id: str, body: GenerateQuestionsRequest, preset: dict[str, Any] | None,
    groups: list[list[Any]], group_types: list[str],
) -> None:
    from adapters.ollama_generator import OllamaGenerator

    def _emit(event: dict[str, Any]) -> None:
        _progress.setdefault(job_id, []).append(event)

    # One instance for the whole batch — model doesn't change mid-request.
    # Constructed ad-hoc, per-call: this must NOT touch core.registry's
    # shared "ollama" generator (used by live /query traffic and controlled
    # by PUT /settings/model) — a one-off generation task picking its own
    # model has nothing to do with the platform's active chat model.
    generator = OllamaGenerator(model=body.model)

    log.info(
        "generate_questions.start", realm_id=body.realm_id, corpus_id=body.corpus_id,
        model=body.model, type_counts=[tc.model_dump() for tc in body.type_counts], n_groups=len(groups),
    )
    _emit({"type": "start", "total": len(groups)})

    drafts: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()
    for i, (group, question_type) in enumerate(zip(groups, group_types, strict=True), start=1):
        chunk_ids = [c.chunk_id for c in group]
        try:
            raw = await asyncio.to_thread(_generate_one_sync, generator, preset, question_type, group)
        except Exception as e:
            log.warning("generate_questions.item_failed", index=i, total=len(groups), reason=str(e))
            failed.append({"reason": str(e), "chunk_ids": chunk_ids})
            _emit({"type": "progress", "processed": i, "total": len(groups), "generated": len(drafts), "failed": len(failed)})
            continue
        parsed = _parse_llm_json(raw)
        if parsed is None:
            # A preset replaces the built-in template wholesale, so one that
            # does not ask for the JSON object this parses back fails every
            # item identically. That is the likely cause when a preset is in
            # play, and naming it turns "20 skipped" into something the
            # operator can act on.
            reason = "unparseable_response"
            if preset is not None:
                reason = (
                    "unparseable_response: the preset must ask the model for "
                    '{"question": "...", "reference_answer": "..."} and nothing else'
                )
            log.warning("generate_questions.item_failed", index=i, total=len(groups), reason=reason)
            failed.append({"reason": reason, "chunk_ids": chunk_ids})
            _emit({"type": "progress", "processed": i, "total": len(groups), "generated": len(drafts), "failed": len(failed)})
            continue

        log.info("generate_questions.item_done", index=i, total=len(groups))
        drafts.append({
            "question": parsed["question"],
            "reference_answer": _clean_reference_answer(parsed["reference_answer"]),
            "question_type": question_type,
            "article_refs": _article_refs(group),
            "provenance": {
                "origin": "generated",
                "model": body.model,
                "corpus_id": body.corpus_id,
                "chunk_id": chunk_ids if len(chunk_ids) > 1 else chunk_ids[0],
                "preset_id": body.preset_id,
                "generated_at": now,
            },
        })
        # processed/generated/total drives the frontend's "N of M questions
        # generated" progress display (found live: a bare elapsed-seconds
        # timer gave no sense of how much of a multi-minute batch was left).
        _emit({"type": "progress", "processed": i, "total": len(groups), "generated": len(drafts), "failed": len(failed)})

    log.info("generate_questions.done", n_drafts=len(drafts), n_failed=len(failed))
    _emit({"type": "done", "drafts": drafts, "failed": failed})


@router.post("/generate/questions")
async def generate_questions(body: GenerateQuestionsRequest) -> dict[str, Any]:
    """Samples real chunks from the chosen corpus and kicks off a background
    job that asks the chosen Ollama model to draft a question+answer per
    sample using the chosen preset (or a built-in fallback template — see
    _BUILTIN_TEMPLATES — when none is chosen) — nothing is written to Mongo
    here (see module docstring). Returns immediately with a job_id; the frontend
    listens on the paired WebSocket (/generate/questions/{job_id}/progress)
    for live progress and the final {"drafts": [...], "failed": [...]}
    payload — the same async-job-plus-WS-poll shape corpus.py's ingestion and
    experiments.py's runs already use. A batch is one blocking LLM call per
    sampled chunk group and can run minutes; running it synchronously on the
    request would tie up the event loop for the whole batch, starving this
    same job's own progress socket (found live: with no progress signal at
    all, a multi-minute batch looked identical to a hung request).

    `body.type_counts` lets one batch mix question types (e.g. 5 closed + 3
    open + 2 clarifying) instead of one uniform type for the whole n_questions
    — each entry is sampled independently via _sample_chunk_groups (so a
    comparative/navigational entry still gets its own pair/range sampling,
    not just a label on a single-chunk group) and the resulting groups are
    concatenated with a parallel list of which type each group belongs to,
    consumed by _run_generation_background.

    `body.preset_id` is optional — see _BUILTIN_TEMPLATES above for the
    domain-neutral fallback used when it's absent."""
    import adapters.mongodb as mdb
    from services.api_gateway.routers.corpus import _get_realm_resource, _resolve_qdrant

    preset: dict[str, Any] | None = None
    if body.preset_id:
        preset = await mdb.find_one(_PRESETS_COLLECTION, {"id": body.preset_id})
        if not preset:
            raise HTTPException(status_code=404, detail=f"Preset {body.preset_id!r} not found")

    qdrant_cfg = await _get_realm_resource(body.realm_id, "qdrant") if body.realm_id else None
    try:
        qdrant = _resolve_qdrant(body.corpus_id, body.strategy, body.embedder, qdrant_cfg, body.realm_id)
        chunks = qdrant.scroll_all()
    except Exception as e:
        log.error(
            "generate_questions.qdrant_unavailable", realm_id=body.realm_id, corpus_id=body.corpus_id,
            qdrant_cfg=qdrant_cfg, error=str(e), exc_info=True,
        )
        raise HTTPException(status_code=503, detail=f"Corpus index unavailable: {e}") from e
    if not chunks:
        log.warning("generate_questions.empty_corpus", realm_id=body.realm_id, corpus_id=body.corpus_id)
        raise HTTPException(status_code=503, detail=f"Corpus {body.corpus_id!r} has no indexed chunks")

    groups: list[list[Any]] = []
    group_types: list[str] = []
    for tc in body.type_counts:
        type_groups = _sample_chunk_groups(chunks, tc.question_type, tc.n_questions)
        groups.extend(type_groups)
        group_types.extend([tc.question_type] * len(type_groups))

    job_id = str(uuid.uuid4())[:8]
    _progress[job_id] = []
    asyncio.create_task(_run_generation_background(job_id, body, preset, groups, group_types))

    return {"job_id": job_id, "status": "started", "n_groups": len(groups)}


@router.websocket("/generate/questions/{job_id}/progress")
async def generate_questions_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        sent = 0
        while True:
            events = _progress.get(job_id, [])
            for event in events[sent:]:
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
                sent += 1
                if event.get("type") in ("done", "error"):
                    return
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
