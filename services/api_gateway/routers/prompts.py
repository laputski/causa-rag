"""Prompts REST router — CRUD for prompt templates.

Read order:  MongoDB primary → file fallback.
Write order: file (sync, used by pipeline) + MongoDB (async dual-write).
The pipeline reads prompts synchronously from files via core.prompt_store,
so files remain the source of truth for the running pipeline.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import adapters.mongodb as mdb
from core.prompt_store import prompt_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/prompts", tags=["prompts"])


async def _mongo_upsert(data: dict[str, Any]) -> None:
    try:
        await mdb.upsert_one("prompts", {"id": data["id"]}, data)
    except Exception as exc:
        logger.warning("MongoDB prompt write failed: %s", exc)


async def _mongo_delete(prompt_id: str) -> None:
    try:
        await mdb.delete_one("prompts", {"id": prompt_id})
    except Exception as exc:
        logger.warning("MongoDB prompt delete failed: %s", exc)


class PromptCreateRequest(BaseModel):
    name: str
    description: str = ""
    template: str
    # Realm scoping, mirroring datasets/corpus_ingests/external_rags:
    # optional so an old client / the file-fallback path still validates.
    realm_id: str | None = None


@router.get("")
async def list_prompts(realm_id: str | None = None) -> list[dict[str, Any]]:
    # Filtered by realm_id when given, same convention as GET /datasets and
    # GET /corpus — a prompt with no realm_id (pre-scoping) is invisible
    # under any Realm filter rather than shown to every Realm (see
    # the design notes). Which prompt is *active* is now also Realm-scoped
    # (core.prompt_store.get_active/set_active take realm_id) — this list
    # just needs to agree with the file store on realm_id, which create_prompt
    # now writes to both.
    try:
        query = {"realm_id": realm_id} if realm_id else {}
        docs = await mdb.find_many("prompts", query=query, sort=[("version", 1)])
        if docs:
            return [{k: v for k, v in d.items() if k != "_id"} for d in docs]
        if realm_id:
            return []
    except Exception as exc:
        logger.warning("MongoDB prompts read failed, falling back to files: %s", exc)
    if realm_id:
        return []
    return [t.to_dict() for t in prompt_store.list()]


@router.get("/{prompt_id}")
async def get_prompt(prompt_id: str) -> dict[str, Any]:
    try:
        doc = await mdb.find_one("prompts", {"id": prompt_id})
        if doc:
            return {k: v for k, v in doc.items() if k != "_id"}
    except Exception as exc:
        logger.warning("MongoDB prompt get failed, falling back to file: %s", exc)
    t = prompt_store.get(prompt_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id!r} not found") from None
    return t.to_dict()


@router.post("", status_code=201)
async def create_prompt(body: PromptCreateRequest) -> dict[str, Any]:
    # Version is Realm-scoped, computed from Mongo (where realm_id lives) —
    # not from prompt_store.list(), which is a shared, Realm-agnostic file
    # store (see the design notes "Prompts scoping"). Found live: a Realm's
    # first-ever prompt landed on prompt_v5 because another Realm already had 4
    # prompts in that same shared file store. Same fallback shape as
    # list_prompts: an unscoped request trusts Mongo's global count, but
    # only falls back to the file store's global count when Mongo has
    # nothing to say (down, or empty with no realm_id to scope by) — a
    # realm-scoped request with zero Mongo docs legitimately means "this
    # Realm's first prompt", not "go count everyone else's".
    try:
        query = {"realm_id": body.realm_id} if body.realm_id else {}
        docs = await mdb.find_many("prompts", query=query)
        if docs or body.realm_id:
            next_version = max((d.get("version", 0) for d in docs), default=0) + 1
        else:
            next_version = max((t.version for t in prompt_store.list()), default=0) + 1
    except Exception as exc:
        logger.warning("MongoDB prompts read (versioning) failed, falling back to global file count: %s", exc)
        next_version = max((t.version for t in prompt_store.list()), default=0) + 1

    # The generated id must still be globally unique in the shared file
    # store regardless of Realm, so a Realm-scoped version number can
    # collide with another Realm's existing file (e.g. two Realms' first
    # prompt both computing version 1) — disambiguate with a suffix.
    existing_ids = {t.id for t in prompt_store.list()}
    prompt_id = f"prompt_v{next_version}"
    suffix = 1
    while prompt_id in existing_ids:
        prompt_id = f"prompt_v{next_version}-{suffix}"
        suffix += 1
    data = {
        "id": prompt_id,
        "name": body.name,
        "version": next_version,
        "description": body.description,
        "is_active": False,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "template": body.template,
        # Written to the FILE now too, not just Mongo — core.prompt_store's
        # get_active/set_active scope by this field (see its module
        # docstring), so it has to be the pipeline's own synchronous source
        # of truth, not metadata bolted on afterward.
        "realm_id": body.realm_id,
    }
    t = prompt_store.save(data)
    await _mongo_upsert(t.to_dict())
    return t.to_dict()


@router.put("/{prompt_id}/activate")
async def activate_prompt(prompt_id: str) -> dict[str, Any]:
    try:
        active = prompt_store.set_active(prompt_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id!r} not found") from None
    # Sync activation state to MongoDB for every prompt PromptStore.set_active
    # touched (its own Realm group only — see core.prompt_store's module
    # docstring). A pre-migration file with no realm_id of its own falls back
    # to whatever Mongo already had for it, so an un-migrated prompt doesn't
    # lose its Mongo-only realm_id tag the first time any prompt is activated.
    existing_realm_ids: dict[str, str | None] = {}
    try:
        for doc in await mdb.find_many("prompts"):
            existing_realm_ids[doc["id"]] = doc.get("realm_id")
    except Exception as exc:
        logger.warning("MongoDB prompts read (pre-activate) failed: %s", exc)
    for t in prompt_store.list():
        await _mongo_upsert({**t.to_dict(), "realm_id": t.realm_id or existing_realm_ids.get(t.id)})
    return {**active.to_dict(), "realm_id": active.realm_id or existing_realm_ids.get(active.id)}


@router.delete("/{prompt_id}", status_code=204)
async def delete_prompt(prompt_id: str) -> None:
    try:
        prompt_store.delete(prompt_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id!r} not found") from None
    await _mongo_delete(prompt_id)


# ── AI-drafted prompt (pre-fills the new-prompt form, doesn't save anything) ──
# Found live: prompts were only ever written by hand — no way to get a
# first draft tailored to a specific corpus's actual content/domain and a
# specific model's own quirks. Mirrors services/api_gateway/routers/
# generation.py's question generator (same "sample real chunks, ask the
# chosen Ollama model, tolerant JSON parse" shape) — a single one-shot call,
# not a background job, since there's exactly one output to wait for
# (generate_questions' async-job-plus-WS-progress pattern exists specifically
# for a multi-minute BATCH of many LLM calls, not one).

class PromptGenerateRequest(BaseModel):
    realm_id: str
    corpus_id: str
    model: str
    strategy: str = "structure_aware"
    embedder: str = "bge_m3"


# The meta-prompt: what the LLM is asked in order to write a system prompt for a
# realm. Written in Russian, like the built-in generation templates in
# generation.py, and kept that way for the same settled reason: the realms it
# serves hold Russian corpora, and a realm working in another language edits the
# drafted prompt, which it does anyway.
_PROMPT_GENERATE_META_TEMPLATE = """Ты помогаешь настроить RAG-систему (retrieval-augmented generation).

Вот несколько примерных фрагментов из корпуса документов, с которым будет работать система:

{excerpts}

Составь системный промпт-шаблон для LLM-модели "{model}", которая будет отвечать на вопросы пользователя по этому корпусу, используя найденные в нём фрагменты текста в качестве контекста.

ВАЖНО: итоговый шаблон ДОЛЖЕН содержать ровно два плейсхолдера — буквально `{{context}}` (сюда будет подставлен найденный контекст) и `{{query}}` (сюда будет подставлен вопрос пользователя). Не заменяй эти плейсхолдеры реальным текстом из примеров выше — они должны остаться в шаблоне буквально, как есть: система сама подставит в них нужный текст при каждом запросе.

Ответь СТРОГО в формате JSON, без пояснений до или после:
{{"name": "короткое название промпта", "description": "краткое описание назначения (1 предложение)", "template": "полный текст шаблона, обязательно содержащий {{context}} и {{query}}"}}
"""


@router.post("/generate")
async def generate_prompt_draft(body: PromptGenerateRequest) -> dict[str, Any]:
    """Samples a few real chunks from the chosen corpus and asks the chosen
    Ollama model to draft {name, description, template} for the new-prompt
    form — the user still reviews/edits before saving, this never writes
    anything itself (same "draft only" posture as /generate/questions)."""
    import asyncio
    import random

    from adapters.ollama_generator import OllamaGenerator
    from services.api_gateway.routers.corpus import _get_realm_resource, _resolve_qdrant
    from services.api_gateway.routers.generation import _parse_llm_json

    qdrant_cfg = await _get_realm_resource(body.realm_id, "qdrant") if body.realm_id else None
    try:
        qdrant = _resolve_qdrant(body.corpus_id, body.strategy, body.embedder, qdrant_cfg, body.realm_id)
        chunks = qdrant.scroll_all()
    except Exception as e:
        logger.error("generate_prompt_draft.qdrant_unavailable: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Corpus index unavailable: {e}") from e
    if not chunks:
        raise HTTPException(status_code=503, detail=f"Corpus {body.corpus_id!r} has no indexed chunks")

    sample = random.sample(chunks, k=min(3, len(chunks)))
    excerpts = "\n\n---\n\n".join(c.text[:600] for c in sample)
    prompt = _PROMPT_GENERATE_META_TEMPLATE.format(excerpts=excerpts, model=body.model)

    generator = OllamaGenerator(model=body.model)
    try:
        raw = await asyncio.to_thread(generator.generate, prompt, response_format="json", temperature=0.4)
    except Exception as e:
        logger.error("generate_prompt_draft.llm_call_failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}") from e

    parsed = _parse_llm_json(raw, required_keys=("template",))
    if parsed is None or "{context}" not in parsed["template"] or "{query}" not in parsed["template"]:
        raise HTTPException(
            status_code=502,
            detail="Model did not return a usable template (unparseable response, or missing "
            "{context}/{query} placeholders) — try again or a different model",
        )
    return {
        "name": parsed.get("name") or "",
        "description": parsed.get("description") or "",
        "template": parsed["template"],
    }
