"""Settings router — active model, embedder mode, retriever config."""
from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["settings"])

_DEFAULT_SETTINGS = {
    "active_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    "embedder_mode": "real" if os.getenv("USE_REAL_BGE_M3") else "stub",
    "retriever_mode": "hybrid_rrf",
    "active_packs": [],  # Domain packs loaded at gateway startup
}

_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _settings_doc_id(realm_id: str | None) -> str:
    # Settings are scoped per Realm; "global" is the legacy
    # singleton used when no Realm is selected (back-compat, env-var path).
    return realm_id or "global"


async def _get_settings_doc(realm_id: str | None = None) -> dict[str, Any]:
    try:
        import adapters.mongodb as mdb
        doc = await mdb.find_one("settings", {"_id": _settings_doc_id(realm_id)})
        if doc:
            doc.pop("_id", None)
            return {**_DEFAULT_SETTINGS, **doc}
    except Exception:
        pass
    return dict(_DEFAULT_SETTINGS)


async def _save_settings(data: dict[str, Any], realm_id: str | None = None) -> None:
    try:
        import adapters.mongodb as mdb
        doc_id = _settings_doc_id(realm_id)
        await mdb.upsert_one("settings", {"_id": doc_id}, {"_id": doc_id, **data})
    except Exception:
        pass


# ── Models ────────────────────────────────────────────────────────────────────

class ModelSelectRequest(BaseModel):
    model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/settings")
async def get_settings(realm_id: str | None = None) -> dict[str, Any]:
    doc = await _get_settings_doc(realm_id)
    # `embedder_mode` describes the process answering this request, not a choice
    # anybody saved. It only ever reached the stored document by being part of
    # the defaults, and once stored it outlived the condition it described: a
    # gateway started with the real weights kept reporting "stub" because a
    # document written months earlier said so.
    #
    # `tools/doctor.py` and `install.sh` both decide "is this install degraded"
    # from this field, so a stale value there is a check that lies, which is the
    # exact failure this platform exists to catch. Computed fresh every time.
    doc["embedder_mode"] = "real" if os.getenv("USE_REAL_BGE_M3") else "stub"
    return doc


@router.put("/settings/model")
async def set_active_model(body: ModelSelectRequest, realm_id: str | None = None) -> dict[str, Any]:
    settings = await _get_settings_doc(realm_id)
    settings["active_model"] = body.model
    await _save_settings(settings, realm_id)
    # update in-memory generator if possible
    try:
        from core.registry import registry
        gen = registry.resolve("generator", "ollama")
        gen._model = body.model
    except Exception:
        pass
    return {"active_model": body.model, "status": "updated"}


@router.get("/models")
async def list_models(completion_only: bool = False, realm_id: str | None = None,
                      provider: str | None = None) -> list[dict[str, Any]]:
    """The models the Realm's own server offers
    (services/api_gateway/model_servers.py), or the gateway's local server's
    when no Realm is named. It was the gateway's for everyone, so a Realm
    served elsewhere was offered models it could not run.

    `provider` names which of the Realm's servers to ask, for a Realm that
    declares more than one; unnamed, the rule every other caller follows
    decides."""
    # `completion_only` — the question generator (services/api_gateway/
    # routers/generation.py) needs to offer only models that can actually
    # generate text, not the embedding models Ollama also serves locally
    # (e.g. BGE-M3/nomic-embed-text report capabilities=["embedding"], no
    # "completion" — confirmed live against a real local Ollama instance).
    import asyncio

    from services.api_gateway.model_servers import (
        OPENAI_COMPATIBLE,
        generator_for,
        model_resources,
        provider_of,
    )

    resources = await model_resources(realm_id)
    provider = provider_of(resources, provider)
    try:
        generator = generator_for(resources, provider=provider)
        if provider == OPENAI_COMPATIBLE:
            # This API lists names and says nothing of what a model can do,
            # so `completion_only` has nothing to filter by and filters nothing.
            names = await asyncio.to_thread(generator.list_models)
            return [{"name": name, "size_gb": 0, "modified_at": "", "capabilities": [],
                     "provider": provider} for name in names]
        base = generator.served_from()["address"] if any(resources.values()) else _OLLAMA_BASE
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 1),
                    "modified_at": m.get("modified_at", ""),
                    "capabilities": m.get("capabilities", []),
                    "provider": provider,
                }
                for m in data.get("models", [])
            ]
            if completion_only:
                models = [m for m in models if "completion" in m["capabilities"]]
            return models
    except Exception as e:
        if provider == OPENAI_COMPATIBLE:
            # The name below is the local default's. A hosted server that
            # could not be asked offers nothing, which is what is known.
            return []
        return [{"name": os.getenv("OLLAMA_MODEL", "qwen3:8b"), "size_gb": 0, "error": str(e)}]


# ── How long runs keep their retrieval windows ────────────────────────────────

class DebugRetentionRequest(BaseModel):
    """At least one run kept per realm and at least a day: a rule that
    cleared a run the moment it was saved would leave nobody anything to
    debug, and a store that small has lost the feature it was kept for."""

    keep_last_per_realm: int = Field(ge=1, le=1000)
    max_age_days: int = Field(ge=1, le=36500)
    max_mib: int = Field(ge=16, le=1024 * 1024)


@router.get("/settings/debug-retention")
async def get_debug_retention() -> dict[str, Any]:
    from dataclasses import asdict

    from services.api_gateway.run_retention import enabled, retention
    # Whether the pass runs by itself on this installation, beside the rule:
    # a rule that is saved and never applied reads as one that is in force.
    return {**asdict(await retention()), "automatic": enabled()}


@router.put("/settings/debug-retention")
async def set_debug_retention(body: DebugRetentionRequest) -> dict[str, Any]:
    from core.run_retention import Retention
    from services.api_gateway.run_retention import save_retention
    await save_retention(Retention(**body.model_dump()))
    return body.model_dump()


@router.post("/settings/debug-retention/preview")
async def preview_debug_retention() -> dict[str, Any]:
    """What the next pass would clear, without clearing it."""
    from services.api_gateway.run_retention import prune
    return await prune(write=False)


# ── Who judges answers against their statements ──────────────────────────────

class JudgeRequest(BaseModel):
    """Which of the Realm's model servers judges, and with which model.

    An empty provider leaves the choice to the rule every other caller follows
    (services/api_gateway/model_servers.py); an empty model means the
    platform's default, which exists on the local server only.
    """

    provider: Literal["", "ollama", "openai_compatible"] = ""
    model: str = ""


@router.get("/settings/judge")
async def get_judge(realm_id: str | None = None) -> dict[str, Any]:
    from eval.judge_model import ASSERTION_JUDGE_MODEL
    stored = (await _get_settings_doc(realm_id)).get("judge") or {}
    return {"provider": stored.get("provider") or "", "model": stored.get("model") or "",
            "default_model": ASSERTION_JUDGE_MODEL}


@router.put("/settings/judge")
async def set_judge(body: JudgeRequest, realm_id: str | None = None) -> dict[str, Any]:
    settings = await _get_settings_doc(realm_id)
    settings["judge"] = body.model_dump()
    await _save_settings(settings, realm_id)
    return body.model_dump()
