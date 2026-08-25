"""Settings router — active model, embedder mode, retriever config."""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

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
async def list_models(completion_only: bool = False) -> list[dict[str, Any]]:
    # `completion_only` — the question generator (services/api_gateway/
    # routers/generation.py) needs to offer only models that can actually
    # generate text, not the embedding models Ollama also serves locally
    # (e.g. BGE-M3/nomic-embed-text report capabilities=["embedding"], no
    # "completion" — confirmed live against a real local Ollama instance).
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 1),
                    "modified_at": m.get("modified_at", ""),
                    "capabilities": m.get("capabilities", []),
                }
                for m in data.get("models", [])
            ]
            if completion_only:
                models = [m for m in models if "completion" in m["capabilities"]]
            return models
    except Exception as e:
        return [{"name": os.getenv("OLLAMA_MODEL", "qwen3:8b"), "size_gb": 0, "error": str(e)}]
