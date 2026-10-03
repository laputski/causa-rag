"""Models served over the OpenAI-compatible HTTP API.

The API many servers speak: hosted providers, and local ones too (a local
model server usually offers it beside its own). One adapter reaches any of
them by an address, a key and a model name.

What this cannot promise is that two servers behave alike behind the same
API. A seed is sent when a run fixed one, and whether the server honours it
is the server's affair, so each call keeps what was sent and what the reply
said about itself (`last_call`), and a run records it. Nothing here claims a
repeat run gives the same answers.
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx

from adapters.connections import is_local

PROVIDER = "openai_compatible"


class ModelServerNotConfigured(RuntimeError):
    """No address to call. Raised instead of calling a default, because the
    only default there could be is somebody else's service."""


def _stated(resource: dict[str, Any] | None, name: str, fallback: Any) -> Any:
    value = (resource or {}).get(name)
    return fallback if value is None or value == "" else value


class _Server:
    """An address and a key, and how a request to it is made."""

    def __init__(self, base_url: str | None, api_key: str | None, timeout: float) -> None:
        self.base_url = (base_url or "").rstrip("/") or None
        self.api_key = api_key or None
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @property
    def remote(self) -> bool:
        return not is_local(urlparse(self.base_url or "").hostname or "")

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise ModelServerNotConfigured(
                "no OpenAI-compatible server is configured: give the Realm a resource of that type "
                "with its base_url, or set OPENAI_BASE_URL for the gateway")
        return f"{self.base_url}{path}"

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(self._url(path), json=payload, headers=self.headers, timeout=self.timeout)
        self._raise_for_status(response)
        return dict(response.json())

    def get(self, path: str) -> dict[str, Any]:
        response = httpx.get(self._url(path), headers=self.headers, timeout=self.timeout)
        self._raise_for_status(response)
        return dict(response.json())

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        # The status and the server's own words, and never the request: the
        # library's error names the URL, and the headers travel with it.
        if response.status_code >= 400:
            raise RuntimeError(
                f"the model server answered {response.status_code}: {response.text[:300]}")

    def list_models(self) -> list[str]:
        return [str(m["id"]) for m in self.get("/models").get("data") or [] if m.get("id")]


class OpenAICompatibleGenerator:
    """A generator calling `/chat/completions` on an OpenAI-compatible server."""

    generator_id = PROVIDER

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        seed: int | None = None,
        from_environment: bool = False,
    ) -> None:
        if from_environment:
            base_url = base_url or os.getenv("OPENAI_BASE_URL")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            model = model or os.getenv("OPENAI_MODEL")
        self._server = _Server(base_url, api_key, timeout)
        self._model = model or ""
        self._seed = seed
        #: What the last call sent and what the reply said about itself.
        self.last_call: dict[str, Any] = {}

    def _copy(self, **changed: Any) -> OpenAICompatibleGenerator:
        current = {"base_url": self._server.base_url, "api_key": self._server.api_key,
                   "model": self._model, "timeout": self._server.timeout, "seed": self._seed}
        return OpenAICompatibleGenerator(**{**current, **changed})

    def with_model(self, model: str) -> Any:
        """See `core.interfaces.ChoosingItsModel`."""
        return self if not model or model == self._model else self._copy(model=model)

    def with_seed(self, seed: int) -> Any:
        """See `core.interfaces.FixingItsSampling`. Sent, and not promised."""
        return self if seed is None or seed == self._seed else self._copy(seed=seed)

    def for_realm(self, resources: dict[str, dict[str, Any] | None] | None) -> Any:
        """A copy calling the Realm's own server. See
        `core.interfaces.ServedByARealm`."""
        resource = (resources or {}).get(PROVIDER)
        if not resource:
            return self
        return self._copy(
            base_url=_stated(resource, "base_url", self._server.base_url),
            api_key=_stated(resource, "api_key", self._server.api_key),
            model=_stated(resource, "model", self._model))

    def served_from(self) -> dict[str, Any]:
        """Where answers come from, in words that carry no key."""
        return {"provider": PROVIDER, "address": self._server.base_url or "", "model": self._model,
                "remote": self._server.remote}

    def list_models(self) -> list[str]:
        return self._server.list_models()

    def generate(self, prompt: str, **params: Any) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.1),
            "max_tokens": params.get("max_tokens", 1024),
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        if params.get("response_format"):
            payload["response_format"] = {"type": "json_object"}
        data = self._server.post("/chat/completions", payload)
        usage = data.get("usage") or {}
        self._last_token_counts = {
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
        self.last_call = {
            "seed_sent": self._seed is not None,
            "served_model": data.get("model") or "",
            "system_fingerprint": data.get("system_fingerprint") or "",
        }
        choices = data.get("choices") or [{}]
        return str((choices[0].get("message") or {}).get("content") or "").strip()


def _slug(model: str) -> str:
    """A model's name as it may stand inside a collection's name."""
    return "".join(ch if ch.isalnum() else "-" for ch in model.lower()).strip("-")


class OpenAICompatibleEmbedder:
    """An embedder calling `/embeddings` on an OpenAI-compatible server.

    Its identifier carries the model's name, so a corpus loaded with it gets
    a collection of its own and never shares one with vectors from another
    model, which is what the collection's name is for. The size of its
    vectors is asked of the server once, since the API does not state it.
    """

    version = "remote"
    #: Never a stand-in: the vectors come from whatever model the server runs.
    is_real_model = True

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        batch: int = 64,
        from_environment: bool = False,
    ) -> None:
        if from_environment:
            base_url = base_url or os.getenv("OPENAI_BASE_URL")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            model = model or os.getenv("OPENAI_EMBEDDING_MODEL")
        self._server = _Server(base_url, api_key, timeout)
        self._model = model or ""
        self._batch = batch
        self._vector_size: int | None = None
        self._texts_embedded = 0

    @property
    def embedder_id(self) -> str:
        return f"{PROVIDER}__{_slug(self._model)}" if self._model else PROVIDER

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = len(self.embed(["a"])[0])
        return self._vector_size

    def for_realm(self, resources: dict[str, dict[str, Any] | None] | None) -> Any:
        """A copy calling the Realm's own server. See
        `core.interfaces.ServedByARealm`. The model is the resource's
        `embedding_model`, which is not the model that writes answers."""
        resource = (resources or {}).get(PROVIDER)
        if not resource:
            return self
        return OpenAICompatibleEmbedder(
            base_url=_stated(resource, "base_url", self._server.base_url),
            api_key=_stated(resource, "api_key", self._server.api_key),
            model=_stated(resource, "embedding_model", self._model),
            timeout=self._server.timeout, batch=self._batch)

    def served_from(self) -> dict[str, Any]:
        return {"provider": PROVIDER, "address": self._server.base_url or "", "model": self._model,
                "remote": self._server.remote}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._model:
            raise ModelServerNotConfigured(
                "no embedding model is named: set the Realm resource's embedding_model, or "
                "OPENAI_EMBEDDING_MODEL for the gateway")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            part = texts[start:start + self._batch]
            data = self._server.post("/embeddings", {"model": self._model, "input": part}).get("data") or []
            if len(data) != len(part):
                raise RuntimeError(
                    f"the model server returned {len(data)} vectors for {len(part)} texts")
            # By the index each vector carries, since the API does not promise
            # the order of its reply.
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            vectors.extend([float(x) for x in item["embedding"]] for item in ordered)
        self._texts_embedded += len(texts)
        return vectors

    def cache_stats(self) -> dict[str, int | float]:
        """Nothing is remembered between calls, and the shape is the local
        embedder's, which a load prints."""
        return {"hits": 0, "misses": self._texts_embedded, "hit_ratio": 0.0, "size": 0}
