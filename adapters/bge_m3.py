"""BGE-M3 embedder adapter.

Without USE_REAL_BGE_M3 the model is not loaded: a deterministic stub is used so the
full pipeline can run without GPU.  The real sentence-transformers call is
behind a flag so it can be enabled when the model weights are available.
"""
from __future__ import annotations

import hashlib
from typing import Any

_VECTOR_DIM = 1024  # BGE-M3 output dimension


def _stub_vector(text: str) -> list[float]:
    """Deterministic pseudo-embedding for testing (no GPU needed)."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)  # noqa: S324
    rng_state = seed
    result: list[float] = []
    for _ in range(_VECTOR_DIM):
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        # map to [-1, 1]
        result.append((rng_state / 0xFFFFFFFFFFFFFFFF) * 2 - 1)
    return result


class BgeM3Embedder:
    """Embedder backed by BGE-M3 (a stub unless USE_REAL_BGE_M3 is set).

    Embedding cache: in-memory dict keyed by sha256(text).
    The same cache key is used by Redis in later stages (same hash).
    """

    embedder_id = "bge_m3"
    version = "1.0.0"

    @property
    def is_real_model(self) -> bool:
        """Whether the vectors come from the model or from a hash.

        The id and the version above are class constants, so a corpus indexed
        by the stub and one indexed by the model are indistinguishable by
        either: the record says bge_m3 1.0.0 in both cases. That is why the
        catalogue's entry about a stub embedder had no reliable evidence, and
        this is the evidence, recorded where the indexing happened and never
        inferred from scores afterwards.
        """
        return self._use_real_model

    def __init__(self, use_real_model: bool | None = None) -> None:
        """`use_real_model` unset reads the environment; passed, it decides.

        It used to be `use_real_model or os.getenv(...)`, so the argument could
        turn the real model on and could never turn it off: a caller asking for
        the stub in a process where USE_REAL_BGE_M3=true was set received the
        real model and no indication of it. Found while staging a
        stub-embedder failure on the proving ground, where the half that was
        supposed to be broken retrieved exactly as well as the healthy one.

        Unset is the default and keeps every existing caller reading the
        environment, which is how the gateway and the ingestion command both
        pass their choice today.
        """
        import os
        self._cache: dict[str, list[float]] = {}
        self._use_real_model = (
            os.getenv("USE_REAL_BGE_M3", "").lower() == "true"
            if use_real_model is None else use_real_model
        )
        self._model: Any = None
        self._hits: int = 0
        self._misses: int = 0
        if self._use_real_model:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._model = SentenceTransformer("BAAI/bge-m3")
        except ImportError as exc:
            raise RuntimeError("sentence-transformers not installed") from exc

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _compute(self, text: str) -> list[float]:
        if self._use_real_model and self._model is not None:
            vec: list[float] = self._model.encode(text).tolist()
            return vec
        return _stub_vector(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            key = self._cache_key(text)
            if key in self._cache:
                self._hits += 1
            else:
                self._misses += 1
                self._cache[key] = self._compute(text)
            result.append(self._cache[key])
        return result

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def cache_hit_ratio(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def cache_stats(self) -> dict[str, int | float]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": self.cache_hit_ratio,
            "size": self.cache_size,
        }
