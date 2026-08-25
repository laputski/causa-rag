"""LLM-judge chunk coherence — scores a SAMPLE of already-indexed chunks for
internal coherence (does this chunk read as one self-contained, sensible
unit of text, or does it look cut off mid-thought / mid-sentence / merged
from unrelated fragments).

This is deliberately a different question from the final-answer judges
(DeepEval/Ragas/TruLens) — those score generation quality given retrieved
context; this scores the CHUNKING decision itself, independent of any
question or retrieval. Catches chunking-strategy regressions (e.g. the bare-
numeral and mid-word-split bugs found earlier in this corpus) that a
downstream answer-quality metric wouldn't directly point back to.

Sampled (not exhaustive — see corpus_health.py's near-duplicate/language
detectors for the same reasoning): an LLM call per chunk is too slow to run
over an entire corpus, so this reports an estimate over N chunks, not an
audit.

Uses a local Ollama model as judge — same dependency-free, offline approach
as adapters/ollama_generator.py and eval/deepeval_runner.py.

Usage:
    from eval.chunk_coherence_judge import ChunkCoherenceJudge
    judge = ChunkCoherenceJudge()  # judge model: eval/judge_model.py
    result = judge.run(chunks)  # chunks: list[core.models.Chunk] or dict-like
    # result.metrics = {"avg_coherence": 0.78, "n_incoherent": 12}
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from eval.judge_model import JUDGE_MODEL

_SAMPLE_SIZE = 150
_SCORE_RE = re.compile(r"\b([0-5])\b")

_JUDGE_PROMPT = """Ты оцениваешь качество разбиения текста на фрагменты (chunking) для системы поиска.
Прочитай фрагмент ниже и оцени по шкале от 0 до 5, насколько он выглядит как ЦЕЛЬНАЯ, самодостаточная единица текста:
- 5: фрагмент полный и понятный сам по себе, не обрезан ни в начале, ни в конце.
- 3: фрагмент в целом понятен, но есть небольшая шероховатость на границе.
- 0: фрагмент явно обрезан посередине слова/предложения, либо состоит только из служебного текста (номер, заголовок без содержания) без самостоятельного смысла.

Фрагмент:
---
{text}
---

Ответь ОДНИМ числом от 0 до 5, без пояснений."""


def _chunk_text(c: Any) -> str:
    return (c.get("text") if isinstance(c, dict) else getattr(c, "text", "")) or ""


def _chunk_id_of(c: Any) -> str:
    if isinstance(c, dict):
        return c.get("chunk_id", "") or c.get("id", "")
    return getattr(c, "chunk_id", "") or ""


@dataclass
class ChunkCoherenceResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_chunk: list[dict[str, Any]] = field(default_factory=list)
    sample_size: int = 0
    timestamp: float = field(default_factory=time.time)


class ChunkCoherenceJudge:
    def __init__(
        self,
        ollama_model: str = JUDGE_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        sample_size: int = _SAMPLE_SIZE,
        results_dir: str | pathlib.Path = "eval/results",
    ) -> None:
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url
        self._sample_size = sample_size
        self._results_dir = pathlib.Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def _build_generator(self) -> Any:
        from adapters.ollama_generator import OllamaGenerator
        return OllamaGenerator(base_url=self._ollama_base_url, model=self._ollama_model)

    def _score_one(self, generator: Any, text: str) -> int | None:
        if not text.strip():
            return 0
        response = generator.generate(
            _JUDGE_PROMPT.format(text=text[:2000]), temperature=0.0, max_tokens=8,
        )
        match = _SCORE_RE.search(response)
        return int(match.group(1)) if match else None

    def run(self, chunks: list[Any], *, corpus_id: str = "") -> ChunkCoherenceResult:
        eligible = [c for c in chunks if _chunk_text(c).strip()]
        sample = (
            random.sample(eligible, self._sample_size)
            if len(eligible) > self._sample_size
            else eligible
        )
        if not sample:
            return ChunkCoherenceResult(metrics={"avg_coherence": 0.0, "n_incoherent": 0.0}, sample_size=0)

        generator = self._build_generator()
        scores: list[int] = []
        per_chunk: list[dict[str, Any]] = []
        for c in sample:
            score = self._score_one(generator, _chunk_text(c))
            if score is None:
                continue
            scores.append(score)
            per_chunk.append({
                "chunk_id": _chunk_id_of(c),
                "score": score,
                "text_preview": _chunk_text(c)[:120],
            })

        n_incoherent = sum(1 for s in scores if s <= 2)
        avg_coherence = (sum(scores) / len(scores) / 5.0) if scores else 0.0

        result = ChunkCoherenceResult(
            metrics={"avg_coherence": avg_coherence, "n_incoherent": float(n_incoherent)},
            per_chunk=per_chunk,
            sample_size=len(scores),
        )

        out_file = self._results_dir / f"chunk_coherence_{corpus_id or 'default'}_{int(result.timestamp)}.json"
        out_file.write_text(
            json.dumps(
                {
                    "corpus_id": corpus_id,
                    "timestamp": result.timestamp,
                    "metrics": result.metrics,
                    "sample_size": result.sample_size,
                    "per_chunk": per_chunk,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return result
