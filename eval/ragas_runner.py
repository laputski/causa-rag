"""Ragas runner for RAG platform evaluation — kept alongside DeepEvalRunner
deliberately: six independent chunk-quality diagnostics. Both run
the same questions through their own independent LLM-judge implementations;
the value is the cross-check (agreement/disagreement between judges), not
consolidating into one number.

Uses Ollama as local LLM judge (no external API needed), via langchain-ollama
— ragas's own LLM wrapper interface expects a Langchain chat model. Ragas's
ResponseRelevancy metric additionally needs an embedding model — not every
Ollama deployment serves embeddings on its chat endpoint (observed live:
"This server does not support embeddings"), so embeddings default to this
platform's own BgeM3Embedder (see _BgeM3LangchainEmbeddings below) rather
than Ollama, unless an embedder is passed in explicitly.

Metrics: LLMContextPrecisionWithoutReference (does retrieved context look
relevant to the question, judged without a reference answer — matches what
we actually have, since the golden set's `ground_truth` is sparse),
LLMContextRecall (does the context cover what the reference answer needed),
ResponseRelevancy (does the answer address the question).

Usage:
    from eval.ragas_runner import RagasRunner
    runner = RagasRunner(pipeline=pipeline)  # judge model: eval/judge_model.py
    result = runner.run(dataset)
    # result.metrics = {"context_precision": 0.8, "context_recall": 0.7, "answer_relevancy": 0.75}
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

from eval.dataset import EvalDataset
from eval.judge_model import JUDGE_MODEL


@dataclass
class RagasResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    dataset_name: str = ""
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""


class _BgeM3LangchainEmbeddings:
    """Adapts adapters/bge_m3.py's BgeM3Embedder to the langchain_core
    Embeddings interface ragas expects. Not every Ollama deployment serves
    an embeddings-capable model on its chat endpoint (observed live:
    "This server does not support embeddings" from the same chat model
    used as judge) — BGE-M3 is the embedder this whole platform already
    relies on elsewhere, so reusing it here needs no extra Ollama model
    pulled and no separate embeddings-serving config."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]


class RagasRunner:
    def __init__(
        self,
        pipeline: Any,
        ollama_model: str = JUDGE_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        embedder: Any | None = None,
        results_dir: str | pathlib.Path = "eval/results",
    ) -> None:
        self._pipeline = pipeline
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url
        self._embedder = embedder
        self._results_dir = pathlib.Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def _build_judge(self) -> tuple[Any, Any]:
        try:
            from langchain_ollama import ChatOllama
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
        except ImportError as exc:
            raise RuntimeError(
                "ragas/langchain-ollama not installed: pip install -e '.[judges]'"
            ) from exc
        llm = LangchainLLMWrapper(ChatOllama(
            model=self._ollama_model, base_url=self._ollama_base_url,
            # Same qwen3-thinking-mode fix an external RAG applies in its
            # own generator — a thinking model makes ragas's many short judge
            # calls slow enough to risk timing out, and ChatOllama's own
            # `reasoning` field maps straight to Ollama's native `think` API
            # param, unlike deepeval's OllamaModel which has no such
            # passthrough (see eval/judge_model.py).
            reasoning=False,
        ))
        embedder = self._embedder
        if embedder is None:
            from adapters.bge_m3 import BgeM3Embedder
            embedder = BgeM3Embedder(use_real_model=True)
        embeddings = LangchainEmbeddingsWrapper(_BgeM3LangchainEmbeddings(embedder))
        return llm, embeddings

    def _run_pipeline(self, question: str) -> tuple[str, list[str]]:
        from core.models import QueryRequest

        req = QueryRequest(text=question)
        answer = self._pipeline.run(req)
        contexts: list[str] = []
        if answer.source_refs:
            contexts = [ref.text for ref in answer.source_refs if hasattr(ref, "text") and ref.text]
        if not contexts:
            contexts = [answer.text] if answer.text else [""]
        return answer.text or "", contexts

    def run(self, dataset: EvalDataset, *, max_questions: int | None = None, run_id: str = "") -> RagasResult:
        try:
            import warnings

            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.metrics import (
                LLMContextPrecisionWithoutReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError as exc:
            raise RuntimeError("ragas not installed: pip install -e '.[judges]'") from exc

        llm, embeddings = self._build_judge()
        metrics = [
            LLMContextPrecisionWithoutReference(llm=llm),
            LLMContextRecall(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=embeddings),
        ]

        questions = dataset.questions
        if max_questions is not None:
            questions = questions[:max_questions]

        samples: list[Any] = []
        per_question: list[dict[str, Any]] = []
        for q in questions:
            answer_text, contexts = self._run_pipeline(q["question"])
            reference = q.get("ground_truth", "") or ""
            samples.append(SingleTurnSample(
                user_input=q["question"],
                response=answer_text,
                retrieved_contexts=contexts,
                reference=reference,
            ))
            per_question.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "actual_output": answer_text,
                "expected_output": reference,
            })

        ragas_dataset = EvaluationDataset(samples=samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            eval_result = evaluate(dataset=ragas_dataset, metrics=metrics, show_progress=False)

        df = eval_result.to_pandas()
        name_map = {
            "llm_context_precision_without_reference": "context_precision",
            "context_precision": "context_precision",
            "llm_context_recall": "context_recall",
            "context_recall": "context_recall",
            "answer_relevancy": "answer_relevancy",
            "response_relevancy": "answer_relevancy",
        }
        avg_metrics: dict[str, float] = {}
        for col in df.columns:
            key = name_map.get(col)
            if key is None:
                continue
            series = df[col].dropna()
            if len(series):
                avg_metrics[key] = float(series.mean())

        result = RagasResult(
            metrics=avg_metrics,
            per_question=per_question,
            dataset_name=dataset.name,
            run_id=run_id,
        )

        out_file = self._results_dir / f"ragas_{dataset.name}_{int(result.timestamp)}.json"
        out_file.write_text(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "timestamp": result.timestamp,
                    "metrics": avg_metrics,
                    "per_question": per_question,
                    "run_id": run_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return result
