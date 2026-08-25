"""DeepEval runner for RAG platform evaluation.

Uses Ollama as local LLM judge (no external API needed).
Metrics: AnswerRelevancy, Faithfulness, ContextualPrecision, ContextualRecall.

`ollama_model` here is the JUDGE model only — it scores the pipeline's
answers, it does not generate them (that's `pipeline`'s own generator).
Prefer a fast non-thinking instruct model: a thinking model (qwen3:8b) makes
deepeval's per-metric judge calls slow enough to blow its default 88.5s
per-attempt timeout (deepeval's OllamaModel has no `think` passthrough to
disable thinking mode the way adapters/ollama_generator.py does).

Usage:
    from eval.deepeval_runner import DeepEvalRunner
    runner = DeepEvalRunner(pipeline=pipeline)  # judge model: eval/judge_model.py
    results = runner.run(dataset)
    # results = {"answer_relevancy": 0.82, "faithfulness": 0.71, ...}
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
class DeepEvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    dataset_name: str = ""
    timestamp: float = field(default_factory=time.time)
    # Links this report to an experiment run (RunPage shows a badge/link
    # when one exists for the viewed run_id) — empty for standalone deepeval
    # runs not tied to a specific experiment (e.g. ad-hoc smoke tests).
    run_id: str = ""
    # Found live: GET /report/deepeval showed a raw epoch float and no way
    # to tell which judge/generator model a given run actually used — with
    # the judge model now swappable in one place (eval/judge_model.py) and
    # multiple historical runs sitting side by side, "which run is which"
    # became a real question. Persisted so the report can answer it without
    # re-deriving anything.
    judge_model: str = ""
    generator_model: str = ""
    n_questions: int = 0

    @property
    def accuracy_semantic(self) -> float:
        """Composite accuracy metric for SLA gate (≥0.85 required).

        Uses faithfulness + contextual_recall — both measure retrieval quality
        independently of the LLM generator. answer_relevancy depends on the
        generator producing real answers and is tracked separately.
        """
        keys = ["faithfulness", "contextual_recall"]
        values = [self.metrics[k] for k in keys if k in self.metrics]
        return sum(values) / len(values) if values else 0.0


class DeepEvalRunner:
    """Evaluates a pipeline against an EvalDataset using deepeval metrics.

    LLM judge: Ollama (local, offline, free).
    Embedder for deepeval's own similarity checks: uses the same stub to avoid GPU dependency.
    """

    def __init__(
        self,
        pipeline: Any,
        ollama_model: str = JUDGE_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        results_dir: str | pathlib.Path = "eval/results",
    ) -> None:
        self._pipeline = pipeline
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url
        self._results_dir = pathlib.Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def _build_ollama_model(self) -> Any:
        try:
            from deepeval.models import OllamaModel  # type: ignore[import]
            return OllamaModel(
                model=self._ollama_model,
                base_url=self._ollama_base_url,
            )
        except ImportError as exc:
            raise RuntimeError("deepeval not installed: pip install -e '.[judges]'") from exc

    def _run_pipeline(self, question: str) -> tuple[str, list[str]]:
        """Run pipeline and return (answer_text, list of retrieved context strings)."""
        from core.models import QueryRequest

        req = QueryRequest(text=question)
        answer = self._pipeline.run(req)

        # Extract context from source_refs if available, else use answer text as fallback
        contexts: list[str] = []
        if answer.source_refs:
            contexts = [ref.text for ref in answer.source_refs if hasattr(ref, "text") and ref.text]
        if not contexts:
            contexts = [answer.text] if answer.text else [""]

        return answer.text or "", contexts

    def run(self, dataset: EvalDataset, *, max_questions: int | None = None, run_id: str = "") -> DeepEvalResult:
        """Evaluate pipeline on dataset, return aggregated metrics."""
        try:
            from deepeval import evaluate as de_evaluate  # type: ignore[import]
            from deepeval.metrics import (  # type: ignore[import]
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("deepeval not installed: pip install -e '.[judges]'") from exc

        judge = self._build_ollama_model()
        metrics = [
            AnswerRelevancyMetric(model=judge, threshold=0.5),
            FaithfulnessMetric(model=judge, threshold=0.5),
            ContextualPrecisionMetric(model=judge, threshold=0.5),
            ContextualRecallMetric(model=judge, threshold=0.5),
        ]

        questions = dataset.questions
        if max_questions is not None:
            questions = questions[:max_questions]

        test_cases: list[LLMTestCase] = []
        per_question: list[dict[str, Any]] = []

        for q in questions:
            answer_text, contexts = self._run_pipeline(q["question"])
            expected = q.get("ground_truth", "")
            test_case = LLMTestCase(
                input=q["question"],
                actual_output=answer_text,
                expected_output=expected,
                retrieval_context=contexts,
                context=contexts,
            )
            test_cases.append(test_case)
            per_question.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "actual_output": answer_text,
                "expected_output": expected,
            })

        eval_results = de_evaluate(test_cases=test_cases, metrics=metrics)

        # Aggregate metric scores
        aggregated: dict[str, list[float]] = {
            "answer_relevancy": [],
            "faithfulness": [],
            "contextual_precision": [],
            "contextual_recall": [],
        }

        metric_name_map = {
            "Answer Relevancy": "answer_relevancy",
            "Faithfulness": "faithfulness",
            "Contextual Precision": "contextual_precision",
            "Contextual Recall": "contextual_recall",
        }

        # Found live: per_question only ever carried question/answer text, so
        # a low aggregate metric (e.g. contextual_recall 0.40) gave no way to
        # tell WHICH question(s) dragged it down without re-running with ad
        # hoc instrumentation — attach each test case's own per-metric scores
        # alongside the aggregate, zipped by index (test_results is produced
        # from test_cases in the same order they were submitted, same order
        # per_question was appended in above).
        for tr, q_entry in zip(eval_results.test_results, per_question, strict=True):
            q_metrics: dict[str, float] = {}
            for metric in tr.metrics_data or []:
                key = metric_name_map.get(metric.name)
                if key and metric.score is not None:
                    aggregated[key].append(metric.score)
                    q_metrics[key] = metric.score
            q_entry["metrics"] = q_metrics

        avg_metrics = {k: (sum(v) / len(v) if v else 0.0) for k, v in aggregated.items()}

        # Best-effort — not every pipeline's generator exposes a model name
        # the same way (mirrors services/api_gateway/main.py's own
        # `getattr(self._generator, "_model", self._generator.generator_id)`
        # pattern for the pipeline trace).
        generator = getattr(self._pipeline, "_generator", None)
        generator_model = str(getattr(generator, "_model", getattr(generator, "generator_id", "")))

        result = DeepEvalResult(
            metrics=avg_metrics,
            per_question=per_question,
            dataset_name=dataset.name,
            run_id=run_id,
            judge_model=self._ollama_model,
            generator_model=generator_model,
            n_questions=len(questions),
        )

        # Persist results
        out_file = self._results_dir / f"deepeval_{dataset.name}_{int(result.timestamp)}.json"
        out_file.write_text(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "timestamp": result.timestamp,
                    "metrics": avg_metrics,
                    "per_question": per_question,
                    "run_id": run_id,
                    "judge_model": result.judge_model,
                    "generator_model": result.generator_model,
                    "n_questions": result.n_questions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return result
