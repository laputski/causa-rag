"""TruLens runner — RAG triad (context relevance / groundedness / answer
relevance), kept as a fourth independent cross-check alongside DeepEval and
Ragas (see eval/ragas_runner.py module docstring for why all three+ stay
separate rather than consolidating).

Calls the LiteLLM provider's feedback functions directly (context_relevance,
groundedness_measure_with_cot_reasons, relevance) rather than going through
TruLens's full app-instrumentation/dashboard machinery — we just need three
numbers per question, not a tracing UI.

Uses Ollama via LiteLLM's `ollama_chat/<model>` provider string (LiteLLM's
own Ollama integration, talks directly to the Ollama server — no separate
client library needed beyond what litellm already pulls in).

Usage:
    from eval.trulens_runner import TruLensRunner
    runner = TruLensRunner(pipeline=pipeline)  # judge model: eval/judge_model.py
    result = runner.run(dataset)
    # result.metrics = {"context_relevance": 0.7, "groundedness": 0.8, "answer_relevance": 0.75}
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

from eval.dataset import EvalDataset
from eval.judge_model import JUDGE_MODEL

_nltk_punkt_ready = False


def _ensure_nltk_punkt() -> None:
    """groundedness_measure_with_cot_reasons (called below) splits the
    statement into sentences via nltk's punkt tokenizer, downloaded on
    first use. On a stock macOS python.org install, the system trust store
    is missing an intermediate issuer cert nltk's download host needs,
    producing CERTIFICATE_VERIFY_FAILED — observed live. Pointing the
    default HTTPS context at certifi's CA bundle (already a transitive
    dependency here) fixes the missing-issuer gap WITHOUT disabling
    certificate verification — this is the same fix python.org's own
    "Install Certificates.command" applies, just done from code instead of
    requiring the user to run it manually.
    """
    global _nltk_punkt_ready
    if _nltk_punkt_ready:
        return
    import functools
    import ssl

    import certifi
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
        _nltk_punkt_ready = True
        return
    except LookupError:
        pass

    previous_context_factory = ssl._create_default_https_context
    ssl._create_default_https_context = functools.partial(
        ssl.create_default_context, cafile=certifi.where()
    )
    try:
        nltk.download("punkt_tab", quiet=True)
    finally:
        ssl._create_default_https_context = previous_context_factory
    _nltk_punkt_ready = True


@dataclass
class TruLensResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    dataset_name: str = ""
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""


class TruLensRunner:
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

    def _build_provider(self) -> Any:
        try:
            from trulens.providers.litellm import LiteLLM
        except ImportError as exc:
            raise RuntimeError(
                "trulens not installed: pip install -e '.[judges]'"
            ) from exc
        _ensure_nltk_punkt()
        return LiteLLM(
            model_engine=f"ollama_chat/{self._ollama_model}",
            completion_kwargs={
                "api_base": self._ollama_base_url,
                # Same qwen3-thinking-mode fix an external RAG applies in
                # its own generator, and eval/ragas_runner.py's ChatOllama
                # reasoning=False — litellm's ollama_chat provider maps
                # reasoning_effort not in {"low","medium","high"} to Ollama's
                # native `think: false`, the only passthrough it exposes
                # (see litellm.llms.ollama.chat.transformation.map_openai_params).
                "reasoning_effort": "none",
            },
        )

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

    def run(self, dataset: EvalDataset, *, max_questions: int | None = None, run_id: str = "") -> TruLensResult:
        provider = self._build_provider()

        questions = dataset.questions
        if max_questions is not None:
            questions = questions[:max_questions]

        aggregated: dict[str, list[float]] = {
            "context_relevance": [], "groundedness": [], "answer_relevance": [],
        }
        per_question: list[dict[str, Any]] = []

        for q in questions:
            answer_text, contexts = self._run_pipeline(q["question"])
            joined_context = "\n\n".join(contexts)

            # All three feedback functions return (score, reasons_dict) in
            # this trulens version, despite type hints suggesting a bare
            # float for context_relevance/relevance — confirmed live (the
            # un-unpacked tuple caused "unsupported operand type(s) for +:
            # 'int' and 'tuple'" in sum() when aggregating).
            ctx_relevance, _ = provider.context_relevance(question=q["question"], context=joined_context)
            groundedness, _ = provider.groundedness_measure_with_cot_reasons(
                source=joined_context, statement=answer_text,
            )
            answer_relevance, _ = provider.relevance(prompt=q["question"], response=answer_text)

            aggregated["context_relevance"].append(ctx_relevance)
            aggregated["groundedness"].append(groundedness)
            aggregated["answer_relevance"].append(answer_relevance)
            per_question.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "actual_output": answer_text,
                "context_relevance": ctx_relevance,
                "groundedness": groundedness,
                "answer_relevance": answer_relevance,
            })

        avg_metrics = {k: (sum(v) / len(v) if v else 0.0) for k, v in aggregated.items()}

        result = TruLensResult(
            metrics=avg_metrics,
            per_question=per_question,
            dataset_name=dataset.name,
            run_id=run_id,
        )

        out_file = self._results_dir / f"trulens_{dataset.name}_{int(result.timestamp)}.json"
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
