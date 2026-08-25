"""eval/judge_model.py — single source of truth for the eval-suite judge model.

Found live: the judge model was a literal string duplicated across 6 files;
changing it meant editing up to 6 places (see module docstring). Pins that
the env var override actually works and that every runner/test file reads
from this one module rather than its own hardcoded default.
"""
from __future__ import annotations

import importlib

import eval.judge_model as judge_model


def test_default_judge_model_used_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    importlib.reload(judge_model)
    assert judge_model.JUDGE_MODEL == judge_model.DEFAULT_JUDGE_MODEL


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_JUDGE_MODEL", "some-other-model:latest")
    importlib.reload(judge_model)
    assert judge_model.JUDGE_MODEL == "some-other-model:latest"
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    importlib.reload(judge_model)


def test_every_runner_defaults_to_the_shared_judge_model():
    """Regression guard against a runner silently reverting to its own
    hardcoded literal instead of importing JUDGE_MODEL."""
    import inspect

    from eval.chunk_coherence_judge import ChunkCoherenceJudge
    from eval.deepeval_runner import DeepEvalRunner
    from eval.ragas_runner import RagasRunner
    from eval.trulens_runner import TruLensRunner

    for cls in (DeepEvalRunner, RagasRunner, TruLensRunner, ChunkCoherenceJudge):
        default = inspect.signature(cls.__init__).parameters["ollama_model"].default
        assert default == judge_model.JUDGE_MODEL, f"{cls.__name__} not wired to eval.judge_model.JUDGE_MODEL"
