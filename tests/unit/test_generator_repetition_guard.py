"""adapters/ollama_generator.py + adapters/vllm.py — repetition-penalty guard.

Locks in a real generation bug found in run a914f2eb: with no explicit
repeat_penalty, qwen3:8b (via Ollama, temperature=0.1) degenerated into a
runaway "1.1.1.1...." token loop on ~3% of answers, running all the way to
the num_predict cap instead of stopping (e.g. pikoap_closed_001,
brak_closed_002, budget_closed_005, upk_open_004). Retrieval was correct in
every one of these (recall_at_k=1.0) — purely a generation-decoding issue.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.ollama_generator import OllamaGenerator
from adapters.vllm import VllmGenerator


def test_ollama_generator_sends_repeat_penalty_by_default() -> None:
    gen = OllamaGenerator()
    with patch("adapters.ollama_generator.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"response": "ok", "prompt_eval_count": 1, "eval_count": 1},
        )
        gen.generate("a question")
    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"]["repeat_penalty"] == 1.3


def test_ollama_generator_repeat_penalty_is_overridable() -> None:
    gen = OllamaGenerator()
    with patch("adapters.ollama_generator.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"response": "ok", "prompt_eval_count": 1, "eval_count": 1},
        )
        gen.generate("a question", repeat_penalty=1.5)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"]["repeat_penalty"] == 1.5


def test_vllm_generator_sends_repetition_penalty_by_default() -> None:
    gen = VllmGenerator()
    with patch("adapters.vllm.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"choices": [{"text": "ok"}]},
        )
        gen.generate("a question")
    payload = mock_post.call_args.kwargs["json"]
    assert payload["repetition_penalty"] == 1.3


def test_vllm_generator_repetition_penalty_is_overridable() -> None:
    gen = VllmGenerator()
    with patch("adapters.vllm.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"choices": [{"text": "ok"}]},
        )
        gen.generate("a question", repetition_penalty=1.5)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["repetition_penalty"] == 1.5
