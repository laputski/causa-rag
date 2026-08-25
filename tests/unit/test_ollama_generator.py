"""adapters/ollama_generator.py#OllamaGenerator — constrained-decoding support
added for the question generator (services/api_gateway/routers/generation.py),
which needs a parseable JSON response rather than free-form prose."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.ollama_generator import OllamaGenerator


def _mock_response(text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": text, "prompt_eval_count": 1, "eval_count": 1}
    return resp


def test_generate_omits_format_by_default() -> None:
    gen = OllamaGenerator(model="qwen3:8b")
    with patch("httpx.post", return_value=_mock_response()) as mock_post:
        gen.generate("hello")
    payload = mock_post.call_args.kwargs["json"]
    assert "format" not in payload


def test_generate_passes_response_format_as_ollama_format_field() -> None:
    gen = OllamaGenerator(model="qwen3:8b")
    with patch("httpx.post", return_value=_mock_response()) as mock_post:
        gen.generate("hello", response_format="json")
    payload = mock_post.call_args.kwargs["json"]
    assert payload["format"] == "json"
