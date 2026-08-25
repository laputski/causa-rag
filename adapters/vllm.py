"""vLLM generator adapter — calls the OpenAI-compatible vLLM HTTP server."""
from __future__ import annotations

from typing import Any

import httpx


class VllmGenerator:
    """Calls a running vLLM server (OpenAI-compatible /v1/completions)."""

    generator_id = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "default",
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(self, prompt: str, **params: Any) -> str:
        max_tokens: int = params.get("max_tokens", 512)
        temperature: float = params.get("temperature", 0.0)
        # Same runaway-repetition guard as adapters/ollama_generator.py —
        # vLLM's OpenAI-compatible endpoint accepts this as an extension field.
        repetition_penalty: float = params.get("repetition_penalty", 1.3)

        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }
        response = httpx.post(
            f"{self._base_url}/v1/completions",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"]
