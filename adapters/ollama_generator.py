"""Ollama generator adapter — calls local Ollama HTTP API."""
from __future__ import annotations

import os
from typing import Any

import httpx


class OllamaGenerator:
    """Calls a running Ollama server (/api/generate endpoint).

    Drop-in replacement for GeneratorStub and VllmGenerator.
    Requires Ollama running locally (docker compose up -d or native).
    """

    generator_id = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        seed: int | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self._timeout = timeout
        # None means "sample as the server would", which is what every caller
        # got before a run's seed reached here, and what the chat path still
        # gets: a conversation is not a measurement and nobody repeats one
        # expecting the same words.
        self._seed = seed

    def with_model(self, model: str) -> Any:
        """A copy of this running another model. See
        `core.interfaces.ChoosingItsModel`.

        The pipelines the gateway registers are built once at start-up with
        whatever model was current then, so a run asking for another one used
        the start-up model and the only way to change it was a process-wide
        setting that moved every Realm's chat with it.
        """
        if not model or model == self._model:
            return self
        return OllamaGenerator(base_url=self._base_url, model=model,
                               timeout=self._timeout, seed=self._seed)

    def with_seed(self, seed: int) -> Any:
        """A copy of this sampling the same way every time. See
        `core.interfaces.FixingItsSampling`.

        Measured on the model this platform runs: identical requests
        can give different answers and a seed removes it, occasionally
        and depending on the question. The numbers are in
        `tests/unit/test_a_run_repeats_itself.py`.
        """
        if seed is None or seed == self._seed:
            return self
        return OllamaGenerator(base_url=self._base_url, model=self._model,
                               timeout=self._timeout, seed=seed)

    def generate(self, prompt: str, **params: Any) -> str:
        temperature: float = params.get("temperature", 0.1)
        num_predict: int = params.get("max_tokens", 1024)
        # Default repeat_penalty (1.1) wasn't enough to stop qwen3:8b from
        # degenerating into runaway "1.1.1.1...." token loops at low
        # temperature (observed in run a914f2eb — ~3% of answers ran to the
        # num_predict cap repeating a digit/dot pattern instead of stopping).
        repeat_penalty: float = params.get("repeat_penalty", 1.3)

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # disable thinking mode (qwen3, deepseek-r1, etc.)
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
            },
        }
        # Sent only when a run asked for one. Absent, the server samples as it
        # would have, which keeps every caller that never had a seed exactly
        # where it was.
        if self._seed is not None:
            payload["options"]["seed"] = self._seed
        # Constrained decoding for callers that need a parseable structure
        # (e.g. the question generator, services/api_gateway/routers/
        # generation.py) rather than free-form prose — Ollama's own
        # /api/generate "format" field, not a second hand-rolled HTTP path.
        response_format = params.get("response_format")
        if response_format:
            payload["format"] = response_format
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        self._last_token_counts = {
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
        }
        return data["response"].strip()
