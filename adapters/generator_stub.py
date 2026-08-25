"""Generator stub for unit/integration tests (no GPU, no network)."""
from __future__ import annotations

from typing import Any


class GeneratorStub:
    """Returns a deterministic canned answer built from the prompt."""

    generator_id = "stub"

    def generate(self, prompt: str, **params: Any) -> str:
        preview = prompt[:80].replace("\n", " ")
        return f"[STUB] Generated answer for: {preview}"
