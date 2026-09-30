"""The question reaches the prompt, whatever a template calls it.

Found live: both seeded realms wrote `{question}` where the substitution
filled only `{query}`, so every answer in them was generated from a prompt
reading "Question: {question}", and the model replied that no question had
been asked. A run of six questions showed it as empty answers and wrong
refusals; the prompt itself said nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.prompt_store import PromptTemplate

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("placeholder", ["{query}", "{question}"])
def test_either_placeholder_carries_the_question(placeholder: str) -> None:
    template = PromptTemplate({"id": "p", "name": "p", "version": 1,
                               "template": f"Context:\n{{context}}\n\nQuestion: {placeholder}"})
    rendered = template.render("How often is it calibrated?", ["a fragment"])
    assert "How often is it calibrated?" in rendered and placeholder not in rendered


def test_every_seeded_prompt_carries_the_question() -> None:
    from tools import seed_demo, seed_proving_ground

    for template in (seed_demo._PROMPT_TEMPLATE, seed_proving_ground._PROMPT_TEMPLATE):
        rendered = PromptTemplate({"id": "p", "name": "p", "version": 1, "template": template}).render(
            "QUESTION-MARK", ["c"])
        assert "QUESTION-MARK" in rendered


def test_the_shipped_prompt_carries_the_question() -> None:
    """The one prompt file the repository ships, and the demo bundle that
    carries it to a fresh installation."""
    data = json.loads((ROOT / "prompts" / "demo_prompt_v1.json").read_text(encoding="utf-8"))
    assert "QUESTION-MARK" in PromptTemplate(data).render("QUESTION-MARK", ["c"])
    bundle = json.loads((ROOT / "ui" / "public" / "demo.realm.json").read_text(encoding="utf-8"))
    for prompt in bundle.get("prompts") or []:
        rendered = PromptTemplate({**prompt, "name": prompt.get("name", "p"),
                                   "version": prompt.get("version", 1)}).render("QUESTION-MARK", ["c"])
        assert "QUESTION-MARK" in rendered


def test_a_prompt_without_the_question_is_refused() -> None:
    import asyncio

    from fastapi import HTTPException

    from services.api_gateway.routers import prompts

    class _NoStore:
        """Any write reaching the store fails the test instead of landing in it:
        an earlier version of this test, run with the check disabled to see it
        fail, wrote a real prompt into the running installation."""

        def __getattr__(self, name):
            raise AssertionError(f"the store was touched ({name}) before the template was checked")

    body = prompts.PromptCreateRequest(name="p", template="Context:\n{context}", realm_id="acme")
    original_mdb, original_store = prompts.mdb, prompts.prompt_store
    prompts.mdb, prompts.prompt_store = _NoStore(), _NoStore()
    try:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(prompts.create_prompt(body))
    finally:
        prompts.mdb, prompts.prompt_store = original_mdb, original_store
    assert exc.value.status_code == 400 and "{query}" in exc.value.detail
