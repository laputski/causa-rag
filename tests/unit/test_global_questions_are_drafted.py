"""Drafting questions about the corpus as a whole.

The model is stood in for: what is checked is how fragments are chosen, which
template a group gets, and what a reply turns into. Whether the model writes a
good question was measured before any of this, and the figure is in the
template's comment in `services/api_gateway/routers/generation.py`.
"""
from __future__ import annotations

import json
import math
from random import Random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers import generation as G


def _chunk(doc: str, n: int, text: str | None = None):
    return SimpleNamespace(chunk_id=f"{doc}-{n}", text=text or f"{doc} fragment {n}",
                           doc_id=doc, structural_path="", metadata={})


def _close_but_distinct(texts: list[str]) -> list[list[float]]:
    """Vectors a little apart from each other, as related fragments are.
    Identical vectors would read as the same text under another name, which
    the sampler refuses to group, and rightly."""
    return [[math.cos(0.4 * i), math.sin(0.4 * i)] for i, _ in enumerate(texts)]


# ── Choosing the fragments ────────────────────────────────────────────────────

def test_a_group_is_drawn_from_as_many_documents_as_it_has_fragments() -> None:
    chunks = [_chunk(f"d{i}", j, f"alpha d{i} {j}") for i in range(6) for j in range(3)]
    groups = G._sample_global(chunks, 3, Random(1), _close_but_distinct, k=4)
    assert groups
    for group in groups:
        assert len(group) == 4 and len({c.doc_id for c in group}) == 4


def test_near_identical_fragments_are_not_grouped_together() -> None:
    """Four copies of one template with different names make a question about
    the template, which is what the first probe produced."""
    chunks = [_chunk(f"card{i}", 0, f"card template {i}") for i in range(6)]
    chunks += [_chunk(f"rule{i}", 0, f"rule text {i}") for i in range(6)]

    def embed(texts: list[str]) -> list[list[float]]:
        # Every card the same vector; the rules close to each other and apart.
        return [[1.0, 0.0, 0.0] if t.startswith("card") else
                [0.6, 0.8 * math.cos(i), 0.8 * math.sin(i)] for i, t in enumerate(texts)]
    groups = G._sample_global(chunks, 4, Random(3), embed, k=2)
    assert groups
    for group in groups:
        assert sum(c.text.startswith("card") for c in group) <= 1


def test_no_two_questions_are_written_from_one_group() -> None:
    chunks = [_chunk(f"d{i}", 0, f"alpha {i}") for i in range(4)]
    groups = G._sample_global(chunks, 10, Random(5), lambda texts: [
        [math.cos(0.4 * i), math.sin(0.4 * i)] for i, _ in enumerate(texts)], k=4)
    assert len({frozenset(c.chunk_id for c in g) for g in groups}) == len(groups) == 1


def test_a_corpus_with_too_few_documents_yields_no_group() -> None:
    chunks = [_chunk("d0", j) for j in range(10)] + [_chunk("d1", j) for j in range(10)]
    assert G._sample_global(chunks, 3, Random(0), lambda t: [[1.0] for _ in t], k=4) == []


# ── The template a group gets ─────────────────────────────────────────────────

def test_a_global_group_of_two_gets_the_global_template_numbered() -> None:
    """Dispatched by shape, a group of two from two documents would have been
    offered a comparison of two fragments of one document."""
    group = [_chunk("d1", 0, "first"), _chunk("d2", 0, "second")]
    template = G._resolve_template(None, group, "global")
    assert template == G._BUILTIN_TEMPLATES["global"]
    prompt = G._render_prompt(template, "global", group)
    assert "[1]\nfirst" in prompt and "[2]\nsecond" in prompt and "{k}" not in prompt
    assert "Тебе даны 2 фрагментов" in prompt


def test_the_english_template_fills_the_same_placeholders() -> None:
    group = [_chunk("d1", 0, "first"), _chunk("d2", 0, "second"), _chunk("d3", 0, "third")]
    prompt = G._render_prompt(G.GLOBAL_TEMPLATE_EN, "global", group)
    assert "given 3 fragments" in prompt and "[3]\nthird" in prompt and "{" not in prompt.split("Return")[0]


# ── What a reply turns into ───────────────────────────────────────────────────

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(G.router)
    with TestClient(app) as c:
        yield c


def _run(client, reply: str | list[str], chunks, type_counts, *, cut: bool = False, **extra) -> dict:
    qdrant = MagicMock()
    qdrant.scroll_all.return_value = chunks
    replies = iter(reply if isinstance(reply, list) else [reply] * 50)
    embedder = SimpleNamespace(embed=lambda texts: [
        [math.cos(0.4 * i), math.sin(0.4 * i)] for i, _ in enumerate(texts)])
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=qdrant), \
         patch("core.registry.registry.resolve", return_value=embedder), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", side_effect=lambda *a, **k: next(replies)), \
         patch("adapters.ollama_generator.OllamaGenerator.prompt_was_cut", return_value=cut):
        resp = client.post("/generate/questions", json={
            "realm_id": "demo", "corpus_id": "handbook", "model": "qwen3:8b",
            "type_counts": type_counts, **extra})
        job_id = resp.json()["job_id"]
        with client.websocket_connect(f"/generate/questions/{job_id}/progress") as ws:
            while True:
                event = ws.receive_json()
                if event["type"] == "done":
                    return event


_GLOBAL_REPLY = json.dumps({
    "question": "Какие сроки поверки у приборов?", "reference_answer": "Разные.",
    "assertions": [{"fragment": 1, "statement": "Поверка ХР-310 раз в 12 месяцев."},
                   {"fragment": 9, "statement": "Поверка СФ-210 раз в 24 месяца."}],
}, ensure_ascii=False)


def _corpus(documents: int = 6):
    return [_chunk(f"d{i}", 0, f"alpha {i}") for i in range(documents)]


def test_a_global_draft_carries_its_scope_assertions_and_their_fragments(client) -> None:
    done = _run(client, _GLOBAL_REPLY, _corpus(), [{"question_type": "global", "n_questions": 1}])
    draft = done["drafts"][0]
    assert draft["scope"] == "global" and draft["question_type"] == "global"
    assert draft["assertions"] == ["Поверка ХР-310 раз в 12 месяцев.", "Поверка СФ-210 раз в 24 месяца."]
    fragments = draft["provenance"]["assertion_fragments"]
    # Fragment 9 does not exist in a group of four, and is not guessed at.
    assert fragments[0] and fragments[1] == ""
    assert len(draft["article_refs"]) == 4


def test_a_global_reply_without_assertions_is_a_failure_and_not_a_draft(client) -> None:
    reply = json.dumps({"question": "q", "reference_answer": "a"})
    done = _run(client, reply, _corpus(), [{"question_type": "global", "n_questions": 1}])
    assert done["drafts"] == []
    assert done["failed"][0]["reason"].startswith("unparseable_response")


def test_a_prompt_the_server_cut_is_a_failure_and_not_a_draft(client) -> None:
    done = _run(client, _GLOBAL_REPLY, _corpus(), [{"question_type": "global", "n_questions": 1}], cut=True)
    assert done["drafts"] == [] and done["failed"][0]["reason"] == "prompt_cut"


def test_a_shortfall_is_reported_and_never_padded_with_single_fragments(client) -> None:
    """The old samplers filled a shortfall with single fragments carrying the
    type's label, which for a global question would be a local one wearing
    the wrong scope."""
    done = _run(client, _GLOBAL_REPLY, _corpus(documents=2), [{"question_type": "global", "n_questions": 3}])
    assert done["drafts"] == []
    assert done["failed"] == [{"reason": "not_enough_documents", "chunk_ids": [],
                               "missing": 3, "question_type": "global"}]


def test_a_local_question_carries_assertions_only_when_asked(client) -> None:
    reply = json.dumps({"question": "q", "reference_answer": "a", "assertions": ["a fact"]})
    chunks = [_chunk("d1", 0, "text")]
    asked = _run(client, reply, chunks, [{"question_type": "closed", "n_questions": 1}], with_assertions=True)
    assert asked["drafts"][0]["assertions"] == ["a fact"]
    assert "scope" not in asked["drafts"][0]


def test_a_word_spelled_in_two_alphabets_is_pointed_out() -> None:
    """Found live: "Замena" and "Zамena" for «Замена» in a draft's assertions.
    A word wholly in one alphabet, a model code like ХР-310, and a Latin
    abbreviation beside Cyrillic words are all left alone."""
    assert G._mixed_script_words(["Замena воздушного фильтра", "Zамena окна"]) == ["Zамena", "Замena"]
    assert G._mixed_script_words(["Поверка ХР-310 раз в 12 мес", "калибровка по ISO 17025"]) == []


def test_the_draft_carries_the_words_for_the_reviewer(client) -> None:
    reply = json.dumps({"question": "Какие сроки?", "reference_answer": "Разные.",
                        "assertions": [{"fragment": 1, "statement": "Замena фильтра раз в 6 мес."}]},
                       ensure_ascii=False)
    done = _run(client, reply, _corpus(), [{"question_type": "global", "n_questions": 1}])
    assert done["drafts"][0]["provenance"]["mixed_script_words"] == ["Замena"]
