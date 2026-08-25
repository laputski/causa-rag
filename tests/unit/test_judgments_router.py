"""services/api_gateway/routers/judgments.py.

Replaces test_pins_router.py. The boundary being tested moved from MongoDB to
a file, so what matters is that a reviewer's verdict survives the round trip,
that a judgment ruling on nothing is refused rather than stored as a silent
no-op, and that promotion to a golden question refuses when it would produce
a test that can never fail.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from core.judgments import JudgedChunk
from services.api_gateway.routers import judgments as router


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "_JUDGMENTS_DIR", tmp_path)
    return tmp_path


def _body(**overrides):
    base = dict(
        realm_id="acme", corpus_id="handbook_01",
        question="What is the appeal deadline?", relevant=["c1"], irrelevant=["c9"],
    )
    base.update(overrides)
    return router.JudgmentWriteRequest(**base)


def _unresolvable_chunks(corpus_id, realm_id, chunk_ids):
    """Stands in for an index that cannot enrich the ids — the degraded path
    the router is explicitly designed to keep working."""
    return [JudgedChunk(chunk_id=cid) for cid in chunk_ids]


async def test_a_judgment_ruling_on_nothing_is_refused(store) -> None:
    with pytest.raises(HTTPException) as exc:
        await router.create_judgment(_body(relevant=[], irrelevant=[]))
    assert exc.value.status_code == 400


async def test_a_created_judgment_is_listed_back_for_its_realm_and_corpus(store) -> None:
    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_unresolvable_chunks)):
        created = await router.create_judgment(_body())
    listed = await router.list_judgments(realm_id="acme", corpus_id="handbook_01")
    assert [j["id"] for j in listed] == [created["id"]]
    assert listed[0]["question"] == "What is the appeal deadline?"


async def test_another_realm_does_not_see_it(store) -> None:
    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_unresolvable_chunks)):
        await router.create_judgment(_body())
    assert await router.list_judgments(realm_id="other", corpus_id="handbook_01") == []


async def test_retiring_a_judgment_keeps_it_on_record(store) -> None:
    """A retired judgment is not deleted. It remains evidence of what a
    reviewer once observed, which is the difference between retiring a
    verdict and pretending it was never made."""
    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_unresolvable_chunks)):
        created = await router.create_judgment(_body())
    updated = await router.update_judgment(
        created["id"], router.JudgmentPatchRequest(status="retired"),
        realm_id="acme", corpus_id="handbook_01",
    )
    assert updated["status"] == "retired"
    assert len(await router.list_judgments(realm_id="acme", corpus_id="handbook_01")) == 1


async def test_an_unknown_status_is_refused(store) -> None:
    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_unresolvable_chunks)):
        created = await router.create_judgment(_body())
    with pytest.raises(HTTPException) as exc:
        await router.update_judgment(
            created["id"], router.JudgmentPatchRequest(status="whatever"),
            realm_id="acme", corpus_id="handbook_01",
        )
    assert exc.value.status_code == 400


async def test_deleting_removes_only_the_named_judgment(store) -> None:
    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_unresolvable_chunks)):
        first = await router.create_judgment(_body())
        second = await router.create_judgment(_body(question="another question"))
    await router.delete_judgment(first["id"], realm_id="acme", corpus_id="handbook_01")
    remaining = await router.list_judgments(realm_id="acme", corpus_id="handbook_01")
    assert [j["id"] for j in remaining] == [second["id"]]


async def test_a_missing_judgment_is_a_404(store) -> None:
    with pytest.raises(HTTPException) as exc:
        await router.get_judgment("nope", realm_id="acme", corpus_id="handbook_01")
    assert exc.value.status_code == 404


# ── Promotion to a golden question ──────────────────────────────────────


async def test_promotion_refuses_when_the_test_could_never_fail(store) -> None:
    """Without a ref id there is nothing a retrieval metric could check, and
    the emitted question would classify out_of_scope — a green test that
    proves nothing is worse than no test."""
    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_unresolvable_chunks)):
        created = await router.create_judgment(_body())
    with pytest.raises(HTTPException) as exc:
        await router.promote_to_golden_question(
            created["id"],
            router.GoldenQuestionRequest(dataset_id="ds1", reference_answer="an answer"),
            realm_id="acme", corpus_id="handbook_01",
        )
    assert exc.value.status_code == 409


async def test_promotion_writes_the_expected_sources_into_the_dataset(store) -> None:
    async def _resolved(corpus_id, realm_id, chunk_ids):
        return [JudgedChunk(chunk_id=cid, ref_id=f"S/{cid}") for cid in chunk_ids]

    with patch.object(router, "_resolve_chunks", AsyncMock(side_effect=_resolved)):
        created = await router.create_judgment(_body())

    add_question = AsyncMock(return_value={"ok": True})
    with patch("services.api_gateway.routers.datasets.add_question", add_question):
        await router.promote_to_golden_question(
            created["id"],
            router.GoldenQuestionRequest(dataset_id="ds1", reference_answer="an answer"),
            realm_id="acme", corpus_id="handbook_01",
        )

    payload = add_question.await_args.args[1]
    assert payload.question == "What is the appeal deadline?"
    assert payload.reference_answer == "an answer"
    # The relevant chunk becomes an expected source; the irrelevant one must
    # not, or the test would demand retrieval of what the reviewer rejected.
    assert payload.model_dump()["article_refs"] == ["S/c1"]
