"""POST /experiments/{run_id}/questions/{question_id}/feedback/triage and
its /decide follow-up. Mocks the generator call and
`adapters.mongodb` directly — same style as test_feedback_promote.py — so
the deterministic parts (parsing, cross-check, taxonomy lookup) run for
real against a controlled model response.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.feedback import router

_RUN = {
    "run_id": "run1",
    "realm_id": "",
    "question_results": [
        {
            "question_id": "q1", "question": "What is the limitation period?",
            "reference_answer": "3 years", "generated_answer": "1 year",
            "answerability": "answerable",
            "metrics": {"retrieval_recall_at_k": 0.0},
            "source_refs": [{"article_no": "art-10"}],
            "pre_rerank_source_refs": [],
        },
    ],
}

_FEEDBACK_WITH_COMMENT = {
    "id": "fb1", "run_id": "run1", "question_id": "q1", "rating": "bad",
    "comment": "The answer is imprecise and cites the wrong article; it should be art-10",
}

_VALID_MODEL_RESPONSE = json.dumps({
    "error_classes": ["wrong_citation"],
    "target_spans": ["1 year"],
    "expected_refs": ["art-10"],
    "severity": "medium",
    "confidence": 0.85,
})

_FAKE_TAXONOMY = [
    {
        "id": "wrong_citation", "definition": "wrong article cited", "typical_layer": "generation_citation",
        "candidate_levers": ["fix_citation_mapping"],
    },
]


def _mdb_side_effect(feedback_doc=None):
    async def find_one(collection, query):
        if collection == "experiment_runs":
            return _RUN if query.get("run_id") == "run1" else None
        if collection == "answer_feedback":
            return feedback_doc
        return None
    return find_one


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_generator():
    gen = MagicMock()
    gen.generate.return_value = _VALID_MODEL_RESPONSE
    return gen


@pytest.fixture(autouse=True)
def _mock_taxonomy():
    """Every test in this file exercises the parsing/cross-check/proposal
    logic against a controlled taxonomy — resolving a real domain pack's
    registry-provided one is a separate concern (see
    TestTriageFeedback.test_409_when_no_domain_pack_provides_a_taxonomy,
    which overrides this with None)."""
    with patch(
        "services.api_gateway.routers.feedback._resolve_error_taxonomy",
        AsyncMock(return_value=_FAKE_TAXONOMY),
    ) as mock:
        yield mock


class TestTriageFeedback:
    def test_409_when_no_domain_pack_provides_a_taxonomy(
        self, client: TestClient, _mock_taxonomy,
    ) -> None:
        """409, and not 503: nothing is down. The Realm has no domain pack
        active, and the caller fixes that on the Domain packs page. A 503 tells
        them to come back later, which is the one thing that will not help. It
        is what the demo realm answered on a first run: it ships with no pack
        active."""
        _mock_taxonomy.return_value = None
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(_FEEDBACK_WITH_COMMENT)),
        ):
            resp = client.post("/experiments/run1/questions/q1/feedback/triage")
        assert resp.status_code == 409

    def test_triages_the_stored_comment_and_saves_the_proposal(
        self, client: TestClient, fake_generator,
    ) -> None:
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(_FEEDBACK_WITH_COMMENT)),
        ), patch("adapters.mongodb.upsert_one", AsyncMock()) as upsert_mock, patch(
            "services.api_gateway.routers.feedback._resolve_triage_generator",
            AsyncMock(return_value=fake_generator),
        ):
            resp = client.post("/experiments/run1/questions/q1/feedback/triage")

        assert resp.status_code == 200
        body = resp.json()
        assert body["triage_status"] == "proposed"
        triage = body["triage_result"]
        assert triage["error_classes"] == ["wrong_citation"]
        assert triage["expected_refs"] == ["art-10"]
        # art-10 IS in this question's source_refs -> cross_check confirms it.
        assert triage["verified_refs"] == {"art-10": True}
        assert triage["diagnosis_confirmed"] is True
        assert triage["proposal"]["primary_lever"] == "fix_citation_mapping"
        upsert_mock.assert_awaited_once()

    def test_404_when_run_does_not_exist(self, client: TestClient) -> None:
        with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect())):
            resp = client.post("/experiments/missing-run/questions/q1/feedback/triage")
        assert resp.status_code == 404

    def test_404_when_question_not_in_run(self, client: TestClient) -> None:
        with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect())):
            resp = client.post("/experiments/run1/questions/does-not-exist/feedback/triage")
        assert resp.status_code == 404

    def test_400_when_no_feedback_comment_exists_yet(self, client: TestClient) -> None:
        with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(None))):
            resp = client.post("/experiments/run1/questions/q1/feedback/triage")
        assert resp.status_code == 400

    def test_400_when_feedback_exists_but_comment_is_empty(self, client: TestClient) -> None:
        no_comment = {**_FEEDBACK_WITH_COMMENT, "comment": None}
        with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(no_comment))):
            resp = client.post("/experiments/run1/questions/q1/feedback/triage")
        assert resp.status_code == 400

    def test_502_when_the_generator_call_itself_fails(self, client: TestClient) -> None:
        broken_generator = MagicMock()
        broken_generator.generate.side_effect = RuntimeError("connection refused")
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(_FEEDBACK_WITH_COMMENT)),
        ), patch(
            "services.api_gateway.routers.feedback._resolve_triage_generator",
            AsyncMock(return_value=broken_generator),
        ):
            resp = client.post("/experiments/run1/questions/q1/feedback/triage")
        assert resp.status_code == 502

    def test_malformed_model_response_still_returns_200_flagged_for_manual_review(
        self, client: TestClient,
    ) -> None:
        broken_generator = MagicMock()
        broken_generator.generate.return_value = "I cannot help with that"
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(_FEEDBACK_WITH_COMMENT)),
        ), patch("adapters.mongodb.upsert_one", AsyncMock()), patch(
            "services.api_gateway.routers.feedback._resolve_triage_generator",
            AsyncMock(return_value=broken_generator),
        ):
            resp = client.post("/experiments/run1/questions/q1/feedback/triage")
        assert resp.status_code == 200
        assert resp.json()["triage_result"]["needs_manual_review"] is True


class TestDecideTriage:
    _PROPOSED = {
        **_FEEDBACK_WITH_COMMENT,
        "triage_result": {"error_classes": ["wrong_citation"], "proposal": {"primary_lever": "fix_citation_mapping"}},
        "triage_status": "proposed",
    }

    def test_confirm_records_the_decision(self, client: TestClient) -> None:
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(self._PROPOSED)),
        ), patch("adapters.mongodb.upsert_one", AsyncMock()) as upsert_mock:
            resp = client.post(
                "/experiments/run1/questions/q1/feedback/triage/decide", json={"action": "confirm"},
            )
        assert resp.status_code == 200
        assert resp.json()["triage_status"] == "confirmed"
        upsert_mock.assert_awaited_once()

    def test_reject_records_the_decision(self, client: TestClient) -> None:
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(self._PROPOSED)),
        ), patch("adapters.mongodb.upsert_one", AsyncMock()):
            resp = client.post(
                "/experiments/run1/questions/q1/feedback/triage/decide", json={"action": "reject"},
            )
        assert resp.json()["triage_status"] == "rejected"

    def test_edit_merges_the_edited_result(self, client: TestClient) -> None:
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(self._PROPOSED)),
        ), patch("adapters.mongodb.upsert_one", AsyncMock()):
            resp = client.post(
                "/experiments/run1/questions/q1/feedback/triage/decide",
                json={"action": "edit", "edited_result": {"error_classes": ["incomplete"]}},
            )
        body = resp.json()
        assert body["triage_status"] == "edited"
        assert body["triage_result"]["error_classes"] == ["incomplete"]
        # untouched fields survive the merge
        assert body["triage_result"]["proposal"]["primary_lever"] == "fix_citation_mapping"

    def test_edit_without_edited_result_400s(self, client: TestClient) -> None:
        with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(self._PROPOSED))):
            resp = client.post(
                "/experiments/run1/questions/q1/feedback/triage/decide", json={"action": "edit"},
            )
        assert resp.status_code == 400

    def test_404_when_no_proposal_exists_yet(self, client: TestClient) -> None:
        with patch(
            "adapters.mongodb.find_one", AsyncMock(side_effect=_mdb_side_effect(_FEEDBACK_WITH_COMMENT)),
        ):
            resp = client.post(
                "/experiments/run1/questions/q1/feedback/triage/decide", json={"action": "confirm"},
            )
        assert resp.status_code == 404
