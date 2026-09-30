"""What a question set is made of, read before a run is spent on it."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.eval.composition import profile
from services.api_gateway.routers.datasets import router


def test_a_set_of_single_facts_shows_that_it_is_one() -> None:
    """The proving ground's sets are nineteen questions about single facts,
    and nothing said so."""
    rows = [{"question_type": "closed", "article_refs": [f"base-ru/{i:02d}"]} for i in range(1, 18)]
    rows += [{"question_type": "closed", "article_refs": []}] * 2
    p = profile(rows)
    assert p["by_scope"] == {"local": 19}
    assert p["by_type"] == {"closed": 19}
    assert p["by_answerability"] == {"answerable": 17, "out_of_scope": 2}
    assert p["distinct_references"] == 17
    assert p["with_assertions"] == 0


def test_global_questions_and_assertions_are_counted() -> None:
    rows = [
        {"scope": "global", "article_refs": ["a/1", "a/2"], "assertions": ["x", "y", " "]},
        {"article_refs": ["a/1"], "answerability": "uncovered"},
    ]
    p = profile(rows)
    assert p["by_scope"] == {"global": 1, "local": 1}
    assert p["by_type"] == {"unlabelled": 2}
    assert p["by_answerability"] == {"answerable": 1, "uncovered": 1}
    assert (p["distinct_references"], p["with_assertions"], p["assertions"]) == (2, 1, 2)


def test_the_profile_is_served_for_a_stored_set() -> None:
    app = FastAPI()
    app.include_router(router)
    doc = {"id": "ds1", "realm_id": "demo", "questions": [{"article_refs": ["a/1"], "scope": "global"}]}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=doc)), TestClient(app) as client:
        body = client.get("/datasets/ds1/profile").json()
    assert body["dataset_id"] == "ds1" and body["by_scope"] == {"global": 1}
