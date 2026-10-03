"""The rule for clearing old runs' windows is read and written as a setting.

Bounded below on purpose: a rule that cleared a run the moment it was saved
would leave nobody anything to debug.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers import settings as S
from tests.unit._fake_mongo import FakeMongo


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("CAUSA_RUN_RETENTION", raising=False)
    FakeMongo().install(monkeypatch)
    app = FastAPI()
    app.include_router(S.router)
    with TestClient(app) as c:
        yield c


def test_the_defaults_are_served_before_anything_is_set(client) -> None:
    assert client.get("/settings/debug-retention").json() == {
        "keep_last_per_realm": 2, "max_age_days": 30, "max_mib": 512, "automatic": False}


def test_what_is_written_is_what_is_read(client) -> None:
    rule = {"keep_last_per_realm": 5, "max_age_days": 7, "max_mib": 2048}
    assert client.put("/settings/debug-retention", json=rule).status_code == 200
    assert client.get("/settings/debug-retention").json() == {**rule, "automatic": False}


def test_the_rule_says_whether_it_is_applied_by_itself(client, monkeypatch) -> None:
    """A rule that is saved and never applied reads as one that is in force."""
    monkeypatch.setenv("CAUSA_RUN_RETENTION", "on")
    assert client.get("/settings/debug-retention").json()["automatic"] is True


@pytest.mark.parametrize("field,value", [
    ("keep_last_per_realm", 0), ("max_age_days", 0), ("max_mib", 1)])
def test_a_rule_that_would_leave_nothing_to_debug_is_refused(client, field, value) -> None:
    rule = {"keep_last_per_realm": 2, "max_age_days": 30, "max_mib": 512, field: value}
    assert client.put("/settings/debug-retention", json=rule).status_code == 422


# ── The judge of statements is a Realm's setting ─────────────────────────────

def test_a_realm_with_no_judge_named_reports_the_platform_s_default(client) -> None:
    from eval.judge_model import ASSERTION_JUDGE_MODEL
    judge = client.get("/settings/judge?realm_id=acme").json()
    assert judge == {"provider": "", "model": "", "default_model": ASSERTION_JUDGE_MODEL}


def test_a_realm_s_judge_is_kept_and_read_back(client) -> None:
    asked = {"provider": "openai_compatible", "model": "a-judge"}
    assert client.put("/settings/judge?realm_id=acme", json=asked).status_code == 200
    stored = client.get("/settings/judge?realm_id=acme").json()
    assert (stored["provider"], stored["model"]) == ("openai_compatible", "a-judge")


def test_saving_a_judge_leaves_the_realm_s_other_settings_alone(client) -> None:
    client.put("/settings/model?realm_id=acme", json={"model": "an-answering-model"})
    client.put("/settings/judge?realm_id=acme", json={"provider": "", "model": "a-judge"})
    assert client.get("/settings?realm_id=acme").json()["active_model"] == "an-answering-model"


def test_a_server_the_platform_does_not_know_is_refused(client) -> None:
    assert client.put("/settings/judge?realm_id=acme",
                      json={"provider": "somewhere", "model": "m"}).status_code == 422
