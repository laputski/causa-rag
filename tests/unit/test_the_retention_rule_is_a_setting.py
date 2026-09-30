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
    FakeMongo().install(monkeypatch)
    app = FastAPI()
    app.include_router(S.router)
    with TestClient(app) as c:
        yield c


def test_the_defaults_are_served_before_anything_is_set(client) -> None:
    assert client.get("/settings/debug-retention").json() == {
        "keep_last_per_realm": 2, "max_age_days": 30, "max_mib": 512}


def test_what_is_written_is_what_is_read(client) -> None:
    rule = {"keep_last_per_realm": 5, "max_age_days": 7, "max_mib": 2048}
    assert client.put("/settings/debug-retention", json=rule).status_code == 200
    assert client.get("/settings/debug-retention").json() == rule


@pytest.mark.parametrize("field,value", [
    ("keep_last_per_realm", 0), ("max_age_days", 0), ("max_mib", 1)])
def test_a_rule_that_would_leave_nothing_to_debug_is_refused(client, field, value) -> None:
    rule = {"keep_last_per_realm": 2, "max_age_days": 30, "max_mib": 512, field: value}
    assert client.put("/settings/debug-retention", json=rule).status_code == 422
