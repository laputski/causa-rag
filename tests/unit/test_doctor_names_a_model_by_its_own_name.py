"""The installation check reports a model pulled only when that model is.

Found while adding the assertion judge's row: the check matched on the family
alone, so with qwen3:8b on the server it reported qwen3:32b as present, and a
run needing the larger model would have found nothing where the report said
it was.
"""
from __future__ import annotations

from tools import doctor


def _rows(monkeypatch, pulled: list[str]) -> dict[str, str]:
    monkeypatch.setattr(doctor, "_http", lambda url: (True, {"models": [{"name": n} for n in pulled]}))
    return {c.component: c.status for c in doctor.check_ollama().children}


def test_a_sibling_of_the_same_family_is_not_the_model(monkeypatch) -> None:
    from eval.judge_model import ASSERTION_JUDGE_MODEL

    rows = _rows(monkeypatch, ["qwen3:8b"])
    assert rows[ASSERTION_JUDGE_MODEL] != doctor.OK


def test_the_model_itself_is(monkeypatch) -> None:
    from eval.judge_model import ASSERTION_JUDGE_MODEL

    rows = _rows(monkeypatch, ["qwen3:8b", ASSERTION_JUDGE_MODEL])
    assert rows[ASSERTION_JUDGE_MODEL] == doctor.OK
