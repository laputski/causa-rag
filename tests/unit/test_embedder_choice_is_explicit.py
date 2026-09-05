"""Asking for the stub has to give the stub.

`BgeM3Embedder` used to compute `use_real_model or os.getenv("USE_REAL_BGE_M3")`,
so the argument could turn the real model on and could never turn it off. A
caller asking for the stub inside a process where the variable was set received
the real model, and nothing said so.

Found while staging a stub-embedder failure on the proving ground: the half
that was supposed to be broken retrieved exactly as well as the healthy one,
because the session had set the variable to load the model once and the
"stub" query embedder was the real model all along. The same shape as every
other decorative setting this platform has been finding: accepted, and inert.
"""
from __future__ import annotations

import pytest

from adapters.bge_m3 import BgeM3Embedder


def test_asking_for_the_stub_gives_the_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_REAL_BGE_M3", "true")
    assert BgeM3Embedder(use_real_model=False)._use_real_model is False


def test_asking_for_the_real_model_gives_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half. An argument that could only ever turn the model off
    would be the same defect facing the other way.

    The weight loading is stubbed out, since what is under test is the
    decision and not the download. Written first by reimplementing the
    decision inside the test, which would have passed against any constructor
    at all.
    """
    monkeypatch.delenv("USE_REAL_BGE_M3", raising=False)
    loaded: list[bool] = []
    monkeypatch.setattr(BgeM3Embedder, "_load_model", lambda self: loaded.append(True))
    assert BgeM3Embedder(use_real_model=True)._use_real_model is True
    assert loaded == [True], "the real model was chosen and never loaded"


def test_saying_nothing_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default every existing caller relies on: the gateway and the
    ingestion command both express their choice through the variable."""
    monkeypatch.delenv("USE_REAL_BGE_M3", raising=False)
    assert BgeM3Embedder()._use_real_model is False


def test_the_stub_is_deterministic_and_unrelated_between_texts() -> None:
    """What makes it usable as a defect at all: an index built with it is
    perfectly well formed and carries no meaning."""
    import math

    stub = BgeM3Embedder(use_real_model=False)
    first = stub.embed(["Замена детектора"])[0]
    again = BgeM3Embedder(use_real_model=False).embed(["Замена детектора"])[0]
    assert first == again, "two loads of the same corpus would differ, which is a second defect"
    other = stub.embed(["Detector replacement"])[0]
    dot = sum(a * b for a, b in zip(first, other, strict=True))
    norms = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in other))
    assert abs(dot / norms) < 0.1, "the stub relates two texts, so it would not ruin retrieval"


def test_the_stub_has_the_width_the_real_model_has() -> None:
    """Why nothing downstream refuses it, and why the failure is silent."""
    from adapters.bge_m3 import _VECTOR_DIM

    assert len(BgeM3Embedder(use_real_model=False).embed(["x"])[0]) == _VECTOR_DIM


def test_saying_nothing_still_lets_the_environment_turn_the_model_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bait for the test above, which deletes the variable and so could not
    tell "reads the environment" from "ignores it and answers no".

    This is how the gateway and the ingestion command choose today, so losing
    it would silently run every ingest against the stub.
    """
    monkeypatch.setenv("USE_REAL_BGE_M3", "true")
    loaded: list[bool] = []
    monkeypatch.setattr(BgeM3Embedder, "_load_model", lambda self: loaded.append(True))
    assert BgeM3Embedder()._use_real_model is True
    assert loaded == [True]
