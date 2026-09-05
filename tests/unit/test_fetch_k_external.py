"""Follow-up — the knobs an external RAG declares reading actually reach it.

Found by re-verifying the plan end to end: `fetch_k` was threaded into the
two in-process branches only, so a run against an external RAG accepted the
field, recorded it in the config and changed nothing — the same shape
`merge_strategy` had until a later change.

`rrf_k` arrived later and would have repeated that defect one release apart,
so the forwarding is written over a list of field names and these tests run
over the same list instead of naming one field.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.experiment.runner import _FORWARDED_TO_EXTERNAL, _params_with_declared_fields


def _config(**fields: object) -> SimpleNamespace:
    """A stand-in carrying only the fields the forwarding reads.

    A real `ExperimentConfig` would work and would also pin four unrelated
    required fields into every case below, which is what makes such a test
    fail for reasons that have nothing to do with its name.
    """
    return SimpleNamespace(**{name: None for name in _FORWARDED_TO_EXTERNAL} | fields)


@pytest.mark.parametrize("field_name", _FORWARDED_TO_EXTERNAL)
def test_forwarded_when_the_rag_declared_it(field_name: str) -> None:
    assert _params_with_declared_fields(None, _config(**{field_name: 50}), [field_name]) == {field_name: 50}


@pytest.mark.parametrize("field_name", _FORWARDED_TO_EXTERNAL)
def test_not_forwarded_when_the_rag_never_declared_it(field_name: str) -> None:
    """The platform's own rule: never send a knob absent from the declared
    list, since a silently ignored parameter is how a configuration change
    looks applied while doing nothing."""
    assert _params_with_declared_fields(None, _config(**{field_name: 50}), []) is None
    assert _params_with_declared_fields({"a": 1}, _config(**{field_name: 50}), ["temperature"]) == {"a": 1}


@pytest.mark.parametrize("field_name", _FORWARDED_TO_EXTERNAL)
def test_not_forwarded_when_the_declaration_is_unknown(field_name: str) -> None:
    # A RAG registered before capabilities were probed has no list at all;
    # absence of a declaration is not permission.
    assert _params_with_declared_fields({"a": 1}, _config(**{field_name: 50}), None) == {"a": 1}


@pytest.mark.parametrize("field_name", _FORWARDED_TO_EXTERNAL)
def test_an_unset_field_changes_nothing(field_name: str) -> None:
    assert _params_with_declared_fields({"a": 1}, _config(), [field_name]) == {"a": 1}
    assert _params_with_declared_fields(None, _config(), [field_name]) is None


@pytest.mark.parametrize("field_name", _FORWARDED_TO_EXTERNAL)
def test_a_hand_typed_value_wins_over_the_config_field(field_name: str) -> None:
    """A caller who typed a value meant it; overwriting it silently would
    make the raw-params escape hatch unreliable."""
    assert _params_with_declared_fields({field_name: 7}, _config(**{field_name: 50}), [field_name]) == {field_name: 7}


@pytest.mark.parametrize("field_name", _FORWARDED_TO_EXTERNAL)
def test_existing_params_are_preserved_and_not_mutated(field_name: str) -> None:
    original = {"temperature": 0.2}
    merged = _params_with_declared_fields(original, _config(**{field_name: 50}), [field_name])
    assert merged == {"temperature": 0.2, field_name: 50}
    assert original == {"temperature": 0.2}


def test_two_declared_knobs_travel_together() -> None:
    """Written because the forwarding returns early on the first field it
    cannot send, and an early return placed one line higher would have
    dropped every knob after the first undeclared one."""
    config = _config(fetch_k=50, rrf_k=10)
    assert _params_with_declared_fields(None, config, ["fetch_k", "rrf_k"]) == {"fetch_k": 50, "rrf_k": 10}
    assert _params_with_declared_fields(None, config, ["rrf_k"]) == {"rrf_k": 10}


def test_every_forwarded_name_is_a_real_field_of_a_configuration() -> None:
    """The list is read with getattr, which answers None for a name nobody
    ever added, so a typo here would forward nothing and say nothing."""
    from core.experiment.config import ExperimentConfig

    for name in _FORWARDED_TO_EXTERNAL:
        assert name in ExperimentConfig.model_fields, f"{name!r} is not a field of ExperimentConfig"
