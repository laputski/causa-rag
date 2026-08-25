"""Follow-up — `fetch_k` reaches an external RAG that reads it.

Found by re-verifying the plan end to end: `fetch_k` was threaded into the
two in-process branches only, so a run against an external RAG accepted the
field, recorded it in the config and changed nothing — the same shape
`merge_strategy` had until a later change.
"""
from __future__ import annotations

from core.experiment.runner import _params_with_fetch_k


def test_forwarded_when_the_rag_declared_it() -> None:
    assert _params_with_fetch_k(None, 50, ["fetch_k"]) == {"fetch_k": 50}


def test_not_forwarded_when_the_rag_never_declared_it() -> None:
    """The platform's own rule: never send a knob absent from the declared
    list, since a silently ignored parameter is how a configuration change
    looks applied while doing nothing."""
    assert _params_with_fetch_k(None, 50, []) is None
    assert _params_with_fetch_k({"a": 1}, 50, ["temperature"]) == {"a": 1}


def test_not_forwarded_when_the_declaration_is_unknown() -> None:
    # A RAG registered before capabilities were probed has no list at all;
    # absence of a declaration is not permission.
    assert _params_with_fetch_k({"a": 1}, 50, None) == {"a": 1}


def test_an_unset_field_changes_nothing() -> None:
    assert _params_with_fetch_k({"a": 1}, None, ["fetch_k"]) == {"a": 1}
    assert _params_with_fetch_k(None, None, ["fetch_k"]) is None


def test_a_hand_typed_value_wins_over_the_config_field() -> None:
    """A caller who typed a value meant it; overwriting it silently would
    make the raw-params escape hatch unreliable."""
    assert _params_with_fetch_k({"fetch_k": 7}, 50, ["fetch_k"]) == {"fetch_k": 7}


def test_existing_params_are_preserved_and_not_mutated() -> None:
    original = {"temperature": 0.2}
    merged = _params_with_fetch_k(original, 50, ["fetch_k"])
    assert merged == {"temperature": 0.2, "fetch_k": 50}
    assert original == {"temperature": 0.2}
