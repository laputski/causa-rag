"""a later change — the retrieval-pin
overlay is opt-in and off by default.

Pins them as behaviour rather than as intent: "off by default" is only worth
anything if a config built the ordinary way really has it off, and if turning
the field on did not quietly renumber every historical run's config_hash.
"""
from __future__ import annotations

from core.experiment.config import ComponentRef, ExperimentConfig


def _config(**overrides) -> ExperimentConfig:
    base = dict(
        name="run",
        chunking_strategy=ComponentRef(kind="chunker", component_id="structure_aware"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_pins_are_off_unless_a_run_asks_for_them() -> None:
    assert _config().retrieval_pins_enabled is False


def test_enabling_pins_is_a_different_configuration() -> None:
    # A run with the overlay applied is not the same system as one without,
    # so it must not collide with the other's hash — otherwise the two would
    # be treated as reruns of one configuration and compared as if equal.
    assert _config().config_hash != _config(retrieval_pins_enabled=True).config_hash


def test_adding_the_field_did_not_renumber_historical_runs() -> None:
    """The field is in _BACKCOMPAT_DEFAULTS, so at its default it drops out of
    the hash payload entirely. Without that, every baseline run recorded
    before that change would have stopped matching its own config_hash and every
    regression comparison against a pinned baseline would have broken at once
    — silently, since a hash mismatch reads as "different configuration"
    rather than as an error."""
    payload = _config()._compute()
    dumped = _config().model_dump(exclude={"config_hash"})
    assert "retrieval_pins_enabled" in dumped, "field must exist on the model"

    # Same config, hashed as it would have been before the field existed.
    import hashlib
    import json

    from core.experiment.config import _BACKCOMPAT_DEFAULTS

    legacy = {k: v for k, v in dumped.items() if k not in _BACKCOMPAT_DEFAULTS}
    for key, default in _BACKCOMPAT_DEFAULTS.items():
        if key in dumped and dumped[key] != default:
            legacy[key] = dumped[key]
    canonical = json.dumps(legacy, sort_keys=True, ensure_ascii=False)
    assert payload == hashlib.sha256(canonical.encode()).hexdigest()[:16]
