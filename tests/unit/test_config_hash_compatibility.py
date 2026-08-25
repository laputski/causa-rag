"""config_hash stays backward compatible.

Adding the new optional fields must NOT change the hash of a config that does not
set them, so historical experiment_runs remain comparable.
"""
from __future__ import annotations

from core.experiment.config import ComponentRef, ExperimentConfig


def _base() -> dict:
    return dict(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )


def test_unset_new_fields_keep_prior_hash():
    old = ExperimentConfig(**_base())
    explicit_defaults = ExperimentConfig(
        **_base(), grounding=None, route_policy=None, config_schema_version="1.0"
    )
    assert old.config_hash == explicit_defaults.config_hash


def test_setting_grounding_changes_hash():
    old = ExperimentConfig(**_base())
    grounded = ExperimentConfig(
        **_base(), grounding=ComponentRef(kind="grounder", component_id="token_overlap")
    )
    assert grounded.config_hash != old.config_hash


def test_setting_route_policy_changes_hash():
    old = ExperimentConfig(**_base())
    routed = ExperimentConfig(
        **_base(), route_policy=ComponentRef(kind="route_policy", component_id="naive")
    )
    assert routed.config_hash != old.config_hash


def test_hash_is_deterministic():
    assert ExperimentConfig(**_base()).config_hash == ExperimentConfig(**_base()).config_hash


def test_unset_params_keeps_prior_hash():
    """params={} (the default) must not change the
    hash, same back-compat contract as every other optional field here."""
    old = ExperimentConfig(**_base())
    explicit_empty = ExperimentConfig(**_base(), params={})
    assert old.config_hash == explicit_empty.config_hash


def test_setting_params_changes_hash():
    old = ExperimentConfig(**_base())
    with_params = ExperimentConfig(**_base(), params={"temperature": 0.2})
    assert with_params.config_hash != old.config_hash
