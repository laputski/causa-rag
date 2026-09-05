"""A configuration's fingerprint has to describe the configuration beside it.

`config_hash` is what deduplication, cache boundaries and every "same
configuration" comparison are keyed on, and it is computed once, by a validator,
at construction. Assigning to a hashed field afterwards leaves a stored run
whose fingerprint belongs to a configuration that no longer exists.

Found live, by recomputing the fingerprint of all forty-one stored runs: five
reproduced and thirty-six did not. All thirty-six came from one eleven-minute
window, the MIRACL grid, whose reporting appended the question count to the
name after the run had finished. Everything saved before and after that window
reproduces, which is what ruled out a schema drift and pointed at the sweep.
"""
from __future__ import annotations

from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig


def _config(**fields: Any) -> ExperimentConfig:
    return ExperimentConfig(
        name="miracl-ru-hybrid_rrf-k10",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="never_called"),
        **fields,
    )


def test_a_fresh_configuration_describes_itself() -> None:
    assert _config().fingerprint_matches_fields()


def test_assigning_to_a_hashed_field_breaks_the_description() -> None:
    """The defect itself, stated as the property that catches it. Kept as a
    test and not as a prohibition because two backfills the runner performs
    after hashing are deliberate: they exist so a registered system's URL
    rotating over time does not change the identity of the same run against
    the same system."""
    config = _config()
    config.name = "miracl-ru-hybrid_rrf-k10-demo50"
    assert not config.fingerprint_matches_fields()


def test_renaming_keeps_the_description_true() -> None:
    renamed = _config().renamed("miracl-ru-hybrid_rrf-k10-demo50")
    assert renamed.name == "miracl-ru-hybrid_rrf-k10-demo50"
    assert renamed.fingerprint_matches_fields()


def test_renaming_changes_the_fingerprint_and_says_so() -> None:
    """The name is a hash input, so two runs differing only by name are two
    configurations by this platform's own definition. A rename that kept the
    fingerprint would be the same lie from the other side."""
    original = _config()
    assert original.renamed("other").config_hash != original.config_hash


def test_renaming_carries_every_other_field_across() -> None:
    """Rebuilding from a dump is where a field silently goes missing, and a
    lost field would change the fingerprint for a second, unrelated reason."""
    original = _config(top_k=17, rrf_k=10, retrieval_only=True, merge_alpha=0.3)
    renamed = original.renamed("x")
    before = original.model_dump(exclude={"config_hash", "name"})
    after = renamed.model_dump(exclude={"config_hash", "name"})
    assert before == after


def test_the_sweep_renames_through_the_helper_and_not_by_assignment() -> None:
    """The site the defect was found at. Read out of the source because the
    behaviour needs a whole grid run and an index to observe end to end."""
    import pathlib

    source = (pathlib.Path(__file__).parents[2] / "eval" / "miracl" / "report.py").read_text(encoding="utf-8")
    assert "result.config.name =" not in source, (
        "the sweep assigns to a hashed field again; the fingerprint of every run it saves will be stale"
    )
    assert "result.config.renamed(" in source
