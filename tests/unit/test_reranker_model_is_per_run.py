"""Which model reranks is a property of the run, and used not to be.

`ComponentRef.params` reached nothing: the pipeline builder resolves a reranker
out of the registry by `component_id` and drops the params beside it. Observed
directly, by asking for a model and reading back what got built.

The consequence is not cosmetic. The local reranker defaults to
`cross-encoder/ms-marco-MiniLM-L-6-v2`, which is English only, while the default
embedder is multilingual. On a Russian corpus that pairing reorders by noise,
and it was the platform's own default with no way for a run to say otherwise.
The MIRACL sweep names a multilingual model for exactly this reason and had to
construct its own component to do it.
"""
from __future__ import annotations

from adapters.reranker import (
    DEFAULT_LOCAL_MODEL,
    CrossEncoderRerankerLocal,
    CrossEncoderRerankerStub,
)
from core.experiment.runner import _on_the_model

MULTILINGUAL = "BAAI/bge-reranker-v2-m3"


def test_the_model_a_run_asks_for_is_the_model_it_gets() -> None:
    rebound = _on_the_model(CrossEncoderRerankerLocal(), MULTILINGUAL)
    assert rebound._model_name == MULTILINGUAL


def test_the_platform_default_is_english_only_which_is_what_makes_this_matter() -> None:
    """Pinned so the sentence above stays true. Should the default become
    multilingual, this test is the place that says the reasoning moved."""
    assert DEFAULT_LOCAL_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert CrossEncoderRerankerLocal()._model_name == DEFAULT_LOCAL_MODEL


def test_asking_for_nothing_keeps_what_was_registered() -> None:
    registered = CrossEncoderRerankerLocal()
    assert _on_the_model(registered, None) is registered
    assert _on_the_model(registered, "") is registered


def test_asking_for_what_is_already_bound_does_not_rebuild() -> None:
    """A cross-encoder loads its weights on first use, so a needless rebuild
    throws away a loaded model and pays for it again on the next question."""
    registered = CrossEncoderRerankerLocal(model_name=MULTILINGUAL)
    assert _on_the_model(registered, MULTILINGUAL) is registered


def test_a_reranker_with_no_model_to_swap_is_left_alone() -> None:
    """The stub takes no model name. Raising here would fail a whole run over
    a field that cannot apply to the component the registry handed back."""
    stub = CrossEncoderRerankerStub()
    assert _on_the_model(stub, MULTILINGUAL) is stub


def test_a_component_with_no_way_to_change_its_model_is_left_alone() -> None:
    """It used to be attempted anyway: the builder read the model off any
    component carrying the attribute and called its constructor, catching
    whatever came back. Whether a component can change its model is the
    component's own answer now, and one that has no answer is left as it is.
    """
    class Awkward:
        _model_name = "a"

    awkward = Awkward()
    assert _on_the_model(awkward, MULTILINGUAL) is awkward


def test_the_pipeline_builder_threads_the_params_through() -> None:
    """The rebind existing is not the same as the builder calling it, and the
    two were a separate change. Read out of the source, because observing it
    end to end needs a registry, a corpus and an index."""
    import pathlib

    source = (pathlib.Path(__file__).parents[2] / "core" / "experiment" / "runner.py").read_text(encoding="utf-8")
    assert "_on_the_model(\n                _maybe(\"reranker\", config.reranker)" in source, (
        "the pipeline builder resolves a reranker without applying the run's params"
    )


def test_nothing_else_reads_component_ref_params_yet() -> None:
    """Five other component references carry `params` that still reach nothing:
    grounding, route_policy, scorer, mask_engine and refusal_policy. Recorded
    here, so it does not become a surprise for whoever sets one and watches it do
    nothing. This test says what is true today; delete a name from it when that
    reference starts applying its params.
    """
    import pathlib

    source = (pathlib.Path(__file__).parents[2] / "core" / "experiment" / "runner.py").read_text(encoding="utf-8")
    for kind in ("grounder", "route_policy", "scorer", "mask_engine", "refusal"):
        assert f'_maybe("{kind}", config.' in source, f"{kind} no longer resolves this way; revisit this note"


def test_the_stand_in_used_above_matches_the_real_constructor() -> None:
    """`Awkward` stands in for a reranker whose constructor refuses the
    keyword. If the real one grew a different attribute name the rebind would
    silently stop applying, and every test above would still pass."""
    import inspect

    signature = inspect.signature(CrossEncoderRerankerLocal.__init__)
    assert "model_name" in signature.parameters
    assert hasattr(CrossEncoderRerankerLocal(), "_model_name")
