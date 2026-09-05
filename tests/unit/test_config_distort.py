"""One named defect in a healthy configuration, and nothing else.

The companion of the corpus mutator's tests, for the settings. Three properties
carry the whole tool: exactly the declared fields move, a distortion that cannot
bite refuses, and what the configuration cannot say is asked for instead of
guessed.
"""
from __future__ import annotations

from typing import Any

import pytest

from core.experiment.config import ComponentRef, ExperimentConfig
from tools.config_distort import (
    DISTORTIONS,
    ENGLISH_ONLY_RERANKER,
    MERGING_PIPELINES,
    SINGLE_SOURCE_PIPELINES,
    ConfigurationCannotCarryDistortion,
    distort,
)

RERANKER = ComponentRef(kind="reranker", component_id="cross_encoder_local")


def _control(**fields: Any) -> ExperimentConfig:
    """A healthy hybrid configuration that reranks: the widest shape, so a
    distortion refusing here refuses for its own reason and not for want of a
    field to move."""
    return ExperimentConfig(**{
        "name": "proving-ground-control",
        "chunking_strategy": ComponentRef(kind="chunker", component_id="structure_aware"),
        "embedder": ComponentRef(kind="embedder", component_id="bge_m3"),
        "generator": ComponentRef(kind="generator", component_id="ollama"),
        "pipeline_id": "hybrid_rrf",
        "corpus_id": "base-ru",
        "top_k": 5,
        "fetch_k": 50,
        "reranker": RERANKER,
        **fields,
    })


# ── the guarantee ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_exactly_the_declared_fields_move(distortion: Any) -> None:
    """Anything else and the bait would be proving that two configurations
    differ, which they always do."""
    control = _control()
    broken = distort(control, distortion.name, corpus_language="ru")
    assert set(control.diff(broken)) == set(distortion.changes)


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_the_control_is_never_touched(distortion: Any) -> None:
    control = _control()
    before = control.model_dump()
    distort(control, distortion.name, corpus_language="ru")
    assert control.model_dump() == before


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_the_fingerprint_moves_with_the_configuration(distortion: Any) -> None:
    """A distorted configuration sharing its control's fingerprint would be
    filed as a rerun of it, and the pair would compare against itself."""
    control = _control()
    broken = distort(control, distortion.name, corpus_language="ru")
    assert broken.config_hash != control.config_hash
    assert broken.fingerprint_matches_fields()


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_every_distortion_names_a_catalogue_entry_that_exists(distortion: Any) -> None:
    from core.eval.atlas import FAILURES

    known = {f.id for f in FAILURES}
    for failure_id in distortion.provokes:
        assert failure_id in known, f"{distortion.name} names {failure_id}, which is not in the catalogue"
    assert distortion.provokes, f"{distortion.name} provokes nothing, so nothing needs it"


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_every_entry_it_names_is_one_a_setting_can_stage(distortion: Any) -> None:
    """A distortion pointing at a corpus defect or an external system would
    stage nothing, and its bait would then measure the wrong instrument."""
    from core.eval.atlas import FAILURES

    by_id = {f.id: f for f in FAILURES}
    for failure_id in distortion.provokes:
        entry = by_id[failure_id]
        allowed = (entry.instrument, *entry.also_staged_by)
        assert "config" in allowed, (
            f"{failure_id} is staged by {allowed}, and a setting is not among them"
        )


# ── refusing, instead of provoking nothing ────────────────────────────────────

def test_narrowing_a_selection_of_one_refuses() -> None:
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(top_k=1), "narrow_the_selection")


def test_outweighing_one_half_refuses_on_a_single_source() -> None:
    """One retrieval half cannot be outweighed, so the setting would change a
    word in the configuration and nothing in the run."""
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(pipeline_id="naive"), "let_one_half_decide")


def test_pinning_the_fusion_constant_refuses_when_nothing_fuses_by_rank() -> None:
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(pipeline_id="hybrid_weighted", merge_strategy="weighted"),
                "pin_the_fusion_constant")


def test_closing_the_window_refuses_without_a_reranker() -> None:
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(reranker=None), "close_the_candidate_window")


def test_closing_an_already_closed_window_refuses() -> None:
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(fetch_k=5), "close_the_candidate_window")


def test_the_wrong_language_reranker_refuses_when_nobody_said_the_language() -> None:
    """A configuration names a corpus and not its language. Guessing from the
    corpus identifier would work on this proving ground and quietly stop
    working on anybody else's."""
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(), "rerank_in_another_language")


def test_the_wrong_language_reranker_refuses_on_an_english_corpus() -> None:
    """An English reranker on English text is the right tool, so the pairing
    would provoke nothing."""
    with pytest.raises(ConfigurationCannotCarryDistortion):
        distort(_control(corpus_id="base-en"), "rerank_in_another_language", corpus_language="en")


def test_an_unknown_distortion_names_the_ones_that_exist() -> None:
    with pytest.raises(KeyError) as raised:
        distort(_control(), "make_it_worse")
    assert "narrow_the_selection" in str(raised.value)


# ── what each distortion actually writes ──────────────────────────────────────

def test_narrowing_leaves_one_fragment() -> None:
    assert distort(_control(), "narrow_the_selection").top_k == 1


def test_outweighing_one_half_puts_the_whole_weight_on_it() -> None:
    """Measured on this platform's own score ranges: at a weight of one half
    each the merged top three took one document from the lexical half, exactly
    as rank fusion did, because the weighted merge normalises each half by its
    own maximum first. Only the edge of the weight makes one half rule."""
    broken = distort(_control(), "let_one_half_decide")
    assert broken.merge_strategy == "weighted"
    assert broken.merge_alpha == 1.0


def test_pinning_sets_the_constant_to_one() -> None:
    assert distort(_control(), "pin_the_fusion_constant").rrf_k == 1


def test_closing_the_window_makes_it_equal_the_selection() -> None:
    broken = distort(_control(), "close_the_candidate_window")
    assert broken.fetch_k == broken.top_k == 5


def test_the_wrong_language_reranker_names_an_english_only_model() -> None:
    broken = distort(_control(), "rerank_in_another_language", corpus_language="ru")
    assert broken.reranker is not None
    assert broken.reranker.params["model_name"] == ENGLISH_ONLY_RERANKER
    assert broken.reranker.component_id == RERANKER.component_id, "the registry entry must not move too"


# ── the classification of pipelines stays true ────────────────────────────────

def test_every_registered_pipeline_is_classified() -> None:
    """A pipeline nobody classified counts as single-source, so every
    distortion needing two halves would refuse on it without saying why. Read
    out of the gateway's registrations, since building the registry needs the
    adapters up."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).parents[2] / "services" / "api_gateway" / "main.py").read_text(encoding="utf-8")
    registered = set(re.findall(r'register\(\s*"pipeline",\s*"([^"]+)"', source))
    registered |= {"naive"} if 'register("pipeline", naive_pipeline.pipeline_id' in source else set()
    registered |= set(re.findall(r'pipeline_id\s*=\s*"(graph)"', source))
    classified = set(MERGING_PIPELINES) | set(SINGLE_SOURCE_PIPELINES)
    assert registered <= classified, f"unclassified pipelines: {sorted(registered - classified)}"


def test_the_two_classifications_do_not_overlap() -> None:
    assert not set(MERGING_PIPELINES) & set(SINGLE_SOURCE_PIPELINES)


def test_the_configurations_own_retriever_list_is_not_trusted() -> None:
    """It is written by nothing and read by nothing: across all forty-one
    stored runs it holds zero or one entry, eighteen genuinely hybrid runs
    included. A hybrid configuration must be recognised without it."""
    assert _is_hybrid_by_id()


def _is_hybrid_by_id() -> bool:
    from tools.config_distort import _is_hybrid

    return _is_hybrid(_control(retrievers=[]))


def test_the_wrong_language_reranker_keeps_the_parameters_already_there() -> None:
    """Written after a bait failed to bite: the control's reranker carried no
    parameters, so a distortion that dropped every one of them looked
    identical to a distortion that added one. A run pinning a batch size and
    then asking for this defect would have lost the batch size silently.
    """
    control = _control(reranker=ComponentRef(
        kind="reranker", component_id="cross_encoder_local",
        params={"batch_size": 16},
    ))
    broken = distort(control, "rerank_in_another_language", corpus_language="ru")
    assert broken.reranker is not None
    assert broken.reranker.params == {"batch_size": 16, "model_name": ENGLISH_ONLY_RERANKER}


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_a_distortion_staging_only_the_effect_says_why(distortion: Any) -> None:
    """A bait built on such a distortion proves the signal works and does not
    prove the entry was reproduced. Without the note, the first reading passes
    for the second."""
    assert distortion.stages in ("cause", "effect")
    if distortion.stages == "effect":
        assert distortion.stages_note, f"{distortion.name} stages only the effect and does not say why"
    else:
        assert not distortion.stages_note, f"{distortion.name} explains a limitation it does not have"


def test_the_weighted_merge_normalises_which_is_why_one_distortion_stages_the_effect() -> None:
    """The measurement the note above rests on, kept as a test so the note
    cannot outlive the behaviour it describes.

    Real ranges from this platform's own runs: cosine reaching 0.89 against a
    lexical score reaching 58.1. If the merge ever stops normalising, the
    lexical half will take the whole top three and this test says so.
    """
    from core.models import Chunk, ScoredChunk
    from core.retrieval.hybrid import HybridRetriever

    def ranked(pairs: list[tuple[str, float]]) -> list:
        return [ScoredChunk(chunk=Chunk(chunk_id=i, doc_id="d", text=i, structural_path="p"),
                            score=s, retriever_id="r") for i, s in pairs]

    class _Half:
        def __init__(self, out: list) -> None:
            self._out = out

        def retrieve(self, **kwargs: Any) -> list:
            return self._out

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1]] * len(texts)

    dense = _Half(ranked([("d1", 0.89), ("d2", 0.71), ("d3", 0.55)]))
    sparse = _Half(ranked([("s1", 58.1), ("s2", 31.0), ("s3", 12.4)]))
    merged = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, embedder=_Half([]),
                             merge="weighted", alpha=0.5).retrieve("q", k=3)
    from_lexical = sum(1 for sc in merged if sc.chunk.chunk_id.startswith("s"))
    assert from_lexical == 1, (
        f"the weighted merge stopped normalising: {from_lexical} of the top three come from the "
        "half with the larger numbers, so switching the strategy now stages the cause after all"
    )
