"""One named defect in the loading of a corpus, and nothing else.

These four defects live in the index and nowhere else: an analyser is chosen
when an index is created and every query inherits it, and the model a corpus
was embedded with is fixed at load time and no later setting revisits it. A run
configuration can say nothing about either, which is why they need an
instrument of their own.

The properties asserted here are the same three the other two instruments carry,
plus one this one adds: a plan is a pair, and its two halves must land in two
namespaces. A broken load into the control's namespace would destroy the half it
was supposed to be measured against, and the run afterwards would compare a
corpus with itself and report no difference.
"""
from __future__ import annotations

from typing import Any

import pytest

from tools.ingest_distort import (
    DISTORTIONS,
    LoadCannotCarryDistortion,
    Step,
    plan,
)
from tools.seed_proving_ground import CORPORA, GROUND_DIR

_RU = next(c for c in CORPORA if c.corpus_id == "base-ru")
_EN = next(c for c in CORPORA if c.corpus_id == "base-en")


def _plan(distortion: str, corpus: Any = _RU, strategy: str = "structure_aware") -> Any:
    return plan(distortion, str(GROUND_DIR / corpus.directory), corpus.corpus_id,
                corpus.language, strategy)


def _strategy_for(distortion: Any) -> str:
    return "fixed" if distortion.name == "reingest_with_another_chunk_size" else "structure_aware"


# ── a plan is a pair, and its halves are separate ─────────────────────────────

@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_the_two_halves_land_in_two_namespaces(distortion: Any) -> None:
    built = _plan(distortion.name, strategy=_strategy_for(distortion))
    assert built.control_corpus_id != built.distorted_corpus_id


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_every_command_of_a_half_targets_that_half(distortion: Any) -> None:
    """A plan whose second command drifted into the other namespace would look
    correct and break the control."""
    built = _plan(distortion.name, strategy=_strategy_for(distortion))
    for steps, namespace in ((built.control, built.control_corpus_id),
                             (built.distorted, built.distorted_corpus_id)):
        for step in steps:
            if not step.is_command:
                continue
            argv = list(step.argv)
            assert argv[argv.index("--corpus-id") + 1] == namespace


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_both_halves_read_the_same_documents(distortion: Any) -> None:
    """The defect must be the only difference. Two halves reading two corpora
    would differ in everything, and a signal reddening would say nothing about
    which difference caused it."""
    built = _plan(distortion.name, strategy=_strategy_for(distortion))
    paths = {step.argv[4] for half in (built.control, built.distorted)
             for step in half if step.is_command}
    assert len(paths) == 1, paths


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_both_halves_stay_inside_the_proving_ground_realm(distortion: Any) -> None:
    """A load landing in the demo realm would break the installation check on a
    realm that exists to answer whether the installation works."""
    built = _plan(distortion.name, strategy=_strategy_for(distortion))
    for half in (built.control, built.distorted):
        for step in half:
            if step.is_command:
                argv = list(step.argv)
                assert argv[argv.index("--realm-id") + 1] == "proving-ground"


# ── each half is a load that differs by one thing ─────────────────────────────

def test_the_reingest_pair_differs_only_in_the_second_load() -> None:
    """The broken half is the control's load followed by a second one. The
    first must be identical, otherwise the pair carries two differences."""
    built = _plan("reingest_with_another_chunk_size", strategy="fixed")
    assert len(built.control) == 1
    assert len(built.distorted) == 2
    control = [a for a in built.control[0].argv if a != built.control_corpus_id]
    first = [a for a in built.distorted[0].argv if a != built.distorted_corpus_id]
    assert control == first, "the broken half's first load is not the control's load"
    assert built.distorted[0].argv[-1] != built.distorted[1].argv[-1], "both loads use one chunk size"


def test_the_stub_pair_differs_only_in_the_model() -> None:
    built = _plan("index_with_the_stub_embedder")
    assert dict(built.control[0].env)["USE_REAL_BGE_M3"] == "true"
    assert dict(built.distorted[0].env)["USE_REAL_BGE_M3"] == "false"


def test_the_analyser_pair_crosses_the_two_languages() -> None:
    """Each corpus is loaded under the other's analyser, so the defect is
    reproduced twice independently and not argued from one observation."""
    assert _analyser_of(_plan("index_under_the_other_analyser", _RU).distorted[0]) == "en"
    assert _analyser_of(_plan("index_under_the_other_analyser", _EN).distorted[0]) == "ru_be"
    assert _analyser_of(_plan("index_under_the_other_analyser", _RU).control[0]) == "ru_be"
    assert _analyser_of(_plan("index_under_the_other_analyser", _EN).control[0]) == "en"


def _analyser_of(step: Step) -> str:
    argv = list(step.argv)
    return argv[argv.index("--language") + 1]


def test_the_model_mismatch_pair_says_its_second_half_is_not_a_command() -> None:
    """The mismatch lives between the load and the service that answers, so
    half of it cannot be an invocation. Written as a shell comment first, which
    produced a plan that looked runnable end to end and was not."""
    built = _plan("index_and_query_with_different_models")
    instructions = [step for step in built.distorted if not step.is_command]
    assert len(instructions) == 1
    assert instructions[0].note
    with pytest.raises(ValueError):
        instructions[0].as_shell()


# ── refusing, instead of provoking nothing ────────────────────────────────────

def test_reingesting_under_the_structural_strategy_refuses() -> None:
    """Measured: under the structural strategy a chunk size change gave three
    chunks against three with every identifier shared, so the second load
    replaces the first and the collection is left exactly as it was."""
    with pytest.raises(LoadCannotCarryDistortion):
        _plan("reingest_with_another_chunk_size", strategy="structure_aware")


def test_an_unknown_distortion_names_the_ones_that_exist() -> None:
    with pytest.raises(KeyError) as raised:
        _plan("break_everything")
    assert "index_under_the_other_analyser" in str(raised.value)


# ── the catalogue entries these claim ─────────────────────────────────────────

@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_every_entry_named_exists_in_the_catalogue(distortion: Any) -> None:
    from core.eval.atlas import FAILURES

    known = {f.id for f in FAILURES}
    assert distortion.provokes
    for failure_id in distortion.provokes:
        assert failure_id in known, f"{distortion.name} names {failure_id}, absent from the catalogue"


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_a_distortion_proving_only_silence_says_why(distortion: Any) -> None:
    """A pair that proves silence proves that nothing speaks, which is a claim
    about the platform and not about the run. Read as a passing bait it would
    say the opposite of what it shows."""
    assert distortion.proves in ("signal", "silence")
    if distortion.proves == "silence":
        assert distortion.proves_note
    else:
        assert not distortion.proves_note


def test_the_entries_proving_silence_are_the_ones_the_catalogue_calls_undetectable() -> None:
    """The two agree, or one of them is out of date. The catalogue says these
    entries have no signal; this tool says its pair can only show silence, and
    the day a signal is written for them, one of these two must fail."""
    from core.eval.atlas import FAILURES

    by_id = {f.id: f for f in FAILURES}
    for distortion in DISTORTIONS:
        for failure_id in distortion.provokes:
            entry = by_id[failure_id]
            if distortion.proves == "silence":
                assert not entry.signals, (
                    f"{failure_id} now names a signal, so the pair no longer only shows silence"
                )
            else:
                assert entry.signals, (
                    f"{failure_id} names no signal, so the pair cannot show one reddening"
                )


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_every_entry_it_names_is_one_a_load_can_stage(distortion: Any) -> None:
    """`ingest` was split off from `config` because both tools existed and both
    claimed entries marked `config`, so a reader could not tell which applied.
    A distortion here pointing at a run's field would stage nothing, and its
    bait would then measure the wrong instrument."""
    from core.eval.atlas import FAILURES

    by_id = {f.id: f for f in FAILURES}
    for failure_id in distortion.provokes:
        entry = by_id[failure_id]
        allowed = (entry.instrument, *entry.also_staged_by)
        assert "ingest" in allowed, (
            f"{failure_id} is staged by {allowed}, and a load is not among them"
        )


def test_the_two_instruments_together_cover_what_used_to_be_one() -> None:
    """The split must lose nothing. Ten entries carried `config` before it, and
    the two tools between them have to claim exactly those ten."""
    from core.eval.atlas import FAILURES
    from tools.config_distort import DISTORTIONS as CONFIG_DISTORTIONS

    by_instrument = {"config": set(), "ingest": set()}
    for failure in FAILURES:
        if failure.instrument in by_instrument:
            by_instrument[failure.instrument].add(failure.id)

    claimed_by_config = {i for d in CONFIG_DISTORTIONS for i in d.provokes}
    claimed_by_ingest = {i for d in DISTORTIONS for i in d.provokes}
    assert claimed_by_config == by_instrument["config"], (
        f"config entries unclaimed: {sorted(by_instrument['config'] - claimed_by_config)}"
    )
    assert claimed_by_ingest == by_instrument["ingest"], (
        f"ingest entries unclaimed: {sorted(by_instrument['ingest'] - claimed_by_ingest)}"
    )
    assert len(claimed_by_config | claimed_by_ingest) == 10
