"""The healthy configuration of the proving ground, and the pairs built from it.

Every distortion is a difference from this one. Two things therefore have to
hold, and neither is obvious enough to leave unchecked: the control has to be
healthy, and each distortion has to have a field on it to move.

The first is the sharper of the two. The local reranker defaults to an
English-only model, so a control that said nothing about the model would carry,
on the Russian corpus, exactly the failure one of the distortions stages. The
pair would then measure the difference between two broken halves and read as a
signal that stayed silent.
"""
from __future__ import annotations

import pytest

from tools.config_distort import DISTORTIONS, ConfigurationCannotCarryDistortion, distort
from tools.seed_proving_ground import CONTROL_RERANKER_MODEL, CORPORA, control_config

_BY_ID = {corpus.corpus_id: corpus for corpus in CORPORA}


# ── the control is healthy ────────────────────────────────────────────────────

@pytest.mark.parametrize("corpus_id", sorted(_BY_ID))
def test_the_control_names_a_reranker_that_covers_its_corpus(corpus_id: str) -> None:
    """The platform default is English only. Saying nothing here would put the
    defect into the half that exists to be free of it."""
    from adapters.reranker import DEFAULT_LOCAL_MODEL

    reranker = control_config(corpus_id).reranker
    assert reranker is not None
    assert reranker.params.get("model_name") == CONTROL_RERANKER_MODEL
    assert CONTROL_RERANKER_MODEL != DEFAULT_LOCAL_MODEL, (
        "the control now relies on the platform default, so it stops proving anything"
    )


@pytest.mark.parametrize("corpus_id", sorted(_BY_ID))
def test_the_control_fetches_wider_than_it_selects(corpus_id: str) -> None:
    """A window equal to the selection is one of the distortions, so a control
    starting there would leave that pair with no difference in it."""
    control = control_config(corpus_id)
    assert control.fetch_k is not None
    assert control.fetch_k > control.top_k


@pytest.mark.parametrize("corpus_id", sorted(_BY_ID))
def test_the_control_merges_two_halves_by_rank(corpus_id: str) -> None:
    control = control_config(corpus_id)
    assert control.pipeline_id == "hybrid_rrf"
    assert control.merge_strategy == "rrf"


@pytest.mark.parametrize("corpus_id", sorted(_BY_ID))
def test_the_control_points_at_the_question_set_of_its_own_corpus(corpus_id: str) -> None:
    """A configuration measuring one corpus against another corpus's questions
    reads as a total retrieval failure, which is what a broken system looks
    like too."""
    control = control_config(corpus_id)
    assert control.corpus_id == corpus_id
    assert control.dataset_name.startswith(corpus_id)


@pytest.mark.parametrize("corpus_id", sorted(_BY_ID))
def test_the_control_describes_itself(corpus_id: str) -> None:
    assert control_config(corpus_id).fingerprint_matches_fields()


def test_the_two_controls_are_two_configurations() -> None:
    """They differ by corpus and question set, so they must not share a
    fingerprint; a shared one would file the second run as a repeat."""
    hashes = {control_config(corpus.corpus_id).config_hash for corpus in CORPORA}
    assert len(hashes) == len(CORPORA)


# ── every distortion has something to move ────────────────────────────────────

@pytest.mark.parametrize("corpus_id", sorted(_BY_ID))
@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_each_distortion_either_applies_or_refuses_for_a_stated_reason(
    distortion: object, corpus_id: str,
) -> None:
    control = control_config(corpus_id)
    language = _BY_ID[corpus_id].language
    try:
        broken = distort(control, distortion.name, corpus_language=language)  # type: ignore[attr-defined]
    except ConfigurationCannotCarryDistortion as refused:
        assert distortion.requires in str(refused)  # type: ignore[attr-defined]
        return
    assert set(control.diff(broken)) == set(distortion.changes)  # type: ignore[attr-defined]


def test_the_pairs_the_proving_ground_can_actually_stage() -> None:
    """Counted, so that a distortion quietly refusing everywhere is visible as
    a number and not as an absence. Nine today: five on the Russian corpus and
    four on the English one, where pairing an English reranker with English
    text is the right tool and the distortion says so."""
    staged = [
        (corpus.corpus_id, distortion.name)
        for corpus in CORPORA
        for distortion in DISTORTIONS
        if _applies(distortion, corpus)
    ]
    assert len(staged) == 11, staged
    assert ("base-en", "rerank_in_another_language") not in staged


def _applies(distortion: object, corpus: object) -> bool:
    try:
        distort(control_config(corpus.corpus_id), distortion.name,  # type: ignore[attr-defined]
                corpus_language=corpus.language)  # type: ignore[attr-defined]
    except ConfigurationCannotCarryDistortion:
        return False
    return True


@pytest.mark.parametrize("distortion", DISTORTIONS, ids=lambda d: d.name)
def test_no_distortion_refuses_on_both_corpora(distortion: object) -> None:
    """One that refused everywhere would be a distortion the proving ground
    cannot stage at all, and its catalogue entry would sit at "claim unproven"
    with nothing saying why."""
    assert any(_applies(distortion, corpus) for corpus in CORPORA), (
        f"{distortion.name} applies to neither corpus"  # type: ignore[attr-defined]
    )


def test_the_control_carries_no_setting_nothing_reads() -> None:
    """`params` is a no-op for an in-process run apart from the model key, and
    it is hashed. A language recorded there was written first and removed: it
    moved the fingerprint of both controls and reached no code at all, which is
    the decorative-field trap this whole phase exists to close."""
    for corpus in CORPORA:
        assert control_config(corpus.corpus_id).params == {}


def test_an_unknown_corpus_is_refused_and_not_silently_configured() -> None:
    """It used to look the corpus up in a dictionary, so a typo raised a
    KeyError from inside a comprehension with no mention of what was wrong."""
    with pytest.raises(KeyError) as raised:
        control_config("base-de")
    assert "base-ru" in str(raised.value)


# ── the same settings pointed at a distorted index ────────────────────────────

def test_the_same_settings_can_be_pointed_at_a_load_level_pair() -> None:
    """The load-level defects leave the documents and the settings alone, so
    their pair is one configuration against two indexes. Without this the pair
    could not be expressed: a distorted namespace is not one of the corpora
    this seed knows, and asking for it was refused."""
    control = control_config("base-ru")
    distorted = control_config("base-ru", indexed_as="base-ru-stubbed")
    assert sorted(control.diff(distorted)) == ["corpus_id", "name"]
    assert distorted.corpus_id == "base-ru-stubbed"


def test_the_question_set_stays_the_one_written_for_these_documents() -> None:
    """Only the index differs; the questions are the same. A pair that also
    moved the question set would compare two measurements of two things."""
    assert control_config("base-ru", indexed_as="base-ru-stubbed").dataset_name == \
        control_config("base-ru").dataset_name


def test_the_two_halves_of_a_load_pair_are_two_configurations() -> None:
    """They must not share a fingerprint, or the second run is filed as a
    repeat of the first and the comparison has one side."""
    assert control_config("base-ru").config_hash != \
        control_config("base-ru", indexed_as="base-ru-stubbed").config_hash


def test_pointing_at_another_corpus_index_is_refused() -> None:
    """The Russian questions asked of the English index return nothing, and
    every retrieval metric reads zero, which is what a ruined index looks like
    too. A pair built that way would credit the defect with a failure it did
    not cause."""
    with pytest.raises(ValueError):
        control_config("base-ru", indexed_as="base-en-stubbed")


def test_every_namespace_the_load_distortions_produce_is_accepted() -> None:
    """The two tools have to fit together. One invents namespaces and the other
    has to be able to run against them, and nothing but this checks that."""
    from tools.ingest_distort import DISTORTIONS as LOAD_DISTORTIONS
    from tools.ingest_distort import plan
    from tools.seed_proving_ground import GROUND_DIR

    for corpus in CORPORA:
        for distortion in LOAD_DISTORTIONS:
            strategy = "fixed" if distortion.name == "reingest_with_another_chunk_size" \
                else "structure_aware"
            built = plan(distortion.name, str(GROUND_DIR / corpus.directory),
                         corpus.corpus_id, corpus.language, strategy)
            for namespace in (built.control_corpus_id, built.distorted_corpus_id):
                config = control_config(corpus.corpus_id, indexed_as=namespace)
                assert config.corpus_id == namespace
