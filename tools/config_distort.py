"""Put one known defect into a healthy configuration, and nothing else.

The companion of `tools/corpus_mutate.py`, working on the settings where that
one works on the documents. A proving ground is only as good as the difference between its two
halves, and a hand-written broken configuration drifts away from its control
the first time somebody edits one of them. So the broken one is derived from
the healthy one, by a named change, on demand.

Three properties are enforced here, and none of them merely intended:

**Exactly the declared fields differ.** Every distortion names the fields it
touches, and the result is compared against its control before being handed
back. A distortion that quietly moved a second setting would make its bait
prove that two configurations differ, and nothing more.

**A distortion that cannot bite refuses.** Pinning the rank-fusion constant on
a configuration that merges by weight changes a number nothing reads. Half the
defects of the corpus mutator were written without this guard and produced
corpora that looked mutated and provoked nothing, which is the failure this
whole apparatus exists to stop somebody shipping.

**What the configuration cannot say is asked for.** A configuration names a
corpus and not its language, so the distortion that pairs a reranker with a
language it does not cover refuses when nobody says which language the corpus
is in. Guessing from the corpus identifier would work on this proving ground
and quietly stop working on anybody else's.

    python3 -m tools.config_distort --list
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig

# A reranker trained on English alone. Named here because it is what the local
# reranker defaults to, so this distortion asks for the platform's own default
# and the *control* is the configuration that has to name something else.
ENGLISH_ONLY_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MULTILINGUAL_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class Distortion:
    """One named way to break a configuration, and what it is meant to provoke."""

    name: str
    #: What the configuration says afterwards, in the words a reader would use.
    describes: str
    #: The catalogue entries this distortion is meant to make observable. Named
    #: so a distortion nobody has a use for is visible as such.
    provokes: tuple[str, ...]
    #: The fields it is allowed to move. Checked against the result, so a
    #: distortion touching a second setting fails here and not in a bait.
    changes: tuple[str, ...]
    apply: Callable[[ExperimentConfig], ExperimentConfig]
    #: What the configuration must already be for this distortion to bite, and
    #: an empty string when anything will do.
    requires: str = ""
    admits: Callable[[ExperimentConfig, str | None], bool] | None = None
    #: Whether this reproduces the cause the catalogue entry names, or only the
    #: effect its signal reads. A bait built on an "effect" distortion proves
    #: the signal works and does not prove the entry was reproduced, and the two
    #: are worth telling apart before somebody reads the first as the second.
    stages: str = "cause"
    #: Why only the effect, required whenever `stages` says so.
    stages_note: str = ""


# Which pipelines merge two retrieval halves, and which run on one. Maintained
# by hand, and the union is checked against the gateway's own registrations, so
# a pipeline nobody classified fails a test instead of silently counting as
# single-source.
#
# The count of `retrievers` on a configuration would be the honest field to read
# and is unusable: it is written by nothing and read by nothing. Across all
# forty-one stored runs it holds zero or one entry, eighteen genuinely hybrid
# runs included, and no code in core, services or adapters reads it. The
# identifier below is what the platform actually resolves a pipeline from, so
# on this platform it is the applied fact and not a label.
MERGING_PIPELINES = ("hybrid_rrf", "hybrid_weighted")
SINGLE_SOURCE_PIPELINES = ("naive", "graph")


def _is_hybrid(config: ExperimentConfig) -> bool:
    """Whether two retrieval halves are merged at all."""
    return config.pipeline_id in MERGING_PIPELINES or len(config.retrievers) > 1


def _narrow_the_selection(config: ExperimentConfig) -> ExperimentConfig:
    """Cut the selection to a single fragment.

    The right fragment then has to rank first or be lost, so every question
    whose answer sat at rank two becomes a miss with no trace of how close it
    came.
    """
    return _replace(config, top_k=1)


def _let_one_half_decide(config: ExperimentConfig) -> ExperimentConfig:
    """Weight the merge entirely onto the semantic half.

    The lexical half then contributes nothing to the order, which is what a
    ranking ruled by one source looks like from the outside, and it is what the
    dominance signal reads.

    It is deliberately not the cause the catalogue entry names. That cause is
    two halves on incomparable scales, and this platform's weighted merge
    divides each half by its own maximum before weighting, so the scales cannot
    disagree here. Measured on the real ranges of this platform's own runs,
    cosine reaching 0.89 against a lexical score reaching 58.1: at a weight of
    one half each, the merged top three took one document from the lexical half,
    exactly as rank fusion did. Switching the strategy alone stages nothing, and
    the plan that said it would was reading the field name.
    """
    return _replace(config, merge_strategy="weighted", merge_alpha=1.0)


def _pin_the_fusion_constant(config: ExperimentConfig) -> ExperimentConfig:
    """Set the rank-fusion constant to one.

    Reciprocal rank fusion scores a document as 1/(rrf_k + rank). At sixty the
    two lists have nearly equal say; at one the first place in either list
    outweighs everything below it, so the merge stops being a merge.
    """
    return _replace(config, rrf_k=1)


def _close_the_candidate_window(config: ExperimentConfig) -> ExperimentConfig:
    """Fetch exactly as many candidates as reach the answer.

    The reranker can then reorder what retrieval already chose and can rescue
    nothing, because nothing outside the selection was ever fetched. The run
    looks like it reranks and pays for a reranker that cannot change the set.
    """
    return _replace(config, fetch_k=config.top_k)


def _rerank_in_another_language(config: ExperimentConfig) -> ExperimentConfig:
    """Rerank with a model trained on English alone.

    Against a corpus in any other language the reranker reorders by noise, and
    the visible result is a rerank step that loses the source retrieval had
    already found.
    """
    reranker = config.reranker
    assert reranker is not None  # guarded by `admits`
    return _replace(config, reranker=ComponentRef(
        kind=reranker.kind,
        component_id=reranker.component_id,
        params={**reranker.params, "model_name": ENGLISH_ONLY_RERANKER},
    ))


# Every segmentation this platform can name on a configuration. A corpus is cut
# by one of these when it is loaded, and the index carries the name of the one
# that cut it, so naming a different one on a run is what the distortion below
# does.
SEGMENTATIONS = ("fixed", "structure_aware", "sentence", "paragraph")


def _name_another_segmentation(config: ExperimentConfig) -> ExperimentConfig:
    """Name a segmentation the index was not built with.

    The field selects nothing: a corpus is cut at load time and a run reads
    what the cut produced. That is the whole failure. The run records the name
    all the same, so two runs differing only in it are two runs of one
    strategy while every screen and every stored document says they are two.

    Any other name will do, since none of them is the one the index carries
    once this has run. The first that differs is taken, so the same
    configuration always distorts the same way and a bait can be repeated.
    """
    named = config.chunking_strategy.component_id
    other = next(s for s in SEGMENTATIONS if s != named)
    return _replace(config, chunking_strategy=ComponentRef(kind="chunker", component_id=other))


def _replace(config: ExperimentConfig, **fields: Any) -> ExperimentConfig:
    """A copy carrying the changes, with the fingerprint recomputed.

    Rebuilt, never assigned to: the fingerprint is computed once, at
    construction, and assigning to a hashed field leaves a configuration whose
    fingerprint describes one that no longer exists. Thirty-six stored runs
    carry exactly that, from one such assignment.
    """
    payload = config.model_dump(exclude={"config_hash"})
    for key, value in fields.items():
        payload[key] = value.model_dump() if isinstance(value, ComponentRef) else value
    return ExperimentConfig(**payload)


DISTORTIONS: tuple[Distortion, ...] = (
    Distortion("narrow_the_selection",
               "only one fragment reaches the answer",
               ("F18",), ("top_k",), _narrow_the_selection,
               requires="selects more than one fragment, so narrowing it is a change",
               admits=lambda c, _: c.top_k > 1),
    Distortion("let_one_half_decide",
               "the merge is weighted entirely onto the semantic half",
               ("F21",), ("merge_strategy", "merge_alpha"), _let_one_half_decide,
               requires="merges two retrieval halves, since a single half cannot be outweighed",
               admits=lambda c, _: _is_hybrid(c) and (c.merge_strategy, c.merge_alpha) != ("weighted", 1.0),
               stages="effect",
               stages_note=(
                   "this platform's weighted merge divides each half by its own maximum before "
                   "weighting, so the incomparable scales the entry names cannot be staged by a "
                   "setting here; what a setting stages is the dominance the signal reads"
               )),
    Distortion("pin_the_fusion_constant",
               "the rank-fusion constant is one, so the top of either list decides the order",
               ("F22",), ("rrf_k",), _pin_the_fusion_constant,
               requires="merges by rank fusion, since the constant is what rank fusion divides by",
               admits=lambda c, _: _is_hybrid(c) and c.merge_strategy == "rrf" and c.rrf_k != 1),
    Distortion("close_the_candidate_window",
               "retrieval fetches exactly what reaches the answer, so reranking can rescue nothing",
               ("F25",), ("fetch_k",), _close_the_candidate_window,
               requires="reranks, since a window matters only to a step that reorders",
               admits=lambda c, _: c.reranker is not None and c.fetch_k != c.top_k),
    Distortion("name_another_segmentation",
               "the run names a segmentation its index was not built with",
               ("F44",), ("chunking_strategy",), _name_another_segmentation,
               requires="names a segmentation at all, since the distortion is to name another",
               admits=lambda c, _: bool(c.chunking_strategy.component_id)),
    Distortion("rerank_in_another_language",
               "the reranker is a model trained on English alone",
               ("F24",), ("reranker",), _rerank_in_another_language,
               requires=(
                   "reranks a corpus of a declared language other than English, since an "
                   "English reranker on English text is simply the right tool"
               ),
               admits=lambda c, language: (
                   c.reranker is not None and language is not None and not language.startswith("en")
               )),
)

_BY_NAME = {d.name: d for d in DISTORTIONS}


class ConfigurationCannotCarryDistortion(ValueError):
    """The configuration is not the shape this distortion needs.

    Raised instead of returning a configuration that looks distorted and
    provokes nothing.
    """


class DistortionMovedMoreThanItDeclared(AssertionError):
    """The result differs from its control in a field the distortion did not name.

    A bug in this module, caught here so it cannot reach a bait, where it would
    look like evidence about a signal.
    """


def distort(
    config: ExperimentConfig, distortion: str, corpus_language: str | None = None,
) -> ExperimentConfig:
    """The configuration with one named defect in it. Never mutates the input.

    `corpus_language` is what the configuration itself cannot say. Left unset,
    a distortion that depends on it refuses, on the same rule the applicability
    predicates follow: a coordinate nobody stated is not a coordinate that
    failed.
    """
    try:
        chosen = _BY_NAME[distortion]
    except KeyError:
        raise KeyError(f"Unknown distortion {distortion!r}. Known: {sorted(_BY_NAME)}") from None
    if chosen.admits is not None and not chosen.admits(config, corpus_language):
        raise ConfigurationCannotCarryDistortion(
            f"{distortion!r} needs a configuration that {chosen.requires}. This one is not, so "
            "the result would look distorted and provoke nothing."
        )
    result = chosen.apply(config)
    moved = set(config.diff(result))
    if moved != set(chosen.changes):
        raise DistortionMovedMoreThanItDeclared(
            f"{distortion!r} declares it changes {sorted(chosen.changes)} and changed {sorted(moved)}"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--list", action="store_true", help="the distortions this tool can apply")
    parser.parse_args(argv)

    width = max(len(d.name) for d in DISTORTIONS)
    print("Distortions, and the catalogue entries each is meant to make observable:\n")
    for distortion in DISTORTIONS:
        print(f"  {distortion.name:<{width}}  {', '.join(distortion.provokes)}")
        print(f"  {'':<{width}}  {distortion.describes}")
        if distortion.requires:
            print(f"  {'':<{width}}  needs a configuration that {distortion.requires}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
