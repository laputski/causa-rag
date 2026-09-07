"""Two failures of the measuring, staged by measuring honestly.

Neither needs anything broken. A number reported after a search over
configurations is partly the search, and how much cannot be recovered from
the number; a difference smaller than the scatter of the questions is not a
difference the set can see. Both are properties of ordinary work, which is
what makes them worth catching: nothing looks wrong while either is
happening.

The runs go through the store, because that is where both answers live. The
provenance of a number is a fact about the other runs on the same question
set, and a comparison is a fact about two runs; a pair built from results
held in memory would prove the arithmetic and not the path.

Every store call happens inside one event loop. The Mongo client caches
itself against the first loop it sees, so a second `asyncio.run` in one
session raises, and a helper catching that would report an empty store as an
honest one.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.proving_ground.conftest import LANGUAGE, record, retrieval_only, run_on

pytestmark = pytest.mark.proving_ground

CORPUS = "base-ru"


def _config(name: str, **overrides: Any) -> Any:
    from core.experiment.config import ExperimentConfig
    from tools.seed_proving_ground import control_config

    base = control_config(CORPUS).model_dump(exclude={"config_hash"})
    base["name"] = f"proving-ground-{name}"
    base["chunking_strategy"] = {"kind": "chunker", "component_id": "structure_aware",
                                 "params": {}}
    base.update(overrides)
    return retrieval_only(ExperimentConfig(**base))


def _run(embedder: Any, name: str, **overrides: Any) -> dict[str, Any]:
    return run_on(embedder, _config(name, **overrides), CORPUS, LANGUAGE)


@pytest.fixture(scope="module")
def searched(embedder: Any) -> dict[str, Any]:
    """Five configurations over one question set, stored and read back.

    Five, because the detector's own line between a note and a warning falls
    at four, and a pair that never crossed it would prove the quieter half
    only.
    """
    import asyncio

    from core.experiment.runner import ExperimentResult
    from services.api_gateway.routers.experiments import (
        _get_results,
        _parse_result,
        _save,
        _tuning_provenance,
    )

    runs = [_run(embedder, f"search-k{k}", top_k=k) for k in (3, 4, 5, 6, 7)]

    async def _through_the_store() -> dict[str, Any]:
        parsed: list[ExperimentResult] = []
        for i, payload in enumerate(runs):
            # A start time of its own, so "the runs before this one" has an
            # order to mean anything by. Two runs of a suite can otherwise
            # share a timestamp to the second.
            payload = {**payload, "started_at": f"2026-09-0{i + 1}T10:00:00+00:00"}
            result = _parse_result(payload)
            assert result is not None, "a run of this pair did not parse"
            result.run_id = f"proving-ground-search-{i}"
            parsed.append(result)
            await _save(result)
        stored = await _get_results()
        return {
            "ids": [r.run_id for r in parsed],
            "provenance": {r.run_id: _tuning_provenance(r.run_id, stored) for r in parsed},
            "stored": stored,
        }

    return asyncio.run(_through_the_store())


def test_F37_the_fifth_configuration_on_one_question_set_says_so(
    searched: dict[str, Any],
) -> None:
    """The first run of a set searched over nothing; the fifth is the best of
    five, and which part of it is the search cannot be recovered afterwards."""
    from core.eval.detectors import detect_tuned_on_the_measurement_set

    first, last = searched["ids"][0], searched["ids"][-1]
    quiet = detect_tuned_on_the_measurement_set(
        {"tuning_provenance": searched["provenance"][first]})
    spoke = detect_tuned_on_the_measurement_set(
        {"tuning_provenance": searched["provenance"][last]})

    assert quiet is None, (
        f"the first configuration on this set is reported as tuned: {quiet and quiet.detail}"
    )
    assert spoke is not None, (
        f"four configurations preceded this one and nothing said so: "
        f"{searched['provenance'][last]}"
    )
    assert spoke.severity == "warn", (
        f"five configurations on one set and the finding is only a note: {spoke.severity}"
    )
    record("F37", "the fifth configuration on one question set, and the number is the best of it",
           configurations_before_the_first=searched["provenance"][first]["configurations_before"],
           configurations_before_the_last=searched["provenance"][last]["configurations_before"],
           dataset=searched["provenance"][last]["dataset_name"],
           severity=spoke.severity)


def test_F36_a_difference_this_question_set_cannot_resolve(
    embedder: Any, searched: dict[str, Any],
) -> None:
    """A movement smaller than the scatter of the questions themselves.

    Both halves are real runs and neither is broken. Taking the reranker out
    raises recall from 0.94 to 1.00, a single question of seventeen, and the
    set cannot tell that movement from its own noise: somebody could ship
    "we removed the reranker and recall improved" on evidence the questions
    do not support. Against a corpus indexed by the stub the movement is far
    larger than the noise, and the warning stays silent so the reader can
    read the difference.

    Which lever moves a metric slightly was measured and not chosen. Varying
    top_k moves recall by exactly nothing on this corpus, because the right
    source is always in the first three, so no pair built from it could have
    staged this at all.

    The silent half is the same configuration run twice. A comparison that
    moved a metric a great deal was tried there first and refused: the check
    speaks per metric, and a pair of runs carries several, so a difference
    large in one is nearly always small in another. That is the check being
    right about a metric, and it is why it reports which metrics, never a
    verdict on the comparison.
    """
    from core.experiment.compare import check_comparability

    control = _parse_result_of(_run(embedder, "resolution-control"))
    again = _parse_result_of(_run(embedder, "resolution-control"))
    without_the_reranker = _parse_result_of(_run(embedder, "resolution-norerank", reranker=None))

    subtle = {f"compare:{w.id}" for w in check_comparability(control, without_the_reranker)}
    quiet = {f"compare:{w.id}" for w in check_comparability(control, again)}

    assert "compare:below_the_sets_resolution" in subtle, (
        f"a metric moved by one question of seventeen and nothing said the set cannot "
        f"resolve it: {sorted(subtle)}"
    )
    # `same_run` and not `below_the_sets_resolution`: two runs of one
    # configuration carry one identifier, and saying so is a different
    # warning about a different thing.
    assert "compare:below_the_sets_resolution" not in quiet, (
        f"nothing moved and the set is still told it cannot resolve the movement: "
        f"{sorted(quiet)}"
    )
    record("F36", "a difference smaller than the scatter of the questions, and none at all",
           signals_on_the_small_difference=sorted(subtle),
           signals_on_no_difference=sorted(quiet),
           recall_control=control.aggregate_metrics.get("retrieval_recall_at_k"),
           recall_without_the_reranker=without_the_reranker.aggregate_metrics.get(
               "retrieval_recall_at_k"),
           questions=len(control.question_results))


def _parse_result_of(payload: dict[str, Any]) -> Any:
    from services.api_gateway.routers.experiments import _parse_result

    result = _parse_result(payload)
    assert result is not None, "a run of this pair did not parse"
    return result
