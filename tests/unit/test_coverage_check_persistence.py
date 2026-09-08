"""one task of that change — the coverage flag must survive persistence.

The flag is useless if it is computed and then dropped on the way to the
reader, which is exactly what happened to `resample_attempted`/
`noise_filtered` earlier in this project: computed server-side, never
declared in the client type, never rendered. This pins the round trip.
"""
from core.experiment.config import ExperimentConfig
from core.experiment.runner import ExperimentResult
from services.api_gateway.routers.experiments import _parse_result

_CFG = ExperimentConfig(
    name="t",
    chunking_strategy={"kind": "chunker", "component_id": "fixed"},
    embedder={"kind": "embedder", "component_id": "bge_m3"},
    generator={"kind": "generator", "component_id": "ollama"},
)


def test_coverage_check_survives_to_dict_and_back() -> None:
    r = ExperimentResult(config=_CFG, run_id="r1")
    r.coverage_check = {"checked": False, "reason": "index unreachable"}
    back = _parse_result(r.to_dict(), "r1")
    assert back is not None
    assert back.coverage_check == {"checked": False, "reason": "index unreachable"}


def test_run_without_the_field_round_trips_as_empty() -> None:
    back = _parse_result(ExperimentResult(config=_CFG, run_id="r2").to_dict(), "r2")
    assert back is not None
    assert back.coverage_check == {}


# ── Found live: an unreadable index looked like an empty one ─────────────
# QdrantRetriever.scroll against a collection that does not exist returns an
# empty page and raises nothing. Building a ref index from that would mark
# every ref "absent" and every question "uncovered" — the same silent loss of
# measurement that was removed, with a new cause. Caught by running a real
# experiment against a bogus corpus_id, not by any unit test.

def test_empty_read_is_reported_as_unverified_not_as_empty_corpus() -> None:
    import asyncio
    from unittest.mock import patch

    class _EmptyRetriever:
        _corpus_id = "x"

        def scroll(self, offset=None, limit=0):
            return [], None

    class _Pipeline:
        _retriever = _EmptyRetriever()

    from services.api_gateway.routers import experiments as ex

    with patch.object(ex.registry, "resolve", return_value=_Pipeline()), \
         patch("core.experiment.runner._for_the_corpus", lambda r, *a, **k: r):
        resolver = asyncio.run(ex._build_ref_resolver(realm_id="r", corpus_id="missing"))

    assert resolver.checked is False
    assert resolver.presence("any/ref") == "unknown"
