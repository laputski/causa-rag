"""A corpus loaded into a realm has to reach that realm's own list.

The loading command takes a realm identifier and, until now, the realm learned
nothing from it: the index existed, every metric could be computed against it,
and it appeared in no list on any screen. Found while writing a manual check of
the proving ground, where a deliberately damaged corpus was loaded and then
could not be selected to look at, which is half of what that realm is for.

The registry record is what the interface lists, and the helper that writes it
declares itself idempotent and safe to call on every successful load. The
gateway's own ingest route has always called it; the command line had not.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from services.ingestion.cli import _register_in_the_realm

RESULT = {
    "files": 10, "chunks": 220,
    "qdrant_collection": "proving-ground__base-ru__structure_aware__bge_m3",
    "opensearch_index": "rag__proving-ground__base-ru__structure_aware",
}


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {}


def _register(result: dict[str, Any] = RESULT) -> tuple[_Recorder, str]:
    recorder = _Recorder()
    with patch("services.api_gateway.routers.corpus._register_corpus", recorder):
        said = _register_in_the_realm("proving-ground", "base-ru", result)
    return recorder, said


def test_the_realm_is_told_which_corpus_was_loaded() -> None:
    recorder, said = _register()
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["realm_id"] == "proving-ground"
    assert recorder.calls[0]["corpus_id"] == "base-ru"
    assert "base-ru" in said and "proving-ground" in said


def test_both_indexes_are_recorded_by_the_names_the_load_actually_used() -> None:
    """Derived here would mean deriving the collection name a second time, and
    a second derivation drifts from the first. The loader returns what it
    wrote."""
    backends = _register()[0].calls[0]["backends"]
    assert backends["qdrant"]["collection"] == RESULT["qdrant_collection"]
    assert backends["opensearch"]["index"] == RESULT["opensearch_index"]


def test_a_load_without_the_lexical_half_is_recorded_as_dense_only() -> None:
    """One of the proving ground's own distortions loads the semantic half
    alone. Recording it as hybrid would describe an index that is not there."""
    call = _register({**RESULT, "opensearch_index": None})[0].calls[0]
    assert call["storage_type"] == "dense_only"
    assert "opensearch" not in call["backends"]


def test_a_load_with_both_halves_is_recorded_as_hybrid() -> None:
    assert _register()[0].calls[0]["storage_type"] == "hybrid"


def test_a_registry_that_refuses_does_not_fail_the_load() -> None:
    """The documents are in the index either way, and a load that succeeded is
    not undone by a database that was unreachable a second later. What must not
    happen is silence: the failure is returned for the caller to print."""
    async def refuse(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("mongodb unreachable")

    with patch("services.api_gateway.routers.corpus._register_corpus", refuse):
        said = _register_in_the_realm("proving-ground", "base-ru", RESULT)
    assert "not registered" in said
    assert "mongodb unreachable" in said


def test_the_command_line_calls_it_only_when_a_realm_was_named() -> None:
    """A load with no realm belongs to no realm's list. Read out of the source,
    because reaching this branch end to end needs a whole stack up."""
    import pathlib

    source = (pathlib.Path(__file__).parents[2] / "services" / "ingestion" / "cli.py").read_text(
        encoding="utf-8")
    assert "if args.realm_id:" in source
    assert "_register_in_the_realm(args.realm_id" in source
