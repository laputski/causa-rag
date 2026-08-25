"""services/api_gateway/routers/experiments.py:_load_active_pins.

Rewritten from the MongoDB version. The overlay is now built from the Realm's
judgments file, which is the substance of the change rather than a storage
preference: `the design notes` forbids anything a served system
needs at query time from living in the platform's database, and a file is an
artefact that can simply be handed over.

These tests pin the two properties that make the file version correct where
the Mongo version was not — the reviewer's own verdict is what becomes the
overlay, and the query vector is derived rather than stored.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.judgments import JudgedChunk, RelevanceJudgment
from core.judgments.store import append_judgment, judgments_path
from services.api_gateway.routers.experiments import _load_active_pins


class _StubEmbedder:
    """Records what it was asked to embed, so a test can prove the vector is
    computed from the stored question text rather than read from the file."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def _write(dir_path: Path, judgment: RelevanceJudgment) -> None:
    append_judgment(judgments_path(dir_path, judgment.realm_id, judgment.corpus_id), judgment)


def _judgment(**overrides) -> RelevanceJudgment:
    base = dict(
        id="j1",
        realm_id="demo",
        corpus_id="handbook",
        question="a reviewer's question",
        relevant=(JudgedChunk(chunk_id="c1", doc_id="d1", text="t", structural_path="art-1"),),
        irrelevant=(JudgedChunk(chunk_id="c2"),),
    )
    base.update(overrides)
    return RelevanceJudgment(**base)


class TestLoadActivePins:
    async def test_a_judgment_becomes_the_overlay_for_its_own_realm_and_corpus(self, tmp_path) -> None:
        _write(tmp_path, _judgment())
        embedder = _StubEmbedder()
        with patch("services.api_gateway.routers.judgments._JUDGMENTS_DIR", tmp_path), \
             patch("core.registry.registry.resolve", return_value=embedder):
            pins = await _load_active_pins("demo", "handbook")

        assert len(pins) == 1
        pin = pins[0]
        assert pin.id == "j1"
        # The reviewer said "relevant" and "irrelevant"; the overlay reads
        # that as pin and demote rather than as a separately authored order.
        assert pin.pin_chunks[0].chunk_id == "c1"
        assert pin.pin_chunks[0].text == "t"
        assert pin.demote_chunk_ids == ["c2"]

    async def test_the_query_vector_is_embedded_not_stored(self, tmp_path) -> None:
        """A vector belongs to one embedding model. Storing it in an artefact
        meant to outlive model changes would expire the file the day the
        model changes, silently and with no way to tell from the file."""
        _write(tmp_path, _judgment())
        embedder = _StubEmbedder()
        with patch("services.api_gateway.routers.judgments._JUDGMENTS_DIR", tmp_path), \
             patch("core.registry.registry.resolve", return_value=embedder):
            pins = await _load_active_pins("demo", "handbook")

        assert embedder.seen == ["a reviewer's question"]
        assert pins[0].query_vec == [1.0, 0.0]

    async def test_another_realms_file_is_never_read(self, tmp_path) -> None:
        _write(tmp_path, _judgment(realm_id="other-realm"))
        with patch("services.api_gateway.routers.judgments._JUDGMENTS_DIR", tmp_path), \
             patch("core.registry.registry.resolve", return_value=_StubEmbedder()):
            assert await _load_active_pins("demo", "handbook") == []

    async def test_retired_and_empty_judgments_are_not_applied(self, tmp_path) -> None:
        _write(tmp_path, _judgment(id="retired", status="retired"))
        _write(tmp_path, _judgment(id="empty", relevant=(), irrelevant=()))
        _write(tmp_path, _judgment(id="live"))
        with patch("services.api_gateway.routers.judgments._JUDGMENTS_DIR", tmp_path), \
             patch("core.registry.registry.resolve", return_value=_StubEmbedder()):
            pins = await _load_active_pins("demo", "handbook")
        assert [p.id for p in pins] == ["live"]

    async def test_no_file_yet_is_an_empty_overlay_not_a_failure(self, tmp_path) -> None:
        with patch("services.api_gateway.routers.judgments._JUDGMENTS_DIR", tmp_path):
            assert await _load_active_pins("demo", "handbook") == []

    async def test_an_unavailable_embedder_degrades_to_no_overlay(self, tmp_path) -> None:
        """Losing the overlay costs a what-if, not a measurement. Failing the
        whole run over it would trade something cheap for something dear."""
        _write(tmp_path, _judgment())
        with patch("services.api_gateway.routers.judgments._JUDGMENTS_DIR", tmp_path), \
             patch("core.registry.registry.resolve", side_effect=KeyError("no embedder")):
            assert await _load_active_pins("demo", "handbook") == []
