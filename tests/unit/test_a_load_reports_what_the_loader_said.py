"""The ingest route reads the loader's own words, and does not invent them.

Found live, loading four documents from the upload page: the page said
"Done: 0 chunks" and the registry recorded nought, while the index held
twelve. The gateway was started with `FORCE_COLOR=1`, the loader inherits the
environment, and its log lines then carry colour codes between `chunks=` and
the number, so the count was never read. The same load showed a second fault:
the lexical store refused to create an index, the loader said so and went on
with the semantic half, and the route registered the corpus as having both
halves, because it derived the index name instead of reading what was written.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from services.api_gateway.routers import corpus as C

_ESC = "\x1b"


def _coloured(event: str, **fields: Any) -> bytes:
    said = " ".join(f"{_ESC}[36m{k}{_ESC}[0m={_ESC}[35m{v}{_ESC}[0m" for k, v in fields.items())
    return f"{_ESC}[2m2026-11-06 12:41:07{_ESC}[0m [{_ESC}[32minfo{_ESC}[0m] {_ESC}[1m{event}{_ESC}[0m {said}\n".encode()


class _Proc:
    def __init__(self, lines: list[bytes]) -> None:
        self.returncode = 0
        self._lines = lines

        async def stream():
            for line in self._lines:
                yield line
        self.stdout = stream()

    async def wait(self) -> int:
        return 0


def _run(monkeypatch, tmp_path: Path, lines: list[bytes]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    async def spawn(*args: Any, **kwargs: Any) -> _Proc:
        return _Proc(lines)

    registered: list[dict[str, Any]] = []

    async def register(**kwargs: Any) -> dict[str, Any]:
        registered.append(kwargs)
        return {}

    async def save(doc: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(C.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(C, "_register_corpus", register)
    monkeypatch.setattr(C, "_save_ingest", save)
    C._progress["job"] = []
    doc: dict[str, Any] = {"n_files": 2, "realm_id": "acme"}
    asyncio.run(C._run_ingestion("job", tmp_path, "structure_aware", 512, 64, [], doc, "acme-corpus", {}))
    return C._progress["job"], registered, doc


LOADED = [
    _coloured("ingest.file_done", chunks=3, file="01.md", strategy="structure_aware"),
    _coloured("ingest.file_done", chunks=4, file="02.md", strategy="structure_aware"),
    _coloured("ingest.done", files=2, hit_ratio=0.25, hits=1, misses=3, size=0, total_chunks=7),
]


def test_a_count_in_coloured_lines_is_read(monkeypatch, tmp_path) -> None:
    events, _, doc = _run(monkeypatch, tmp_path, LOADED)
    assert events[-1] == {"type": "done", "job_id": "job", "n_chunks": 7, "hit_ratio": 0.25}
    assert doc["n_chunks"] == 7
    # A line is shown as the loader meant it, without the codes.
    assert all(_ESC not in str(e.get("message", "")) for e in events)
    assert [e["n_chunks"] for e in events if e["type"] == "progress"] == [3, 7]


def test_a_load_with_both_halves_is_registered_with_both(monkeypatch, tmp_path) -> None:
    _, registered, _ = _run(monkeypatch, tmp_path, LOADED)
    assert registered[0]["storage_type"] == "dense_sparse"
    assert "opensearch" in registered[0]["backends"]


def test_a_lexical_half_the_store_refused_is_said_and_not_registered(monkeypatch, tmp_path) -> None:
    refused = [_coloured("ingest.opensearch_unavailable",
                         error="AuthorizationException(403, 'index_create_block_exception')"), *LOADED]
    events, registered, _ = _run(monkeypatch, tmp_path, refused)
    warnings = [e for e in events if e["type"] == "warning"]
    assert [w["code"] for w in warnings] == ["lexical_index_not_written"]
    assert "index_create_block_exception" in warnings[0]["message"]
    assert registered[0]["storage_type"] == "dense_only"
    assert "opensearch" not in registered[0]["backends"]


@pytest.mark.parametrize("value", ["1", "true"])
def test_the_loader_is_not_asked_for_colour(monkeypatch, tmp_path, value) -> None:
    """Stripping the codes is the reading's defence; not asking for them is
    the cause removed, so a line read by anyone else is plain too."""
    seen: dict[str, Any] = {}

    async def spawn(*args: Any, **kwargs: Any) -> _Proc:
        seen.update(kwargs.get("env") or {})
        return _Proc(LOADED)

    monkeypatch.setenv("FORCE_COLOR", value)
    monkeypatch.setattr(C.asyncio, "create_subprocess_exec", spawn)

    async def nothing(*a: Any, **k: Any) -> Any:
        return {}
    monkeypatch.setattr(C, "_register_corpus", nothing)
    monkeypatch.setattr(C, "_save_ingest", nothing)
    C._progress["job"] = []
    asyncio.run(C._run_ingestion("job", tmp_path, "structure_aware", 512, 64, [], {"n_files": 2}, "c", {}))
    assert "FORCE_COLOR" not in seen and seen.get("NO_COLOR") == "1"
