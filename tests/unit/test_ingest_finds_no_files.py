"""A load that matched no files is a mistake, and it used to be a traceback.

Found while staging the catalogue's entry for an index holding nothing: the
failure could not be reached through the platform's own loader, because the
loader died before the index could exist. The branch for "no files here"
returned three keys, one of them under a name nothing else uses, and the line
printing the cache hit ratio raised `KeyError: 'hit_ratio'`.

Two things are checked, and the second is the one the catalogue cares about.
The result of an empty load has the shape every other load has, so a caller
reading any of its keys gets a number and not an exception. And the command
says plainly that nothing was indexed and refuses, because the quiet version
of this is exactly the entry being staged: an empty index answers everything
with nothing, every retrieval metric comes back at zero, and zero is also the
shape a catastrophically bad retriever produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from services.ingestion.cli import ingest, main


def test_an_empty_load_returns_the_shape_every_load_returns(tmp_path: Path) -> None:
    result = ingest(source=tmp_path)
    assert result["files"] == 0
    assert result["chunks"] == 0
    # Every key the printing and the realm registration read, named one by one
    # and never compared against a constant, so adding a key to the real path
    # and not to this one stays a change somebody has to make on purpose.
    for key in ("hit_ratio", "hits", "misses", "size", "manifest",
                "qdrant_collection", "opensearch_index"):
        assert key in result, f"an empty load returns no {key!r}, so a caller reading it raises"
    assert f"{result['hit_ratio']:.2%}" == "0.00%", "the ratio is not a number the caller can format"


def test_the_command_says_nothing_was_indexed_and_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["cli.py", "ingest", str(tmp_path)]
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exited:
            main()
    assert exited.value.code == 1, "an empty load reports success, so a script would carry on"
    said = capsys.readouterr().out
    assert "No documents found" in said
    assert str(tmp_path) in said, "the message does not say which path matched nothing"
