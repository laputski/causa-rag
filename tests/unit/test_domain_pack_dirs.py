"""A pack can live outside this repository.

A pack carries one installation's subject-area vocabulary: the refusal wording
its readers expect, the parsing of its headings, its own error classes. Code
like that belongs in that installation's tree and not in a shared repository,
and before path-based loading it could not: `import_module("domain_packs.<id>")`
reaches only what sits inside this package.
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from core.domain import loader
from core.registry import ComponentRegistry


def _write_pack(root: Path, pack_id: str, body: str = "", manifest_id: str | None = None) -> Path:
    d = root / pack_id
    d.mkdir(parents=True)
    (d / "pack.yaml").write_text(
        f"id: {manifest_id or pack_id}\nversion: '1.0.0'\ndisplay_name: {pack_id}\n"
        "exported_kinds:\n  - scorer\n",
        encoding="utf-8",
    )
    (d / "__init__.py").write_text(textwrap.dedent(body or """
        def register(registry, settings):
            registry.register("scorer", "outside", object())
    """), encoding="utf-8")
    return d


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(loader._PACKS_DIR_ENV, raising=False)
    yield monkeypatch


# @lat: [[domain-packs#Каталог пакетов — не один]]
def test_a_pack_outside_the_repository_is_discovered(tmp_path, clean_env) -> None:
    _write_pack(tmp_path, "acme_support")
    clean_env.setenv(loader._PACKS_DIR_ENV, str(tmp_path))
    ids = {p.id for p in loader.discover_packs()}
    assert "acme_support" in ids
    # The built-in ones have not gone anywhere: an external directory adds to
    # them, and replaces nothing.
    assert "manuals" in ids


# @lat: [[domain-packs#Каталог пакетов — не один]]
def test_a_pack_outside_the_repository_loads_and_registers(tmp_path, clean_env) -> None:
    _write_pack(tmp_path, "acme_support")
    clean_env.setenv(loader._PACKS_DIR_ENV, str(tmp_path))
    reg = ComponentRegistry()
    loader.load_pack("acme_support", reg, {})
    assert reg.list_all()["scorer"] == ["outside"]
    sys.modules.pop("acme_support", None)


# @lat: [[domain-packs#Каталог пакетов — не один]]
def test_several_directories_are_read_in_order(tmp_path, clean_env) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _write_pack(a, "one")
    _write_pack(b, "two")
    clean_env.setenv(loader._PACKS_DIR_ENV, os.pathsep.join([str(a), str(b)]))
    ids = {p.id for p in loader.discover_packs()}
    assert {"one", "two"} <= ids


# @lat: [[domain-packs#Каталог пакетов — не один]]
def test_the_installation_s_pack_wins_a_name_collision(tmp_path, clean_env) -> None:
    # An installation that named its own pack `manuals` gets its own, and not
    # the example.
    # No warning: whoever wrote a pack chooses its name, and two installations
    # have no way to agree with each other.
    _write_pack(tmp_path, "manuals")
    clean_env.setenv(loader._PACKS_DIR_ENV, str(tmp_path))
    found = {p.id: p for p in loader.discover_packs()}
    assert found["manuals"].path == tmp_path / "manuals"


# @lat: [[domain-packs#Каталог пакетов — не один]]
def test_a_pack_named_nowhere_says_where_it_looked(clean_env) -> None:
    with pytest.raises(ModuleNotFoundError) as exc:
        loader.load_pack("no_such_pack", ComponentRegistry(), {})
    # The message lists the directories: "pack not found" without answering
    # "where did you look" helps nobody on somebody else's machine.
    assert "domain_packs" in str(exc.value)


# @lat: [[domain-packs#Каталог пакетов — не один]]
def test_an_explicit_directory_overrides_everything(tmp_path, clean_env) -> None:
    # A directory passed directly means "look here and nowhere else": tests and
    # tools bypass both the repository and the environment variable.
    _write_pack(tmp_path, "only_this")
    clean_env.setenv(loader._PACKS_DIR_ENV, "/nonexistent")
    assert [p.id for p in loader.discover_packs(tmp_path)] == ["only_this"]
