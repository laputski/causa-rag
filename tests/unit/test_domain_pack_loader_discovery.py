"""core/domain/loader.py: directory-scan discovery."""
from __future__ import annotations

from pathlib import Path

from core.domain.loader import PackInfo, discover_packs, load_active_packs, load_pack
from core.registry import ComponentRegistry


def test_discover_packs_finds_the_packs_that_ship_with_the_platform() -> None:
    # The subject used to be a pack carrying one installation's own subject
    # area, which has moved to that installation's tree. What ships with the platform is the example
    # matching its own demo realm.
    ids = {p.id for p in discover_packs()}
    assert "manuals" in ids


def test_discover_packs_skips_dirs_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "no_manifest").mkdir()
    (tmp_path / "with_manifest").mkdir()
    (tmp_path / "with_manifest" / "pack.yaml").write_text("id: with_manifest\n", encoding="utf-8")

    packs = discover_packs(tmp_path)

    assert [p.id for p in packs] == ["with_manifest"]


def test_discover_packs_manifest_defaults(tmp_path: Path) -> None:
    pack_dir = tmp_path / "minimal"
    pack_dir.mkdir()
    (pack_dir / "pack.yaml").write_text("id: minimal\n", encoding="utf-8")

    packs = discover_packs(tmp_path)

    assert [p.id for p in packs] == ["minimal"]
    assert packs[0].path == tmp_path / "minimal"
    assert packs[0] == PackInfo(id="minimal", version="", display_name="minimal",
                                description="", exported_kinds=[], path=tmp_path / "minimal")


def test_discover_packs_invalid_manifest_is_skipped_not_raised(tmp_path: Path) -> None:
    pack_dir = tmp_path / "broken"
    pack_dir.mkdir()
    (pack_dir / "pack.yaml").write_text("id: [this is not valid: yaml: :::", encoding="utf-8")

    packs = discover_packs(tmp_path)

    assert packs == []


def test_load_pack_calls_register_with_registry_and_settings() -> None:
    calls = []

    class _FakeModule:
        @staticmethod
        def register(registry, settings):
            calls.append((registry, settings))

    import sys
    sys.modules["domain_packs._fake_test_pack"] = _FakeModule  # type: ignore[assignment]
    try:
        reg = ComponentRegistry()
        load_pack("_fake_test_pack", reg, {"k": "v"})
        assert calls == [(reg, {"k": "v"})]
    finally:
        del sys.modules["domain_packs._fake_test_pack"]


def test_load_pack_without_register_is_a_noop() -> None:
    class _FakeModuleNoRegister:
        pass

    import sys
    sys.modules["domain_packs._fake_noop_pack"] = _FakeModuleNoRegister  # type: ignore[assignment]
    try:
        reg = ComponentRegistry()
        load_pack("_fake_noop_pack", reg, {})  # must not raise
    finally:
        del sys.modules["domain_packs._fake_noop_pack"]


def test_load_active_packs_skips_broken_pack_without_crashing() -> None:
    reg = ComponentRegistry()
    loaded = load_active_packs(reg, {"active_packs": ["this_pack_does_not_exist"]})
    assert loaded == []


def test_load_active_packs_loads_a_real_pack() -> None:
    reg = ComponentRegistry()
    loaded = load_active_packs(reg, {"active_packs": ["manuals"]})
    assert loaded == ["manuals"]
    assert reg.resolve("route_policy", "manual_question_type") is not None
