"""Domain pack discovery and loading.

Discovery is directory-scan only: every immediate subdirectory of a pack
directory that holds a pack.yaml manifest is a discoverable pack. Python
entry-points (importlib.metadata) are deferred to a future packaging-split
stage.

There is more than one pack directory. The one in this repository holds the
examples that ship with the platform; `CAUSA_DOMAIN_PACKS` names any others,
separated the way the operating system separates paths. A pack belonging to
one installation — its own subject area, its own vocabulary, its own
customer — lives in that installation's tree and never enters this
repository. That was not possible while loading went through
`import_module("domain_packs.<id>")`, which can only reach what sits inside
this package; packs are loaded from their manifest's own directory now.

Pure filesystem + dynamic import — no network calls anywhere in this module
(air-gap). Listing discovered packs never imports their Python
code (manifest-only); only load_pack()/load_active_packs() import and call
register() for packs explicitly named (e.g. from settings.active_packs).

This module itself contains zero references to any concrete pack — it knows
nothing about any specific domain, only the generic mechanism.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import structlog
import yaml

from core.registry import ComponentRegistry

log = structlog.get_logger()

_BUILTIN_PACKS_DIR = Path(__file__).parents[2] / "domain_packs"
_PACKS_DIR_ENV = "CAUSA_DOMAIN_PACKS"


def pack_dirs(extra: Path | None = None) -> list[Path]:
    """Where packs are looked for: this repository's, then the installation's.

    Order is the answer to a name collision: an installation whose pack has
    the same id as one shipped here gets its own, not the example. Nothing
    warns about the collision — a pack id is a name chosen by whoever wrote
    it, and two installations have no way to coordinate.
    """
    dirs = [extra] if extra is not None else [_BUILTIN_PACKS_DIR]
    if extra is None:
        for raw in os.environ.get(_PACKS_DIR_ENV, "").split(os.pathsep):
            if raw.strip():
                dirs.append(Path(raw.strip()).expanduser())
    return dirs


@dataclass
class PackInfo:
    id: str
    version: str = ""
    display_name: str = ""
    description: str = ""
    exported_kinds: list[str] = field(default_factory=list)
    # The pack's own directory: loading happens from here, and not by a name
    # inside `domain_packs`, which would require every pack to live in this
    # repository.
    path: Path | None = None


def discover_packs(domain_packs_dir: Path | None = None) -> list[PackInfo]:
    """List packs discoverable under domain_packs/ — manifest-only, never
    imports the pack's Python module. A subdirectory without pack.yaml is
    not a pack and is silently skipped.
    """
    packs: dict[str, PackInfo] = {}
    for root in pack_dirs(domain_packs_dir):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("__"):
                continue
            manifest_path = entry / "pack.yaml"
            if not manifest_path.is_file():
                continue
            try:
                raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                log.warning("domain_pack.manifest_invalid", pack_dir=str(entry), error=str(exc))
                continue
            pack_id = raw.get("id") or entry.name
            packs[pack_id] = PackInfo(
                id=pack_id,
                version=str(raw.get("version", "")),
                display_name=raw.get("display_name", pack_id),
                description=raw.get("description", ""),
                exported_kinds=list(raw.get("exported_kinds", [])),
                path=entry,
            )
    return sorted(packs.values(), key=lambda p: p.id)


def _import_pack_module(pack_id: str) -> Any:
    """Import a pack by finding it, rather than by assuming where it lives.

    A pack inside this repository is imported as `domain_packs.<id>`, so its
    own `from domain_packs.<id>.x import y` keeps working. One outside is
    loaded from its directory under the same module name — and its directory's
    parent goes on `sys.path` first, so a pack of several modules can import
    its own siblings exactly as an in-repo one does.
    """
    for root in pack_dirs():
        entry = root / pack_id
        if not (entry / "pack.yaml").is_file():
            continue
        if root == _BUILTIN_PACKS_DIR:
            return import_module(f"domain_packs.{pack_id}")
        parent = str(root.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return import_module(pack_id)
    # No directory holds a manifest, and yet the module may already be
    # importable: installed as a Python package, or put in place by a test.
    # Failing here would mean a "pack" has to be a directory, and the loader
    # has no use for that restriction.
    try:
        return import_module(f"domain_packs.{pack_id}")
    except ModuleNotFoundError:
        pass
    raise ModuleNotFoundError(
        f"Domain pack {pack_id!r} found in none of: "
        + ", ".join(str(d) for d in pack_dirs())
    )


def load_pack(pack_id: str, registry: ComponentRegistry, settings: dict[str, Any]) -> None:
    """Find the pack, import it and call its register(registry, settings) if
    defined. A pack registering nothing (no register(), or a no-op one)
    is valid: the minimal required set is empty.
    """
    module = _import_pack_module(pack_id)
    register_fn = getattr(module, "register", None)
    if register_fn is None:
        log.info("domain_pack.no_register", pack_id=pack_id)
        return
    register_fn(registry, settings)
    log.info("domain_pack.registered", pack_id=pack_id)


# What each pack registered: {pack_id: {kind: [component_id, ...]}}.
#
# There is one registry per process while pack activation is per Realm, and the
# registry does not remember which pack a component came from. That is why the
# new-run form on one Realm offered the internals of a pack a different Realm
# had enabled. Computed as the difference in the registry before and after the
# load, because a pack reports nothing about itself.
PACK_COMPONENTS: dict[str, dict[str, list[str]]] = {}


def load_active_packs(registry: ComponentRegistry, settings: dict[str, Any]) -> list[str]:
    """Load every pack named in settings["active_packs"]. A pack that fails
    to import/register is logged and skipped — one broken pack must not take
    down gateway startup.
    """
    loaded: list[str] = []
    for pack_id in settings.get("active_packs", []):
        before = {kind: set(ids) for kind, ids in registry.list_all().items()}
        try:
            load_pack(pack_id, registry, settings)
            loaded.append(pack_id)
        except Exception as exc:
            log.warning("domain_pack.load_failed", pack_id=pack_id, error=str(exc))
            continue
        added: dict[str, list[str]] = {}
        for kind, ids in registry.list_all().items():
            new_ids = [i for i in ids if i not in before.get(kind, set())]
            if new_ids:
                added[kind] = new_ids
        PACK_COMPONENTS[pack_id] = added
    return loaded


def components_outside_packs(
    registry: ComponentRegistry, allowed_packs: list[str],
) -> dict[str, list[str]]:
    """The registry without other packs' internals.

    What remains is everything no pack contributed, plus what the permitted ones
    did. A pack absent from `PACK_COMPONENTS` was never loaded, so there is
    nothing to subtract.
    """
    hidden: set[tuple[str, str]] = set()
    for pack_id, kinds in PACK_COMPONENTS.items():
        if pack_id in allowed_packs:
            continue
        for kind, ids in kinds.items():
            hidden.update((kind, i) for i in ids)
    return {
        kind: [i for i in ids if (kind, i) not in hidden]
        for kind, ids in registry.list_all().items()
    }
