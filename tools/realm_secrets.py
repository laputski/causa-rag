"""Seal the secrets stored before sealing existed, or reseal them under a new key.

Dry by default: prints what it would change and changes nothing. `--write`
changes it, then reads every changed document back and opens each secret it
sealed, so a write that stored something unreadable is reported instead of
discovered at the next connection.

    python3 -m tools.realm_secrets seal                    # what is still in the clear
    python3 -m tools.realm_secrets seal --write
    python3 -m tools.realm_secrets rotate --old-key K       # reseal under CAUSA_SECRET_KEY
    python3 -m tools.realm_secrets rotate --old-key K --write

Covers a realm's resources (the fields its connector type declares secret, and
any field whose name reads as one) and a registered RAG's headers.
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any

from core.secrets import SEALED, SecretBox, is_sealed


def _reseal(value: Any, new: SecretBox, old: SecretBox | None) -> tuple[Any, str | None]:
    """The value as it should be stored, and the plain secret when it changed."""
    if value in (None, ""):
        return value, None
    if is_sealed(value):
        if old is None:
            return value, None
        plain = old.open(value[SEALED])
        return {SEALED: new.seal(plain)}, plain
    plain = str(value)
    return {SEALED: new.seal(plain)}, plain


def resealed_realm(realm: dict[str, Any], new: SecretBox, old: SecretBox | None,
                   secret_fields: Any) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """The realm with its secrets sealed under `new`, and what changed as
    (path, plain) pairs for the read-back."""
    changed: list[tuple[str, str]] = []
    resources = []
    for resource in realm.get("resources") or []:
        copy = dict(resource)
        for name in sorted(secret_fields(resource.get("type", ""), resource)):
            copy[name], plain = _reseal(copy.get(name), new, old)
            if plain is not None:
                changed.append((f"resources.{resource.get('type')}.{name}", plain))
        resources.append(copy)
    return {**realm, "resources": resources}, changed


def resealed_rag(rag: dict[str, Any], new: SecretBox, old: SecretBox | None,
                 ) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    changed: list[tuple[str, str]] = []
    headers = {}
    for name, value in (rag.get("headers") or {}).items():
        headers[name], plain = _reseal(value, new, old)
        if plain is not None:
            changed.append((f"headers.{name}", plain))
    return {**rag, "headers": headers}, changed


async def run(write: bool, old_key: str | None) -> int:
    import adapters.mongodb as mdb
    from adapters.fernet_secrets import FernetSecretBox
    from services.api_gateway.routers.realms import _secret_fields

    new = FernetSecretBox()
    if not new.available:
        print(f"cannot seal: {new.reason}")
        return 2
    old = FernetSecretBox(key=old_key) if old_key else None
    if old is not None and not old.available:
        print(f"cannot open under the old key: {old.reason}")
        return 2

    plans: list[tuple[str, dict[str, Any], list[tuple[str, str]]]] = []
    for realm in await mdb.find_many("realms"):
        updated, changed = resealed_realm(realm, new, old, _secret_fields)
        if changed:
            plans.append(("realms", updated, changed))
    for rag in await mdb.find_many("external_rags"):
        updated, changed = resealed_rag(rag, new, old)
        if changed:
            plans.append(("external_rags", updated, changed))

    for collection, doc, changed in plans:
        for path, _plain in changed:
            print(f"{'sealing' if write else 'would seal'} {collection}/{doc.get('id')}: {path}")
    if not plans:
        print("nothing to seal")
        return 0
    if not write:
        print("dry run: nothing written; pass --write")
        return 0

    unreadable = 0
    for collection, doc, changed in plans:
        field = "resources" if collection == "realms" else "headers"
        await mdb.update_one(collection, {"id": doc["id"]}, {"$set": {field: doc[field]}})
        stored = await mdb.find_one(collection, {"id": doc["id"]}) or {}
        for path, plain in changed:
            if _read_back(stored, path, new) != plain:
                unreadable += 1
                print(f"READ-BACK FAILED {collection}/{doc['id']}: {path}")
    print(f"sealed {sum(len(c) for _, _, c in plans)} secret(s); read-back failures: {unreadable}")
    return 1 if unreadable else 0


def _read_back(stored: dict[str, Any], path: str, box: SecretBox) -> str | None:
    parts = path.split(".")
    if parts[0] == "resources":
        resource = next((r for r in stored.get("resources") or [] if r.get("type") == parts[1]), {})
        value = resource.get(parts[2])
    else:
        value = (stored.get("headers") or {}).get(parts[1])
    return box.open(value[SEALED]) if is_sealed(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["seal", "rotate"])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--old-key", default=None, help="the key the stored secrets were sealed with")
    args = parser.parse_args()
    if args.action == "rotate" and not args.old_key:
        parser.error("rotate needs --old-key")
    return asyncio.run(run(args.write, args.old_key if args.action == "rotate" else None))


if __name__ == "__main__":
    raise SystemExit(main())
