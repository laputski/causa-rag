"""Secrets a realm's resources carry, sealed where they are stored.

A secret is a value that grants access to a service: a password, an API key, a
header that authorises. It is stored sealed, travels out of the API as a mask
and never travels out of the platform in an export, and it is opened only where
a connection is made.

What makes a field secret is the connector type's schema declaring it so, and
not a pattern over its name: a name says nothing about what its value opens.

The mask doubles as "keep what is stored": the interface edits the list it
read, and what it read is the mask, so writing the mask back must not replace
the secret with dots. An empty string clears the secret; an absent field keeps
it.

A value stored before sealing existed is a plain string and is read as it is,
so an installation keeps working until its secrets are sealed.
"""
from __future__ import annotations

from typing import Any, Protocol

#: What a secret looks like on the way out of the API.
MASK = "••••••••"

#: A sealed value is stored as `{SEALED: token}`, so a plain string that
#: happens to look like a token is never mistaken for one.
SEALED = "sealed"


class SecretUnavailable(Exception):
    """A secret was needed and could not be had: no key to seal or open it,
    or a value sealed with another key."""


class SecretBox(Protocol):
    #: Whether a key is present, so a caller can say so before trying.
    available: bool

    def seal(self, plain: str) -> str: ...

    def open(self, token: str) -> str: ...


def is_sealed(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get(SEALED), str)


def for_storage(incoming: dict[str, Any], stored: dict[str, Any] | None,
                secret_fields: set[str], box: SecretBox) -> dict[str, Any]:
    """The resource as it should be stored, given what was sent and what was.

    Raises SecretUnavailable when a new secret arrives and there is no key to
    seal it with: storing it in the clear instead would be the failure this
    module exists to prevent, done quietly.
    """
    out = dict(incoming)
    for name in secret_fields:
        sent = incoming.get(name)
        kept = (stored or {}).get(name)
        if sent is None or sent == MASK:
            if kept in (None, ""):
                out.pop(name, None)
            else:
                out[name] = kept
        elif sent == "":
            out.pop(name, None)
        else:
            out[name] = {SEALED: box.seal(str(sent))}
    return out


def masked(resource: dict[str, Any], secret_fields: set[str]) -> dict[str, Any]:
    """The resource as the API shows it: each secret that is set is the mask."""
    out = dict(resource)
    for name in secret_fields:
        if out.get(name) not in (None, ""):
            out[name] = MASK
    return out


def opened(resource: dict[str, Any], secret_fields: set[str], box: SecretBox) -> dict[str, Any]:
    """The resource as a connection needs it: each sealed secret opened.

    Raises SecretUnavailable when a sealed secret cannot be opened. An empty
    field would be quietly replaced by a default further down (the graph
    adapter falls back to its own), and a connection made with somebody
    else's default is worse than one that fails and says why.
    """
    out = dict(resource)
    for name in secret_fields:
        value = out.get(name)
        if isinstance(value, dict) and is_sealed(value):
            out[name] = box.open(str(value[SEALED]))
    return out
