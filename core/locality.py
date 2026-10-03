"""Whether an address is this machine or a neighbour of it.

Asked wherever the platform says where data goes: a store or a model server
that is not here receives what a run sends it, and a run records that.
"""
from __future__ import annotations

from urllib.parse import urlparse

#: Names that mean this machine or a container beside it.
_HERE = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})


def is_local(host: str) -> bool:
    """A name with no dot is a service on the same network, which is how a
    compose file names its own."""
    host = (host or "").strip("[]").lower()
    return host in _HERE or "." not in host or host.endswith((".localhost", ".local", ".internal"))


def address_of(url: str) -> str:
    """Scheme, host and port: an address without what a URL may carry
    besides, a path, a query or credentials."""
    parsed = urlparse(url)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname or ''}{port}"


def is_remote_url(url: str) -> bool:
    return not is_local(urlparse(url).hostname or "")
