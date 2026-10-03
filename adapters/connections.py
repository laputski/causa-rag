"""Where a store is and how it is reached, read in one place.

Each store's address used to be read wherever a client was built: the adapter,
its rebinding to another corpus, the realm's connection test, three clients in
the corpus router, the ingestion command and the health check. Every one of
them read a host and a port and nothing else, so a store behind TLS with a key
could be reached from none of them, and a field a realm declared was dropped
on the way to whichever of them forgot it. The rebinding of the lexical index
forgot the credentials it had been constructed with.

A target is built from a realm's resource (its secrets already opened) or,
with none, from the environment; it builds the client, and it says where the
data goes in words that carry no secret.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse

from core.locality import is_local

__all__ = ["Neo4jTarget", "OpenSearchTarget", "QdrantTarget", "is_local"]


def _flag(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _given(resource: Mapping[str, Any] | None, name: str, fallback: Any) -> Any:
    """A field the resource states, else the fallback. An empty string is not
    a statement: a form submitted with a field left blank must not replace a
    working host by nothing."""
    value = (resource or {}).get(name)
    return fallback if value is None or value == "" else value


@dataclass(frozen=True)
class QdrantTarget:
    host: str = "localhost"
    port: int = 6333
    https: bool = False
    api_key: str | None = None
    #: A whole address, as a hosted cluster gives it. Wins over host and port.
    url: str | None = None

    @classmethod
    def from_environment(cls) -> QdrantTarget:
        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", "6333")),
            https=_flag(os.getenv("QDRANT_HTTPS"), False),
            api_key=os.getenv("QDRANT_API_KEY") or None, url=os.getenv("QDRANT_URL") or None)

    @classmethod
    def of(cls, resource: Mapping[str, Any] | None, fallback: QdrantTarget | None = None) -> QdrantTarget:
        """The target a resource describes; what it leaves unsaid comes from
        `fallback`, and with no resource at all, from the environment."""
        base = fallback or cls.from_environment()
        if not resource:
            return base
        return replace(
            base,
            host=str(_given(resource, "host", base.host)), port=int(_given(resource, "port", base.port)),
            https=_flag(resource.get("https"), base.https),
            api_key=_given(resource, "api_key", base.api_key), url=_given(resource, "url", base.url))

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url.rstrip("/")
        return f"{'https' if self.https else 'http'}://{self.host}:{self.port}"

    @property
    def headers(self) -> dict[str, str]:
        return {"api-key": self.api_key} if self.api_key else {}

    @property
    def is_remote(self) -> bool:
        return not is_local(urlparse(self.base_url).hostname or self.host)

    def client(self, timeout: float = 60) -> Any:
        from qdrant_client import QdrantClient
        if self.url:
            return QdrantClient(url=self.url, api_key=self.api_key, timeout=int(timeout),
                                check_compatibility=False)
        return QdrantClient(host=self.host, port=self.port, https=self.https, api_key=self.api_key,
                            timeout=int(timeout), check_compatibility=False)


@dataclass(frozen=True)
class OpenSearchTarget:
    host: str = "localhost"
    port: int = 9200
    use_ssl: bool = False
    verify_certs: bool = True
    user: str | None = None
    password: str | None = None

    @classmethod
    def from_environment(cls) -> OpenSearchTarget:
        return cls(
            host=os.getenv("OPENSEARCH_HOST", "localhost"), port=int(os.getenv("OPENSEARCH_PORT", "9200")),
            use_ssl=_flag(os.getenv("OPENSEARCH_USE_SSL"), False),
            verify_certs=_flag(os.getenv("OPENSEARCH_VERIFY_CERTS"), True),
            user=os.getenv("OPENSEARCH_USER") or None, password=os.getenv("OPENSEARCH_PASSWORD") or None)

    @classmethod
    def of(cls, resource: Mapping[str, Any] | None,
           fallback: OpenSearchTarget | None = None) -> OpenSearchTarget:
        base = fallback or cls.from_environment()
        if not resource:
            return base
        return replace(
            base,
            host=str(_given(resource, "host", base.host)), port=int(_given(resource, "port", base.port)),
            use_ssl=_flag(resource.get("use_ssl"), base.use_ssl),
            verify_certs=_flag(resource.get("verify_certs"), base.verify_certs),
            user=_given(resource, "user", base.user), password=_given(resource, "password", base.password))

    @property
    def base_url(self) -> str:
        return f"{'https' if self.use_ssl else 'http'}://{self.host}:{self.port}"

    @property
    def auth(self) -> tuple[str, str] | None:
        return (self.user, self.password or "") if self.user else None

    @property
    def is_remote(self) -> bool:
        return not is_local(self.host)

    def client(self, timeout: float | None = None) -> Any:
        from opensearchpy import OpenSearch
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        return OpenSearch(
            hosts=[{"host": self.host, "port": self.port}], http_auth=self.auth,
            use_ssl=self.use_ssl, verify_certs=self.verify_certs if self.use_ssl else False,
            ssl_show_warn=False, **extra)


@dataclass(frozen=True)
class Neo4jTarget:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "ragplatform"
    #: The database inside the server. None is the server's own default, which
    #: is the only one a community edition has and not the one a hosted
    #: instance necessarily gives its tenant.
    database: str | None = None

    @classmethod
    def from_environment(cls) -> Neo4jTarget:
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"), user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "ragplatform"),
            database=os.getenv("NEO4J_DATABASE") or None)

    @classmethod
    def of(cls, resource: Mapping[str, Any] | None, fallback: Neo4jTarget | None = None) -> Neo4jTarget:
        base = fallback or cls.from_environment()
        if not resource:
            return base
        return replace(
            base, uri=str(_given(resource, "uri", base.uri)), user=str(_given(resource, "user", base.user)),
            password=str(_given(resource, "password", base.password)),
            database=_given(resource, "database", base.database))

    @property
    def base_url(self) -> str:
        """The address without the credentials a URI may carry inside it."""
        parsed = urlparse(self.uri)
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.hostname or ''}{port}"

    @property
    def is_remote(self) -> bool:
        return not is_local(urlparse(self.uri).hostname or "")
