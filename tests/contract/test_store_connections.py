"""A store that is not on this machine is reached with what the realm says.

Checked against servers started inside the test, which record what each
request carried. That is a contract check and no more: it shows the key and
the credentials leave with the request and survive a rebinding to another
corpus. Whether a hosted service accepts them is something only a live check
with real keys can say.
"""
from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from adapters.connections import Neo4jTarget, OpenSearchTarget, QdrantTarget, is_local


class _Recorder:
    """A server that answers what the adapters ask on construction and keeps
    the headers of every request."""

    def __init__(self, kind: str) -> None:
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                recorder.requests.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}))
                length = int(self.headers.get("content-length") or 0)
                if length:
                    self.rfile.read(length)
                if kind == "qdrant":
                    body: Any = {"result": {"exists": True, "collections": [], "count": 0},
                                 "status": "ok", "time": 0}
                elif self.command == "HEAD":
                    body = None
                else:
                    body = {"cluster_name": "fake", "acknowledged": True}
                payload = b"" if body is None else json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_HEAD = do_PUT = do_POST = _answer

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def headers(self, name: str) -> list[str]:
        return [h[name] for _, _, h in self.requests if name in h]

    def clear(self) -> None:
        self.requests.clear()


@pytest.fixture
def qdrant() -> Iterator[_Recorder]:
    recorder = _Recorder("qdrant")
    yield recorder
    recorder.server.shutdown()


@pytest.fixture
def opensearch() -> Iterator[_Recorder]:
    recorder = _Recorder("opensearch")
    yield recorder
    recorder.server.shutdown()


# ── What a target is built from ───────────────────────────────────────────────

def test_a_resource_states_its_own_fields_and_the_rest_come_from_the_fallback() -> None:
    base = QdrantTarget(host="first", port=1, api_key="k")
    target = QdrantTarget.of({"host": "second"}, base)
    assert (target.host, target.port, target.api_key) == ("second", 1, "k")


def test_a_field_left_blank_is_not_a_statement() -> None:
    """A form saved with a field empty must not replace a working host by
    nothing, nor a stored key by no key."""
    base = QdrantTarget(host="first", api_key="k")
    assert QdrantTarget.of({"host": "", "api_key": ""}, base) == base


def test_no_resource_at_all_is_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_HOST", "from-env")
    monkeypatch.setenv("QDRANT_API_KEY", "env-key")
    monkeypatch.setenv("OPENSEARCH_USE_SSL", "true")
    monkeypatch.setenv("NEO4J_DATABASE", "tenant")
    assert QdrantTarget.of(None).host == "from-env" and QdrantTarget.of(None).api_key == "env-key"
    assert OpenSearchTarget.of(None).use_ssl is True
    assert Neo4jTarget.of(None).database == "tenant"


def test_an_address_is_said_without_what_opens_it() -> None:
    assert QdrantTarget(host="h.example.org", port=6333, https=True, api_key="k").base_url == "https://h.example.org:6333"
    assert QdrantTarget(url="https://c.example.org:6333/", api_key="k").base_url == "https://c.example.org:6333"
    assert OpenSearchTarget(host="s.example.org", port=443, use_ssl=True, user="u", password="p").base_url \
        == "https://s.example.org:443"
    assert Neo4jTarget(uri="neo4j+s://user:secret@g.example.org:7687").base_url == "neo4j+s://g.example.org:7687"


@pytest.mark.parametrize("host,local", [
    ("localhost", True), ("127.0.0.1", True), ("qdrant", True), ("host.docker.internal", True),
    ("cluster.example.org", False), ("10.0.0.7", False)])
def test_a_store_on_this_machine_or_beside_it_is_not_remote(host: str, local: bool) -> None:
    assert is_local(host) is local
    assert QdrantTarget(host=host).is_remote is (not local)


# ── Qdrant ────────────────────────────────────────────────────────────────────

def test_the_key_leaves_with_every_request(qdrant: _Recorder) -> None:
    QdrantTarget(host="127.0.0.1", port=qdrant.port, api_key="the-key").client().get_collections()
    assert qdrant.headers("api-key") == ["the-key"]


def test_no_key_sends_no_key(qdrant: _Recorder) -> None:
    QdrantTarget(host="127.0.0.1", port=qdrant.port).client().get_collections()
    assert qdrant.requests and qdrant.headers("api-key") == []


def test_a_whole_address_is_used_as_given(qdrant: _Recorder) -> None:
    QdrantTarget(url=f"http://127.0.0.1:{qdrant.port}", api_key="k").client().get_collections()
    assert qdrant.headers("api-key") == ["k"]


def test_tls_is_asked_for_when_the_resource_says_so() -> None:
    client = QdrantTarget(host="h.example.org", port=6333, https=True).client()
    assert client._client.rest_uri.startswith("https://h.example.org:6333")


def test_the_retriever_reaches_its_store_with_the_key(qdrant: _Recorder) -> None:
    from adapters.qdrant import QdrantRetriever
    QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port, api_key="the-key")).count()
    assert qdrant.requests and set(qdrant.headers("api-key")) == {"the-key"}


def test_a_copy_for_another_corpus_keeps_the_key(qdrant: _Recorder) -> None:
    from adapters.qdrant import QdrantRetriever
    first = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port, api_key="the-key"))
    qdrant.clear()
    first.for_corpus("another", "a-realm").count()
    assert qdrant.requests and set(qdrant.headers("api-key")) == {"the-key"}


def test_a_copy_for_a_realm_s_own_store_uses_that_realm_s_key(qdrant: _Recorder) -> None:
    from adapters.qdrant import QdrantRetriever
    first = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port, api_key="gateway-key"))
    qdrant.clear()
    first.for_corpus("another", "a-realm", resources={"qdrant": {
        "host": "127.0.0.1", "port": qdrant.port, "api_key": "realm-key"}}).count()
    assert set(qdrant.headers("api-key")) == {"realm-key"}


# ── OpenSearch ────────────────────────────────────────────────────────────────

def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_the_credentials_leave_with_every_request(opensearch: _Recorder) -> None:
    OpenSearchTarget(host="127.0.0.1", port=opensearch.port, user="u", password="p").client().info()
    assert opensearch.headers("authorization") == [_basic("u", "p")]


def test_tls_and_its_verification_are_what_the_resource_says() -> None:
    kept = OpenSearchTarget(host="s.example.org", port=443, use_ssl=True).client()
    relaxed = OpenSearchTarget(host="s.example.org", port=443, use_ssl=True, verify_certs=False).client()
    connection = kept.transport.get_connection()
    assert connection.use_ssl is True and connection.pool.cert_reqs == "CERT_REQUIRED"
    assert relaxed.transport.get_connection().pool.cert_reqs == "CERT_NONE"


def test_the_lexical_retriever_reaches_its_store_with_the_credentials(opensearch: _Recorder) -> None:
    from adapters.opensearch import OpenSearchRetriever
    OpenSearchRetriever(target=OpenSearchTarget(host="127.0.0.1", port=opensearch.port, user="u", password="p"))
    assert opensearch.requests and set(opensearch.headers("authorization")) == {_basic("u", "p")}


def test_a_copy_of_the_lexical_retriever_keeps_the_credentials(opensearch: _Recorder) -> None:
    """It dropped them: the copy was built from a host and a port alone, so a
    run naming another corpus on a store that asks for a password was refused
    by it, and a store that asks for none hid the defect."""
    from adapters.opensearch import OpenSearchRetriever
    first = OpenSearchRetriever(host="127.0.0.1", port=opensearch.port, http_auth=("u", "p"))
    opensearch.clear()
    first.for_corpus("another", "a-realm")
    assert opensearch.requests and set(opensearch.headers("authorization")) == {_basic("u", "p")}


def test_a_copy_for_a_realm_s_own_lexical_store_uses_that_realm_s_credentials(opensearch: _Recorder) -> None:
    from adapters.opensearch import OpenSearchRetriever
    first = OpenSearchRetriever(host="127.0.0.1", port=opensearch.port)
    opensearch.clear()
    first.for_corpus("another", "a-realm", resources={"opensearch": {
        "host": "127.0.0.1", "port": opensearch.port, "user": "realm", "password": "secret"}})
    assert set(opensearch.headers("authorization")) == {_basic("realm", "secret")}


# ── Whose vectors a retriever reads ───────────────────────────────────────────

class _Embedder:
    embedder_id = "openai_compatible__some-model"
    vector_size = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 1.0] for _ in texts]


def test_a_retriever_bound_to_another_embedder_reads_that_embedder_s_collection(qdrant: _Recorder) -> None:
    """A run naming another embedder embedded its queries with it and went on
    searching the collection the gateway's own embedder had filled."""
    from adapters.qdrant import QdrantRetriever, _collection_name
    first = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port, api_key="k"),
                            strategy_id="fixed", corpus_id="c", realm_id="r")
    qdrant.clear()
    bound = first.for_embedder(_Embedder())
    wanted = _collection_name("fixed", "openai_compatible__some-model", "c", "r")
    assert bound is not first and bound._collection == wanted and bound._vector_size == 3
    bound.count()
    assert any(wanted in path for _, path, _ in qdrant.requests)
    assert set(qdrant.headers("api-key")) == {"k"}, "the copy reaches the same store the same way"


def test_a_retriever_already_reading_that_embedder_s_collection_is_itself(qdrant: _Recorder) -> None:
    from adapters.qdrant import QdrantRetriever
    first = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port),
                            embedder_id="openai_compatible__some-model", vector_size=3)
    assert first.for_embedder(_Embedder()) is first


def test_a_retriever_nobody_used_touches_no_store(qdrant: _Recorder) -> None:
    """Found live: a run naming another embedder left an empty collection
    behind for the gateway's own. Binding a retriever to a corpus and then to
    an embedder made a copy in between, and a retriever used to create its
    collection the moment it was built, so the copy nobody queried created
    one all the same."""
    from adapters.qdrant import QdrantRetriever
    first = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port))
    first.for_corpus("another", "a-realm").for_embedder(_Embedder())
    assert qdrant.requests == []


def test_a_retriever_makes_sure_of_its_collection_once_and_on_first_use(qdrant: _Recorder) -> None:
    from adapters.qdrant import QdrantRetriever
    retriever = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=qdrant.port))
    retriever.count()
    retriever.count()
    assert sum(1 for _, path, _ in qdrant.requests if path.endswith("/exists")) == 1
