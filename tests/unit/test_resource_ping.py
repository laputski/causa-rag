"""A realm resource check must answer the question it was asked.

Written after `POST /realms/{id}/resources/test` returned `{"status": "ok"}` for
a Neo4j nobody had started: the ping called `Neo4jGraphRetriever.is_available()`,
which checks whether the `neo4j` package imports and never touches the network.

There were no tests for this, which is how the defect survived until somebody
checked by hand.
"""
import importlib.util

import pytest

from adapters.neo4j_graph import Neo4jGraphRetriever
from services.api_gateway.routers.realms import _ping_resource

# A port with certainly nothing on it. Taken from the dynamic range and matching
# no port in deploy/compose.
CLOSED_PORT = 7699

# Probing a closed port needs a driver to do the probing with. Without one the
# ping still answers with an error, which is what this file is about, but it is
# "driver not installed" and not "unreachable", and says nothing about the
# network. A fresh clone with the `dev` extra alone has no driver, and `pytest`
# there must be green.
_needs_neo4j = pytest.mark.skipif(
    importlib.util.find_spec("neo4j") is None,
    reason="needs the neo4j driver: pip install -e '.[graph]'",
)


# @lat: [[shell#Проверка ресурсов#is_available отвечает не на тот вопрос]]
@_needs_neo4j
def test_is_available_says_nothing_about_the_server():
    """The distinction the rewrite exists for: the package is always installed,
    the server is not. The assertion deliberately pins the inconvenient behaviour
    rather than fixing it, because `is_available()` has a legitimate consumer —
    choosing between the graph retriever and a stub — that needs no socket."""
    graph = Neo4jGraphRetriever(uri=f"bolt://localhost:{CLOSED_PORT}", password="x")
    assert graph.is_available() is True
    assert graph.verify() is False


# @lat: [[shell#Проверка ресурсов#Проверка на закрытом порту возвращает ошибку]]
@pytest.mark.asyncio
@_needs_neo4j
async def test_ping_neo4j_on_closed_port_reports_error():
    """The ping must answer with an error rather than "ok". This is the case the
    status screen exists for: the one way of telling a working installation from
    a broken one must not be capable of lying."""
    with pytest.raises(Exception) as exc:
        await _ping_resource("neo4j", {"uri": f"bolt://localhost:{CLOSED_PORT}", "password": "x"})
    assert "unreachable" in str(exc.value).lower()


# @lat: [[shell#Проверка ресурсов#Проверка укладывается в отведённое время]]
@pytest.mark.asyncio
async def test_ping_neo4j_is_bounded_in_time():
    """The driver's own timeout is thirty seconds, which turns "check everything"
    on an installation with one service down into half a minute of silence. The
    ping is bounded at five seconds, like the other three resource types."""
    import time
    started = time.monotonic()
    with pytest.raises(Exception):  # noqa: B017 — any error will do; the point is that it arrives in time
        await _ping_resource("neo4j", {"uri": f"bolt://localhost:{CLOSED_PORT}", "password": "x"})
    assert time.monotonic() - started < 10.0
