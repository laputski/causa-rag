"""A graph store's own database is named on every session, and what the
server lacks is said before an operation needs it.

A hosted graph database gives a tenant one named database and not always the
server's default, and may come without the two procedure libraries this
platform's graph path calls. Checked against a driver that stands in for the
real one and records what it is asked, which is a contract check: it shows
what leaves, and not what a hosted service answers.
"""
from __future__ import annotations

from typing import Any

import pytest

from adapters.connections import Neo4jTarget


class _Result(list):
    """What a query returns: rows to read, and a summary to wait for."""

    def consume(self) -> None:
        return None

    def single(self) -> Any:
        return self[0] if self else None


class _Session:
    def __init__(self, driver: _Driver, database: str | None) -> None:
        self._driver = driver
        driver.sessions.append(database)

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def run(self, query: str, **params: Any) -> Any:
        self._driver.queries.append(query)
        if query.startswith("SHOW PROCEDURES"):
            return _Result({"name": name} for name in self._driver.procedures)
        return _Result()


class _Driver:
    def __init__(self, procedures: list[str]) -> None:
        self.procedures = procedures
        self.sessions: list[str | None] = []
        self.queries: list[str] = []

    def verify_connectivity(self) -> None:
        return None

    def session(self, **kwargs: Any) -> _Session:
        return _Session(self, kwargs.get("database"))

    def close(self) -> None:
        return None


@pytest.fixture
def driver(monkeypatch) -> _Driver:
    import neo4j

    fake = _Driver(["apoc.periodic.iterate", "gds.graph.project", "gds.leiden.write"])
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda uri, **kwargs: fake)
    return fake


def _retriever(**target: Any) -> Any:
    from adapters.neo4j_graph import Neo4jGraphRetriever
    return Neo4jGraphRetriever(target=Neo4jTarget(uri="bolt://graph.example.org:7687", **target))


def test_the_named_database_is_asked_for_on_every_session(driver: _Driver) -> None:
    graph = _retriever(database="tenant")
    graph.missing_extensions()
    graph.clear()
    assert driver.sessions and set(driver.sessions) == {"tenant"}


def test_no_named_database_leaves_the_choice_to_the_server(driver: _Driver) -> None:
    _retriever().clear()
    assert driver.sessions and set(driver.sessions) == {None}


def test_a_realm_s_own_database_on_the_same_server_is_another_store(driver: _Driver) -> None:
    """Keyed by the address alone, two realms with a database each on one
    server would have been handed one retriever, and one realm's graph."""
    first = _retriever(database="one")
    same = first.for_corpus("c", "r", resources={"neo4j": {"uri": "bolt://graph.example.org:7687", "database": "one"}})
    other = first.for_corpus("c", "r", resources={"neo4j": {"uri": "bolt://graph.example.org:7687", "database": "two"}})
    assert same is first and other is not first
    other.clear()
    assert driver.sessions[-1] == "two"


def test_a_server_with_both_libraries_lacks_nothing(driver: _Driver) -> None:
    assert _retriever().missing_extensions() == {}


def test_a_server_without_the_libraries_says_which_operations_are_gone(driver: _Driver) -> None:
    driver.procedures = ["db.labels"]
    missing = _retriever().missing_extensions()
    assert set(missing) == {"apoc", "gds"}
    assert all(operations for operations in missing.values())


def test_one_library_absent_names_only_its_own_operations(driver: _Driver) -> None:
    driver.procedures = ["apoc.periodic.iterate"]
    assert set(_retriever().missing_extensions()) == {"gds"}
