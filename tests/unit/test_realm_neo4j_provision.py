"""POST/GET /realms/{id}/neo4j/provision.

Self-service, UI-triggered equivalent of the manual two-step
`python3 -m tools.generate_realm_neo4j_compose` + `docker compose up -d`
process (the per-realm Neo4j notes). Mirrors the existing
_run_experiment_background convention (tests/unit/test_async_job_model.py)
— the background function is exercised directly with asyncio.run(), the
endpoint's own branching logic is exercised through the FastAPI route
functions with asyncio.create_task mocked out so no real subprocess/docker
call ever happens during the test.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from services.api_gateway.routers import realms as realms_module


class _FakeProc:
    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"", self._stderr)


@pytest.fixture(autouse=True)
def _clear_status():
    realms_module._neo4j_provision_status.clear()
    yield
    realms_module._neo4j_provision_status.clear()


# ── POST /realms/{id}/neo4j/provision — branching logic ─────────────────────

@pytest.mark.asyncio
async def test_provision_404_when_realm_not_found(monkeypatch):
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as exc:
        await realms_module.provision_realm_neo4j("nope")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_provision_not_needed_for_default_realm(monkeypatch):
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value={"id": "demo"}))
    monkeypatch.setattr(realms_module, "generate_neo4j_overlay", AsyncMock(
        return_value={"demo": {"status": "default", "uri": "bolt://localhost:7687"}},
    ))
    monkeypatch.setattr(realms_module.shutil, "which", lambda _: "/usr/bin/docker")

    result = await realms_module.provision_realm_neo4j("demo")

    assert result["status"] == "not_needed"


@pytest.mark.asyncio
async def test_provision_400_when_docker_cli_missing(monkeypatch):
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value={"id": "acme"}))
    monkeypatch.setattr(realms_module.shutil, "which", lambda _: None)

    with pytest.raises(HTTPException) as exc:
        await realms_module.provision_realm_neo4j("acme")
    assert exc.value.status_code == 400
    assert "docker" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_provision_starts_background_task_for_a_new_realm(monkeypatch):
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value={"id": "acme"}))
    monkeypatch.setattr(realms_module.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(realms_module, "generate_neo4j_overlay", AsyncMock(return_value={
        "acme": {
            "status": "provisioned", "uri": "bolt://localhost:7476",
            "service_name": "neo4j-acme", "http_port": 7475, "bolt_port": 7476,
        },
    }))

    with patch("services.api_gateway.routers.realms.asyncio.create_task") as mock_create_task:
        mock_create_task.side_effect = lambda coro: coro.close()  # never actually run it
        result = await realms_module.provision_realm_neo4j("acme")

    assert result["status"] == "running"
    assert result["service_name"] == "neo4j-acme"
    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_provision_returns_cached_running_status_without_relaunching(monkeypatch):
    """A second POST while a job is already in flight must not spawn a
    duplicate docker compose invocation — same idempotency guarantee the
    underlying script already provides."""
    realms_module._neo4j_provision_status["acme"] = {"status": "running", "service_name": "neo4j-acme"}
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value={"id": "acme"}))

    with patch("services.api_gateway.routers.realms.asyncio.create_task") as mock_create_task:
        result = await realms_module.provision_realm_neo4j("acme")

    assert result["status"] == "running"
    mock_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_provision_re_runs_when_already_migrated_but_container_unhealthy(monkeypatch):
    """resources[] can point at a neo4j URI from an earlier provision whose
    container was since removed (docker compose down / volume prune) — don't
    just trust the Mongo record, confirm the container is actually healthy
    before reporting done."""
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value={"id": "acme"}))
    monkeypatch.setattr(realms_module.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(realms_module, "generate_neo4j_overlay", AsyncMock(return_value={
        "acme": {"status": "already_migrated", "uri": "bolt://localhost:7476", "service_name": "neo4j-acme"},
    }))
    monkeypatch.setattr(realms_module, "_container_health", AsyncMock(return_value="unknown"))

    with patch("services.api_gateway.routers.realms.asyncio.create_task") as mock_create_task:
        mock_create_task.side_effect = lambda coro: coro.close()
        result = await realms_module.provision_realm_neo4j("acme")

    assert result["status"] == "running"
    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_provision_reports_done_when_already_migrated_and_healthy(monkeypatch):
    monkeypatch.setattr(realms_module.mdb, "find_one", AsyncMock(return_value={"id": "acme"}))
    monkeypatch.setattr(realms_module.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(realms_module, "generate_neo4j_overlay", AsyncMock(return_value={
        "acme": {"status": "already_migrated", "uri": "bolt://localhost:7476", "service_name": "neo4j-acme"},
    }))
    monkeypatch.setattr(realms_module, "_container_health", AsyncMock(return_value="healthy"))

    with patch("services.api_gateway.routers.realms.asyncio.create_task") as mock_create_task:
        result = await realms_module.provision_realm_neo4j("acme")

    assert result["status"] == "done"
    mock_create_task.assert_not_called()


# ── GET /realms/{id}/neo4j/provision ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_status_404_when_no_job_started():
    with pytest.raises(HTTPException) as exc:
        await realms_module.get_neo4j_provision_status("never-provisioned")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_status_returns_current_state():
    realms_module._neo4j_provision_status["acme"] = {"status": "done", "service_name": "neo4j-acme"}
    result = await realms_module.get_neo4j_provision_status("acme")
    assert result["status"] == "done"


# ── _run_neo4j_provision (background task body) ─────────────────────────────

@pytest.mark.asyncio
async def test_run_neo4j_provision_marks_done_when_container_becomes_healthy(monkeypatch):
    monkeypatch.setattr(
        realms_module.asyncio, "create_subprocess_exec",
        AsyncMock(return_value=_FakeProc(returncode=0)),
    )
    monkeypatch.setattr(realms_module, "_container_health", AsyncMock(return_value="healthy"))

    await realms_module._run_neo4j_provision("acme", "neo4j-acme", "bolt://localhost:7476")

    status = realms_module._neo4j_provision_status["acme"]
    assert status["status"] == "done"
    assert status["uri"] == "bolt://localhost:7476"


@pytest.mark.asyncio
async def test_run_neo4j_provision_marks_error_on_nonzero_compose_exit(monkeypatch):
    monkeypatch.setattr(
        realms_module.asyncio, "create_subprocess_exec",
        AsyncMock(return_value=_FakeProc(returncode=1, stderr=b"no such service")),
    )

    await realms_module._run_neo4j_provision("acme", "neo4j-acme", "bolt://localhost:7476")

    status = realms_module._neo4j_provision_status["acme"]
    assert status["status"] == "error"
    assert "no such service" in status["detail"]


@pytest.mark.asyncio
async def test_run_neo4j_provision_marks_error_when_container_reports_unhealthy(monkeypatch):
    monkeypatch.setattr(
        realms_module.asyncio, "create_subprocess_exec",
        AsyncMock(return_value=_FakeProc(returncode=0)),
    )
    monkeypatch.setattr(realms_module, "_container_health", AsyncMock(return_value="unhealthy"))

    await realms_module._run_neo4j_provision("acme", "neo4j-acme", "bolt://localhost:7476")

    status = realms_module._neo4j_provision_status["acme"]
    assert status["status"] == "error"


@pytest.mark.asyncio
async def test_run_neo4j_provision_times_out_if_never_healthy(monkeypatch):
    monkeypatch.setattr(
        realms_module.asyncio, "create_subprocess_exec",
        AsyncMock(return_value=_FakeProc(returncode=0)),
    )
    monkeypatch.setattr(realms_module, "_container_health", AsyncMock(return_value="starting"))
    monkeypatch.setattr(realms_module.asyncio, "sleep", AsyncMock(return_value=None))

    await realms_module._run_neo4j_provision("acme", "neo4j-acme", "bolt://localhost:7476")

    status = realms_module._neo4j_provision_status["acme"]
    assert status["status"] == "error"
    assert "60s" in status["detail"]
