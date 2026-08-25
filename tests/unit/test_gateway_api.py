"""gateway API contract tests (no running server needed — imports only)."""
from services.api_gateway.routers.datasets import router as ds_router
from services.api_gateway.routers.experiments import router as exp_router


def test_experiments_router_prefix():
    assert exp_router.prefix == "/experiments"


def test_datasets_router_prefix():
    assert ds_router.prefix == "/datasets"


def test_experiments_routes_exist():
    paths = {r.path for r in exp_router.routes}
    assert any(p.endswith("/experiments") or p == "/experiments" for p in paths)
    assert any("{run_id}" in p for p in paths)
    assert any("compare" in p for p in paths)


def test_datasets_routes_exist():
    paths = {r.path for r in ds_router.routes}
    assert any(p.endswith("/datasets") or p == "/datasets" for p in paths)
    assert any("{filename}" in p for p in paths)


def test_panels_env_override(monkeypatch):
    monkeypatch.setenv("LANGFUSE_EXTERNAL_HOST", "http://langfuse.example.com:3000")
    # Re-import to pick up env (simulate fresh start)
    import importlib

    import services.api_gateway.main as gw
    importlib.reload(gw)
    assert gw._PANELS["langfuse"] == "http://langfuse.example.com:3000"
