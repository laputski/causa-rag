import pytest

from core.registry import ComponentRegistry


def test_register_and_resolve():
    reg = ComponentRegistry()
    obj = object()
    reg.register("embedder", "bge_m3", obj)
    assert reg.resolve("embedder", "bge_m3") is obj


def test_resolve_missing_raises():
    reg = ComponentRegistry()
    with pytest.raises(KeyError):
        reg.resolve("embedder", "missing")


def test_list_kind():
    reg = ComponentRegistry()
    reg.register("chunker", "fixed", object())
    reg.register("chunker", "structure", object())
    assert set(reg.list_kind("chunker")) == {"fixed", "structure"}


def test_list_all():
    reg = ComponentRegistry()
    reg.register("embedder", "bge_m3", object())
    reg.register("retriever", "qdrant", object())
    all_ = reg.list_all()
    assert "embedder" in all_
    assert "retriever" in all_


def test_overwrite_registration():
    reg = ComponentRegistry()
    a, b = object(), object()
    reg.register("gen", "vllm", a)
    reg.register("gen", "vllm", b)
    assert reg.resolve("gen", "vllm") is b
