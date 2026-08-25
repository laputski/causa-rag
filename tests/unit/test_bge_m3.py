from adapters.bge_m3 import _VECTOR_DIM, BgeM3Embedder
from core.interfaces import Embedder


def test_satisfies_protocol():
    assert isinstance(BgeM3Embedder(), Embedder)


def test_embed_returns_correct_shape():
    emb = BgeM3Embedder()
    vecs = emb.embed(["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == _VECTOR_DIM for v in vecs)


def test_embed_deterministic():
    emb = BgeM3Embedder()
    v1 = emb.embed(["test"])[0]
    v2 = emb.embed(["test"])[0]
    assert v1 == v2


def test_embed_different_texts_differ():
    emb = BgeM3Embedder()
    v1 = emb.embed(["foo"])[0]
    v2 = emb.embed(["bar"])[0]
    assert v1 != v2


def test_cache_hit():
    emb = BgeM3Embedder()
    emb.embed(["cached text"])
    assert emb.cache_size == 1
    emb.embed(["cached text"])
    assert emb.cache_size == 1  # no duplicate


def test_cache_grows_for_new_text():
    emb = BgeM3Embedder()
    emb.embed(["a"])
    emb.embed(["b"])
    assert emb.cache_size == 2


def test_embedder_ids():
    emb = BgeM3Embedder()
    assert emb.embedder_id == "bge_m3"
    assert emb.version == "1.0.0"
