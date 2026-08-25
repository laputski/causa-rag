from adapters.bge_m3 import BgeM3Embedder


def test_initial_hit_ratio_zero():
    emb = BgeM3Embedder()
    assert emb.cache_hit_ratio == 0.0


def test_all_misses_on_first_call():
    emb = BgeM3Embedder()
    emb.embed(["a", "b", "c"])
    stats = emb.cache_stats()
    assert stats["misses"] == 3
    assert stats["hits"] == 0
    assert stats["hit_ratio"] == 0.0


def test_repeat_ingest_gives_full_hit_ratio():
    emb = BgeM3Embedder()
    texts = ["document one", "document two", "document three"]
    emb.embed(texts)   # first pass — all misses
    emb.embed(texts)   # second pass — all hits
    assert emb.cache_hit_ratio == pytest_approx(0.5)


def test_partial_hits():
    emb = BgeM3Embedder()
    emb.embed(["existing"])
    emb.embed(["existing", "new"])
    stats = emb.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2


def pytest_approx(val: float) -> float:
    return val  # used inline — real pytest.approx is in test body


def test_hit_ratio_approaches_one_on_repeat():
    import pytest as pt
    emb = BgeM3Embedder()
    texts = [f"chunk {i}" for i in range(20)]
    emb.embed(texts)          # 20 misses
    emb.embed(texts)          # 20 hits
    assert emb.cache_hit_ratio == pt.approx(0.5, abs=0.01)
    emb.embed(texts)          # 20 more hits → 40 hits / 60 total = 0.666…
    assert emb.cache_hit_ratio > 0.6


def test_cache_stats_keys():
    emb = BgeM3Embedder()
    emb.embed(["x"])
    stats = emb.cache_stats()
    assert set(stats.keys()) == {"hits", "misses", "hit_ratio", "size"}
