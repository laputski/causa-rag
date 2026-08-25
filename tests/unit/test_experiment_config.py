from core.experiment.config import ComponentRef, ExperimentConfig


def _base_cfg(**overrides: object) -> ExperimentConfig:
    defaults: dict = dict(
        name="test",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        retrievers=[ComponentRef(kind="retriever", component_id="qdrant_dense")],
        generator=ComponentRef(kind="generator", component_id="stub"),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_config_hash_is_computed():
    cfg = _base_cfg()
    assert cfg.config_hash
    assert len(cfg.config_hash) == 16


def test_same_config_same_hash():
    a = _base_cfg()
    b = _base_cfg()
    assert a.config_hash == b.config_hash


def test_different_config_different_hash():
    a = _base_cfg(top_k=5)
    b = _base_cfg(top_k=10)
    assert a.config_hash != b.config_hash


def test_hash_stable_across_instances():
    cfg1 = _base_cfg(name="exp1", seed=7)
    cfg2 = _base_cfg(name="exp1", seed=7)
    assert cfg1.config_hash == cfg2.config_hash


def test_reranker_change_changes_hash():
    a = _base_cfg()
    b = _base_cfg(reranker=ComponentRef(kind="reranker", component_id="cross_encoder"))
    assert a.config_hash != b.config_hash


def test_diff_detects_changed_fields():
    a = _base_cfg(top_k=5, merge_strategy="rrf")
    b = _base_cfg(top_k=10, merge_strategy="weighted")
    d = a.diff(b)
    assert "top_k" in d
    assert d["top_k"]["before"] == 5
    assert d["top_k"]["after"] == 10
    assert "merge_strategy" in d


def test_diff_empty_when_equal():
    a = _base_cfg()
    b = _base_cfg()
    assert a.diff(b) == {}


def test_serialise_deserialise_roundtrip():
    cfg = _base_cfg(name="roundtrip")
    data = cfg.model_dump()
    cfg2 = ExperimentConfig(**{k: v for k, v in data.items() if k != "config_hash"})
    assert cfg2.config_hash == cfg.config_hash
