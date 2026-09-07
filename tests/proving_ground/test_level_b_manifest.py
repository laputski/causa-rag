"""What a load records about itself, proved by loading.

Two entries rest on the corpus manifest, and a manifest is written by the
ingestion path and by nothing else. A unit test can build one from a list of
hashes; only a load can show that the path writes it, that the registry keeps
it, and that a run made afterwards carries it.

Both pairs load into corpus ids of their own, so the base corpus every other
suite here measures against is never touched. Each load costs one pass of the
real embedding model over two hundred and twenty fragments, which is why
these are two pairs and not five.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.proving_ground.conftest import REALM, corpus_manifest, record

pytestmark = pytest.mark.proving_ground

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / "proving-ground" / "base-ru"
LANGUAGE = "ru_be"


def _load(source: Path, corpus_id: str, strategy: str = "structure_aware") -> None:
    """One load through the command a person would type.

    The command and not the function beneath it: the manifest is assembled at
    the end of the load and registered by the caller after it, so a test that
    called the loader directly would prove the half that never fails.
    """
    result = subprocess.run(
        [sys.executable, "-m", "services.ingestion.cli", "ingest", str(source),
         "--strategy", strategy, "--corpus-id", corpus_id,
         "--language", LANGUAGE, "--realm-id", REALM],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        env={**os.environ, "USE_REAL_BGE_M3": "true"},
    )
    assert result.returncode == 0, (
        f"the load of {corpus_id} failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
    )


def _mutated(defect: str, into: Path) -> Path:
    from tools.corpus_mutate import mutate, read_corpus, write_corpus

    write_corpus(mutate(read_corpus(CORPUS), defect), into)
    return into


def test_F03_a_flattened_corpus_makes_the_strategy_break_its_own_promise(
    stack: None, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The strategy named for structure, on documents that have none.

    It produces exactly what the plain fixed-window strategy would, under
    another name, and every other trace of that load looks correct: the
    fragments are well formed, the index takes them, retrieval works, and the
    setting still reads structure_aware. What says otherwise is the
    strategy's own promise, checked at the end of the load and kept on the
    corpus record.
    """
    flat = _mutated("flatten_headings", tmp_path_factory.mktemp("flat") / "base-ru")
    _load(flat, "base-ru-flat")

    broken = corpus_manifest("base-ru-flat")
    healthy = corpus_manifest("base-ru")
    assert broken, "the load wrote no manifest at all, so this pair proves nothing"
    assert broken.get("post_conditions_unmet"), (
        f"a corpus with no headings was loaded by the structural strategy and its own "
        f"check passed: {broken}"
    )
    assert not healthy.get("post_conditions_unmet"), (
        f"the healthy corpus is reported as breaking the same promise: {healthy}"
    )
    record("F03", "the structural strategy on documents with no structure, and it says so",
           promise_unmet=broken["post_conditions_unmet"],
           chunks_broken=broken.get("chunk_count"), chunks_control=healthy.get("chunk_count"),
           control_promise_unmet=healthy.get("post_conditions_unmet") or [])


def test_F09_two_loads_of_a_changed_corpus_are_not_the_same_corpus(
    stack: None, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """One corpus id over two different sets of documents.

    The platform said for a long time that it could not tell: the comparison
    check named this in its own docstring as undetectable from run data. Every
    document's content hash was computed at load time and thrown away, and a
    digest over the set is kept now, so two runs naming one corpus can be
    asked whether they queried the same documents.
    """
    from core.experiment.compare import check_comparability
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult

    scratch = tmp_path_factory.mktemp("changed") / "base-ru"
    _mutated("drop_a_numbered_document", scratch)
    _load(CORPUS, "base-ru-changing")
    before = corpus_manifest("base-ru-changing")
    _load(scratch, "base-ru-changing")
    after = corpus_manifest("base-ru-changing")

    assert before and after, "a load wrote no manifest, so this pair proves nothing"
    assert before["documents_digest"] != after["documents_digest"], (
        f"a document was dropped and the digest did not move: {before['document_count']} "
        f"documents against {after['document_count']}"
    )

    def _result(manifest: dict[str, Any]) -> ExperimentResult:
        result = ExperimentResult(
            config=ExperimentConfig(
                name="r", corpus_id="base-ru-changing", dataset_name="base-ru.v1.fast.jsonl",
                chunking_strategy=ComponentRef(kind="chunker", component_id="structure_aware"),
                embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
                generator=ComponentRef(kind="generator", component_id="ollama"),
            ),
            dataset_name="base-ru.v1.fast.jsonl",
        )
        result.question_results = [
            QuestionResult(question_id=f"q{i}", question="?", reference_answer="",
                           generated_answer="an answer", metrics={"retrieval_recall_at_k": 1.0})
            for i in range(1, 20)
        ]
        result.corpus_manifest = manifest
        return result

    spoke = {f"compare:{w.id}" for w in check_comparability(_result(before), _result(after))}
    quiet = {f"compare:{w.id}" for w in check_comparability(_result(before), _result(before))}
    assert "compare:corpus_changed" in spoke, (
        f"the corpus behind one name changed between two runs and nothing said so: "
        f"{sorted(spoke)}"
    )
    assert "compare:corpus_changed" not in quiet, (
        "two runs against one loading of the corpus are reported as incomparable"
    )
    record("F09", "one corpus id over two different sets of documents, and the runs say so",
           documents_before=before["document_count"], documents_after=after["document_count"],
           digest_before=before["documents_digest"][:16],
           digest_after=after["documents_digest"][:16],
           signals=sorted(spoke), signals_on_one_loading=sorted(quiet))
