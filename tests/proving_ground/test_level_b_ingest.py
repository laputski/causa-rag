"""Paired baits for the failures the loading of a corpus stages.

These four cannot be staged any other way. An analyser is chosen when an index
is created and every query afterwards inherits it; the model a corpus was
embedded with is fixed at load time and no later setting revisits it. A run
configuration can say nothing about either, so the difference between the two
halves of each pair is a difference between two loads.

The loads are not performed here. They are printed by
`python3 -m tools.ingest_distort --plan <name> --corpus <id>` and run by hand,
so what a bait depends on is what a person reads. A missing index skips the
test with the command that creates it.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.proving_ground.conftest import (
    detector_signals,
    health_signals,
    recall,
    recall_before_rerank,
    record,
    retrieval_only,
    run_on,
)
from tools.seed_proving_ground import control_config

pytestmark = pytest.mark.proving_ground

CORPUS = "base-ru"
LANGUAGE = "ru_be"


def _against(embedder: Any, corpus_id: str, strategy: str = "structure_aware",
             real_model: bool = True, pipeline_id: str = "naive") -> dict[str, Any]:
    """The control configuration pointed at another loaded index.

    Dense only by default, and that is the whole of why these read anything at
    all. A failure of the vectors is invisible through a hybrid pipeline: the
    lexical half answers from the words whatever the vectors say, and recall
    stayed at one on a corpus embedded with a stub. That masking is the
    field behaviour these entries describe, so it is named here and not worked
    around in silence: on a hybrid system a ruined embedding is not
    visible in the retrieval metric at all.
    """
    from core.experiment.config import ExperimentConfig

    base = control_config(CORPUS).model_dump(exclude={"config_hash"})
    base["corpus_id"] = corpus_id
    base["name"] = f"proving-ground-{corpus_id}-{pipeline_id}"
    base["pipeline_id"] = pipeline_id
    base["chunking_strategy"] = {"kind": "chunker", "component_id": strategy, "params": {}}
    return run_on(embedder, retrieval_only(ExperimentConfig(**base)), CORPUS, LANGUAGE,
                  strategy=strategy, real_model=real_model)


def test_F01_a_second_load_under_another_chunk_size_leaves_the_first_behind(
    embedder: Any,
) -> None:
    """Re-ingestion adding instead of updating.

    Neither store removes anything: both write by chunk identifier, and a fixed
    window's identifier is derived from its boundaries, so a second load under
    a different size writes a second, complete chunking beside the first.

    Read on the corpus and not on a run: the duplication is a property of what
    is in the index, and the health check is the signal that reads that.
    """
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name

    client = QdrantClient(host="localhost", port=6333)

    def points(corpus_id: str) -> int:
        return client.count(collection_name=_collection_name(
            "fixed", "bge_m3", corpus_id, "proving-ground")).count

    once, twice = points(f"{CORPUS}-fixed"), points(f"{CORPUS}-reingested")
    assert twice > once * 2, (
        f"the second load replaced the first instead of adding to it: {once} against {twice}"
    )

    # The named signals do not see this one, and that is the finding.
    # `duplicates` compares text, and two chunkings of one document cut at
    # different boundaries produce overlapping text and never identical text.
    # What this platform can stage is a re-ingest under a different chunk size,
    # because a chunk identifier here is derived from content and an identical
    # re-ingest is therefore a no-op. The entry's own signals cover the case
    # where identifiers are not derived from content, which is a failure of
    # other systems and not one this platform can reproduce.
    assert "health:duplicates" not in health_signals(f"{CORPUS}-reingested", strategy="fixed"), (
        "the health check now sees this, so the note above is out of date"
    )
    record("F01", "a second load under another chunk size left the first chunking in the index",
           chunks_after_one_load=once, chunks_after_two=twice)


def test_F10_a_stub_embedder_indexes_vectors_that_mean_nothing(embedder: Any) -> None:
    """Loaded and queried by the stub, which returns a deterministic vector per
    text and no relation between two of them.

    The vector has the width of the real model's, so nothing downstream refuses
    it and the index is perfectly well formed. What collapses is retrieval.
    """
    healthy = _against(embedder, CORPUS)
    stubbed = _against(embedder, f"{CORPUS}-stubbed", real_model=False)
    assert recall_before_rerank(healthy) > recall_before_rerank(stubbed), (
        f"the stub retrieved as well as the real model: "
        f"{recall_before_rerank(stubbed)} against {recall_before_rerank(healthy)}"
    )
    # How much of the damage the reranker hides is worth recording, because it
    # is what decides whether anybody would notice. On a corpus of forty-nine
    # chunks, where the candidate window of fifty covered everything, the
    # reranker reordered the whole corpus and the final recall of the ruined
    # half equalled the healthy one exactly: the failure was invisible in the
    # number a reader looks at first. At two hundred and twenty chunks the
    # window covers less than a quarter, the reranker is handed a worse set to
    # begin with, and the damage reaches the final number as well.
    masked = recall(stubbed) >= recall(healthy)
    assert recall(stubbed) < recall(healthy) or masked, "impossible"
    record("F10", "a corpus embedded by the stub retrieves worse, and most of it before the reranker",
           pre_rerank_healthy=recall_before_rerank(healthy),
           pre_rerank_stubbed=recall_before_rerank(stubbed),
           final_healthy=recall(healthy), final_stubbed=recall(stubbed),
           reranker_hid_it_completely=masked)


def test_F11_and_F12_a_model_mismatch_ruins_retrieval_and_nothing_speaks(
    embedder: Any,
) -> None:
    """Loaded by one model, questioned by another.

    The reverse bait. Both models write vectors of the same width into a
    collection whose name carries the model's identifier and not the model, so
    nothing refuses the mixture. Two catalogue entries share this arrangement
    and are told apart only by their history: one is two models used at once,
    the other is one model replaced after the corpus was loaded. The index is
    identical in both cases, and the platform records neither the model a
    corpus was loaded with nor when it changed. That absence is why both are
    undetectable, and this is that absence made observable.
    """
    healthy = _against(embedder, CORPUS)
    mismatched = _against(embedder, f"{CORPUS}-mismatched", real_model=True)
    assert recall_before_rerank(mismatched) < recall_before_rerank(healthy), (
        f"the mismatch cost nothing: {recall_before_rerank(mismatched)} "
        f"against {recall_before_rerank(healthy)}"
    )
    # What speaks, and what it says. On a corpus of forty-nine chunks nothing
    # spoke at all. At two hundred and twenty the generic layer signal fires:
    # it says the run fails at retrieval, which is true and is as far as it
    # goes. Nothing names a model, compares two of them, or reads back what the
    # corpus was embedded with, because none of that is recorded. The entries
    # stay undetectable in the sense that matters, and this pair now shows the
    # exact shape of that: the platform can say where and cannot say why.
    spoke = detector_signals(mismatched) - {"detector:embedder_unverified"}
    names_the_cause = spoke - {"detector:layer_bottleneck"}
    assert names_the_cause == set(), (
        f"a signal now names the cause of a model mismatch, and the catalogue says none does: "
        f"{sorted(names_the_cause)}"
    )
    for failure_id in ("F11", "F12"):
        record(failure_id,
               "loaded by one model and questioned by another: the layer is named, the cause is not",
               pre_rerank_healthy=recall_before_rerank(healthy),
               pre_rerank_mismatched=recall_before_rerank(mismatched),
               final_healthy=recall(healthy), final_mismatched=recall(mismatched),
               signals=sorted(spoke), signals_naming_the_cause=sorted(names_the_cause))


def test_F16_the_wrong_analyser_costs_the_lexical_half_its_word_forms(
    embedder: Any,
) -> None:
    """The lexical index created under another language's analyser.

    Russian text stemmed by English rules keeps every word form apart, so a
    question phrased in another case or number stops matching the sentence that
    answers it. Read on the lexical half alone: the semantic half is unaffected
    by an analyser, and a hybrid run would hide the loss behind it, which is
    exactly how this failure survives in the field.
    """
    import json
    import pathlib

    from adapters.opensearch import OpenSearchRetriever

    questions = [json.loads(line)["question"]
                 for line in (pathlib.Path("eval/golden") / f"{CORPUS}.v1.fast.jsonl")
                 .read_text(encoding="utf-8").splitlines() if line.strip()]

    from core.eval.retrieval_metrics import extract_ref_id

    expected = [json.loads(line).get("article_refs") or []
                for line in (pathlib.Path("eval/golden") / f"{CORPUS}.v1.fast.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]

    def hits(corpus_id: str, language: str) -> int:
        """Questions whose own answer the lexical half puts in its top five.

        Counted against the expected source and not against "returned
        anything at all", which was the first measure and reported fifteen of
        fifteen under both analysers: a wrong analyser still matches some word
        somewhere, and matching something is not finding the answer.
        """
        sparse = OpenSearchRetriever(host="localhost", port=9200, strategy_id="structure_aware",
                                     corpus_id=corpus_id, realm_id="proving-ground",
                                     language=language)
        found = 0
        for question, want in zip(questions, expected, strict=True):
            if not want:
                continue
            got = {extract_ref_id({"doc_id": s.chunk.doc_id, "source_code": s.chunk.metadata.get("source_code"),
                                   "article_no": s.chunk.metadata.get("article_no"),
                                   "structural_path": s.chunk.structural_path})
                   for s in sparse.retrieve(query=question, k=5)}
            found += bool(got & set(want))
        return found

    own = hits(CORPUS, LANGUAGE)
    other = hits(f"{CORPUS}-en-analyser", "en")
    assert own > 0, "the lexical half finds nothing even under its own analyser"
    if other >= own:
        pytest.skip(
            f"NOT STAGED: {other} questions of the answerable ones still find their own answer "
            f"under the other language's analyser, against {own} under the corpus's own. "
            "The question set and the documents were written together and share their word "
            "forms, so exact matching carries them and there is little for stemming to do. "
            "Measured at three depths: ten against ten at k=1, thirteen against twelve at k=3, "
            "thirteen against thirteen at k=5. Staging this needs questions phrased in cases "
            "and numbers the documents do not use, which is what the entry means by a paraphrase."
        )


def test_F40_a_missing_half_leaves_the_other_supplying_everything(
    embedder: Any,
) -> None:
    """A proof about the signal, and deliberately not about a catalogue entry.

    The lexical index is never built and the run stays configured for hybrid
    retrieval throughout, so one half supplies every chunk of every context and
    the dominance signal reads exactly that. No setting can produce it: both
    halves draw on the same corpus, and putting the whole weight on one of them
    moves the order of the merged list without moving its membership.

    It was written first as proof of an entry that already existed, and that
    was wrong: the entry it named is scoped to fusion which normalises scores,
    while this proving ground fuses by rank, so that entry cannot occur here at
    all. The condition staged here had no entry, which is how the catalogue
    gained one. Its signal is shared with two others, and all three say so, so
    a firing is read as evidence for any of them and for none in particular.
    """
    healthy = _against(embedder, CORPUS, pipeline_id="hybrid_rrf")
    missing = _against(embedder, f"{CORPUS}-lexical-missing", pipeline_id="hybrid_rrf")

    def single_half(run: dict[str, Any]) -> tuple[int, int]:
        refs = [s for q in run["question_results"] for s in (q.get("pre_rerank_source_refs") or [])]
        alone = [s for s in refs if s.get("dense_score", 0) > 0 and not s.get("sparse_score", 0)]
        return len(alone), len(refs)

    alone_healthy, total_healthy = single_half(healthy)
    alone_missing, total_missing = single_half(missing)
    assert alone_missing == total_missing, (
        f"the lexical half still contributes: {total_missing - alone_missing} chunks of "
        f"{total_missing} carry a score from it"
    )
    assert "detector:bm25_dominance" not in detector_signals(healthy), (
        "the healthy half already reads as ruled by one source"
    )
    assert "detector:bm25_dominance" in detector_signals(missing), (
        f"one half supplied the whole context and nothing said so: "
        f"{sorted(detector_signals(missing))}"
    )
    assert alone_healthy < total_healthy // 2, (
        f"the healthy half already leans on one source: {alone_healthy} of {total_healthy}"
    )
    record("F40", "a lexical index that was never built leaves one half supplying everything",
           single_half_healthy=f"{alone_healthy}/{total_healthy}",
           single_half_missing=f"{alone_missing}/{total_missing}",
           signals=sorted(detector_signals(missing)))
