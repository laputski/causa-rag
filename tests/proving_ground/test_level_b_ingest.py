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


def test_F11_and_F12_a_model_mismatch_ruins_retrieval_and_the_load_record_names_it(
    embedder: Any,
) -> None:
    """Loaded by one model, questioned by another.

    This was a reverse bait, on the reading that the platform recorded
    neither the model a corpus was loaded with nor when it changed. It
    records both now, on the corpus and on every run that queries it, so the
    pair shows a cause being named where it used to show a silence.

    What names it here is the load record, and not the comparison of two
    identifiers. That is worth stating plainly: the stub and
    the working model share an identifier and a version, both being class
    constants, so this staging cannot move the identifier comparison at all.
    Staging that would need two genuinely different embedders, and this
    platform has one. The comparison is proven at the unit level, on a
    payload where the identifiers differ, and this pair proves the other
    half: that a corpus whose vectors came from no model is said to be one.
    """
    healthy = _against(embedder, CORPUS)
    mismatched = _against(embedder, f"{CORPUS}-mismatched", real_model=True)
    assert recall_before_rerank(mismatched) < recall_before_rerank(healthy), (
        f"the mismatch cost nothing: {recall_before_rerank(mismatched)} "
        f"against {recall_before_rerank(healthy)}"
    )
    spoke = detector_signals(mismatched)
    assert "detector:stub_embedder" in spoke, (
        "the corpus was loaded by the stub and its own load record did not say so: "
        f"{sorted(spoke)}. A run carrying no manifest cannot know, so check that the "
        "distorted corpus was loaded after manifests existed."
    )
    assert "detector:embedder_unverified" not in spoke, (
        "the check was made and the run still reports that it could not be made"
    )
    for failure_id in ("F11", "F12"):
        record(failure_id,
               "loaded by one model and questioned by another: the load record names the cause",
               pre_rerank_healthy=recall_before_rerank(healthy),
               pre_rerank_mismatched=recall_before_rerank(mismatched),
               final_healthy=recall(healthy), final_mismatched=recall(mismatched),
               signals=sorted(spoke),
               identifier_comparison_not_exercised=(
                   "the stub and the model share an identifier and a version, so this "
                   "staging cannot move that comparison; it is baited at the unit level"
               ))


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
    assert other < own, (
        f"the wrong analyser cost nothing: {other} questions still find their own answer, "
        f"against {own} under the corpus's own analyser"
    )
    record("F16", "an index built under another language's analyser loses the paraphrases",
           found_under_own_analyser=own, found_under_the_other=other,
           questions=len([e for e in expected if e]))


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


def test_F13_the_end_of_a_unit_past_the_model_window_is_in_no_vector(
    embedder: Any,
) -> None:
    """A unit longer than the model reads, and the part it never read.

    Nothing on this platform reaches this by default: the chunker caps a unit
    at its chunk size and the default is a thousand characters, far below the
    window. The pair is therefore two loads of the same documents, one of
    which carries a fact after sixty thousand characters: at a thousand the
    fact sits near the start of a unit, and at sixty-five thousand the whole
    section is one unit whose end the model stops at.

    Measured on the stored vector and never on a ranking, and the difference
    matters. The long unit still comes back for a question about its subject,
    because it is the only unit on that subject and the first eight thousand
    tokens of it are about that subject. What is absent is the end: embedding
    the unit's text again, and embedding it with its tail removed, gives the
    same vector to six decimal places. So retrieval looks correct and answers
    from a unit whose relevant part was never read, which is the worst shape
    this failure has.
    """
    import numpy as np
    from qdrant_client import QdrantClient

    from adapters.qdrant import _collection_name
    from tools.corpus_mutate import THE_FACT_AT_THE_END

    client = QdrantClient(host="localhost", port=6333)

    def the_unit_holding_the_fact(corpus_id: str) -> Any:
        name = _collection_name("structure_aware", "bge_m3", corpus_id, "proving-ground")
        if name not in {c.name for c in client.get_collections().collections}:
            pytest.skip(
                f"NOT RUN: {corpus_id} is not loaded. Print the two loads with "
                f"`python3 -m tools.ingest_distort --plan "
                f"load_a_section_whole_past_the_model_window --corpus {CORPUS}` and run them."
            )
        points, _ = client.scroll(collection_name=name, limit=10_000,
                                  with_payload=True, with_vectors=True)
        holding = [p for p in points
                   if THE_FACT_AT_THE_END in (p.payload.get("text") or "")]
        assert len(holding) == 1, (
            f"{corpus_id} holds {len(holding)} units carrying the fact, and the pair needs one"
        )
        return holding[0], [len(p.payload.get("text") or "") for p in points]

    def cosine(a: Any, b: Any) -> float:
        first, second = np.array(a), np.array(b)
        return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))

    def without_its_tail(text: str) -> str:
        # Twenty words back from the fact, so what is dropped is a sentence and
        # not a token, and a vector that changes has changed for a reason a
        # reader would call a change of meaning.
        return text[:text.index(THE_FACT_AT_THE_END)].rsplit(" ", 20)[0]

    def stored_vector(point: Any) -> Any:
        return point.vector if not isinstance(point.vector, dict) else list(point.vector.values())[0]

    split, split_lengths = the_unit_holding_the_fact(f"{CORPUS}-window-split")
    whole, whole_lengths = the_unit_holding_the_fact(f"{CORPUS}-window-whole")

    # Exactly one unit of the distorted index is longer than the window, so
    # everything below is attributable to that unit and not to the chunk size.
    over_the_window = [n for n in whole_lengths if n > 40_000]
    assert len(over_the_window) == 1, (
        f"{len(over_the_window)} units exceed the window, so more than one thing changed"
    )
    assert max(split_lengths) < 40_000, "the control index also holds a unit past the window"

    split_text = split.payload["text"]
    whole_text = whole.payload["text"]
    moved = cosine(stored_vector(split), embedder.embed([without_its_tail(split_text)])[0])
    unmoved = cosine(stored_vector(whole), embedder.embed([without_its_tail(whole_text)])[0])

    assert moved < 0.99, (
        f"removing the tail of the control's unit left its vector at {moved:.6f}, so this pair "
        "cannot tell a read tail from an unread one"
    )
    assert unmoved > 0.999999, (
        f"removing the tail of a unit of {len(whole_text)} characters moved its vector to "
        f"{unmoved:.6f}, so the model did read past the window and this entry is not staged"
    )

    # The reverse half. Nothing compares a unit's length against the window.
    spoke = health_signals(f"{CORPUS}-window-whole")
    assert spoke == set(), (
        f"something does see this, which would make the entry detectable: {sorted(spoke)}"
    )
    record("F13",
           "a unit of sixty thousand characters is stored with a vector its end never "
           "reached, and every check reports a corpus in order",
           unit_characters_whole=len(whole_text), unit_characters_split=len(split_text),
           units_whole=len(whole_lengths), units_split=len(split_lengths),
           units_over_the_window=len(over_the_window),
           cosine_after_dropping_the_tail_whole=round(unmoved, 6),
           cosine_after_dropping_the_tail_split=round(moved, 6),
           declared_window_tokens=8192, signals_seen=[])
