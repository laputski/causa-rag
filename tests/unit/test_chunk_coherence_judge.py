"""eval/chunk_coherence_judge.py — sampled LLM-judge of chunk coherence.

Unit-tested with a fake generator (no real Ollama) — only the scoring/
sampling/aggregation logic is under test here; the real judge prompt's
actual quality is an integration concern (see tests/eval/ for the live
deepeval/ragas/trulens suites this mirrors).
"""
from __future__ import annotations

from eval.chunk_coherence_judge import ChunkCoherenceJudge


class _C:
    def __init__(self, text, chunk_id=""):
        self.text = text
        self.chunk_id = chunk_id


class _FakeGenerator:
    """Returns a canned score per call_index, cycling — mimics the judge LLM
    without any network call."""

    def __init__(self, scores: list[str]):
        self._scores = scores
        self._i = 0

    def generate(self, prompt, **params):
        score = self._scores[self._i % len(self._scores)]
        self._i += 1
        return score


def _judge(scores: list[str], sample_size: int = 150) -> ChunkCoherenceJudge:
    j = ChunkCoherenceJudge(sample_size=sample_size)
    j._build_generator = lambda: _FakeGenerator(scores)  # type: ignore[method-assign]
    return j


def test_empty_chunk_list_returns_zero_sample():
    result = _judge(["5"]).run([])
    assert result.sample_size == 0


def test_averages_scores_into_0_to_1_range():
    chunks = [_C("text one", "c1"), _C("text two", "c2")]
    result = _judge(["5", "5"]).run(chunks)
    assert result.sample_size == 2
    assert result.metrics["avg_coherence"] == 1.0
    assert result.metrics["n_incoherent"] == 0.0


def test_low_scores_counted_as_incoherent():
    chunks = [_C("truncated text", "c1"), _C("another truncated one", "c2")]
    result = _judge(["1", "2"]).run(chunks)
    assert result.metrics["n_incoherent"] == 2.0


def test_unparseable_response_is_skipped_not_counted():
    chunks = [_C("some text", "c1"), _C("text two", "c2")]
    result = _judge(["not a number at all", "4"]).run(chunks)
    assert result.sample_size == 1


def test_blank_text_chunks_excluded_before_sampling():
    chunks = [_C(""), _C("  "), _C("substantive text here")]
    result = _judge(["5"]).run(chunks)
    assert result.sample_size == 1


def test_sample_capped_at_configured_size():
    chunks = [_C(f"text number {i}", f"c{i}") for i in range(20)]
    result = _judge(["5"], sample_size=5).run(chunks)
    assert result.sample_size == 5


def test_results_written_to_disk(tmp_path):
    j = ChunkCoherenceJudge(results_dir=tmp_path)
    j._build_generator = lambda: _FakeGenerator(["5"])  # type: ignore[method-assign]
    j.run([_C("some text", "c1")], corpus_id="test_corpus")
    saved = list(tmp_path.glob("chunk_coherence_test_corpus_*.json"))
    assert saved
