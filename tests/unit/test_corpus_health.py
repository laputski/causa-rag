"""The length-distribution and structure-coverage counters.

A corpus mean does not tell an even corpus from one half made of headings: both
pictures give the same number.
"""
def test_length_deciles_show_two_clumps_not_one_mean():
    """The mean does not tell an even corpus from one of headings and walls."""
    from core.eval.corpus_health import analyze

    chunks = [{"text": "x" * 80, "structural_path": "a"} for _ in range(10)]
    chunks += [{"text": "x" * 1300, "structural_path": "a"} for _ in range(10)]
    health = analyze(chunks)

    assert len(health.length_deciles) == 10
    # Both piles at the edges and nothing in the middle: exactly what a mean
    # hides.
    assert health.length_deciles[0] == 10
    assert health.length_deciles[-1] == 10
    assert sum(health.length_deciles[1:-1]) == 0


def test_root_path_counts_as_missing():
    """`root` is what the chunker marks a treeless document with, not a path."""
    from core.eval.corpus_health import analyze

    health = analyze([
        {"text": "text one", "structural_path": "root"},
        {"text": "text two", "structural_path": ""},
        {"text": "text three", "structural_path": "sec. 1"},
    ])
    assert health.n_missing_path == 2
    assert any(i.id in ("no_structure", "some_missing_path") for i in health.items)
