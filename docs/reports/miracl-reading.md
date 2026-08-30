Written by hand from the table above, and revised as rows arrive. What is
still being measured is named as such rather than assumed.

### The reranker makes the retrieval mode stop mattering

Dense and hybrid RRF are 0.024 apart on ranking precision in English
without a reranker. With one they are 0.001 apart, and they stay within a
thousandth at all three context sizes: 0.720 against 0.721 at five, 0.793
against 0.794 at ten, 0.799 against 0.800 at twenty.

Russian is the stronger case, and it was measured after this paragraph was
first written. The unreranked gap there is wider, between 0.041 and 0.055
depending on context size. Reranked, it is **0.000 at all three**: 0.788
against 0.788, 0.845 against 0.845, 0.852 against 0.852. A larger
difference to erase, erased more completely.

The mechanism is visible in the conditions rather than hidden in the model.
Both modes hand the reranker the same fifty candidates, because the
candidate window is held at fifty for every row. The reranker sorts them by
its own judgement of the query, and where they came from leaves no trace in
that judgement. Whatever ordering advantage the retrieval mode had, the
reranker overwrites.

For anyone paying for both, the reading is short. **Once a reranker is in
the pipeline, the choice between dense and hybrid stops being a question
about ranking and becomes a question about what enters the window.** Hybrid
is marginally ahead there: 0.998 against 0.996 recall at twenty in English,
and ahead at twenty in Russian and Arabic without a reranker as well.

Two cautions on that. The reranked Arabic rows are still being measured,
and if the sign differs there, the divergence is the finding rather than
this convergence. And the pool is short, which compresses every gap in the
table, this one included.

### Adding BM25 costs ranking and buys recall

Without a reranker the pattern holds in all three languages and in the same
direction. Dense wins mean average precision everywhere (Arabic .815
against .790, Russian .794 against .753, English .742 against .718). Hybrid
wins recall at twenty everywhere (.992 against .983, .993 against .988,
.997 against .992).

Three languages with three different morphologies, one sign. That is
sturdier than any single number in the table.

Read plainly: the sparse half of the merge surfaces a few documents the
vectors missed, and it pays for them by pushing documents the vectors
ranked well further down. Whether that trade is worth making depends
entirely on whether something downstream reorders the result, which brings
it back to the finding above.

### The reranker is worth more than the retrieval mode

English dense at twenty gains .057 mean average precision from reranking.
Hybrid gains .082. Russian dense gains .058. Every one of those is larger
than the .024 that separates the two retrieval modes without a reranker.

The ordering of effort follows: a reranker earns more than a merge
strategy, on this pool, in every language measured so far.

### Arabic was not the hard case

Arabic scored the highest ranking precision of the three languages,
ahead of Russian and English, in every unreranked configuration.

This is worth stating because the expectation ran the other way, mine
included. It says nothing about Arabic retrieval in general: it says that
on this pool, with this embedder and an Arabic analyser on the sparse side,
Arabic was not where the difficulty was. The analyser matters and is not
free. Before it was chosen per corpus, Arabic text was stemmed by Russian
rules in the same index, and that is the version of this row that would
have confirmed the expectation.

### What the recall column is not telling you

Recall passes .98 at k=20 in every row of the table, reranked or not,
in all three languages. Nothing in the retrieval is that good.

Only judged passages are indexed, about 26 thousand of Arabic's 2.06
million, so almost every distractor a real corpus would contain is absent.
The number to carry away from that column is not its height but its
flatness: it has nowhere left to move, which is exactly why the
configurations look so close to each other.

**Every gap in this table is a floor on the real gap, not a measure of it.**
A configuration that ties another here has not been shown to tie on a full
corpus. Restoring the distractors is the next iteration of this work, and
it is the one that decides whether these findings survive.
