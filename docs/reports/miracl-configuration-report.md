# Configuration Report: retrieval on MIRACL

What changes when you change a RAG's retrieval, measured on a public
multilingual benchmark rather than on a demo corpus. Languages: ar, en, ru.
Questions: ar 2896, en 799, ru 1252, every question in the dev split.

## Why this replaced the demo corpus

Every number this platform had published came from a synthetic handbook of
eight documents and fifteen questions. As evidence it fell apart the moment
anyone opened the file.

Half of it was also wrong, and that surfaced only while preparing this
report. **The demo corpus is English, and its sparse index was built with a
Russian analyser** — the platform hardcoded one analyser for every corpus,
whatever language it held. Checked against the running index:
`approving purchases requires evidence` tokenises to
`approving · purchases · requires · evidence`, where an English analyser
gives `approv · purchas · requir · evid`. A query for "approve purchase"
did not match a document saying "approving purchases".

Dense retrieval was unaffected, since the embedder does not consult an
analyser. BM25 and every hybrid configuration on that corpus were measuring
something broken. It is recorded here rather than quietly fixed because a
measurement platform that cannot say where its own numbers came from has
nothing to offer anyone else's.

## Read these three things first

**The pool is the judged passages, not the whole corpus.** Arabic MIRACL
holds 2.06 million passages; the roughly 26 thousand a native speaker
judged for the dev split are what is indexed here, positives and
rejections together. This raises every absolute number and narrows the gaps
between configurations. Two rows that tie here are not shown to tie on the
full corpus.

**Only the dev split.** MIRACL keeps the judgements for test-a and test-b
private, so there is nothing to score against there.

**No generation is measured, in any language.** MIRACL ships relevance
judgements and no reference answers. Every row below stopped before the
model was asked anything, enforced by a generator component that raises if
called. Answer quality needs a benchmark that carries reference answers,
which is a separate piece of work.

## Conditions

Identical for every row unless the row says otherwise.

- Embeddings: BGE-M3, run locally
- Chunking: fixed, 3000 characters, no overlap. Above the 99th
  percentile of passage length in all three languages, so nearly every
  passage is a single chunk and chunk size is not an axis
- Candidate window: 50 for every row, so the reranked and unreranked
  halves see the same passages and differ only in ordering
- Sparse index: one OpenSearch analyser per language, not one for all
- Reranker, where on: BAAI/bge-reranker-v2-m3
- Everything ran on one machine, by hand. None of it runs in CI

## Results

| Language | Mode | top-k | Reranker | Recall@k | Precision@k | MAP | Questions | Failed |
|---|---|---:|---|---:|---:|---:|---:|---:|
| ar | hybrid_rrf | 5 | off | 0.877 | 0.320 | 0.753 | 2896 | 0 |
| ar | hybrid_rrf | 5 | on | 0.911 | 0.339 | 0.814 | 2896 | 0 |
| ar | hybrid_rrf | 10 | off | 0.965 | 0.187 | 0.785 | 2896 | 0 |
| ar | hybrid_rrf | 10 | on | 0.976 | 0.190 | 0.840 | 2896 | 0 |
| ar | hybrid_rrf | 20 | off | 0.992 | 0.097 | 0.790 | 2896 | 0 |
| ar | hybrid_rrf | 20 | on | 0.995 | 0.097 | 0.843 | 2896 | 0 |
| ar | naive | 5 | off | 0.887 | 0.330 | 0.784 | 2896 | 0 |
| ar | naive | 5 | on | 0.908 | 0.338 | 0.813 | 2896 | 0 |
| ar | naive | 10 | off | 0.960 | 0.186 | 0.810 | 2896 | 0 |
| ar | naive | 10 | on | 0.973 | 0.190 | 0.839 | 2896 | 0 |
| ar | naive | 20 | off | 0.983 | 0.096 | 0.815 | 2896 | 0 |
| ar | naive | 20 | on | 0.990 | 0.097 | 0.842 | 2896 | 0 |
| en | hybrid_rrf | 5 | off | 0.771 | 0.400 | 0.620 | 799 | 0 |
| en | hybrid_rrf | 5 | on | 0.837 | 0.440 | 0.721 | 799 | 0 |
| en | hybrid_rrf | 10 | off | 0.958 | 0.277 | 0.707 | 799 | 0 |
| en | hybrid_rrf | 10 | on | 0.978 | 0.283 | 0.794 | 799 | 0 |
| en | hybrid_rrf | 20 | off | 0.997 | 0.145 | 0.718 | 799 | 0 |
| en | hybrid_rrf | 20 | on | 0.998 | 0.145 | 0.800 | 799 | 0 |
| en | naive | 5 | off | 0.784 | 0.408 | 0.652 | 799 | 0 |
| en | naive | 5 | on | 0.837 | 0.440 | 0.720 | 799 | 0 |
| en | naive | 10 | off | 0.955 | 0.273 | 0.730 | 799 | 0 |
| en | naive | 10 | on | 0.977 | 0.283 | 0.793 | 799 | 0 |
| en | naive | 20 | off | 0.992 | 0.144 | 0.742 | 799 | 0 |
| en | naive | 20 | on | 0.996 | 0.145 | 0.799 | 799 | 0 |
| ru | hybrid_rrf | 5 | off | 0.798 | 0.404 | 0.667 | 1252 | 0 |
| ru | hybrid_rrf | 5 | on | 0.873 | 0.449 | 0.788 | 1252 | 0 |
| ru | hybrid_rrf | 10 | off | 0.958 | 0.269 | 0.743 | 1252 | 0 |
| ru | hybrid_rrf | 10 | on | 0.970 | 0.273 | 0.845 | 1252 | 0 |
| ru | hybrid_rrf | 20 | off | 0.993 | 0.141 | 0.753 | 1252 | 0 |
| ru | hybrid_rrf | 20 | on | 0.995 | 0.141 | 0.852 | 1252 | 0 |
| ru | naive | 5 | off | 0.838 | 0.428 | 0.722 | 1252 | 0 |
| ru | naive | 5 | on | 0.873 | 0.448 | 0.788 | 1252 | 0 |
| ru | naive | 10 | off | 0.960 | 0.269 | 0.785 | 1252 | 0 |
| ru | naive | 10 | on | 0.970 | 0.272 | 0.845 | 1252 | 0 |
| ru | naive | 20 | off | 0.988 | 0.141 | 0.794 | 1252 | 0 |
| ru | naive | 20 | on | 0.992 | 0.141 | 0.852 | 1252 | 0 |


Numbers are means over the questions that produced them. A cell reading
"not measured" is a metric this run had no basis to compute, left empty
rather than filled with a zero. The last column counts questions the run
could not answer at all; a row with failures is an average over a subset,
and the count is printed rather than folded into the mean.

## Reading

Written by hand from the table above, and revised as rows arrived. All 36
configurations are measured: 4947 questions, no failures.

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

| Language | Gap without a reranker | With one |
|---|---:|---:|
| en | 0.024 | 0.001 |
| ru | 0.041 to 0.055 | 0.000 |
| ar | 0.024 to 0.031 | 0.001 to 0.002 |

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

Arabic closes it. Its unreranked gap runs 0.024 to 0.031, and reranked it
is 0.001, 0.001, 0.002 at the three context sizes. Three languages, three
morphologies, one result, and the two that were measured last were measured
after this paragraph had already been written about the first.

One caution stands: the pool is short, and that compresses every gap in the
table, this one included. What is being said here is that the reranker
erases a difference that was already small. On a full corpus the difference
it has to erase would be larger, and whether it still erases all of it is
not something this run can answer.

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

Arabic gains 0.027 on dense and 0.053 on hybrid, the smallest of the three
and still larger than the 0.024 that separates its two retrieval modes.

The ordering of effort follows: a reranker earns more than a merge
strategy, on this pool, in all three languages.

### Arabic was not the hard case

Arabic scored the highest ranking precision of the three languages,
ahead of Russian and English, in every unreranked configuration. Reranked,
Russian overtakes it (0.852 against 0.842 at twenty) while English stays
behind both.

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

## Reproducing it

```
make miracl-fetch
make miracl-ingest
make miracl-report
```

Data: MIRACL, Zhang et al., TACL 2023, arXiv:2210.09984, Apache-2.0.
Passage text is Wikipedia under CC BY-SA 3.0 and is not redistributed;
the answer key travels with this repository under `eval/golden/`.
Provenance and the fetcher's own caveats: `eval/miracl/README.md`.
