# MIRACL slices

What `python3 -m eval.miracl.fetch --lang ar` produces, and what the numbers
built on it do and do not mean.

## What arrives

    corpus/miracl-{lang}/{wiki_id}/{passage}.txt     the judged passages
    eval/golden/miracl-{lang}.v1.fast.jsonl          the questions

The layout is not decoration. Retrieval metrics compare
`{source_code}/{article_no}` strings, ingestion reads `source_code` from a
file's parent directory and `article_no` from its numeric stem, and a MIRACL
passage is identified as `151236#1`. Writing it to `151236/1.txt` makes the
golden set's `article_refs` a straight copy of the qrels, with no mapping
layer to go wrong.

`corpus/*/` is gitignored. The golden sets are committed: they are small,
and a report whose answer key has to be re-downloaded to be checked is a
report nobody checks.

## Three things to say before quoting any number

**The pool is the judged passages, not the corpus.** Arabic MIRACL holds
2.06 million passages; about 26 thousand of them were judged for the dev
split, and those are what gets indexed. This **raises every absolute number
and narrows the gaps between configurations.** Two configurations that tie
on this pool are not shown to tie on the real one. Adding unjudged
distractors restores the discriminating power and is the second iteration
of this work, not part of the first.

**Only the dev split.** `test-a` and `test-b` keep their judgements
private. A run against them scores every configuration at zero, which reads
as a broken pipeline, when what is missing is the answer key.

**No generation metrics, in any language.** MIRACL ships relevance
judgements and no reference answers. Answer similarity, faithfulness and
refusal correctness need a reference, so the golden sets carry no
`ground_truth` field and those metrics stay empty, never scoring against
an empty string. Generation is measured separately on
MIRAGE-Bench, which exists because MIRACL does not answer this.

## Volumes, dev split

| Language | Questions | Relevant passages | Pool | Corpus streamed |
|---|---:|---:|---:|---:|
| ar | 2896 | 5658 | 25 881 | 0.32 GB |
| ru | 1252 | 3560 | 12 607 | 1.58 GB |
| en | 799 | 2326 | 7 921 | 5.06 GB |

The shards are streamed and discarded, so the download is transient and only
the passages stay. Shards stop being fetched once the last wanted passage is
found, which is why Arabic usually reads four of its five.

## Running it

Three steps, in order, all on one machine. None of it runs in CI: the grid
needs Qdrant, OpenSearch and the real BGE-M3.

```
make miracl-fetch                  # download and lay out ar, ru, en
make miracl-ingest                 # index each language with its own analyser
make miracl-report LIMIT=200       # smoke run first
make miracl-report                 # the whole grid
```

Do the smoke run first.

**What it costs, measured and not guessed.** The half without the
reranker is four minutes for all three languages and 4947 questions. The
reranked half is about ten hours. Reranking one candidate window takes 4.1
seconds on Arabic and 2.5 on English, and a language's whole cost sits in
its first reranked configuration: the reranker's answers are remembered, so
top-k 5, 10 and 20 ask it the same question and the last two read the first
one's answer.

`--no-reranker` gives the dense-against-hybrid answer in those four
minutes. The reranked rows can be added later; a stopped run keeps both its
finished configurations and the reranker's scores, and re-running the same
command continues.

**Run it under `caffeinate -i`.** The first attempt sat overnight for
nothing: the machine slept at 00:20 and the process collected seven minutes
of CPU in nine hours.

**There is no faster device to move to.** sentence-transformers already
selects `mps` on a Mac, so the ten hours are GPU hours, not CPU hours. The
one remaining lever is `max_length`, 8192 by default: dropping it to 512 is
about 1.5x, and all of that comes from truncating the passages long enough
to dominate a batch through quadratic attention. That changes what is
measured on roughly a twentieth of the passages, so it is not taken.

## The grid, and what is deliberately not in it

Axes: top-k 5 / 10 / 20, mode dense against hybrid RRF, reranker on or off.

**The candidate window is fixed at 50 for every row.** Letting it follow
top-k would move the window and the reranker together, and the reranker's
own effect would stop being readable. Both halves see the same fifty
passages and differ only in the order they put them in.

**Chunk size is fixed at 3000 characters, above the 99th percentile of
passage length in all three languages.** A chunk larger than the passage
cuts nothing, so the axis only exists below that length and would have
measured nothing here.

**Graph retrieval is out of the first iteration.** It needs Neo4j, a
separate ingest pass, and roughly triples the runtime, and the dense
against hybrid comparison answers the main question without it.

**The reranker is not the platform's default.** `cross_encoder_local`
defaults to ms-marco-MiniLM, which is trained on English alone. Scoring
Arabic and Russian with it would have produced a row reading "the reranker
hurts outside English" when the finding is that this particular reranker
only speaks English. The report uses `BAAI/bge-reranker-v2-m3`, which is
multilingual and pairs with the BGE-M3 embeddings already in use, and every
row records which reranker produced it.

**Nothing generates.** The generator slot is filled by a component that
raises if called, so the claim is checked on every row and never merely
asserted here.

## The contract, measured against the pipeline

Every row of the report is produced by the in-process pipeline, and the
platform exists to measure somebody else's retrieval over
`docs/external-rag-contract.md`. A report that never exercises that
contract does not demonstrate the thing it is offered as evidence for.

    python3 -m eval.miracl.verify_contract --lang ar --limit 50

This serves the platform's own retriever behind the contract, on a loopback
port, and asks the same questions both ways. The external side embeds its
own queries, the way any real RAG would: the contract carries text, and
what a RAG does with that text is its business.

**Result: fifty questions in each of ar, ru and en, no disagreements.** The
same sources, in the same order, with the same text, down both paths.

The text is compared as well as the identifiers, and that is why this runs
on Arabic and Russian, not on English alone. Retrieval metrics match on
source_code and article_no, so a passage mangled in a JSON round trip would
reach the reranker and the answer while every number in the report stayed
identical.

The bait shows that exactly. Encoding the served text as UTF-8 and decoding
it as latin-1, the classic mojibake, leaves **every identifier matching and
every passage wrong**: zero order disagreements, ten out of ten on text, in
all three languages. Dropping a source from the response instead makes the
order check fire. Each half of the comparison catches what the other misses.

What that establishes is narrow and worth stating narrowly. It shows that
request shaping, JSON round-tripping, response parsing and SourceRef
construction lose nothing. It does not show that any particular external
RAG is measured correctly, because that depends on what that RAG returns.

## Source and licence

MIRACL, Zhang et al., TACL 2023, [arXiv:2210.09984](https://arxiv.org/abs/2210.09984).
Topics, qrels and corpus from [`miracl/miracl`](https://huggingface.co/datasets/miracl/miracl)
and [`miracl/miracl-corpus`](https://huggingface.co/datasets/miracl/miracl-corpus),
both Apache-2.0. Passage text is Wikipedia, CC BY-SA 3.0, and is not
redistributed here.

Nothing goes through the `datasets` library: `miracl/miracl` sits behind a
loading script, and the dataset viewer answers 501 for it. Topics and qrels
are plain TSV, the corpus is gzipped JSONL, and this module reads both
directly.
