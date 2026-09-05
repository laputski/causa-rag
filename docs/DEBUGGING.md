# RAG Platform Debugging Lessons Learned

Real issues encountered during development and how they were resolved.

## 1. Stub embedder silently indexed wrong vectors

**Symptom:** Every answer came back as the model's "the documents contain no such information" refusal, despite correct questions.

**Root cause:** `adapters/bge_m3.py` had `if use_real_model:` (local variable) instead of `if self._use_real_model:` (instance attribute). The `USE_REAL_BGE_M3=true` env var was read and had no effect: every request used stub random 1024-dim vectors.

**Fix:** Changed to `self._use_real_model`. Re-ingested corpus with real model.

**How to detect:** In API startup logs, look for `Loading weights: 100%|██████████| 391/391`. Absence of this line = stub mode.

---

## 2. Duplicate chunks in OpenSearch (4× expected count)

**Symptom:** OpenSearch doc count was 14,988 but Qdrant had 3,747. After 4 `make ingest` runs, OpenSearch grew linearly.

**Root cause:** `chunk_id` was generated with `uuid4()`, giving a new random UUID on every ingest. OpenSearch `index` (not `update`) creates a new doc each time.

**Fix:** Deterministic `chunk_id = sha256(f"{source}:{strategy}:{start}:{end}")[:32]`. Now repeated ingestion is an upsert.

**Cleanup:** Delete and recreate Qdrant collection and OpenSearch index, then re-ingest:
```bash
# Delete Qdrant collection via REST or Dashboard
# Delete OpenSearch index:
curl -XDELETE http://localhost:9200/chunks_*
# Re-ingest:
USE_REAL_BGE_M3=true python3 -m services.ingestion.cli ...
```

---

## 3. qwen3:8b returning empty strings

**Symptom:** `generated_answer` was `""` for all questions in experiment run.

**Root cause:** qwen3 (and deepseek-r1) have a "thinking" mode where they emit `<think>...</think>` tokens before the answer. With a low `num_predict` budget, the model exhausted the budget on thinking and returned an empty response string.

**Fix:** Add `"think": false` to the Ollama request payload in `adapters/ollama_generator.py`. This disables CoT thinking mode.

---

## 4. BM25 returning keyword-biased results

**Symptom:** Sparse retrieval's top hit was always an article carrying the jurisdiction's name in its title, regardless of the actual question.

**Root cause:** Questions were phrased as "under the law of <jurisdiction>", and BM25 scored any document mentioning that jurisdiction highly.

**Fix:** Use `hybrid-rrf` merge strategy. RRF combines dense (semantic) and sparse (keyword) rankings, so dense retrieval correctly finds the relevant article semantically, and RRF downweights BM25's keyword bias.

---

## 5. Misleading faithfulness metric (0.04 ≠ bad answer)

**Symptom:** `faithfulness` was 0.04 even when the answer was correct and verifiable.

**Root cause:** Faithfulness is implemented as Jaccard overlap between answer tokens and retrieved context tokens. A model that paraphrases well will have low Jaccard even if fully grounded.

**Example:**
- Context: "A legal entity is an organisation that owns property in its own right"
- Answer: "A legal entity is an organisation with its own property and its own liability"
- Jaccard ≈ 0.12 (different word forms, plus the added clause about liability)

**Conclusion:** Trust `reference_overlap` as the primary quality metric. `faithfulness > 0.15` is a useful threshold but low faithfulness alone does not indicate hallucination. Always read the actual generated answer.

---

## 6. structure_aware chunker losing `doc.source` path

**Symptom:** `chunk.metadata["source"]` was empty string; deterministic chunk_id was computed with empty source, causing collisions between chunks from different documents.

**Root cause:** `_traverse()` and `_split_text()` in `core/chunking/structure_aware.py` did not thread `doc.source` through, using only `doc.doc_id`.

**Fix:** Added `source` parameter to `_traverse` and `_split_text`, passed `doc.source` from `chunk()`.

---

## 7. MongoDB Motor async in sync router

**Symptom:** `TypeError: object NoneType can't be used in 'await' expression` in experiments router.

**Root cause:** `_save()` and `_get_results()` helper functions were `def` (sync) but called Motor async methods without `await`.

**Fix:** Changed both to `async def` and updated all callers with `await`.

---

## 8. structure_aware silently degraded to fixed splitting on every real ingest

**Symptom:** Chunks indexed under the `structure_aware` strategy/collection had `structural_path == "root"` and were cut at exactly 512 chars mid-word, which is the signature of `fixed` splitting rather than structure-aware splitting.

**Root cause:** `StructureAwareChunkingStrategy.chunk()` only builds a structural
tree when `doc.structure` is already populated, but `_read_file()` in
`services/ingestion/cli.py` constructed every `Document` with `structure=None`.
No production parser anywhere turned a raw corpus file into a `DocumentNode`
tree: `DocumentNode` was only ever built by hand in unit-test fixtures. So
every `--strategy structure_aware` ingest had been silently falling back to
flat windows for the project's entire history, despite getting its own Qdrant
namespace, which implied it was doing something different.

**Fix:** a structure parser that turns raw corpus files into a real
`DocumentNode` tree. Because the heading convention is specific to a subject
area, the parser belongs to a domain pack and not to `core/*`, and the
CLI selects it by id with `--structure <pack_id>/<parser_id>`, resolved through
the domain-pack registry so the CLI imports no pack directly.
`domain_packs/manuals/structure_parser.py` is the worked example. The
`structure_aware` chunker also derives a tree from markdown headings on its own,
which covers a corpus like the demo handbook with no pack involved. Locked in
with a unit test using a real corpus file, not a hand-built fixture, so the same
regression can't silently recur.

**Correction, found later by the failure catalogue.** The paragraph above says
the `structure_aware` chunker "also derives a tree from markdown headings on its
own, which covers a corpus like the demo handbook with no pack involved". That
half was described and never built. `chunk()` read `if doc.structure is None:`
and returned flat windows, and `make demo` passes no parser, so the demo realm
shipped twenty-two chunks all carrying the path `root`, which is the very
regression this entry says was closed. The test named as the lock exercises the pack's
parser on a hand-written string and never touches that path.

Two further losses came out of fixing it, both of the same shape and neither
reported by anything:

- `_traverse` emitted a chunk only for a leaf, so a section carrying a lead
  paragraph before its subsections lost that paragraph. Eight hundred
  characters of body text over the demo corpus's eight files.
- `parse_manual_section` discarded every line arriving before the first
  heading, and a file with no numbering at all produced a tree with no children
  and therefore **zero chunks**: the whole file absent from the index, no
  exception, no warning. A unit test asserted `tree.children == []` and so
  pinned it.

Deriving the tree then made a third defect reachable: chunk offsets are counted
inside a node, so two sections sharing a heading in one file produced the same
`chunk_id` and ingestion, which upserts by it, kept whichever was written last.
That is entry F02 of the catalogue, introduced by a fix and caught by auditing
the fix. The identifier now carries the node's position in the tree.

`tests/unit/test_structure_aware_derives_a_tree.py` reads real corpus files, as
this entry asked for and its own test did not.

**How to detect:** Browse the corpus (Data → Corpus → Content) and check
`structural_path`. If it is `"root"` for a non-`fixed` strategy, structure was
never populated. The `missing_structural_numbers` health detector (#10 below)
also catches numbering gaps left by a parser that can't recognize a heading.

---

## 9. source_refs silently empty on every experiment-run read, regardless of retrieval quality

**Symptom:** `GET /experiments/{id}` returned `source_refs: []` for every
question on every run, including runs whose generation step clearly used
real retrieved context (the generated answer matched a manual reproduction
word-for-word). Faithfulness/relevancy looked catastrophic (~0.03-0.04)
because the evaluator computed them against an empty context, even though
retrieval itself was working.

**Root cause:** `_parse_result()` in
`services/api_gateway/routers/experiments.py` rebuilds `QuestionResult` from
the MongoDB document (or file fallback) but never passed `source_refs`
through to the constructor, so it defaulted to an empty list. This affected
*every* run fetched via the API, not a specific config, which is why it kept
reappearing across multiple unrelated debugging sessions before the read path
itself was inspected instead of the retrieval pipeline.

**Fix:** Pass `source_refs=qd.get("source_refs", [])` into the
`QuestionResult(...)` call. One line; the write path (`ExperimentRunner.run()`)
was always correct, only the read path dropped the field.

**How to detect:** If a run's generated answers look reasonable but
`source_refs` is empty for *every* question across *every* run (not just a
flaky one), suspect the read and serialisation path before the retriever:
reproduce the exact pipeline call directly in Python and compare its
`Answer.source_refs` against what the API returns for the same run.

---

## 10. Corpus export silently duplicated/dropped same-numbered articles

**Symptom:** A question with a clear, on-topic answer in the corpus
(one asking about company registration procedure) got the "no information" refusal
even though the corresponding article existed verbatim in the aggregated
source (a corpus's own `full.txt`).

**Root cause:** The corpus's own per-file export has the same article number
reused by two structurally different units. For example, `full.txt` has *two*
separate "article 47" headers in a row (one about company registration, the
other untitled and about share capital).
The per-article file `47.txt` only ever captured the second one, so the
first one's content was never indexed under any number, which is not a duplicate in
the index, a silent gap. Separately, several real headings (an article 146 among them)
had no trailing period, which the parser's regex required, so even when the
file *was* indexed its number was lost entirely until the regex was relaxed
(#8 above).

**Fix:** No code fix reaches the source data itself. This is a corpus
data-quality issue, not a platform bug. The platform now has two detectors
that catch each half of the underlying signature deterministically:
`duplicate_structural_numbers` (same number, different content, though it will not
fire here since only one side survived export) and `missing_structural_numbers`
(a gap in sequential numbering, which is what actually catches it, since the
real article's content never made it into the index under any number).

**How to detect:** Data → Corpus → Health → check `missing_structural_numbers`. A number missing from the index that *does* exist on disk in the source files means either the parser couldn't recognize that file's heading, or the export process dropped or overwrote it before the parser ever saw it. Check the raw source file directly.

## 11. Model cited the wrong article number despite retrieving the right chunk

**Symptom:** Visually inspecting top-k retrieval in RunPage confirmed the
correct chunk was retrieved (and even that the answer's *content* was
verbatim from it), but the printed citation named a *different* retrieved
chunk's article number. Answer content matched chunk #1's text
word-for-word, but cited chunk #2's article 43 instead of chunk #1's
article 40. Three successive prompt revisions telling the model "take the
article number ONLY from the fragment's own text, not from the Fragment N
label" did not fix it at scale: on one run (top_k=10, reranker) only ~2.4% of
answers cited the correct number, and a new, worse failure mode appeared:
literal "(Fragment N)" text leaking into ~50% of answers.

**Root cause:** Asking an LLM to transcribe a structured identifier
correctly out of a multi-chunk context is unreliable at scale, no matter how
the instruction is worded. It is a content-to-citation *binding* problem, not a
knowledge problem (the model knows the right content, just attaches the
wrong label to it).

**Fix:** Stop asking the model to transcribe the number at all. The
model may only reference a fragment by its small, unambiguous position
("(Fragment N)"); the platform deterministically substitutes the real structural
label (`core/citation.py:substitute_fragment_markers`, called from
`core/pipeline.py` on every run) before the answer is shown. Verified live over
130 questions: every sampled citation's substituted title now matches the
question's topic.

A prompt has to ask for the marker in the form the substitution recognises, or
the mechanism silently does nothing. The demo realm shipped asking for
"[section N]" and its answers kept a marker that means nothing beside the
context list; `tests/unit/test_demo_realm.py` now checks the demo prompt against
`core/citation.py`'s own pattern.

**How to detect:** RunPage → expand a question's "Retrieval" panel and compare
the model's printed citation against the actual top-k chunk text/labels
shown. `citation_number_coverage` (`ui/src/lib/metricMeta.ts`) is a
secondary deterministic signal, but note it's capped near `1/top_k` for
single-citation answers (it divides by *all* candidate numbers among
retrieved chunks, not just the correct one), so a low absolute value alone
doesn't mean the citation is wrong; check the actual text.

## 12. Thirty-six stored runs carry a fingerprint that does not describe them

**Symptom:** none visible. Every affected run displays, compares and reruns
without complaint.

**Found by:** recomputing `config_hash` from the configuration stored beside it,
for all 41 runs in `eval/results/runs/`. Five reproduced; thirty-six did not.
The check was run while adding an unrelated field, precisely to prove the new
field did not break historical fingerprints. It did not, and the breakage was
already there.

**Root cause:** `config_hash` is computed once, by a validator, at
construction, and `name` is one of the hashed fields. The MIRACL sweep appended
the question count to the name *after* the run finished
(`eval/miracl/report.py`, `result.config.name = ...`), so the stored
fingerprint belongs to a configuration that no longer exists.

The thirty-six are the whole MIRACL grid, saved in one eleven-minute window on
2026-08-29. Runs from 08-25, 08-30 and 08-31 all reproduce, which is what ruled
out a schema drift and pointed at the sweep.

**Consequence:** `config_hash` is what run deduplication, cache-boundary
decisions and every "same configuration" comparison are keyed on. For those
thirty-six, that key describes a configuration whose name differs from the one
on record. Nothing in the platform reads a fingerprint back to check it, so the
disagreement stayed silent for as long as it existed.

**Fix:** `ExperimentConfig.renamed(name)` returns a copy with the fingerprint
recomputed, and the sweep uses it. `ExperimentConfig.fingerprint_matches_fields()`
answers the question directly for anybody who wants to ask it.

Offered as a question, and not enforced by freezing the model, because the runner
performs two backfills after hashing on purpose (`external_rag_name`,
`http_endpoint`), so that a registered system's URL rotating over time does not
change the identity of the same run against the same system. Freezing would
have to undo those.

**Not repaired retroactively.** The thirty-six keep their fingerprints. Editing
stored measurements to make a derived key agree is the wrong trade, and a rerun
of the grid now produces different fingerprints for the same grid points. That is
correct, and it is stated here so that the discontinuity is not read later as a
second defect.

**How to detect:** `config.fingerprint_matches_fields()` on any configuration
read back out of storage.

## 13. The two halves of the hybrid merge never recognised one chunk as one chunk

**Symptom:** none visible. The answer's context looked correct on every screen.

**Found by:** a proving-ground bait that would not fire. Setting the rank-fusion
constant to one changed the retrieved order on none of fifteen questions, so the
staged defect appeared to be no defect. Measuring the two halves' candidate
lists showed twenty dense and nineteen sparse candidates with **zero** in
common, on a corpus of forty-nine chunks.

**Root cause:** Qdrant requires a point identifier to be a UUID or an integer. A
chunk id here is thirty-two hexadecimal characters, which Qdrant accepts,
normalises into dashed UUID form, and returns in that form. OpenSearch stores
the same id verbatim and returns it verbatim. `core/retrieval/hybrid.py` keys
the merge on the chunk id, so one chunk arriving from both halves was two
documents.

**Consequences, all silent:**

- Reciprocal rank fusion adds a document's contribution from each list it
  appears in. No document ever appeared in both, so every score had exactly one
  term and the merge was an interleaving of two disjoint lists.
- `rrf_k` could not change the order. With disjoint lists a document at rank r
  scores 1/(k+r) whatever k is, and the ordering by that value is the ordering
  by r. The constant is inert by construction, not by coincidence.
- `dense_score` and `sparse_score` are annotated by looking a chunk up in the
  other half's score map. That lookup always missed, so a chunk found by both
  halves recorded one of its two scores as zero. Every detector reading that
  split read it wrong.
- The context could hold the same passage twice. Measured: a context of five
  carried four distinct texts.

**Why it stayed quiet:** `core/pipeline.py` drops repeated text before
answering, so the user-visible context was clean. What was lost sat upstream of
that.

**Fix:** the Qdrant payload now carries `chunk_id`, as the OpenSearch document
body already did, and the read takes the id from the payload. Points written
before the field existed fall back to the point identifier, so an old
collection keeps today's behaviour until it is loaded again.

**After the fix,** the same measurement on the same corpus gives ten shared
candidates of twenty, and the fusion constant changes the order on ten of the
fifteen questions.

**Not repaired retroactively.** Collections loaded by an older build keep the
dashed identifiers until they are re-ingested. Runs stored against them keep
whatever they recorded.

**How to detect:** compare the identifier sets the two halves return for one
query. Any overlap at all is the healthy state on a corpus small enough for the
halves to compete.
