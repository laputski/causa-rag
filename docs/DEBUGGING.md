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
