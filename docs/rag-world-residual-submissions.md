# For the RAG World repository: two residual mechanisms and one data question

A brief for whoever works in `rag-world`. It is written to be executed there
and not here: `causa-rag` keeps a checked copy of the dimension codes and does
not edit the schema, because a new dimension belongs to that project's residual
queue and its rule of three mentions.

Everything below was measured against `data/technologies/*.json` on 2026-09-08,
over 76 records, and every count is reproducible with the snippet beside it.

**It was executed, and it came back with two of its three findings overturned.
The last section records what returned and what this repository did about it.
Read that before acting on anything above it.**

---

## Why this is needed

`causa-rag` catalogues failures of RAG systems and gives each entry the
coordinates at which it can occur, so applicability is derived and not
maintained by hand. Five of its entries cannot be scoped truthfully with the
schema as it stands, and each carries a written caveat saying so.

The five are failures of a system that searches a dense index **and** a lexical
one: a keyword search raising a document full of frequent words, the two halves
returning the same thing, the halves on incomparable scales, an analyser
stemming by the wrong language's rules. The nearest expressible scope is "this
system fuses sources", `C3 ≠ none`.

That scope is wider than the truth, and here is what it costs a reader today.
Ten records carry `C3 ≠ none`. Three of them, `hyde`, `multi_query` and
`rag_fusion`, fuse the results of several reformulations of **one** query
against **one** index. A reader who selects one of those three is shown five
failures about a keyword half that those systems do not have.

The schema cannot presently tell the two situations apart, and the reason is
two mechanisms it does not express. Both are below, in the shape the residual
dictionary uses.

---

## Finding 1. Source fusion does not say which sources are fused

**Measured.** Ten records carry `C3 ≠ none`. Not one of them carries
`A5=lexical` or `C1=lexical`, although `standard_hybridrag` and `opensearch`
are dense search together with lexical search by definition. `A5=lexical`
occurs on exactly one record in the whole registry, `bm25_sparse`, which is
lexical and nothing else.

```python
import json, pathlib
recs = {p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in pathlib.Path("data/technologies").glob("*.json")}
fused = {n: r["configuration"] for n, r in recs.items()
         if (r.get("configuration") or {}).get("C3") not in (None, "none")}
print(len(fused), [n for n, c in fused.items()
                   if c.get("A5") == "lexical" or c.get("C1") == "lexical"])
```

**What it means.** `A5` records one primary representation model, so a system
searching a dense index and a lexical index at once is recorded as dense, and
its second source exists nowhere in its coordinates. A predicate over the
schema cannot select the systems in which a lexical failure is possible, and a
reader of a record cannot tell a dense-plus-lexical system from a
dense-plus-dense one.

**Suggested entry** for `data/residual_vocabulary.json`, under `mechanisms`:

```json
{
  "id": "second_source_of_a_fusion",
  "ru": "какой именно второй источник участвует в слиянии",
  "en": "which second source a fusion actually joins",
  "note": "Слияние источников (C3) называет, что выдачи нескольких источников сливаются, и называет арифметику слияния. Оно не называет источники. Слой представления записывает одну основную модель представления, поэтому система, ищущая одновременно по плотному и по лексическому указателю, записана как плотная, и её лексическая половина не существует ни в одной координате.",
  "note_en": "Source fusion (C3) names that the results of more than one source are merged, and names the arithmetic of the merge. It does not name the sources. The representation stratum records one primary representation model, so a system searching a dense index and a lexical index at once is recorded as dense and its lexical half exists in none of its coordinates."
}
```

**Records to add `second_source_of_a_fusion` to.** Two are certain from their
definition: `standard_hybridrag` and `opensearch`. The remaining eight carrying
`C3 ≠ none` are `edge`, `hyde`, `kag`, `magma`, `multi_query`, `rag_anything`,
`rag_fusion`, `replug`; three of them are finding 2 below, and the rest should
be decided against their own sources and never from their coordinates, which
is precisely what this finding says the coordinates cannot settle.

Two mentions is under the rule of three. That is the honest state, and the
third will come from the next dense-plus-lexical record anybody describes.

---

## Finding 2. Source fusion and query fusion are one value

**Measured.** Five records carry `C3=rrf` while `A5=dense_single`, `C1=ann` and
`C4=single_store`, that is, while the record describes a single source:
`edge`, `hyde`, `multi_query`, `rag_fusion`, `standard_hybridrag`.

Reading `B1` splits them. Three reformulate the query and merge the results of
one index: `hyde` (`B1=hyde`), `multi_query` and `rag_fusion`
(`B1=multi_reformulation`). Two carry `B1=identity` and are recorded as
single-source only because of finding 1.

```python
single = {n: c for n, c in fused.items()
          if c.get("A5") == "dense_single" and c.get("C1") == "ann"
          and c.get("C4") == "single_store"}
print({n: c.get("B1") for n, c in single.items()})
```

**What it means.** One value of `C3` answers two questions. Rank fusion over
one index queried several ways is a property of the query stratum and depends
on `B1`; rank fusion over several sources is a property of the retrieval
stratum. Telling them apart today requires a reader to cross-check `B1` and to
know that a lexical half may be missing from the record.

**Suggested entry** for `data/residual_vocabulary.json`:

```json
{
  "id": "fusion_of_reformulations_of_one_query",
  "ru": "слияние выдач по нескольким переформулировкам одного запроса",
  "en": "merging the results of several reformulations of one query",
  "note": "Ранговое слияние выдач одного указателя, опрошенного несколькими переформулировками запроса, есть иной механизм, нежели ранговое слияние выдач нескольких источников, и оба несут одно значение C3. Первое принадлежит слою формулировки запроса и зависит от B1, второе относится к слою поиска.",
  "note_en": "Rank fusion over one index queried by several reformulations is a different mechanism from rank fusion over several sources, and the two carry the same value of C3. The first belongs to the query stratum and depends on B1; the second belongs to the retrieval stratum."
}
```

**Records to add `fusion_of_reformulations_of_one_query` to:** `hyde`,
`multi_query`, `rag_fusion`. That is three mentions, which by the queue's own
rule makes it a candidate for a dimension or for a new value.

**If it becomes one**, the shape that would settle both findings is a
dimension of the retrieval stratum naming what a fusion joins, with values
along the lines of `one_source_several_queries`, `dense_and_lexical`,
`dense_and_graph`, `several_stores`. `causa-rag` would then narrow five
entries from "there is a fusion" to "the fusion joins a lexical source" and
drop their caveats. That consequence is named so the queue can weigh it; the
decision is the schema owner's.

---

## Finding 3. Distribution disagrees between two records of one form

**Measured.** `standard_hybridrag` carries `C4=single_store`; `opensearch`
carries `C4=multiple_local`. Both describe dense search together with lexical
search, both carry `C3=rrf`, `A5=dense_single` and `C1=ann`.

This is not a residual and needs no dictionary entry. It is a question about
two records: does a "store" count engines or indexes? `C4`'s guard says the
dimension is "defined when more than one store is involved", which does not
settle whether one engine holding a vector index and an inverted index is one
store or two. Whichever answer the schema intends, one of the two records
contradicts it today, and a reader comparing them cannot tell which.

**Suggested action.** Decide the reading, write it into `C4`'s guard so the
next record does not have to guess, and make the two records agree.

---

## What this repository does not ask for

No edit to the dimension count, no new value slipped into `C3`, and no change
to any record's configuration beyond finding 3. The residual queue exists so a
schema grows by a count and not by a hunch, and two of the three items above
are exactly that: a count, offered.


---

## What came back, and what it changed here

The brief was executed in `rag-world` and answered in full. Its method beat
this one: it read the three primary sources, where this document inferred from
coordinates. Two of the three findings did not survive that reading.

**Finding 1 undercounted.** Two more records qualify, `magma` and `edge`, so
four fuse a source no coordinate of theirs records, and the mechanism reaches
the rule of three. It is now the only live candidate in that queue.

**Finding 2 is withdrawn, and this repository stopped waiting for it.** One of
its three mentions rested on a value the record's own source contradicts: HyDE
averages the embeddings of its generated documents into one vector and runs a
single search, so no lists arise and nothing is merged. Its `C3` is corrected
to `none`. Two mentions are left, below the threshold.

The reply added a second objection worth more than the count: the two cases are
told apart by reading `B1`, and information another coordinate already carries
is a difficulty of reading and not a gap in the schema. That is true, and
it applies to every entry here that carried the caveat. All five are narrowed
instead of waiting: two of the six query transformations leave one query
standing, the other four make several out of one, and a fusion over those joins
reformulations. Each now carries `("B1", ("identity", "key_extraction"))`
beside its fusion predicate, and the two registry records that fuse
reformulations, `multi_query` and `rag_fusion`, are ruled out of all five. That
is what this brief's own opening asked for: a reader who selects one of those
systems is no longer shown failures about a keyword half they do not have. The
platform's three points record `B1` now, so nothing about them moved.

The scope is only ever the nearest expressible thing, so the narrowing trades
one error for another: a system that fuses two sources **and** asks several
queries is now ruled out where it used to be included. No record of the
seventy-six exhibits that, and two exhibit the error it replaces.

Five entries carried a caveat before this. Two still do, `F16` and `F17`, and
theirs is finding 1: `B1` says whether a fusion joins several queries and says
nothing about whether the sources joined include a lexical one. Their scope is
narrower and their caveat stands.

**Finding 3 falls as a claim and stands as a question.** The two records were
judged by different criteria, and `opensearch` says so in its own
justification, where `standard_hybridrag` justifies nothing. There is no
contradiction in the data. Whether one engine holding a vector index and an
inverted index is one store or two is still unsettled, and belongs in the
guard on `C4`.

### The question the reply asks back: narrow or broad

The reply asks which reading of finding 1 is meant. A source that no
coordinate records, or `C3` never naming its sources at all. **Narrow**, and by
the queue's own criterion, not by preference.

Under the broad reading `kag` and `rag_anything` carry the code, and their
second source is a graph, which `A4=graph` already records. Coding what another
coordinate already carries is the objection that declined the one mechanism
which reached three mentions before this, and it would apply here in the same
words.

What the entries in this repository cannot derive is specifically a lexical
half, because `A5` records only the primary representation model. Under the
narrow reading the exhibited instances are that lexical half, in
`standard_hybridrag`, `opensearch` and `edge`, and MAGMA's filter by time,
which the reply is right to say the sketched value set does not cover. A
mechanism named for what a fusion joins that nothing records has room for both.
