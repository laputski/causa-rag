# Three things this platform could not express in RAG World's schema

`core/eval/rag_space.py` carries a checked copy of RAG World's dimensions and
says, in its own words, that a mechanism the twenty-eight do not express
belongs to that project's residual queue and not to a private vocabulary here.
Three such mechanisms were found while giving every catalogue entry a scope,
and this file is what a person submits there. Nothing in this repository edits
that one: the proposal belongs to its queue and its rule of three mentions.

Measured against `../rag-world/data/technologies/*.json` on 2026-09-08, over
76 records. Each finding names the query that produced it, so whoever reads
this can run it again instead of taking it on trust.

## 1. Source fusion does not say which sources are fused

Ten records carry `C3 ≠ none`. **Not one of them** carries `A5=lexical` or
`C1=lexical`, although `standard_hybridrag` and `opensearch` are dense search
together with lexical search by definition. `A5=lexical` appears only on
`bm25_sparse`, which is lexical and nothing else.

So the whole of a system's hybridity is one value, `C3=rrf`, and the second
source it fuses is invisible in every layer. A reader of a record cannot tell
a dense-plus-lexical system from a dense-plus-dense one, and a predicate over
coordinates cannot select the systems a lexical failure can occur in.

**Suggested residual entry**

    id: the_second_source_of_a_fusion
    ru: какой именно второй источник участвует в слиянии
    en: which second source a fusion actually joins
    note: Source fusion (C3) names that results from more than one source are
          merged and names the arithmetic. It does not name the sources. A
          representation layer records one primary representation, so a
          system searching a dense index and a lexical index at once is
          recorded as dense, and the lexical half exists nowhere in its
          coordinates.

## 2. Source fusion and query fusion are one value

Five records carry `C3=rrf` while `A5=dense_single`, `C1=ann` and
`C4=single_store`, that is, while the record describes a single source:
`edge`, `hyde`, `multi_query`, `rag_fusion`, `standard_hybridrag`.

They are not one case, and reading `B1` separates them. Three reformulate the
query and merge the results of one index: `hyde` (`B1=hyde`), `multi_query`
and `rag_fusion` (`B1=multi_reformulation`). Two do not reformulate anything
(`B1=identity`) and are single-source only because of finding 1.

So `C3` currently answers two questions with one value, and telling them apart
needs a reader to cross-check `B1` and to know that a lexical half may be
missing from the record.

**Suggested residual entry**

    id: fusion_of_reformulations_of_one_query
    ru: слияние выдач по нескольким переформулировкам одного запроса
    en: merging the results of several reformulations of one query
    note: Rank fusion over one index queried several ways is a different
          mechanism from rank fusion over several sources, and the two carry
          the same value of C3. The first is a property of the query layer
          and depends on B1; the second is a property of the retrieval layer.

## 3. Distribution disagrees between two records of one form

`standard_hybridrag` carries `C4=single_store`; `opensearch` carries
`C4=multiple_local`. Both describe dense search together with lexical search,
both carry `C3=rrf`, `A5=dense_single` and `C1=ann`.

One of the two is wrong, and which depends on whether a "store" counts engines
or indexes. This is not a residual: it is a question about two records, and it
belongs to whoever owns them.

## What this platform did in the meantime

Nothing to the schema. Entries whose scope the coordinates cannot express
carry a `scope_caveat` naming what is not expressed, the report counts how
many are outstanding, and the atlas page shows the caveat beside the entry.
Five entries carry one today, and all five are the failures scoped by the
presence of fusion because their lexical half is invisible: F16, F17, F21,
F22, F23. If a dimension ever answers finding 1 or 2, those caveats come off
and `applies_when` gets narrower.
