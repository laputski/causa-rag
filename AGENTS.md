# Working in this repository

Orientation for anyone, human or agent, making a change here. It says where
things live, which rules are enforced by tests rather than by review, and what
"done" means.

Nothing below is aspirational: every rule named here has a test that fails when
it is broken, and the test is named so you can run it.

## The one rule that fails the build

**`core/` is domain-neutral.** No legal, medical, financial or other subject
vocabulary, and no import from a domain pack. Subject knowledge lives only in a
pack.

`tests/unit/test_p1_guardian.py` enforces this with a regex over `core/`'s own
source, comments included. An early draft of a docstring that merely *named* a
pack's module tripped it, which is the rule working: a comment naming a subject
area is how the coupling starts.

Layer isolation is enforced separately by `lint-imports` against `.importlinter`.

## Where things live

| Directory | What belongs there |
|---|---|
| `core/` | Protocols, Pydantic models, the configurable pipeline, chunking strategies, metrics, diagnostics. Abstractions only |
| `adapters/` | One concrete implementation per protocol: Qdrant, OpenSearch, Neo4j, BGE-M3, Ollama, rerankers, MongoDB, the external-RAG HTTP pipeline |
| `domain_packs/` | Subject-area plugins, discovered by directory scan from `pack.yaml`. `manuals/` is the worked example |
| `services/` | `api_gateway` (FastAPI), `ingestion` (CLI), `reference_rag_server` |
| `ui/` | React, TypeScript, Vite, TanStack Query |
| `clients/python/` | `causa-rag-client`, the SDK for connecting an external RAG |
| `eval/` | Golden-set loading, the judge runners, the SLA gate |
| `tests/` | unit, contract, integration, fitness, e2e, eval |
| `deploy/` | Compose for development, a Helm chart as a starting point |

An adapter implements exactly one protocol from `core/interfaces.py` and ships a
stub beside it in the same file, so unit tests need no service running.

## Naming that carries meaning

Two namespacing rules exist because switching a component must not silently
reuse another one's index.

Qdrant collections are `{strategy_id}__{embedder_id}`, plus a `corpus_id`
suffix. OpenSearch indices are `rag__{strategy_id}` with the same suffix rule.
Changing a strategy or a corpus therefore produces a new namespace rather than
mixing two vector spaces in one collection.

Neo4j is **not** partitioned by corpus: the graph is shared. Call
`Neo4jGraphRetriever.clear()` before ingesting a different corpus when isolation
matters.

## Test layers, and which one to reach for

| Command | Layer | Needs |
|---|---|---|
| `pytest tests/unit tests/contract -q` | logic in isolation | nothing |
| `pytest tests/fitness -q -m ""` | architectural invariants | nothing |
| `pytest tests/integration -q -m integration` | real backends | the Docker stack |
| `pytest tests/e2e -q -m e2e` | the full walkthrough | the stack |
| `pytest tests/eval -q -m deepeval` | answer quality | the stack, Ollama, and `pip install -e '.[judges]'` |

Write a unit test when the behaviour can be decided from inputs alone. Reach for
integration when the assertion is about how a real backend responds, and for e2e
only when the point is that several parts work together.

Integration tests skip themselves when their service is absent, which means a
green run does not prove they ran. `tests/fitness/test_suite_collects.py` exists
because of that: it imports every test file, including the suites that markers
filter out, so an import error in them surfaces on the ordinary round rather
than on the day somebody needs the suite.

A real Neo4j holding tens of thousands of nodes makes even a simple `MATCH`
noticeably slower than an empty test database. Integration tests on the graph
taking tens of seconds is normal rather than a hang.

## What "done" means

A change is not finished without a test at the layer that can actually decide
it, and `tests/unit/test_p1_guardian.py` still passing.

A test that pins presentation text is usually pinning the wrong thing. Assert on
the stable identifier a diagnostic carries rather than on the sentence it
renders.

A fixture and the dataset that queries it are one unit. Moving either alone
leaves a suite that still passes its collection step and measures nothing, which
has happened here: an evaluation suite seeded one corpus while its golden set
asked about another, and every retrieval metric would have come back zero.

## The interface

Four constraints, each with a test.

**Air-gapped.** No external CDN. Fonts and icons ship locally.

**WCAG AA.** Minimum contrast 4.5:1. Colours come from the tokens in
`styles.css`, never from a literal: a hex literal looks identical under all five
palettes and both themes, which is the same as having no theming at all.
`ui/src/test/designLanguage.test.ts` checks this, along with type scale, retired
class names, and a ratchet on inline styles.

**The gateway is the only backend.** The interface calls `/api/*` and never
reaches Qdrant, OpenSearch or Neo4j directly.

**The new-run form is built from the registry**, so a newly registered component
appears without an edit here.

Both locales must declare the same keys, and neither may carry the other's
language. `ui/src/test/i18nCoverage.test.ts` enforces both, after two guide
sections were found sitting in the English file written in Russian: the key
parity check had not noticed, because the keys were all present.

## Adding things

**A component** (chunker, embedder, retriever, reranker, grounder, route
policy): implement the protocol in `core/`, write the adapter in `adapters/`,
register it in `services/api_gateway/main.py`, add a contract test.

**An architecture** is not a component and takes five parts, and the first two
cost no code. *Find its point*: the coordinates come from the published schema
the neighbouring project keeps, and if the architecture is not there it is
added there and not here. *Ask what applies*: `python3 -m tools.atlas_report
--point <id>` prints which catalogue entries can occur in it, which
coordinates of the point no entry speaks about, and how many applicable
entries still carry a scope caveat, so the gap is visible before the work
starts. *Make the platform run it*: a protocol in `core/interfaces.py` if none
fits, the adapter and its stub, the registration, and the parameters it varies
by, which means fields in `core/experiment/config.py` **and** their
application in `core/experiment/runner.py`. A field accepted and never applied
is the failure this platform sells the detection of, and five of them lived
here. *Make the platform speak about it*: entries for the coordinates nothing
covers, each with its bait, and applicability on the signals that cannot say
anything there. *Stage it*: a corpus, a golden set and a configuration in the
proving ground, then the level-B pairs. Until those, an entry's state is "not
staged", which is neither caught nor uncaught.

**A domain pack**: a directory with `pack.yaml`, discovered by scan. It
contributes structure parsers, answer masks, refusal policies and error
taxonomies for one subject area. A pack carries one subject area's vocabulary in
one language; a pack for another language is a new pack.

**A page**: create it under `ui/src/pages/`, add the route and the nav entry in
`App.tsx`, write a test in `ui/src/test/`.

**A locale**: produce a file that passes `i18nCoverage`. Partial locales were
tried and removed, because a switcher offering a language that renders English
is worse than one that does not offer it.

## Prose

Comments and identifiers are English. The exceptions are deliberate and each
says so where it sits: prompt templates, refusal wording, a domain pack's own
vocabulary, and the tests whose subject is a Russian literal a parser matches.

Comments explain why a thing is the way it is rather than restating what the
code does. If a comment would only paraphrase the line below it, the line is
better named instead.

When a change fixes a defect found by running the system rather than by reading
it, say so and say what the symptom looked like. Those notes are how the next
person recognises the same failure.

`tools/check_style.py` counts two constructions this project removes on purpose:
an em dash used as a connector, which usually marks a missing verb, and "not X
but Y", which spends a clause on the half readers then remember. Neither is
mechanically fixable, so CI holds a ratchet rather than a gate: the count may
fall and may not rise.
