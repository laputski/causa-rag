# Contributing

Thanks for looking. This file covers how to get the stack running, which test
layer to use for what, and where the extension points are.

## Getting set up

```bash
git clone https://github.com/laputski/causa-rag && cd causa-rag
./install.sh
```

If you only want to run unit tests and linters, you can skip the heavy machine
learning dependencies:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

That must pass with nothing running and no extras installed. If it does not on
a clean checkout, that is a bug worth reporting on its own.

`make doctor` reports the state of everything and says what to do about each
failure. Pasting its output into an issue usually answers the first three
questions a maintainer would ask.

## Test layers

| Command | Layer | Needs | In CI |
|---|---|---|---|
| `make test` | unit and contract | nothing | yes |
| `make test-int` | integration | the Docker stack | no |
| `make test-e2e` | full walkthrough over the demo realm | the stack and Ollama | no |
| `make test-eval` | LLM-judge evaluation | the stack, Ollama, the judge model, and `pip install -e '.[judges]'` | no |
| `make test-all` | everything, with an HTML report | all of the above plus `.[report]` | no |

CI runs the layers that need nothing but a checkout. The rest are run by hand,
against a real stack:

```bash
make infra        # Qdrant, OpenSearch, MongoDB, Redis, Langfuse
make test-int
make test-e2e
```

Run those before opening a pull request that touches retrieval, ingestion or the
run pipeline. A CI runner can start the same containers, but not Ollama, so
generation there falls back to the stub and every assertion about answer quality
skips itself: green would mean less than it looks. On your own machine the same
suites measure the real thing.

Write a unit test when the behaviour can be decided from inputs alone. Reach
for integration when the assertion is about how a real backend responds, and
for e2e only when the point is that several parts work together.

A test that pins presentation text is usually pinning the wrong thing. Assert
on the stable identifier a diagnostic carries rather than on the sentence it
renders.

## Extension points

**Component registry.** Chunkers, embedders, retrievers, rerankers, grounders
and route policies are registered by id at startup in
`services/api_gateway/main.py` and resolved from an `ExperimentConfig`. Adding
one means implementing the protocol in `core/`, writing the adapter in
`adapters/`, and registering it. The protocols live in `core/models.py` and the
respective `core/` modules.

**Domain packs.** A pack contributes structure parsers, answer masks, refusal
policies and error taxonomies for one subject area, discovered by scanning
`domain_packs/` and any directory named in `CAUSA_DOMAIN_PACKS`.
`domain_packs/manuals/` is the worked example, and it is the subject area of
the demo realm, so it can be switched on and its effect seen. A pack carrying
one installation's own vocabulary belongs in that installation's tree, not
here.

**External RAG contract.** Any system reachable over HTTP can be scored by the
same metrics as the built-in pipeline. `docs/external-rag-contract.md` is the
field-by-field reference and `docs/connector-guide.md` covers the Python SDK
route. Two tiers exist: implement the native JSON shape, or declare a JSONPath
mapping from whatever shape you already return.

**Locales.** `ru` is the source language and `en` is a full translation of it.
Those two are what the switcher offers, and `ui/src/i18n/SUPPORTED_LANGUAGES` is
the list. Five more files sit in `ui/src/i18n/locales/` (`be`, `de`, `es`, `fr`,
`zh`) covering 176 keys of two thousand; they are not offered, because a
switcher listing a language that renders English is worse than one that does not
list it.

`ui/src/test/i18nCoverage.test.ts` is what a new language has to pass. It checks
three things: that the key sets match once plural suffixes are stripped, that
every key used in the source resolves, and that a listed language covers at
least 95% of the keys. The suffixes are stripped on purpose, because Russian has
`_few` and `_many` where English has `_other`, and demanding identical suffixes
would demand Russian grammar of every other language.

## Conventions

Comments and identifiers are English. Prose explains why a thing is the way it
is rather than restating what the code does; if a comment would only paraphrase
the line below it, the line is better named instead.

When a change fixes a defect that was found by running the system rather than
by reading it, say so in the commit message and say what the symptom looked
like. Those notes are how the next person recognises the same failure.

Commits are scoped to one change. A pull request that renames things and also
changes behaviour is two pull requests.

## Reporting a defect

Open an issue with `make doctor` output, what you expected, and what happened.
If it involves a run, the run id and the dataset name are usually enough for a
maintainer to reproduce against the demo corpus.

For anything security-sensitive, read [SECURITY.md](SECURITY.md) first.
