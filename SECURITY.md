# Security

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/laputski/causa-rag/security/advisories/new) rather
than as a public issue. You will get an acknowledgement within a few days.

## What this software is designed for

Causa runs on a developer's or a team's own machine and expects to sit behind
whatever boundary already protects that machine. It ships no authentication,
no authorisation and no rate limiting, and the API gateway binds `0.0.0.0` so
that a browser on the same host can reach it.

Exposing the gateway to a network you do not control would let anyone reach
every endpoint, including corpus ingestion and file upload. If you need that,
put an authenticating proxy in front of it and treat the platform as a trusted
backend rather than as a public service.

`RAG_HTTP_ALLOWLIST` restricts which URLs the external-RAG pipeline may call,
as a comma-separated list of prefixes. It is empty by default, which is right
for local development and wrong for anything reachable by others. Setting it
is the single most useful hardening step if the gateway is not alone on a
laptop.

## Credentials in this repository, and why they are not a leak

The compose stack carries several literal credentials. All of them are local
by construction and authenticate nothing beyond the machine running them:

- the Langfuse Postgres password, its `NEXTAUTH_SECRET` and `SALT`
- the `lf-pk-local-*` and `lf-sk-local-*` Langfuse keys, self-issued by that
  same stack at first start
- the `ragplatform` fallback password for Neo4j
- the `admin@example.com` initial Langfuse account and its password

They exist so that a first run needs no configuration. A secret scanner will
flag them, and that flag is a false positive. If you deploy any of these
services somewhere other people can reach, change them, and note that the
compose file is a development convenience rather than a deployment artefact.

Real secrets belong in `.env`, which is gitignored. `.env.example` documents
every variable and holds no values worth protecting.

## Data handling

Nothing leaves the host. Generation and LLM judging call a local Ollama,
embeddings are computed in-process, and every store runs in your own Docker.
The platform makes no outbound call other than to the model and storage
endpoints you configure.

Two exceptions, both opt-in and both visible in the code: `pip install` and
`npm install` reach their registries during setup, and the embedder weights
download once from Hugging Face. `./install.sh --skip-models` covers the
air-gapped case where those caches are pre-seeded.

## Scope

In scope: anything that lets a request reach data or execution it should not,
anything that writes outside the working tree unexpectedly, and any dependency
with a known exploitable vulnerability that this project actually reaches.

Out of scope: the absence of authentication, which is documented above and is
a design decision rather than a defect; the local credentials listed above; and
denial of service against a single-user local install.
