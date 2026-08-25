"""Single source of truth for which Ollama model acts as the LLM judge
across the deepeval/ragas/trulens/chunk-coherence eval runners.

Found live: the judge model used to be a literal string
("qwen2.5:7b-instruct-q4_K_M") duplicated across 6 files (the four runners
below plus both eval test files) — a model never actually pulled locally,
so `make test-eval` always skipped with "judge model not loaded". Two of
the six already read a `DEEPEVAL_JUDGE_MODEL` env var override, four didn't
— changing the judge model meant editing up to 6 places and could easily
land inconsistent. This module is now the one place: change
DEFAULT_JUDGE_MODEL below, or set DEEPEVAL_JUDGE_MODEL to override per-run
without touching code at all.

At the time this module was introduced, none of the models actually pulled
locally were true non-thinking instruct models — every one was qwen3-family,
which "thinks" by default. Where the underlying client supports disabling it
(eval/ragas_runner.py via langchain_ollama's `reasoning=False`, the same fix
an external RAG applies in its own generator; eval/chunk_coherence_judge.py via
adapters/ollama_generator.py's own `think: False`), it's disabled regardless
of which model judges. Where it isn't (eval/deepeval_runner.py — deepeval's
OllamaModel has no passthrough for it at all, see its own module docstring),
the model choice is the only lever — and qwen3:8b (the smallest/fastest
qwen3 available) still weren't enough: confirmed live, a single deepeval
judge call on qwen3:8b exceeded even a 420s per-attempt timeout.

`qwen2.5:7b-instruct-q4_K_M` — pulled specifically to fix this — is the
default now: a genuinely non-thinking instruct model, so deepeval's judge
calls run in the ~seconds range like everything else, not minutes. If it's
ever removed, fall back to qwen3:8b (or another general-purpose model) and
expect deepeval specifically to be slow again for the reason above; prefer
a real non-thinking model over that fallback whenever one is available.
Deliberately still not one of the locally-present domain fine-tunes even
as a fallback, since those would bias judging toward their own domain
rather than judging generically.
"""
from __future__ import annotations

import os

DEFAULT_JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
