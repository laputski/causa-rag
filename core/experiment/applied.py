"""What a configuration field does to the run it configures.

A field a form can set and nothing reads is the trap this whole proving
ground exists to find, and this platform had five of them at once. They were
found by reading, one at a time, years apart, and each was fixed alone. What
was missing was the question asked of every field at once: does anything read
this?

It could not be asked. Which fields the build applies was spread across a
builder and four rebuilding functions, none of which said what it applied, so
there was nothing to compare the configuration against.

So every field of `ExperimentConfig` is named in exactly one of the four
tables below, and no table is believed:

* a field said to be applied while building must change what gets built, and
  `describe_built` below is what that is observed with;
* a field said to be applied elsewhere names the test that observes it, and
  the pointer is resolved against what pytest actually collects;
* a field said to name the configuration, and a field said to be applied by
  nothing, must both leave what gets built untouched.

The last table is the one worth reading. A field in it promises something the
platform does not do, and it stays in the configuration because removing it
would change the fingerprint of every run ever stored.
"""
from __future__ import annotations

from typing import Any

# Attributes read off a built pipeline. Read by name and never required: a
# pipeline that has none of one is described without it, which is how a stub
# and an external system describe themselves beside an in-process run.
_OF_A_PIPELINE = (
    "pipeline_id", "_top_k", "_fetch_k", "_realm_id", "_url", "_corpus_id",
    "_external_pipeline_id", "_reranker_id", "_params", "_timeout",
)
_OF_A_RETRIEVER = (
    "retriever_id", "_corpus_id", "_realm_id", "_strategy_id", "_embedder_id",
    "_language", "_host", "_port", "_merge", "_alpha", "_rrf_k",
    "_graph_weight", "_hops",
)
# The retrievers a retriever is made of. A hybrid holds two, a graph pipeline
# holds a graph beside a base, and each of them can be any of the others.
_INSIDE_A_RETRIEVER = ("_dense", "_sparse", "_base", "_graph")
# The steps a configurable pipeline can carry, named as the configuration
# names them.
_STEPS = {
    "_reranker": "reranker", "_grounder": "grounding", "_route_policy": "route_policy",
    "_scorer": "scorer", "_mask_engine": "mask_engine", "_refusal_policy": "refusal_policy",
}


def _describe_thing(thing: Any) -> dict[str, Any]:
    """What a step is: its type, and the model it was built on where it has
    one. Two rerankers of the same class on different models rerank
    differently, and a description that could not tell them apart would let a
    field claiming to choose the model pass a check for changing what runs."""
    if thing is None:
        return {}
    out: dict[str, Any] = {"is": type(thing).__name__}
    for name in ("_model_name", "_model", "embedder_id", "generator_id", "version"):
        value = getattr(thing, name, None)
        if value not in (None, ""):
            out[name.lstrip("_")] = value
    return out


def describe_retriever(retriever: Any, depth: int = 0) -> dict[str, Any]:
    """What the retriever is made of, all the way down."""
    if retriever is None or depth > 4:
        return {}
    out: dict[str, Any] = {"is": type(retriever).__name__}
    for name in _OF_A_RETRIEVER:
        value = getattr(retriever, name, None)
        if value is not None:
            out[name.lstrip("_")] = value
    for name in _INSIDE_A_RETRIEVER:
        inside = getattr(retriever, name, None)
        if inside is not None:
            out[name.lstrip("_")] = describe_retriever(inside, depth + 1)
    return out


def describe_built(pipeline: Any) -> dict[str, Any]:
    """What the object that will do the work is made of.

    Read off the built pipeline and never off the configuration, which is the
    whole point: a field the build ignores leaves no trace here. That is what
    turns "this setting changes the run" from something claimed into something
    observed, and it is the same reading `ExperimentResult.applied` already
    does for the two embedders, carried out over every part.
    """
    if pipeline is None:
        return {}
    out: dict[str, Any] = {"is": type(pipeline).__name__}
    for name in _OF_A_PIPELINE:
        value = getattr(pipeline, name, None)
        if value is not None:
            out[name.lstrip("_")] = value
    out["retriever"] = describe_retriever(getattr(pipeline, "_retriever", None))
    for attribute, spoken_as in _STEPS.items():
        described = _describe_thing(getattr(pipeline, attribute, None))
        if described:
            out[spoken_as] = described
    for attribute, spoken_as in (("_embedder", "embedder"), ("_generator", "generator")):
        described = _describe_thing(getattr(pipeline, attribute, None))
        if described:
            out[spoken_as] = described
    return out


# ── what each field does ─────────────────────────────────────────────────────

APPLIED_WHILE_BUILDING: dict[str, str] = {
    "pipeline_id": "chooses which pipeline the run is built from",
    "pipeline_source": "chooses between a pipeline here and a system over HTTP",
    "http_endpoint": "the address of that system, when it is named inline",
    "external_rag_id": "the registered system to resolve that address from",
    "corpus_id": "binds every retriever to the corpus this run queries",
    "embedder": "the model the query is embedded with",
    "generator": "the component that writes the answer",
    "reranker": "the rerank step, and through its params the model that reranks",
    "grounding": "adds the grounding step",
    "route_policy": "adds the routing step",
    "scorer": "adds a domain pack's scoring step",
    "mask_engine": "adds a domain pack's masking step",
    "refusal_policy": "adds a domain pack's refusal step",
    "top_k": "how many fragments reach the answer",
    "fetch_k": "how many are fetched before the cut",
    "merge_strategy": "how a hybrid fuses its two lists",
    "merge_alpha": "the weight of that fusion",
    "rrf_k": "the rank-fusion constant",
    "graph_weight": "how much of the ranking a graph gets",
    "hops": "how far from a matched unit a graph may walk",
    "params": "the generation model here, and the declared knobs of a system over HTTP",
}

# A field applied somewhere other than while building the pipeline, and the
# observation that says so. The pointer is resolved against what pytest
# collects, on the rule the failure catalogue already follows: a claim naming
# a test nobody can run is the decorative claim it was meant to replace.
APPLIED_ELSEWHERE: dict[str, tuple[str, str]] = {
    "retrieval_only": (
        "the runner stops at retrieval and never reaches the generator",
        "tests/unit/test_what_a_configuration_field_does.py"
        "::test_a_retrieval_only_run_stops_before_the_generator",
    ),
    "retrieval_pins_enabled": (
        "the services layer skips the pin lookup entirely when it is off, so off "
        "costs nothing at all, where loading pins and ignoring them would cost "
        "a round trip and a scan",
        "tests/unit/test_retrieval_pins_disabled.py"
        "::test_pins_are_off_unless_a_run_asks_for_them",
    ),
    "dataset_name": (
        "names the question set a run was measured on, and the services layer "
        "loads that set again whenever it goes back to the run: to widen one "
        "question's search, and to compare two runs",
        "tests/unit/test_what_a_configuration_field_does.py"
        "::test_a_later_look_at_a_run_loads_the_question_set_it_names",
    ),
}

# Read by nothing that runs, on purpose. These name the configuration and
# say nothing about what it does, so leaving what gets built untouched is
# the correct behaviour and is checked as such.
NAMES_THE_CONFIGURATION: dict[str, str] = {
    "name": "what a reader calls this configuration",
    "version": "the version of the configuration itself",
    "dataset_version": "the version of the question set, recorded beside its name",
    "config_schema_version": "the shape this configuration was written in",
    "config_hash": "the fingerprint, computed from every other field",
    "external_rag_name": "backfilled after the run so a failure names the service",
}

# A field that promises something and does nothing. Each stays because a
# configuration's fingerprint is computed over its fields, so removing one
# would change the identity of every run ever stored.
NOT_APPLIED: dict[str, str] = {
    "chunking_strategy": (
        "a corpus is cut at load time and a run reads what the cut produced, so "
        "naming a strategy here selects nothing. The index a run really read is "
        "recorded as applied.index_chunking_strategy, and the corpus manifest "
        "records the one it was built with, which is what a comparison of the "
        "two reads. Applying it would mean binding the retriever to the index of "
        "the named strategy, and pointing a run at an index nobody built would "
        "return nothing while looking like a search that found nothing"
    ),
    "retrievers": (
        "the pipeline decides which retriever it holds, and this list sits "
        "beside that decision without changing it. Recorded and shown, so what "
        "it says has to match what the pipeline holds, which is what the new-run "
        "form was corrected to read from the registry"
    ),
    "seed": (
        "nothing anywhere reads it. It says a run is reproducible by repetition, "
        "and no part of the platform fixes any source of randomness: a generator "
        "samples as its own settings say, and running the same configuration "
        "twice can give two answers"
    ),
}
