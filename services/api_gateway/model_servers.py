"""Which model server answers for a Realm, decided in one place.

A generator was built in five places, each reading the Realm's resources its
own way or not at all: the chat and the feedback triage asked the Realm's
server, and a run, the drafting of questions, the drafting of prompts and the
judge of assertions asked the one the gateway started with. All of them come
here now, so a Realm's server is either used by everything or by nothing.
"""
from __future__ import annotations

from typing import Any

from adapters.openai_compatible import PROVIDER as OPENAI_COMPATIBLE

__all__ = ["MODEL_RESOURCE_TYPES", "OPENAI_COMPATIBLE", "EmbedderMismatch", "embedder_for",
           "generator_for", "generator_for_realm", "model_resources", "provider_of"]

#: The resource types that are model servers. A Realm keeps at most one of
#: each, which is the platform's limit and not this module's.
MODEL_RESOURCE_TYPES = ("ollama", OPENAI_COMPATIBLE)


async def model_resources(realm_id: str | None) -> dict[str, dict[str, Any] | None]:
    """What the Realm declares for each kind of model server, secrets opened.
    Empty for no Realm, which every caller reads as "the gateway's own"."""
    if not realm_id:
        return {}
    from services.api_gateway.routers.corpus import _get_realm_resource
    return {kind: await _get_realm_resource(realm_id, kind) for kind in MODEL_RESOURCE_TYPES}


def provider_of(resources: dict[str, dict[str, Any] | None] | None, asked: str | None = None) -> str:
    """Which server a call uses when it does not say.

    The one the Realm declares, when it declares exactly one. With both, or
    with neither, the local one: a call that names no provider must not start
    sending text to a hosted service because a second resource was added.
    """
    if asked in MODEL_RESOURCE_TYPES:
        return str(asked)
    declared = [kind for kind in MODEL_RESOURCE_TYPES if (resources or {}).get(kind)]
    return declared[0] if len(declared) == 1 else "ollama"


def generator_for(
    resources: dict[str, dict[str, Any] | None] | None,
    *,
    model: str | None = None,
    provider: str | None = None,
    seed: int | None = None,
    num_ctx: int | None = None,
) -> Any:
    """A generator on the Realm's server, running `model` when one is named.

    `num_ctx` is the local server's own notion and is passed only to it.
    """
    if provider_of(resources, provider) == OPENAI_COMPATIBLE:
        from adapters.openai_compatible import OpenAICompatibleGenerator
        generator: Any = OpenAICompatibleGenerator(from_environment=True, seed=seed).for_realm(resources)
    else:
        from adapters.ollama_generator import OllamaGenerator
        generator = OllamaGenerator(seed=seed, num_ctx=num_ctx).for_realm(resources)
    return generator.with_model(model) if model else generator


class EmbedderMismatch(RuntimeError):
    """A corpus was filled by one embedder and the Realm now offers another."""


def embedder_for(embedder_id: str, resources: dict[str, dict[str, Any] | None] | None, local: Any) -> Any:
    """The embedder that answers to `embedder_id`, which is how the registry
    names the one a corpus was filled with.

    `local` is the gateway's own, returned for its own identifier. For one
    served over HTTP the Realm's server is asked, and it has to answer to the
    same identifier: a collection filled by one model must not be searched
    with another's vectors, so a Realm whose resource has since named another
    model is refused and told which model the corpus needs.
    """
    if not embedder_id.startswith(OPENAI_COMPATIBLE):
        return local
    from adapters.openai_compatible import OpenAICompatibleEmbedder
    embedder = OpenAICompatibleEmbedder(from_environment=True).for_realm(resources)
    if embedder.embedder_id != embedder_id:
        raise EmbedderMismatch(
            f"this corpus was loaded with {embedder_id!r} and the Realm's model server now offers "
            f"{embedder.embedder_id!r}; name the same embedding model in the resource, or load the "
            "corpus again")
    return embedder


async def generator_for_realm(realm_id: str | None, **chosen: Any) -> Any:
    return generator_for(await model_resources(realm_id), **chosen)
