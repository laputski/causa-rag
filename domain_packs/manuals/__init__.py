"""Technical manuals: the worked example pack shipped with the platform.

It shows exactly what packs exist for. The platform does not know what a
section heading looks like in a particular subject area, what separates a
question about a procedure from a question about safety, or what to say when
there is nothing to ground an answer in. The pack knows all of that, and the
core knows nothing about the pack.

The subject area was not chosen at random: the platform's demo realm is a
technical handbook, so this pack can be switched on there and its effect seen.
An example that matches nothing anybody runs does not work as an example.

One caveat about language, and it is worth reading before switching the pack on
over the demo. The question-type keywords in `routing.py` carry English and
Russian, so classification works on the demo's English questions. The answer
masks and the refusal wording are Russian, because they are answer text rather
than interface text: enabled over an English corpus, this pack appends Russian
source labels and disclaimers to English answers.

That is the deliberate position on packs, not an oversight. A pack carries one
subject area's vocabulary in one language, and a pack for another language is a
new pack, which is what makes them pluggable. Copying `templates/answer_masks.yaml`
and `refusal.py` into a pack of your own is the intended route.
"""
from __future__ import annotations

import pathlib
from typing import Any

_MASKS_PATH = pathlib.Path(__file__).parent / "templates" / "answer_masks.yaml"


def load_mask_engine() -> Any:
    import yaml

    from core.answer.mask_engine import MaskEngine

    with _MASKS_PATH.open(encoding="utf-8") as f:
        return MaskEngine.from_dict(yaml.safe_load(f))


def register(registry: Any, settings: dict[str, Any]) -> None:
    from domain_packs.manuals.refusal import ManualsRefusalPolicy
    from domain_packs.manuals.routing import ManualsRoutePolicy
    from domain_packs.manuals.structure_parser import parse_manual_section

    registry.register("structure_parser", "manual_section", parse_manual_section)
    registry.register("mask_engine", "manuals", load_mask_engine())

    route_policy = ManualsRoutePolicy()
    registry.register("route_policy", route_policy.policy_id, route_policy)

    refusal = ManualsRefusalPolicy()
    registry.register("refusal", refusal.policy_id, refusal)

    # The general "what this pack contributes" view: generic code (the /query
    # handler, feedback triage) asks by pack id and knows nothing about the
    # names of that pack's components.
    registry.register("domain_hooks", "manuals", {
        "route_policy": route_policy,
        "mask_engine": registry.resolve("mask_engine", "manuals"),
        "refusal_policy": refusal,
    })
