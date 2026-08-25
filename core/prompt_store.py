"""Prompt template store — file-based, versioned.

Prompts live in prompts/*.json. One prompt has is_active=true per realm_id
(see PromptStore.get_active/set_active) — a generic scoping key, not a
persistence dependency, so this stays inside core/'s domain-neutral boundary the same
way ExperimentConfig's own realm_id field does.

Found live: activation used to be a single GLOBAL is_active flag with no
realm_id concept in the file schema at all (only Mongo carried realm_id,
bolted on by services/api_gateway/routers/prompts.py) — activating a prompt
in one Realm silently deactivated another Realm's own active prompt, since
core/pipeline.py's get_active() had nothing to filter by. Concretely: one
Realm's chat started answering through a second Realm's prompt the moment
someone activated it on that Realm's Prompts page. Fixed by making
realm_id part of the file record itself and scoping both read (get_active)
and write (set_active) to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PromptTemplate:
    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.name: str = data["name"]
        self.version: int = data["version"]
        self.description: str = data.get("description", "")
        self.is_active: bool = data.get("is_active", False)
        self.created_at: str = data.get("created_at", "")
        self.template: str = data["template"]
        # None = no Realm recorded (pre-migration prompt, or a from-scratch
        # install's first prompt before any Realm exists) — deliberately
        # distinct from "" so it never accidentally matches a real realm_id.
        self.realm_id: str | None = data.get("realm_id")

    def render(self, query: str, context_chunks: list[str]) -> str:
        # "Fragment N" as the label, and not "[N]": a bracketed index reads too
        # close to a citation marker and gets confused with a real numbered unit
        # appearing inside the chunk text itself, which is what an earlier
        # bracketed form did.
        #
        # The label used to be the Cyrillic "Фрагмент". core/citation.py matches
        # both scripts, so either works, and the Latin form is the one an
        # English installation should be putting in front of the model. It also
        # removes the cause of the mixed-script artifact core/citation.py
        # documents: a model given a Cyrillic label sometimes answered with
        # "Фragment", one script's first letter on the other's word.
        context = "\n\n".join(f"Fragment {i + 1}:\n{c}" for i, c in enumerate(context_chunks))
        return self.template.replace("{context}", context).replace("{query}", query)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "template": self.template,
            "realm_id": self.realm_id,
        }


class PromptStore:
    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        self._dir = prompts_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[PromptTemplate]:
        result = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                result.append(PromptTemplate(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                pass
        return sorted(result, key=lambda t: t.version)

    def sync_from(self, docs: list[dict[str, Any]]) -> list[str]:
        """Bring the files into line with Mongo and return the ids written.

        Prompts live in two places: in Mongo, which the screens write to, and in
        files, which the pipeline reads synchronously because it has no async
        access. Creating a prompt writes to both. Everything else (realm import,
        migration, an edit made outside the gateway) writes to Mongo alone, and
        no file appears.

        What that led to was visible on a live installation: one
        `demo_prompt_v1.json` in the directory against six prompts across three
        realms in Mongo. For a realm with no file of its own, `get_active` fell
        through to its last resort, the one that prefers anything over nothing,
        and returned the demo prompt. So every realm's chat and runs answered
        through the demo's prompt while the prompts screen showed something
        else, and the only way those two could disagree was silently.

        Creates what is missing and leaves what exists alone. Overwriting would
        be the more logical rule right up until the first run:
        `demo_prompt_v1.json` is in the repository, and the gateway would start
        editing the working tree on every boot. A prompt that already has a file
        is served correctly anyway, because an edit through the screen writes to
        both places at once.
        """
        written: list[str] = []
        for doc in docs:
            pid = doc.get("id")
            if not pid:
                continue
            path = self._dir / f"{pid}.json"
            if path.exists():
                continue
            data = {k: v for k, v in doc.items() if k != "_id"}
            self._dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(pid)
        return written

    def get(self, prompt_id: str) -> PromptTemplate | None:
        p = self._dir / f"{prompt_id}.json"
        if not p.exists():
            return None
        return PromptTemplate(json.loads(p.read_text(encoding="utf-8")))

    def get_active(self, realm_id: str | None = None) -> PromptTemplate | None:
        """Return the active prompt for this realm_id, or a sensible fallback.

        realm_id=None (no Realm context — a from-scratch/pre-Realm caller)
        keeps the old global behavior: highest is_active, else highest
        version, across every prompt regardless of realm_id.

        With a realm_id: prefers that realm's own is_active prompt. Falls
        back to that realm's own highest version (no active one set yet —
        e.g. its first prompt was never explicitly activated), then to the
        old global resolution as a last resort (a realm with zero prompts of
        its own — same "something is better than a hard failure" reasoning
        the rest of the pipeline uses everywhere else).
        """
        templates = self.list()
        if realm_id is None:
            for t in reversed(templates):
                if t.is_active:
                    return t
            return templates[-1] if templates else None

        own = [t for t in templates if t.realm_id == realm_id]
        for t in reversed(own):
            if t.is_active:
                return t
        if own:
            return own[-1]
        return self.get_active(None)

    def save(self, data: dict[str, Any]) -> PromptTemplate:
        pt = PromptTemplate(data)
        p = self._dir / f"{pt.id}.json"
        p.write_text(json.dumps(pt.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return pt

    def set_active(self, prompt_id: str) -> PromptTemplate:
        """Activate the given prompt, deactivating only OTHER prompts that
        share its own realm_id — a different Realm's active prompt is left
        untouched (this is the fix; see module docstring for the incident).
        Prompts with realm_id=None (pre-migration) are their own group, same
        as any real realm_id — mutually exclusive among themselves, but
        without affecting a Realm-tagged prompt or vice versa.
        """
        target = self.get(prompt_id)
        if target is None:
            raise ValueError(f"Prompt {prompt_id!r} not found")
        for t in self.list():
            if t.realm_id != target.realm_id:
                continue
            t.is_active = (t.id == prompt_id)
            self.save(t.to_dict())
        return self.get(prompt_id)  # type: ignore[return-value]

    def delete(self, prompt_id: str) -> None:
        p = self._dir / f"{prompt_id}.json"
        if not p.exists():
            raise FileNotFoundError(f"Prompt {prompt_id!r} not found")
        p.unlink()


prompt_store = PromptStore()
