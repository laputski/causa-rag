"""The prompt files mirror Mongo, or the pipeline answers with another realm's prompt.

Prompts live in two places: in Mongo, which the screens write to, and in files,
which the pipeline reads synchronously. Creating one writes to both; realm
import, migration and any edit made outside the gateway write to Mongo alone.

On a live installation it looked like this: one `demo_prompt_v1.json` in the
directory against six prompts across three realms in Mongo, and every realm's
chat answering through the demo's prompt while the prompts screen showed
something else.
"""
from __future__ import annotations

import json

from core.prompt_store import PromptStore

DOC = {
    "id": "prompt_v1", "name": "Technical assistant", "version": 1,
    "description": "", "is_active": True, "created_at": "2026-01-01T00:00:00Z",
    "template": "Answer from the fragments: {context}\n\nQuestion: {query}",
    "realm_id": "acme",
}


# @lat: [[prompts#Prompt store#Файл отражает базу, а не наоборот]]
def test_a_prompt_only_in_mongo_gets_its_file(tmp_path) -> None:
    store = PromptStore(tmp_path)
    assert store.sync_from([DOC]) == ["prompt_v1"]
    assert json.loads((tmp_path / "prompt_v1.json").read_text(encoding="utf-8")) == DOC


# @lat: [[prompts#Prompt store#Файл отражает базу, а не наоборот]]
def test_after_the_sync_the_realm_gets_its_own_prompt_not_a_stranger_s(tmp_path) -> None:
    store = PromptStore(tmp_path)
    other = {**DOC, "id": "demo_prompt_v1", "realm_id": "demo", "name": "Demo"}
    store.sync_from([other])
    # Before the sync this realm has no file of its own, and the last-resort
    # fallback hands back another realm's active prompt, which is the defect.
    assert store.get_active("acme").id == "demo_prompt_v1"

    store.sync_from([DOC, other])
    assert store.get_active("acme").id == "prompt_v1"
    # And the other realm does not lose its own prompt in the process.
    assert store.get_active("demo").id == "demo_prompt_v1"


# @lat: [[prompts#Prompt store#Файл отражает базу, а не наоборот]]
def test_an_unchanged_file_is_not_rewritten(tmp_path) -> None:
    store = PromptStore(tmp_path)
    store.sync_from([DOC])
    assert store.sync_from([DOC]) == []


# @lat: [[prompts#Prompt store#Файл отражает базу, а не наоборот]]
def test_an_existing_file_is_left_alone(tmp_path) -> None:
    # Creates what is missing and leaves what exists alone: demo_prompt_v1.json
    # is in the repository, and overwriting would mean the gateway edits the
    # working tree on every start.
    store = PromptStore(tmp_path)
    store.sync_from([DOC])
    assert store.sync_from([{**DOC, "template": "other {context} {query}"}]) == []
    assert "other" not in store.get("prompt_v1").template


# @lat: [[prompts#Prompt store#Файл отражает базу, а не наоборот]]
def test_mongo_s_own_key_does_not_travel_into_the_file(tmp_path) -> None:
    # `_id` is a storage detail. Nobody needs it in the file, and it breaks the
    # "the file already matches" comparison.
    store = PromptStore(tmp_path)
    store.sync_from([{**DOC, "_id": "6a88211c26c4c7e7fbfa07d7"}])
    assert "_id" not in json.loads((tmp_path / "prompt_v1.json").read_text(encoding="utf-8"))
