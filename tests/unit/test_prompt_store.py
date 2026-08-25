"""core/prompt_store.py — Realm-scoped prompt activation.

Found live: activation used to be a single GLOBAL is_active flag with no
realm_id in the file schema at all — activating a prompt in one Realm
silently deactivated another Realm's own active prompt: one Realm's chat
started answering through a second Realm's prompt the moment someone activated
it on that second Realm's Prompts page. These pin the fix: get_active/set_active are now
scoped by realm_id, a Realm-tagged prompt never touches another Realm's.
"""
from __future__ import annotations

from core.prompt_store import PromptStore


def _tpl(id_, version, realm_id=None, is_active=False):
    return {
        "id": id_, "name": id_, "version": version, "template": f"t-{id_} {{context}} {{query}}",
        "is_active": is_active, "realm_id": realm_id,
    }


def test_activating_a_prompt_in_one_realm_does_not_deactivate_another_realms(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("handbook_v4", 4, realm_id="demo", is_active=True))
    store.save(_tpl("acme_v1", 1, realm_id="acme", is_active=False))

    store.set_active("acme_v1")

    assert store.get("handbook_v4").is_active is True
    assert store.get("acme_v1").is_active is True


def test_activating_a_second_prompt_in_the_same_realm_deactivates_the_first(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("handbook_v1", 1, realm_id="demo", is_active=True))
    store.save(_tpl("handbook_v2", 2, realm_id="demo", is_active=False))

    store.set_active("handbook_v2")

    assert store.get("handbook_v1").is_active is False
    assert store.get("handbook_v2").is_active is True


def test_get_active_scoped_to_realm_returns_that_realms_own_active_prompt(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("handbook_v4", 4, realm_id="demo", is_active=True))
    store.save(_tpl("acme_v1", 1, realm_id="acme", is_active=True))

    assert store.get_active("demo").id == "handbook_v4"
    assert store.get_active("acme").id == "acme_v1"


def test_get_active_no_realm_context_keeps_old_global_behavior(tmp_path):
    """Back-compat: a caller with no Realm (realm_id=None) still resolves
    the single highest is_active prompt across everything, unchanged."""
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("handbook_v3", 3, realm_id="demo", is_active=False))
    store.save(_tpl("handbook_v4", 4, realm_id="demo", is_active=True))

    assert store.get_active(None).id == "handbook_v4"


def test_get_active_falls_back_to_own_realms_highest_version_when_none_active(tmp_path):
    """A Realm's first prompt was never explicitly activated — still prefer
    THAT Realm's own highest version over a different Realm's active one."""
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("handbook_v4", 4, realm_id="demo", is_active=True))
    store.save(_tpl("acme_v1", 1, realm_id="acme", is_active=False))

    assert store.get_active("acme").id == "acme_v1"


def test_get_active_falls_back_to_global_when_realm_has_no_prompts_of_its_own(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("handbook_v4", 4, realm_id="demo", is_active=True))

    assert store.get_active("brand-new-realm").id == "handbook_v4"


def test_get_active_no_prompts_at_all_returns_none(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    assert store.get_active("acme") is None
    assert store.get_active(None) is None


def test_pre_migration_prompts_with_no_realm_id_are_their_own_group(tmp_path):
    """realm_id=None prompts (pre-migration files) stay mutually exclusive
    among themselves without being scoped by/affecting a real realm_id."""
    store = PromptStore(prompts_dir=tmp_path)
    store.save(_tpl("legacy_v1", 1, realm_id=None, is_active=True))
    store.save(_tpl("legacy_v2", 2, realm_id=None, is_active=False))
    store.save(_tpl("acme_v1", 1, realm_id="acme", is_active=True))

    store.set_active("legacy_v2")

    assert store.get("legacy_v1").is_active is False
    assert store.get("legacy_v2").is_active is True
    assert store.get("acme_v1").is_active is True  # untouched


def test_to_dict_round_trips_realm_id(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    t = store.save(_tpl("acme_v1", 1, realm_id="acme"))
    assert t.to_dict()["realm_id"] == "acme"
    reloaded = store.get("acme_v1")
    assert reloaded.realm_id == "acme"


def test_set_active_unknown_prompt_raises(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    try:
        store.set_active("does-not-exist")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
