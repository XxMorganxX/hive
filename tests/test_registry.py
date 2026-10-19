import pytest

import registry_fixtures
from hive import registry
from hive.registry import RegistryError, make_agent_adapter, make_overseer, make_suite, resolve

FIXTURES = "registry_fixtures"


def test_builtins_name_every_shipped_plugin():
    assert registry.BUILTINS == {
        "agent": {"pi": "hive.agents.pi:PiAdapter", "scripted": "hive.agents.scripted:ScriptedAdapter"},
        "suite": {
            "exact_answer": "hive.suites.exact_answer:ExactAnswerSuite",
            "python_functions": "hive.suites.python_functions:PythonFunctionsSuite",
        },
        "overseer": {
            "claude_code": "hive.overseers.claude_code:ClaudeCodeOverseer",
            "null": "hive.overseers.null:NullOverseer",
        },
    }


def test_resolve_import_path_returns_the_class():
    assert resolve("agent", f"{FIXTURES}:FakeAdapter") is registry_fixtures.FakeAdapter


def test_resolve_import_path_follows_dotted_attribute():
    assert resolve("suite", f"{FIXTURES}:Outer.NestedSuite") is registry_fixtures.Outer.NestedSuite


def test_resolve_builtin_name_uses_the_builtins_table(monkeypatch):
    monkeypatch.setitem(registry.BUILTINS["suite"], "fake", f"{FIXTURES}:FakeSuite")
    assert resolve("suite", "fake") is registry_fixtures.FakeSuite


def test_unknown_kind_lists_known_kinds():
    with pytest.raises(RegistryError, match=r"agent.*overseer.*suite"):
        resolve("judge", "anything")


def test_unknown_builtin_name_lists_builtins_for_that_kind():
    with pytest.raises(RegistryError, match=r"exact_answer.*python_functions"):
        resolve("suite", "no_such_suite")


def test_unknown_builtin_name_mentions_import_path_form():
    with pytest.raises(RegistryError, match="module:Attr"):
        resolve("suite", "no_such_suite")


def test_missing_module_raises_registry_error():
    with pytest.raises(RegistryError, match="no_such_module_xyz"):
        resolve("agent", "no_such_module_xyz:Thing")


def test_missing_attribute_raises_registry_error():
    with pytest.raises(RegistryError, match="NoSuchClass"):
        resolve("agent", f"{FIXTURES}:NoSuchClass")


def test_module_that_fails_while_importing_raises_registry_error(tmp_path, monkeypatch):
    (tmp_path / "plugin_that_crashes_on_import.py").write_text("raise RuntimeError('boom at import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="boom at import"):
        resolve("agent", "plugin_that_crashes_on_import:Thing")


@pytest.mark.parametrize("path", ["registry_fixtures:", ":FakeAdapter", "a:b:c", "bad module:X"])
def test_malformed_import_path_raises_registry_error(path):
    with pytest.raises(RegistryError):
        resolve("agent", path)


def test_attribute_that_is_not_a_class_raises_registry_error():
    with pytest.raises(RegistryError, match="not a class"):
        resolve("agent", f"{FIXTURES}:NOT_A_CLASS")


def test_make_agent_adapter_passes_settings_to_constructor():
    adapter = make_agent_adapter(f"{FIXTURES}:FakeAdapter", {"timeout_s": 30})
    assert adapter.settings == {"timeout_s": 30}


def test_make_agent_adapter_without_settings_passes_empty_mapping():
    adapter = make_agent_adapter(f"{FIXTURES}:FakeAdapter")
    assert adapter.settings == {}


def test_settings_passed_to_plugin_are_a_copy():
    settings = {"timeout_s": 30}
    adapter = make_agent_adapter(f"{FIXTURES}:FakeAdapter", settings)
    adapter.settings["timeout_s"] = 1
    assert settings == {"timeout_s": 30}


def test_make_agent_adapter_rejects_class_without_adapter_contract():
    with pytest.raises(RegistryError, match="AgentAdapter"):
        make_agent_adapter(f"{FIXTURES}:SatisfiesNothing")


def test_make_suite_returns_instance_with_settings():
    suite = make_suite(f"{FIXTURES}:FakeSuite", {"limit": 3})
    assert isinstance(suite, registry_fixtures.FakeSuite)
    assert suite.settings == {"limit": 3}


def test_make_suite_rejects_an_overseer_class():
    with pytest.raises(RegistryError, match="Suite"):
        make_suite(f"{FIXTURES}:FakeOverseer")


def test_make_overseer_returns_instance_with_settings():
    overseer = make_overseer(f"{FIXTURES}:FakeOverseer", {"model": "sonnet"})
    assert isinstance(overseer, registry_fixtures.FakeOverseer)
    assert overseer.settings == {"model": "sonnet"}


def test_make_overseer_rejects_class_without_overseer_contract():
    with pytest.raises(RegistryError, match="Overseer"):
        make_overseer(f"{FIXTURES}:FakeSuite")


def test_constructor_failure_becomes_registry_error_with_cause():
    with pytest.raises(RegistryError, match="timeout_s must be positive"):
        make_agent_adapter(f"{FIXTURES}:RejectsSettings")


def test_agent_adapter_named_by_its_import_path_is_accepted():
    adapter = make_agent_adapter(f"{FIXTURES}:NamedByImportPath")
    assert adapter.name == f"{FIXTURES}:NamedByImportPath"


def test_agent_adapter_whose_name_differs_from_the_key_it_was_resolved_from_is_rejected():
    # Candidates say agent = "<key>", and evaluate/validate compare that with adapter.name, so a
    # mismatch would only surface later as a confusing "candidate belongs to another adapter".
    with pytest.raises(RegistryError, match=r"name is 'tiny'.*agent = .*name = 'registry_fixtures:NamedShort'"):
        make_agent_adapter(f"{FIXTURES}:NamedShort")


def test_builtin_agent_adapters_are_named_by_their_builtin_key():
    for key in registry.BUILTINS["agent"]:
        assert make_agent_adapter(key).name == key
