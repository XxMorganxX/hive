"""The one entry point for choosing plug-in implementations by name (P5).

An experiment config names its agent adapter, suite and overseer as strings. This
module turns those strings into instances, so swapping an implementation is a
config change and the controller never imports a concrete plug-in.

A name is either a built-in short name (see `BUILTINS`) or an import path of the
form "package.module:Attr". The import path means a third-party plug-in works
without editing this file.

An agent adapter's `name` must equal the string it was resolved from, because that
string is what candidates write in `agent = "..."` and what the controller compares
with `adapter.name` (see `hive.contracts.AgentAdapter`). A third-party adapter is
therefore named by its own import path. The check runs here, where the adapter is
built, so a mismatch fails at once with the rule rather than later as a confusing
"candidate belongs to another adapter".
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from hive.contracts import AgentAdapter, Overseer, Suite

# Paths, not imports: a broken or slow plug-in must only fail the experiment that uses it.
BUILTINS: dict[str, dict[str, str]] = {
    "agent": {
        "pi": "hive.agents.pi:PiAdapter",
        "scripted": "hive.agents.scripted:ScriptedAdapter",
    },
    "suite": {
        "exact_answer": "hive.suites.exact_answer:ExactAnswerSuite",
        "python_functions": "hive.suites.python_functions:PythonFunctionsSuite",
    },
    "overseer": {
        "claude_code": "hive.overseers.claude_code:ClaudeCodeOverseer",
        "null": "hive.overseers.null:NullOverseer",
    },
}


class RegistryError(Exception):
    """A plug-in name could not be turned into a working instance."""


def _builtin_list(kind: str) -> str:
    return ", ".join(sorted(BUILTINS[kind])) or "(none)"


def _is_dotted_identifier(text: str) -> bool:
    return bool(text) and all(part.isidentifier() for part in text.split("."))


def _import_path(kind: str, path: str, label: str) -> type:
    """Import `path`; `label` is how errors refer to it (the name the config used)."""
    module_name, separator, attribute_path = path.partition(":")
    if not (separator and _is_dotted_identifier(module_name) and _is_dotted_identifier(attribute_path)):
        raise RegistryError(f"{kind} {label} is not a valid import path; expected 'package.module:Attr'")
    try:
        target: Any = importlib.import_module(module_name)
    except Exception as error:  # any failure inside the plug-in's module is that plug-in's fault
        raise RegistryError(f"{kind} {label}: could not import {module_name!r}: {error}") from error
    for attribute in attribute_path.split("."):
        try:
            target = getattr(target, attribute)
        except AttributeError as error:
            raise RegistryError(f"{kind} {label}: {module_name!r} has no attribute {attribute_path!r}") from error
    if not isinstance(target, type):
        raise RegistryError(f"{kind} {label} is not a class (got {type(target).__name__})")
    return target


def resolve(kind: str, name: str) -> type:
    """Return the class registered as `name` for `kind`, or the class an import path points to."""
    if kind not in BUILTINS:
        raise RegistryError(f"unknown plug-in kind {kind!r}; known kinds: {', '.join(sorted(BUILTINS))}")
    if not isinstance(name, str) or not name:
        raise RegistryError(f"{kind} name must be a non-empty string, got {name!r}")
    if ":" in name:
        return _import_path(kind, name, repr(name))
    if name not in BUILTINS[kind]:
        raise RegistryError(
            f"unknown {kind} {name!r}; built-in {kind}s: {_builtin_list(kind)} "
            "(or give an import path 'package.module:Attr')"
        )
    return _import_path(kind, BUILTINS[kind][name], f"{name!r} ({BUILTINS[kind][name]})")


def _make(kind: str, contract: type, name: str, settings: Mapping[str, Any] | None) -> Any:
    plugin_class = resolve(kind, name)
    # A copy, so a plug-in that mutates its settings can't change the experiment config.
    settings_copy = dict(settings) if settings is not None else {}
    try:
        instance = plugin_class(settings_copy)
    except Exception as error:
        raise RegistryError(f"{kind} {name!r} failed to construct: {error}") from error
    # runtime_checkable only checks that the members exist, not their signatures;
    # that is enough to catch a name pointing at the wrong kind of plug-in.
    if not isinstance(instance, contract):
        raise RegistryError(
            f"{kind} {name!r} ({plugin_class.__module__}.{plugin_class.__qualname__}) "
            f"does not satisfy the {contract.__name__} contract in hive.contracts"
        )
    return instance


def make_agent_adapter(name: str, settings: Mapping[str, Any] | None = None) -> AgentAdapter:
    adapter = _make("agent", AgentAdapter, name, settings)
    if adapter.name != name:
        raise RegistryError(
            f"agent {name!r} resolved to an adapter whose name is {adapter.name!r}; an adapter's name must be exactly "
            f'the string candidates use in agent = "...": the built-in short name for a built-in, or the import '
            f"path for any other, so set name = {name!r} on {type(adapter).__qualname__} (hive.contracts.AgentAdapter)"
        )
    return adapter


def make_suite(name: str, settings: Mapping[str, Any] | None = None) -> Suite:
    return _make("suite", Suite, name, settings)


def make_overseer(name: str, settings: Mapping[str, Any] | None = None) -> Overseer:
    return _make("overseer", Overseer, name, settings)
