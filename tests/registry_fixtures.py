"""Small plug-in classes for the registry tests.

The real built-in plug-ins are written separately, so the registry is tested by
resolving these through the "module:Attr" path instead. Each class records the
settings it was constructed with so tests can check what the registry passed in.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class FakeAdapter:
    name = "fake"

    def __init__(self, settings: Mapping[str, Any] | None = None):
        self.settings = settings

    def validate(self, candidate):
        return []

    def build(self, candidate, model, settings):
        raise NotImplementedError

    def describe_valves(self) -> str:
        return ""


class FakeSuite:
    name = "fake"

    def __init__(self, settings: Mapping[str, Any] | None = None):
        self.settings = settings

    def describe(self) -> str:
        return ""

    def metric(self):
        raise NotImplementedError

    def tasks(self, split):
        return []

    def grade(self, task, result, workdir: Path):
        raise NotImplementedError


class FakeOverseer:
    name = "fake"
    proposes_changes = False

    def __init__(self, settings: Mapping[str, Any] | None = None):
        self.settings = settings

    def propose(self, brief, workspace: Path):
        raise NotImplementedError


class SatisfiesNothing:
    """Constructible like a plug-in but implements no contract."""

    def __init__(self, settings: Mapping[str, Any] | None = None):
        self.settings = settings


class RejectsSettings:
    def __init__(self, settings: Mapping[str, Any] | None = None):
        raise ValueError("timeout_s must be positive")


class Outer:
    class NestedSuite(FakeSuite):
        pass


NOT_A_CLASS = 42
