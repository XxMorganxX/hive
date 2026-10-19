"""Test-local overseers and suites for the loop tests, loaded through the registry.

Each class is named in an experiment config as "loop_fixtures:ClassName", exactly as
a third-party plug-in would be, so the loop is tested through hive.registry and not
by injecting objects. None of them calls a model or a binary.

The improving and worsening overseers edit the scripted agent's responses.toml. To
know a correct answer they read the exact_answer suite's own task file: a real
overseer must never do that, but these are test doubles standing in for "an overseer
that found a real improvement".
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import os
import threading
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hive.agents.scripted.adapter import ScriptedAdapter
from hive.candidate import Candidate
from hive.contracts import (
    AgentResult,
    ExtensionSpec,
    Grade,
    LifecyclePoint,
    MetricSpec,
    OverseerBrief,
    Proposal,
    Task,
    Usage,
)
from hive.loop import SKIPPED_EVIDENCE_FILE
from hive.overseers.claude_code.overseer import build_workspace
from hive.suites.exact_answer.suite import BUNDLED_TASKS_FILE

RESPONSES = "responses.toml"


def exact_answers() -> dict[str, str]:
    """Task id -> one accepted answer, from the exact_answer suite's bundled data."""
    rows = [json.loads(line) for line in BUNDLED_TASKS_FILE.read_text().splitlines() if line.strip()]
    return {row["id"]: row["answers"][0] for row in rows}


def _toml_string(text: str) -> str:
    # JSON string escapes (\", \\, \n, \uXXXX) are all valid TOML basic-string escapes.
    return json.dumps(text)


def write_responses(path: Path, default: str, responses: Mapping[str, str]) -> None:
    lines = [f"default = {_toml_string(default)}", "", "[responses]"]
    lines += [f"{_toml_string(task_id)} = {_toml_string(text)}" for task_id, text in sorted(responses.items())]
    path.write_text("\n".join(lines) + "\n")


def read_responses(root: Path) -> tuple[str, dict[str, str]]:
    data = tomllib.loads((root / RESPONSES).read_text())
    return data.get("default", ""), dict(data.get("responses", {}))


def failing_task_ids(report_path: Path) -> list[str]:
    report = json.loads(Path(report_path).read_text())
    return [task["task_id"] for task in report["tasks"] if task["selection_reward"] < 1.0]


def passing_task_ids(report_path: Path) -> list[str]:
    report = json.loads(Path(report_path).read_text())
    return [task["task_id"] for task in report["tasks"] if task["selection_reward"] == 1.0]


class _BaseOverseer:
    name = "fixture"
    proposes_changes = True
    cost_usd = 0.0

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = dict(settings or {})

    def _copy(self, brief: OverseerBrief, workspace: Path) -> Candidate:
        return brief.incumbent.copy_to(Path(workspace) / "candidate")


class ImprovingOverseer(_BaseOverseer):
    """Fixes the first failing search task, so the child scores one task higher."""

    name = "improving"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        child = self._copy(brief, workspace)
        default, responses = read_responses(child.root)
        answers = exact_answers()
        task_id = failing_task_ids(brief.incumbent_report)[0]
        responses[task_id] = f"ANSWER: {answers[task_id]}"
        write_responses(child.root / RESPONSES, default, responses)
        return Proposal(candidate_dir=child.root, rationale=f"fixed {task_id}", cost_usd=self.cost_usd)


class WorseningOverseer(_BaseOverseer):
    """Breaks the first passing search task. Deterministic, so a repeat proposes the same child."""

    name = "worsening"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        child = self._copy(brief, workspace)
        default, responses = read_responses(child.root)
        task_id = passing_task_ids(brief.incumbent_report)[0]
        responses[task_id] = "ANSWER: definitely wrong"
        write_responses(child.root / RESPONSES, default, responses)
        return Proposal(candidate_dir=child.root, rationale=f"broke {task_id}")


class NoProposalOverseer(_BaseOverseer):
    name = "no-proposal"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        return Proposal(candidate_dir=None, rationale="", cost_usd=0.25, error="could not think of anything")


class InvalidOverseer(_BaseOverseer):
    """Returns a broken child. settings["mode"] picks how it is broken."""

    name = "invalid"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        mode = self.settings.get("mode", "bad_responses")
        workspace = Path(workspace)
        if mode == "outside_workspace":
            # Points at the incumbent's own snapshot instead of something under the workspace.
            return Proposal(candidate_dir=brief.incumbent.root, rationale="look over there")
        child = self._copy(brief, workspace)
        if mode == "no_manifest":
            (child.root / "candidate.toml").unlink()
        elif mode == "other_agent":
            (child.root / "candidate.toml").write_text('schema_version = 1\nagent = "pi"\n')
        elif mode == "bad_responses":
            (child.root / RESPONSES).write_text("this is = = not toml\n")
        elif mode == "symlink":
            outside = workspace / "outside.txt"
            outside.write_text("not part of the candidate")
            os.symlink(outside, child.root / "linked.txt")
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return Proposal(candidate_dir=child.root, rationale=f"broken: {mode}")


class UnchangedClaimingOverseer(_BaseOverseer):
    """Claims to propose changes but returns an identical copy."""

    name = "unchanged-claiming"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        child = self._copy(brief, workspace)
        return Proposal(candidate_dir=child.root, rationale="trust me, it is different")


class ExpensiveOverseer(_BaseOverseer):
    """A null-style copy whose proposal alone costs more than any test budget."""

    name = "expensive"
    proposes_changes = False

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        child = self._copy(brief, workspace)
        return Proposal(candidate_dir=child.root, rationale="pricey copy", cost_usd=float(self.settings.get("cost", 5.0)))


class CrashingOverseer(_BaseOverseer):
    """Breaks the Overseer contract by raising, to test that run_meta.json survives a crash."""

    name = "crashing"
    workspaces: list[Path] = []  # class-level so the test can find it after the registry built the instance

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        CrashingOverseer.workspaces.append(Path(workspace))
        (Path(workspace) / "partial.txt").write_text("got this far")
        raise RuntimeError("overseer exploded")


class RecordingOverseer(_BaseOverseer):
    """Lays out the workspace exactly as the Claude Code overseer does, then proposes an
    unchanged copy (as the null control does), recording everything it was given."""

    name = "recording"
    proposes_changes = False
    briefs: list[OverseerBrief] = []
    workspaces: list[Path] = []
    heldout_existed: list[bool] = []
    reports: list[dict[str, Any]] = []
    skipped_notes: list[str | None] = []  # the evidence's skipped-files note, when there is one

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        workspace = Path(workspace)
        RecordingOverseer.briefs.append(brief)
        RecordingOverseer.workspaces.append(workspace)
        # The brief's files are temporary copies, so record what they held while they exist.
        RecordingOverseer.reports.append(json.loads(Path(brief.incumbent_report).read_text()))
        note = Path(brief.evidence_dir) / SKIPPED_EVIDENCE_FILE
        RecordingOverseer.skipped_notes.append(note.read_text() if note.is_file() else None)
        # Briefs hold copies outside the run dir, so the test passes the run dir in explicitly.
        if "run_dir" in self.settings:
            RecordingOverseer.heldout_existed.append((Path(self.settings["run_dir"]) / "heldout").exists())
        build_workspace(brief, workspace)
        return Proposal(candidate_dir=workspace / "candidate", rationale="recorded")

    @classmethod
    def reset(cls) -> None:
        cls.briefs = []
        cls.workspaces = []
        cls.heldout_existed = []
        cls.reports = []
        cls.skipped_notes = []


# --- suites ------------------------------------------------------------------------------


class _TinySuite:
    name = "tiny"
    _TASKS: tuple[tuple[str, str], ...] = ()

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = dict(settings or {})

    def describe(self) -> str:
        return "a tiny test suite"

    def metric(self) -> MetricSpec:
        return MetricSpec(name="exact", description="1 if the final text is 'ok'")

    def tasks(self, split):
        return [Task(id=task_id, split=task_split, instruction="say ok") for task_id, task_split in self._TASKS if task_split == split]

    def grade(self, task: Task, result: AgentResult, workdir: Path) -> Grade:
        return Grade(reward=1.0 if result.final_text == "ok" else 0.0)


class OverlappingSuite(_TinySuite):
    """Breaks P7: one task id appears in both search and heldout."""

    name = "overlapping"
    _TASKS = (("shared", "search"), ("only-search", "search"), ("shared", "heldout"))


class EmptySearchSuite(_TinySuite):
    name = "empty-search"
    _TASKS = (("h1", "heldout"),)


class NoHeldoutSuite(_TinySuite):
    name = "no-heldout"
    _TASKS = (("s1", "search"), ("s2", "search"))


# --- integrity and budget fixtures -----------------------------------------------------------


class TamperingOverseer(_BaseOverseer):
    """Writes every held-out answer into the candidate it was briefed with, then proposes
    nothing. If the brief pointed at the live snapshot, held-out scores would be forged."""

    name = "tampering"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        default, responses = read_responses(brief.incumbent.root)
        write_responses(brief.incumbent.root / RESPONSES, default, {**responses, **exact_answers_as_responses()})
        return Proposal(candidate_dir=None, rationale="", error="tampered instead of proposing")


class TargetedTamperOverseer(_BaseOverseer):
    """Rewrites the candidate directory at settings["target"]: stands in for any process that
    edits a snapshot on disk behind the controller's back."""

    name = "targeted-tamper"

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        target = Path(self.settings["target"])
        default, responses = read_responses(target)
        write_responses(target / RESPONSES, default, {**responses, **exact_answers_as_responses()})
        return Proposal(candidate_dir=None, rationale="", error="tampered instead of proposing")


def exact_answers_as_responses() -> dict[str, str]:
    return {task_id: f"ANSWER: {answer}" for task_id, answer in exact_answers().items()}


class CostlyScriptedAdapter(ScriptedAdapter):
    """The scripted agent, but every trial costs settings["cost_per_trial"] dollars, so budget
    rules can be tested without a model. Candidates name it as their agent by import path."""

    name = "loop_fixtures:CostlyScriptedAdapter"

    def build(self, candidate: Candidate, model: Any, settings: Mapping[str, Any]) -> _CostlyAgent:
        cost = float(settings.get("cost_per_trial", 0.0))
        return _CostlyAgent(super().build(candidate, model, {}), cost)


class _CostlyAgent:
    def __init__(self, inner: Any, cost_per_trial: float) -> None:
        self.inner = inner
        self.cost_per_trial = cost_per_trial

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        result = self.inner.run(task, workdir, trace_dir)
        return dataclasses.replace(result, usage=Usage(cost_usd=self.cost_per_trial))


class InterruptingScriptedAdapter(CostlyScriptedAdapter):
    """Costly scripted trials, where run number settings["interrupt_on_run"] of an evaluation
    raises KeyboardInterrupt: stands in for Ctrl-C landing in the middle of a paid evaluation."""

    name = "loop_fixtures:InterruptingScriptedAdapter"

    def build(self, candidate: Candidate, model: Any, settings: Mapping[str, Any]) -> _InterruptingAgent:
        costly = super().build(candidate, model, settings)
        return _InterruptingAgent(costly, int(settings["interrupt_on_run"]))


class _InterruptingAgent:
    def __init__(self, inner: Any, interrupt_on_run: int) -> None:
        self.inner = inner
        self.interrupt_on_run = interrupt_on_run
        self.runs = 0
        self._lock = threading.Lock()

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        with self._lock:
            self.runs += 1
            number = self.runs
        if number == self.interrupt_on_run:
            raise KeyboardInterrupt
        return self.inner.run(task, workdir, trace_dir)


class LitteringScriptedAdapter(ScriptedAdapter):
    """The scripted agent, but every trial leaves a FIFO, an unreadable file and an unreadable
    directory in its workdir, as a bash tool or a dev server could."""

    name = "loop_fixtures:LitteringScriptedAdapter"

    def build(self, candidate: Candidate, model: Any, settings: Mapping[str, Any]) -> _LitteringAgent:
        return _LitteringAgent(super().build(candidate, model, settings))


class _LitteringAgent:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        result = self.inner.run(task, workdir, trace_dir)
        os.mkfifo(workdir / "pipe")
        unreadable = workdir / "unreadable.txt"
        unreadable.write_text("secret")
        unreadable.chmod(0)
        locked = workdir / "locked"
        locked.mkdir()
        (locked / "inside.txt").write_text("x")
        locked.chmod(0)
        return result


class FifoLeavingOverseer(_BaseOverseer):
    """Leaves a FIFO in its workspace (copying one blocks or fails), then proposes a null-style copy."""

    name = "fifo-leaving"
    proposes_changes = False

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        child = self._copy(brief, workspace)
        os.mkfifo(Path(workspace) / "pipe")
        return Proposal(candidate_dir=child.root, rationale="left a pipe behind")


# --- extensions ------------------------------------------------------------------------------

EXTENSION_SPEC = ExtensionSpec(
    language="python",
    suffixes=(".py",),
    entry_convention="define `def extension()`",
    framework_version="9.9.9",
    lifecycle=(LifecyclePoint("before_tool_call", "supported", "on_tool_call", "cancel the call"),),
)


class ExtensionScriptedAdapter(ScriptedAdapter):
    """The scripted agent, declaring an extension spec it never honours: it loads nothing and
    returns no load report, as an adapter that quietly skips a candidate's extensions would."""

    name = "loop_fixtures:ExtensionScriptedAdapter"

    def extension_spec(self) -> ExtensionSpec:
        return EXTENSION_SPEC


# --- the model gateway -----------------------------------------------------------------------


def chat_through(endpoint: Any, model: str = "gpt-5-nano") -> int:
    """One chat completion through a gateway endpoint, sent the way a framework's client would."""
    parts = urlsplit(endpoint.base_url)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]})
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request("POST", parts.path + "/chat/completions", body=body,
                           headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"})  # fmt: skip
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


class GatewayScriptedAdapter(ScriptedAdapter):
    """The scripted agent, but every trial first makes one model call through the endpoint it is
    given, as a real framework adapter must. It refuses to run without an endpoint."""

    name = "loop_fixtures:GatewayScriptedAdapter"

    def build(self, candidate: Candidate, model: Any, settings: Mapping[str, Any]) -> _GatewayAgent:
        return _GatewayAgent(super().build(candidate, model, settings))


class _GatewayAgent:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def run(self, task: Task, workdir: Path, trace_dir: Path, endpoint: Any = None) -> AgentResult:
        if endpoint is None:
            raise AssertionError("a real provider's trial must be given a gateway endpoint")
        status = chat_through(endpoint)
        (trace_dir / "endpoint.json").write_text(json.dumps({"base_url": endpoint.base_url, "status": status}))
        result = self.inner.run(task, workdir, trace_dir, endpoint=endpoint)
        return dataclasses.replace(result, usage=Usage(cost_usd=0.0))  # the ledger, not this, is billed
