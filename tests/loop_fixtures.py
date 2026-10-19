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

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hive.candidate import Candidate
from hive.contracts import AgentResult, Grade, MetricSpec, OverseerBrief, Proposal, Task
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

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        workspace = Path(workspace)
        RecordingOverseer.briefs.append(brief)
        RecordingOverseer.workspaces.append(workspace)
        # The incumbent report lives at <run_dir>/evals/gen-NNN/report.json.
        run_dir = Path(brief.incumbent_report).parents[2]
        RecordingOverseer.heldout_existed.append((run_dir / "heldout").exists())
        build_workspace(brief, workspace)
        return Proposal(candidate_dir=workspace / "candidate", rationale="recorded")

    @classmethod
    def reset(cls) -> None:
        cls.briefs = []
        cls.workspaces = []
        cls.heldout_existed = []


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
