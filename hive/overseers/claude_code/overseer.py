"""The Claude Code overseer: headless Claude Code proposes one harness change per generation.

Claude Code runs with its working directory set to a scratch workspace that holds an
editable copy of the incumbent (`candidate/`) and a read-only copy of the evidence
(`context/`). Only `candidate/` and `RATIONALE.md` are harvested afterwards, so the
write fence is the workspace layout itself, not a permission flag (P10). Tools are
limited to file reading and editing (no Bash), and the user's own settings and hooks
are not loaded, so nothing outside this workspace shapes or observes the run.

Every overseer-side failure (missing binary, timeout, bad output, a broken candidate)
comes back as a Proposal with candidate_dir=None and an error, never as an exception.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from hive.candidate import MANIFEST, Candidate, CandidateError
from hive.contracts import OverseerBrief, Proposal

PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompt.md")

# Read and edit files only. Without Bash, the overseer cannot run the suite, reach the
# network, or write outside the workspace by shell tricks.
ALLOWED_TOOLS = "Read,Grep,Glob,Edit,Write"
# "user" is left out on purpose: the user's settings and hooks must not fire inside a proposal.
SETTING_SOURCES = "project,local"

CANDIDATE_DIR = "candidate"
CONTEXT_DIR = "context"
RATIONALE_FILE = "RATIONALE.md"
STDOUT_FILE = "overseer_output.json"
STDERR_FILE = "overseer_stderr.txt"

# How much of the result text or stderr to quote in an error message.
_ERROR_EXCERPT_CHARS = 500


@dataclass(frozen=True)
class ClaudeCodeSettings:
    claude_bin: str = "claude"
    model: str = "sonnet"
    max_usd_per_proposal: float = 1.0
    timeout_s: float = 900.0

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any] | None) -> ClaudeCodeSettings:
        settings = dict(settings or {})
        # A typo in the experiment config should fail loudly, not silently fall back to a default.
        unknown = sorted(set(settings) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown claude_code overseer settings: {', '.join(unknown)}")
        parsed = cls(**settings)
        for name in ("claude_bin", "model"):
            value = getattr(parsed, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"claude_code overseer setting {name!r} must be a non-empty string")
        for name in ("max_usd_per_proposal", "timeout_s"):
            value = getattr(parsed, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"claude_code overseer setting {name!r} must be a positive number")
        return parsed


class ClaudeCodeOverseer:
    name = "claude_code"
    proposes_changes = True

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = ClaudeCodeSettings.from_mapping(settings)

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        workspace = Path(workspace)
        try:
            build_workspace(brief, workspace)
            prompt = render_prompt(brief)
        except (OSError, CandidateError, ValueError, TypeError) as error:
            return _failed(f"could not prepare the overseer workspace {workspace}: {error}")

        try:
            completed = subprocess.run(
                self._command(prompt),
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_s,
            )
        except subprocess.TimeoutExpired as timeout:
            _save_output(workspace, timeout.stdout, timeout.stderr)
            # Any spend before the kill is unknown, because the result object is only printed at the end.
            return _failed(f"claude timed out after {self.settings.timeout_s}s")
        except OSError as error:
            return _failed(f"could not run claude binary {self.settings.claude_bin!r}: {error}")

        _save_output(workspace, completed.stdout, completed.stderr)
        return harvest(brief, workspace, completed)

    def _command(self, prompt: str) -> list[str]:
        return [
            self.settings.claude_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.settings.model,
            "--permission-mode",
            "acceptEdits",
            "--tools",
            ALLOWED_TOOLS,
            "--allowedTools",
            ALLOWED_TOOLS,
            "--setting-sources",
            SETTING_SOURCES,
            "--no-session-persistence",
            "--max-budget-usd",
            str(self.settings.max_usd_per_proposal),
        ]


def build_workspace(brief: OverseerBrief, workspace: Path) -> None:
    """Lay out candidate/ (editable copy) and context/ (evidence) under `workspace`."""
    workspace.mkdir(parents=True, exist_ok=True)
    brief.incumbent.copy_to(workspace / CANDIDATE_DIR)

    context = workspace / CONTEXT_DIR
    context.mkdir()
    (context / "suite.md").write_text(brief.suite_description.rstrip() + "\n")
    (context / "metric.md").write_text(f"# Metric: {brief.metric.name}\n\n{brief.metric.description.rstrip()}\n")
    (context / "valves.md").write_text(brief.valves.rstrip() + "\n")
    (context / "selection_rule.md").write_text(brief.selection_rule.rstrip() + "\n")
    history = [dict(row) for row in brief.history]
    (context / "history.json").write_text(json.dumps(history, indent=2, default=str) + "\n")

    # Copies, not links: the overseer may write anywhere in the workspace, and the
    # experience store's originals must stay exactly as the evaluator wrote them.
    if Path(brief.incumbent_report).is_file():
        shutil.copyfile(brief.incumbent_report, context / "report.json")
    if Path(brief.evidence_dir).is_dir():
        # symlinks=True copies links as links, so a link an agent left in its workdir
        # can't pull outside files into the copy or fail the copy when it dangles.
        shutil.copytree(brief.evidence_dir, context / "evidence", symlinks=True)


def render_prompt(brief: OverseerBrief) -> str:
    """Fill prompt.md. The three required inputs go inline, not only in context/ files."""
    template = Template(PROMPT_TEMPLATE_PATH.read_text())
    return template.substitute(
        generation=brief.generation,
        candidate_toml=(brief.incumbent.root / MANIFEST).read_text().rstrip(),
        metric_name=brief.metric.name,
        metric_description=brief.metric.description.rstrip(),
        suite_description=brief.suite_description.rstrip(),
        selection_rule=brief.selection_rule.rstrip(),
    )


def harvest(brief: OverseerBrief, workspace: Path, completed: subprocess.CompletedProcess[str]) -> Proposal:
    """Turn a finished claude run into a Proposal, checking the edited candidate still loads."""
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = None
    if not isinstance(result, dict):
        return _failed(
            f"claude output is not a JSON result object (exit code {completed.returncode}); "
            f"stderr: {_excerpt(completed.stderr)}"
        )

    cost_usd = _cost(result)
    result_text = result.get("result") if isinstance(result.get("result"), str) else ""
    if completed.returncode != 0:
        return _failed(
            f"claude exited with exit code {completed.returncode}: {_excerpt(result_text or completed.stderr)}",
            cost_usd,
        )
    if result.get("is_error") is True:
        return _failed(f"claude reported an error ({result.get('subtype')}): {_excerpt(result_text)}", cost_usd)

    candidate_dir = workspace / CANDIDATE_DIR
    try:
        child = Candidate.load(candidate_dir)
    except CandidateError as error:
        return _failed(f"the edited candidate no longer loads: {error}", cost_usd)
    # The agent field picks the adapter; a child that switches it is not a harness edit.
    if child.agent != brief.incumbent.agent:
        return _failed(
            f"the edited candidate changed its agent from {brief.incumbent.agent!r} to {child.agent!r}",
            cost_usd,
        )

    rationale_path = workspace / RATIONALE_FILE
    rationale = rationale_path.read_text() if rationale_path.is_file() else result_text
    return Proposal(candidate_dir=candidate_dir, rationale=rationale, cost_usd=cost_usd)


def _cost(result: Mapping[str, Any]) -> float:
    cost = result.get("total_cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return 0.0
    return float(cost)


def _save_output(workspace: Path, stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    (workspace / STDOUT_FILE).write_text(_as_text(stdout))
    (workspace / STDERR_FILE).write_text(_as_text(stderr))


def _as_text(output: str | bytes | None) -> str:
    # A timeout can hand back partial output as bytes even in text mode, or nothing at all.
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _excerpt(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _ERROR_EXCERPT_CHARS else text[:_ERROR_EXCERPT_CHARS] + "..."


def _failed(error: str, cost_usd: float = 0.0) -> Proposal:
    return Proposal(candidate_dir=None, rationale="", cost_usd=cost_usd, error=error)
