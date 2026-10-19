"""The Claude Code overseer: headless Claude Code proposes one harness change per generation.

Claude Code runs with its working directory set to a scratch workspace that holds an
editable copy of the incumbent (`candidate/`) and a copy of the evidence (`context/`).
Only `candidate/` and `RATIONALE.md` are harvested afterwards.

The workspace is also a permission fence, enforced by Claude Code itself (P7, P10). The
overseer may read only inside the workspace and edit only `candidate/**` and
`RATIONALE.md`. The layout alone is not a fence: the suites re-read their runner and
hidden cases from the repo on every grade, and the repo holds the held-out answers, so a
read or a write that escapes the workspace could leak answers or rig the grader. See
`build_command` for the flags and why each one is there (checked against Claude Code
2.1.303's installed bundle).

The copied evidence is rewritten so that no file in the workspace names a location
outside it: paths into the incumbent's trials become relative to `context/evidence/`,
and any other path into the run, the repo or the brief's temp dirs is redacted.

Every overseer-side failure (missing binary, timeout, bad output, a broken candidate)
comes back as a Proposal with candidate_dir=None and an error, never as an exception.
When the spend can't be read (a timeout, or no parseable result), the per-proposal limit
is charged as an upper bound, so the run's dollar cap never undercounts (P12).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Template
from typing import Any

from hive.candidate import MANIFEST, Candidate, CandidateError
from hive.contracts import OverseerBrief, Proposal

PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompt.md")

CANDIDATE_DIR = "candidate"
CONTEXT_DIR = "context"
EVIDENCE_DIR = "evidence"
RATIONALE_FILE = "RATIONALE.md"
STDOUT_FILE = "overseer_output.json"
STDERR_FILE = "overseer_stderr.txt"

# The tools that exist in the session. Read and edit files only: without Bash the overseer
# cannot run the suite, reach the network, or get round the file-tool fence with a shell.
AVAILABLE_TOOLS = "Read,Grep,Glob,Edit,Write"
# The only pre-approved actions. Claude Code checks every file-editing tool (Edit and Write
# alike) against `Edit(...)` path rules; a `Write(...)` path rule is never consulted. For a
# --allowedTools rule, a leading "/" is anchored at the working directory (the workspace),
# not at the filesystem root. Without it a slash-free pattern such as "RATIONALE.md" would
# match at any depth (context/RATIONALE.md too). "/candidate/**" covers every file under
# candidate/. Never pass a bare tool name here: a bare "Edit" or "Read" allows that tool on
# any path on the host, overriding the working-directory check.
EDIT_ALLOW_RULES = (f"Edit(/{CANDIDATE_DIR}/**)", f"Edit(/{RATIONALE_FILE})")

# Stands in for any path outside the workspace that the copied evidence mentions.
OUTSIDE_PLACEHOLDER = "<outside the workspace>"
# What an evidence symlink that leads outside the evidence copy is replaced with.
OUTSIDE_LINK_NOTE = f"This was a symbolic link to {OUTSIDE_PLACEHOLDER}. HIVE removed it from the copy.\n"

# Characters that end a path in free text: whitespace, quotes, brackets, separators, and a
# backslash, which starts a JSON escape that must stay intact.
_PATH_END = r"""\s"'<>()\[\]{},;\\"""

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
                self.build_command(prompt),
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_s,
            )
        except subprocess.TimeoutExpired as timeout:
            _save_output(workspace, timeout.stdout, timeout.stderr)
            # The result object, and with it the spend, is only printed at the end, so the
            # spend before the kill is unknown. Charging nothing would let the dollar cap
            # undercount every timed-out generation (P12).
            charged = self.settings.max_usd_per_proposal
            return _failed(f"claude timed out after {self.settings.timeout_s}s; {_unknown_cost_note(charged)}", charged)
        except OSError as error:
            # The binary never started, so nothing was spent.
            return _failed(f"could not run claude binary {self.settings.claude_bin!r}: {error}")

        _save_output(workspace, completed.stdout, completed.stderr)
        return harvest(brief, workspace, completed, unknown_cost_usd=self.settings.max_usd_per_proposal)

    def build_command(self, prompt: str) -> list[str]:
        """The exact argv `propose` runs, with the workspace as the working directory.

        Each fence flag below was checked against Claude Code 2.1.303's installed bundle,
        not only its --help text. Together they make Claude Code itself refuse any read
        outside the workspace and any edit outside candidate/** and RATIONALE.md.
        """
        return [
            self.settings.claude_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.settings.model,
            # The read and write fence. In the file tools' permission checks, restricted mode
            # denies any path outside the working directories (and any symlink that lands
            # outside them) before allow rules are consulted. It also ignores the user,
            # project and local settings files, so no hooks or permission rules from this
            # machine apply, and it removes every code-running tool that --tools doesn't name.
            "--restricted",
            # Only pre-approved actions run, and every other permission check is denied, not
            # asked. acceptEdits is not used: it would auto-approve edits anywhere in the
            # workspace, context/ included.
            "--permission-mode",
            "dontAsk",
            # Nobody answers permission prompts: anything that would prompt is denied at once.
            # dontAsk already turns asks into denials; this makes sure no host can approve one.
            "--permission-prompts",
            "none",
            "--tools",
            AVAILABLE_TOOLS,
            # Reads inside the working directory are allowed by default, so the only rules
            # needed are the two edit scopes (see EDIT_ALLOW_RULES).
            "--allowedTools",
            ",".join(EDIT_ALLOW_RULES),
            # Restricted mode still loads MCP servers (e.g. user-scope ones in ~/.claude.json),
            # whose tools reach outside the workspace. This loads none.
            "--strict-mcp-config",
            # Skips the user's CLAUDE.md, skills, plugins and hooks, so nothing from this
            # machine steers the proposal or names paths outside the workspace.
            "--safe-mode",
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

    # Copies, not links: the copies are rewritten below, and the experience store's
    # originals must stay exactly as the evaluator wrote them.
    if Path(brief.incumbent_report).is_file():
        shutil.copyfile(brief.incumbent_report, context / "report.json")
    if Path(brief.evidence_dir).is_dir():
        # symlinks=True copies links as links, so a link an agent left in its workdir
        # can't pull outside files into the copy or fail the copy when it dangles.
        shutil.copytree(brief.evidence_dir, context / EVIDENCE_DIR, symlinks=True)

    # The evaluator records absolute paths (every trial_dir, and whatever the agent's trace
    # mentions), and with the shipped configs those lead into the HIVE repo, a few steps
    # from the held-out answers. Claude Code would refuse to read there anyway; this keeps
    # the brief from even naming such places (P7).
    rewriter = _PathRewriter.for_brief(brief)
    _replace_outside_links(context / EVIDENCE_DIR)
    for path in _text_files(context):
        rewriter.rewrite_file(path)
    # The evaluator writes trial_dir relative to its evaluation directory ("trials/<task>/<n>"),
    # and context/evidence/ holds the contents of that trials directory, so drop the first part.
    trials_dir_name = Path(brief.evidence_dir).name
    for path in [context / "report.json", *(context / EVIDENCE_DIR).glob("*/*/outcome.json")]:
        _relativize_trial_dirs(path, trials_dir_name)


class _PathRewriter:
    """Rewrites absolute paths in text: into the trials -> relative to context/evidence/,
    anywhere else the brief came from -> a placeholder that keeps only the file name."""

    def __init__(self, evidence_roots: Iterable[str], outside_roots: Iterable[str]) -> None:
        self._evidence_roots = set(evidence_roots)
        roots = self._evidence_roots | set(outside_roots)
        # Longest first, so a path is matched by its most specific root. The root is
        # matched literally, because real roots contain spaces ("CS 4701"). It must end at
        # a path boundary, and the rest of the path runs to the next _PATH_END character.
        alternatives = "|".join(re.escape(root) for root in sorted(roots, key=len, reverse=True))
        self._pattern = re.compile(rf"(?P<root>{alternatives})(?![^/{_PATH_END}])(?P<rest>/[^{_PATH_END}]*)?")

    @classmethod
    def for_brief(cls, brief: OverseerBrief) -> _PathRewriter:
        # Native traces name paths where the trials were recorded (evidence_origin); the
        # brief's evidence_dir is a copy of that same tree.
        evidence_roots = {str(brief.evidence_dir)}
        if brief.evidence_origin is not None:
            evidence_roots.add(str(brief.evidence_origin))
        # Everything above those roots is outside the workspace too: the run dir, the
        # experience store, the repo, the brief's temp dir.
        brief_paths = {str(brief.incumbent_report), str(brief.incumbent.root), *evidence_roots}
        outside_roots = {str(parent) for path in brief_paths for parent in PurePosixPath(path).parents}
        outside_roots |= brief_paths - evidence_roots
        outside_roots.discard("/")
        return cls(_spellings(evidence_roots), _spellings(outside_roots))

    def rewrite(self, text: str) -> str:
        return self._pattern.sub(self._replacement, text)

    def rewrite_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return  # not text, so it names no path we could rewrite
        rewritten = self.rewrite(text)
        if rewritten != text:
            _make_user_writable(path)
            path.write_text(rewritten, encoding="utf-8")

    def _replacement(self, match: re.Match[str]) -> str:
        rest = match.group("rest") or ""
        if match.group("root") in self._evidence_roots:
            return rest.lstrip("/") or "."
        name = PurePosixPath(rest).name
        return f"{OUTSIDE_PLACEHOLDER}/{name}" if name else OUTSIDE_PLACEHOLDER


def _relativize_trial_dirs(path: Path, trials_dir_name: str) -> None:
    if path.is_symlink() or not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if _strip_trials_prefix(data, trials_dir_name):
        _make_user_writable(path)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _strip_trials_prefix(data: Any, trials_dir_name: str) -> bool:
    """Rewrite every relative trial_dir "<trials>/<task>/<n>" in place; True if any changed."""
    changed = False
    items = data.items() if isinstance(data, dict) else enumerate(data) if isinstance(data, list) else ()
    for key, value in list(items):
        if key == "trial_dir" and isinstance(value, str) and not os.path.isabs(value):
            parts = PurePosixPath(value).parts
            if len(parts) > 1 and parts[0] == trials_dir_name:
                data[key] = PurePosixPath(*parts[1:]).as_posix()
                changed = True
        elif isinstance(value, dict | list):
            changed = _strip_trials_prefix(value, trials_dir_name) or changed
    return changed


def _spellings(paths: Iterable[str]) -> set[str]:
    """Every way the same path may be written: as given, with symlinks resolved (macOS's
    /var is /private/var), and JSON-escaped (json.dumps escapes non-ASCII letters)."""
    spellings: set[str] = set()
    for path in paths:
        forms = {path, os.path.realpath(path)}
        forms |= {form.removeprefix("/private") for form in forms if form.startswith("/private/")}
        for form in forms:
            spellings.add(form)
            spellings.add(json.dumps(form)[1:-1])
    spellings.discard("")
    return spellings


def _replace_outside_links(evidence: Path) -> None:
    """Replace every symlink that leads outside the evidence copy with a short note.

    The target alone names an outside location, and following one would read outside the
    workspace. Links that stay inside the copy are kept as they are.
    """
    if not evidence.is_dir():
        return
    root = os.path.realpath(evidence)
    for directory, dirnames, filenames in os.walk(evidence):
        for name in [*dirnames, *filenames]:
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            target = os.path.realpath(path)
            if os.path.commonpath([root, target]) != root:
                _make_user_writable(path.parent)
                path.unlink()
                path.write_text(OUTSIDE_LINK_NOTE, encoding="utf-8")


def _make_user_writable(path: Path) -> None:
    # An agent may have left read-only files or directories in its workdir, and the copy
    # keeps their modes. Only the copy is changed, never the experience store's original.
    mode = stat.S_IMODE(path.lstat().st_mode)
    if not mode & stat.S_IWUSR:
        path.chmod(mode | stat.S_IWUSR)


def _text_files(directory: Path) -> list[Path]:
    """Regular files under `directory`, never following a symlink."""
    files: list[Path] = []
    for parent, _, filenames in os.walk(directory):
        files.extend(path for name in filenames if not (path := Path(parent) / name).is_symlink())
    return files


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


def harvest(
    brief: OverseerBrief,
    workspace: Path,
    completed: subprocess.CompletedProcess[str],
    unknown_cost_usd: float,
) -> Proposal:
    """Turn a finished claude run into a Proposal, checking the edited candidate still loads.

    `unknown_cost_usd` is charged when the run's spend can't be read from its output: an
    upper bound, so the dollar cap never counts an unreadable run as free (P12).
    """
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = None
    if not isinstance(result, dict):
        # A crash or kill mid-run: the run may have spent up to its limit before dying.
        return _failed(
            f"claude output is not a JSON result object (exit code {completed.returncode}); "
            f"{_unknown_cost_note(unknown_cost_usd)}; stderr: {_excerpt(completed.stderr)}",
            unknown_cost_usd,
        )

    read_cost = _cost(result)
    cost_usd = read_cost if read_cost is not None else unknown_cost_usd
    cost_note = "" if read_cost is not None else f"; {_unknown_cost_note(unknown_cost_usd)}"
    result_text = result.get("result") if isinstance(result.get("result"), str) else ""
    if completed.returncode != 0:
        return _failed(
            f"claude exited with exit code {completed.returncode}: "
            f"{_excerpt(result_text or completed.stderr)}{cost_note}",
            cost_usd,
        )
    if result.get("is_error") is True:
        return _failed(
            f"claude reported an error ({result.get('subtype')}): {_excerpt(result_text)}{cost_note}", cost_usd
        )

    candidate_dir = workspace / CANDIDATE_DIR
    try:
        child = Candidate.load(candidate_dir)
    except CandidateError as error:
        return _failed(f"the edited candidate no longer loads: {error}{cost_note}", cost_usd)
    # The agent field picks the adapter; a child that switches it is not a harness edit.
    if child.agent != brief.incumbent.agent:
        return _failed(
            f"the edited candidate changed its agent from {brief.incumbent.agent!r} to {child.agent!r}{cost_note}",
            cost_usd,
        )

    rationale_path = workspace / RATIONALE_FILE
    rationale = rationale_path.read_text() if rationale_path.is_file() else result_text
    return Proposal(candidate_dir=candidate_dir, rationale=rationale, cost_usd=cost_usd)


def _cost(result: Mapping[str, Any]) -> float | None:
    """The run's reported spend, or None when the result object doesn't carry a usable one."""
    cost = result.get("total_cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        return None
    return float(cost)


def _unknown_cost_note(charged_usd: float) -> str:
    return (
        f"its spend could not be read, so ${charged_usd:.4f} (max_usd_per_proposal) is charged "
        "as an upper bound"
    )


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
