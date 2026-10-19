"""A scripted test double that stands in for a real agent.

This is NOT a real agent: it calls no model and does no reasoning. It replays
canned responses and files from the candidate's `responses.toml`, so the whole
loop (evaluate, grade, propose, select) can run end to end with zero model spend.
That makes it the agent to use for dry runs, tests and checking a new suite.
"""

from __future__ import annotations

import json
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hive.candidate import Candidate, CandidateError
from hive.contracts import AgentResult, ExtensionSpec, ModelSpec, Task, Usage

RESPONSES_FILE = "responses.toml"
TRACE_FILE = "scripted.json"
VALVES_DOC = Path(__file__).with_name("VALVES.md")

# Used when a task has no entry and the candidate sets no `default`. Empty text
# still counts as a completed run; the suite decides it is wrong.
FALLBACK_RESPONSE = ""

_KNOWN_KEYS = {"default", "responses", "files"}


@dataclass(frozen=True)
class Script:
    """The parsed contents of responses.toml."""

    default: str
    responses: Mapping[str, str]
    files: Mapping[str, Mapping[str, str]]  # task id -> relative path -> content


def _path_problem(path: str) -> str | None:
    """Why a scripted file path could land outside the workdir, or None if it is safe."""
    pure = PurePosixPath(path)
    if not path or not pure.parts:
        return "is empty"
    if pure.is_absolute():
        return "is absolute"
    if ".." in pure.parts:
        return "contains '..'"
    return None


def _check_script(data: Mapping[str, Any]) -> list[str]:
    problems = []
    for key in sorted(set(data) - _KNOWN_KEYS):
        problems.append(f"{RESPONSES_FILE}: unknown key {key!r} (expected one of {sorted(_KNOWN_KEYS)})")

    if "default" in data and not isinstance(data["default"], str):
        problems.append(f"{RESPONSES_FILE}: 'default' must be a string")

    responses = data.get("responses", {})
    if not isinstance(responses, dict):
        problems.append(f"{RESPONSES_FILE}: [responses] must be a table of task id -> text")
    else:
        for task_id, text in responses.items():
            if not isinstance(text, str):
                problems.append(f"{RESPONSES_FILE}: response for task {task_id!r} must be a string")

    files = data.get("files", {})
    if not isinstance(files, dict):
        problems.append(f"{RESPONSES_FILE}: [files] must be a table of task id -> {{path -> content}}")
        return problems
    for task_id, task_files in files.items():
        if not isinstance(task_files, dict):
            problems.append(f"{RESPONSES_FILE}: [files.{task_id!r}] must be a table of path -> content")
            continue
        for path, content in task_files.items():
            reason = _path_problem(path)
            if reason:
                problems.append(f"{RESPONSES_FILE}: file path {path!r} for task {task_id!r} {reason}")
            if not isinstance(content, str):
                problems.append(f"{RESPONSES_FILE}: content of {path!r} for task {task_id!r} must be a string")
    return problems


def _load_script(candidate: Candidate) -> tuple[Script | None, list[str]]:
    """Read and check responses.toml. Returns (script, []) or (None, problems); never raises."""
    path = candidate.root / RESPONSES_FILE
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        return None, [f"{RESPONSES_FILE} is missing"]
    except (OSError, UnicodeDecodeError) as error:
        return None, [f"{RESPONSES_FILE} could not be read: {error}"]
    except tomllib.TOMLDecodeError as error:
        return None, [f"{RESPONSES_FILE} is not valid TOML: {error}"]

    problems = _check_script(data)
    if problems:
        return None, problems
    script = Script(
        default=data.get("default", FALLBACK_RESPONSE),
        responses=dict(data.get("responses", {})),
        files={task_id: dict(task_files) for task_id, task_files in data.get("files", {}).items()},
    )
    return script, []


@dataclass(frozen=True)
class ScriptedAgent:
    """Replays a Script. Holds no mutable state, so concurrent runs are safe."""

    script: Script

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        started = time.monotonic()
        response = self.script.responses.get(task.id, self.script.default)
        files = self.script.files.get(task.id, {})
        trace: dict[str, Any] = {"task_id": task.id, "response": response, "files_written": []}
        try:
            root = workdir.resolve()
            for relative in sorted(files):
                target = (root / relative).resolve()
                # validate() already rejects '..' and absolute paths; this also catches
                # a symlink inside the workdir pointing elsewhere.
                if not target.is_relative_to(root):
                    raise ValueError(f"file path {relative!r} escapes the workdir")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(files[relative])
                trace["files_written"].append(relative)
            (trace_dir / TRACE_FILE).write_text(json.dumps(trace, indent=2) + "\n")
        except Exception as error:  # never raise from run(): report the trial as errored
            message = f"{type(error).__name__}: {error}"
            trace["error"] = message
            try:
                (trace_dir / TRACE_FILE).write_text(json.dumps(trace, indent=2) + "\n")
            except OSError:
                pass  # the trace dir itself is the problem; the result still carries the error
            return AgentResult(
                status="errored",
                final_text="",
                usage=Usage(),
                wall_clock_s=time.monotonic() - started,
                error=message,
                error_class="agent_error",
            )
        return AgentResult(
            status="completed",
            final_text=response,
            usage=Usage(),
            wall_clock_s=time.monotonic() - started,
        )


class ScriptedAdapter:
    """Adapter for the scripted test double. Not a real agent: it replays responses.toml."""

    name = "scripted"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        # Nothing to configure: the script is the whole behavior.
        self.settings = dict(settings or {})

    def validate(self, candidate: Candidate) -> list[str]:
        problems = []
        if candidate.agent != self.name:
            problems.append(f"candidate.toml: agent is {candidate.agent!r}, expected {self.name!r}")
        if candidate.valves:
            problems.append(
                f"candidate.toml: the scripted agent has no valves, found {sorted(candidate.valves)}"
            )
        _, script_problems = _load_script(candidate)
        return problems + script_problems

    def build(self, candidate: Candidate, model: ModelSpec, settings: Mapping[str, Any]) -> ScriptedAgent:
        # The model is ignored: nothing here calls one. The script is read once now,
        # so later edits to the candidate directory cannot change a running evaluation.
        problems = self.validate(candidate)
        script, _ = _load_script(candidate)
        if problems or script is None:
            raise CandidateError(f"{candidate.root} is not a valid scripted candidate: " + "; ".join(problems))
        return ScriptedAgent(script)

    def describe_valves(self) -> str:
        return VALVES_DOC.read_text()

    def extension_spec(self) -> ExtensionSpec | None:
        # A replay has no agent loop to hook, so there is nothing an extension could change.
        return None
