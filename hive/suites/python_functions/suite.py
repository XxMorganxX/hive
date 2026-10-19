"""The python_functions suite: implement one small function in solution.py, graded by hidden cases.

It exists to exercise the "code in a directory" task style through the universal Task
shape: the instruction plus a stub `solution.py` go to the agent, and the hidden cases
stay here. Grading runs the agent's code in a separate process (see runner.py) on a
copy of solution.py in a fresh temporary directory, so nothing hidden is ever written
into the agent's workdir, which the overseer later reads as evidence.

Task data lives in tasks/<task-id>/:
    task.toml     id, split, function, description
    stub.py       signature + docstring; the body raises NotImplementedError
    cases.json    hidden: [{"args": [...], "kwargs": {...}, "expected": ...}, ...]
    reference.py  a correct solution, used only by tests for the P2 checks
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hive.contracts import SPLITS, AgentResult, Grade, MetricSpec, Split, Task

PACKAGE_DIR = Path(__file__).parent
DEFAULT_TASKS_DIR = PACKAGE_DIR / "tasks"
RUNNER = PACKAGE_DIR / "runner.py"
INSTRUCTION_TEMPLATE = PACKAGE_DIR / "instruction_template.md"
DESCRIPTION = PACKAGE_DIR / "DESCRIPTION.md"
METRIC_DESCRIPTION = PACKAGE_DIR / "metric_description.md"

SOLUTION_FILE = "solution.py"
TASK_FILES = ("task.toml", "stub.py", "cases.json", "reference.py")
DEFAULT_GRADE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class _TaskData:
    """Everything the suite knows about one task. Only `task` ever leaves the suite."""

    task: Task
    function_name: str
    cases_path: Path
    case_count: int


def _load_task(task_dir: Path, instruction_template: string.Template) -> _TaskData:
    task_id = task_dir.name
    for file_name in TASK_FILES:
        if not (task_dir / file_name).is_file():
            raise ValueError(f"task {task_id!r}: missing {file_name}")
    try:
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"task {task_id!r}: task.toml is not valid TOML: {error}") from error

    # The directory name is the id, so ids are unique by construction; task.toml must agree.
    if config.get("id") != task_id:
        raise ValueError(f"task {task_id!r}: task.toml id {config.get('id')!r} does not match its directory name")
    split = config.get("split")
    if split not in SPLITS:
        raise ValueError(f"task {task_id!r}: split {split!r} is not one of {SPLITS}")
    function_name = config.get("function")
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError(f"task {task_id!r}: 'function' must be a Python identifier, got {function_name!r}")
    description = config.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"task {task_id!r}: 'description' must be a non-empty string")

    cases_path = task_dir / "cases.json"
    try:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"task {task_id!r}: cases.json is not valid JSON: {error}") from error
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"task {task_id!r}: cases.json must be a non-empty list")
    for index, case in enumerate(cases):
        valid = (
            isinstance(case, dict)
            and "expected" in case
            and isinstance(case.get("args", []), list)
            and isinstance(case.get("kwargs", {}), dict)
        )
        if not valid:
            raise ValueError(f"task {task_id!r}: case {index} needs 'expected', a list 'args' and a dict 'kwargs'")

    instruction = instruction_template.substitute(function_name=function_name, description=description.strip())
    task = Task(
        id=task_id,
        split=split,
        instruction=instruction,
        workspace_files={SOLUTION_FILE: (task_dir / "stub.py").read_text(encoding="utf-8")},
    )
    return _TaskData(task=task, function_name=function_name, cases_path=cases_path, case_count=len(cases))


def _load_tasks(tasks_dir: Path) -> dict[str, _TaskData]:
    if not tasks_dir.is_dir():
        raise ValueError(f"tasks_dir {tasks_dir} is not a directory")
    instruction_template = string.Template(INSTRUCTION_TEMPLATE.read_text(encoding="utf-8"))
    task_dirs = sorted(p for p in tasks_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if not task_dirs:
        raise ValueError(f"no tasks found in {tasks_dir}")
    return {task_dir.name: _load_task(task_dir, instruction_template) for task_dir in task_dirs}


def _grade_timeout(settings: Mapping[str, Any]) -> float:
    timeout = settings.get("grade_timeout_s", DEFAULT_GRADE_TIMEOUT_S)
    # bool is an int subclass; `grade_timeout_s = true` is a config mistake, not one second.
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"grade_timeout_s must be a positive, finite number of seconds, got {timeout!r}")
    return float(timeout)


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    # The runner starts its own session, so killing the group also reaches anything the
    # solution spawned; a surviving child would hold the stdout pipe open forever.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:  # already gone
            pass
    else:
        process.kill()


def _parse_runner_output(stdout: str, case_count: int) -> dict[str, Any] | None:
    """The runner's result line, or None if it is missing or inconsistent with the task's cases."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        outcome = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(outcome, dict):
        return None
    if "error" in outcome:
        return outcome if isinstance(outcome["error"], str) else None
    passed, failures = outcome.get("passed"), outcome.get("failures")
    # Every case must be accounted for exactly once, against the suite's own count rather
    # than whatever total the runner reported, so a reward can never leave [0, 1].
    well_formed = (
        isinstance(passed, int)
        and not isinstance(passed, bool)
        and isinstance(failures, list)
        and all(isinstance(kind, str) for kind in failures)
        and passed >= 0
        and passed + len(failures) == case_count
    )
    return outcome if well_formed else None


def _describe_failures(failures: list[str]) -> str:
    counts = Counter(failures)  # insertion-ordered: kinds appear in the order they first failed
    return ", ".join(f"{count} {kind}" for kind, count in counts.items())


class PythonFunctionsSuite:
    name = "python_functions"

    def __init__(self, settings: Mapping[str, Any] | None = None):
        settings = settings or {}
        self._grade_timeout_s = _grade_timeout(settings)
        self._python_bin = str(settings.get("python_bin", sys.executable))
        self._tasks = _load_tasks(Path(settings.get("tasks_dir", DEFAULT_TASKS_DIR)))

    def describe(self) -> str:
        return DESCRIPTION.read_text(encoding="utf-8")

    def metric(self) -> MetricSpec:
        return MetricSpec(
            name="mean_fraction_of_hidden_cases_passed",
            description=METRIC_DESCRIPTION.read_text(encoding="utf-8").strip(),
        )

    def tasks(self, split: Split) -> list[Task]:
        return [data.task for data in self._tasks.values() if data.task.split == split]

    def grade(self, task: Task, result: AgentResult, workdir: Path) -> Grade:
        # Only solution.py is graded; the agent's final text plays no part, so `result` is unused.
        data = self._tasks.get(task.id)
        if data is None:
            raise ValueError(f"task {task.id!r} does not belong to the {self.name} suite")

        solution = Path(workdir) / SOLUTION_FILE
        if not solution.is_file():
            return Grade(reward=0.0, detail=f"{SOLUTION_FILE} not found")

        with tempfile.TemporaryDirectory(prefix="hive-python-functions-") as grading_dir:
            solution_copy = Path(grading_dir) / SOLUTION_FILE
            try:
                shutil.copyfile(solution, solution_copy)
            except OSError as error:
                return Grade(reward=0.0, detail=f"could not read {SOLUTION_FILE}: {type(error).__name__}")
            return self._run_hidden_cases(data, solution_copy, Path(grading_dir))

    def _run_hidden_cases(self, data: _TaskData, solution_copy: Path, grading_dir: Path) -> Grade:
        command = [self._python_bin, "-I", str(RUNNER), str(solution_copy), str(data.cases_path), data.function_name]
        # stderr is discarded: the detail reports only exception type names, and a solution's
        # stderr could be arbitrarily large.
        process = subprocess.Popen(
            command,
            cwd=grading_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=self._grade_timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            process.communicate()
            return Grade(reward=0.0, detail="grading timed out")
        finally:
            # Reap any background processes the solution left behind, even on success.
            _kill_process_tree(process)

        outcome = _parse_runner_output(stdout, data.case_count)
        if outcome is None:
            return Grade(reward=0.0, detail=f"grading runner exited without a result (exit code {process.returncode})")
        if "error" in outcome:
            return Grade(reward=0.0, detail=f"{SOLUTION_FILE} failed to load: {outcome['error']}")

        passed, failures = outcome["passed"], outcome["failures"]
        detail = f"passed {passed}/{data.case_count} hidden cases"
        if failures:
            detail += f"; failures: {_describe_failures(failures)}"
        return Grade(reward=passed / data.case_count, detail=detail)
