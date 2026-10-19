"""The trusted evaluator: one candidate on one suite split, k trials per task (P2, P8).

This is the only code that turns agent runs into numbers, so it is deliberately strict.
Every trial ends as exactly one of two outcomes:

- `scored`: the agent completed and the suite's grader returned a valid reward in [0, 1].
- `errored`: something prevented a measurement (the agent, its environment, the
  workspace or the grader). It carries an error class and is never graded.

Errored trials are left out of `measured_mean` (what we report) but count as 0 in
`selection_score` (what we promote on), so a candidate can't climb by pushing hard
tasks into errors. The report is written to disk as JSON, and loading it back
recomputes every aggregate instead of trusting the stored numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal

from hive.contracts import (
    AGENT_ERROR_CLASSES,
    Agent,
    AgentAdapter,
    AgentResult,
    Grade,
    ModelSpec,
    Split,
    Suite,
    Task,
    Usage,
)

if TYPE_CHECKING:
    from hive.candidate import Candidate

TRIAL_ERROR_CLASSES = AGENT_ERROR_CLASSES + ("grader_error", "workspace_error")

TrialStatus = Literal["scored", "errored"]

REPORT_FILE = "report.json"
OUTCOME_FILE = "outcome.json"
TRIALS_DIR = "trials"

_SAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")
# Long enough to stay readable, short enough to keep the whole trial path well under OS limits.
_MAX_SAFE_ID_PREFIX = 80


class EvaluationError(Exception):
    """The evaluation is misconfigured and no trial should be run."""


@dataclass(frozen=True)
class TrialOutcome:
    task_id: str
    trial: int  # 1-based
    status: TrialStatus
    reward: float | None  # set only when scored
    error_class: str | None  # one of TRIAL_ERROR_CLASSES, set only when errored
    detail: str
    usage: Usage
    wall_clock_s: float
    trial_dir: str  # absolute path; holds workdir/, trace/ and outcome.json

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "usage": asdict(self.usage)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialOutcome:
        return cls(**{**data, "usage": Usage(**data["usage"])})


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    trials: tuple[TrialOutcome, ...]

    @property
    def n_scored(self) -> int:
        return sum(1 for trial in self.trials if trial.status == "scored")

    @property
    def n_errored(self) -> int:
        return sum(1 for trial in self.trials if trial.status == "errored")

    @property
    def measured_mean(self) -> float | None:
        """Mean reward over scored trials only; None when nothing was measured."""
        rewards = [trial.reward for trial in self.trials if trial.status == "scored"]
        return sum(rewards) / len(rewards) if rewards else None

    @property
    def selection_reward(self) -> float:
        """Mean reward over every trial, errored ones counted as 0 (P8)."""
        if not self.trials:
            return 0.0
        total = sum(trial.reward for trial in self.trials if trial.status == "scored")
        return total / len(self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "n_scored": self.n_scored,
            "n_errored": self.n_errored,
            "measured_mean": self.measured_mean,
            "selection_reward": self.selection_reward,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskOutcome:
        return cls(task_id=data["task_id"], trials=tuple(TrialOutcome.from_dict(t) for t in data["trials"]))


@dataclass(frozen=True)
class EvalReport:
    """One candidate on one split. Aggregates are always computed from the trials."""

    candidate_id: str
    suite: str
    split: Split
    model: str  # ModelSpec.qualified
    trials_per_task: int
    tasks: tuple[TaskOutcome, ...]

    @property
    def selection_score(self) -> float:
        """Mean of per-task selection rewards. This is the number promotion uses."""
        if not self.tasks:
            return 0.0
        return sum(task.selection_reward for task in self.tasks) / len(self.tasks)

    @property
    def measured_mean(self) -> float | None:
        """Mean of per-task measured means over tasks with at least one scored trial."""
        means = [task.measured_mean for task in self.tasks if task.measured_mean is not None]
        return sum(means) / len(means) if means else None

    @property
    def stderr(self) -> float:
        """Standard error of selection_score, treating tasks as the sampled unit."""
        if len(self.tasks) < 2:
            return 0.0
        rewards = [task.selection_reward for task in self.tasks]
        return statistics.stdev(rewards) / math.sqrt(len(rewards))

    @property
    def n_trials(self) -> int:
        return sum(len(task.trials) for task in self.tasks)

    @property
    def n_errored(self) -> int:
        return sum(task.n_errored for task in self.tasks)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for task in self.tasks:
            for trial in task.trials:
                total = total + trial.usage
        return total

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "suite": self.suite,
            "split": self.split,
            "model": self.model,
            "trials_per_task": self.trials_per_task,
            # Aggregates are written for human readers and the overseer; from_dict ignores them.
            "selection_score": self.selection_score,
            "measured_mean": self.measured_mean,
            "stderr": self.stderr,
            "n_trials": self.n_trials,
            "n_errored": self.n_errored,
            "usage": asdict(self.usage),
            "cost_usd": self.cost_usd,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalReport:
        return cls(
            candidate_id=data["candidate_id"],
            suite=data["suite"],
            split=data["split"],
            model=data["model"],
            trials_per_task=data["trials_per_task"],
            tasks=tuple(TaskOutcome.from_dict(task) for task in data["tasks"]),
        )


def safe_task_dirname(task_id: str) -> str:
    """A directory name for a task id: deterministic, and distinct for distinct ids.

    Ids that are already safe are kept as-is so trial paths stay readable. Anything
    altered (unsafe characters, '.'/'..', empty, or too long) gets a hash of the
    original id appended, so two ids that sanitize alike still land apart.
    """
    kept = _SAFE_ID_CHARACTERS.sub("_", task_id)
    altered = kept != task_id or kept in {"", ".", ".."} or len(kept) > _MAX_SAFE_ID_PREFIX
    if not altered:
        return task_id
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:10]
    return f"{kept[:_MAX_SAFE_ID_PREFIX]}-{digest}"


def evaluate(
    candidate: Candidate,
    adapter: AgentAdapter,
    suite: Suite,
    split: Split,
    *,
    model: ModelSpec,
    agent_settings: Mapping[str, Any],
    trials_per_task: int,
    out_dir: Path,
    concurrency: int = 1,
) -> EvalReport:
    """Run every task in `split` `trials_per_task` times and write report.json under `out_dir`.

    Raises EvaluationError for configuration bugs, before any agent is built or run.
    Exceptions from `adapter.build` propagate: a candidate that can't be built is an
    environment or config problem, not a trial outcome.
    """
    if trials_per_task < 1:
        raise EvaluationError(f"trials_per_task must be at least 1, got {trials_per_task}")
    if concurrency < 1:
        raise EvaluationError(f"concurrency must be at least 1, got {concurrency}")
    if candidate.agent != adapter.name:
        raise EvaluationError(
            f"candidate {candidate.id} belongs to agent adapter {candidate.agent!r}, not {adapter.name!r}"
        )
    tasks = suite.tasks(split)
    _check_tasks(tasks, suite.name, split)

    out_dir = Path(out_dir).resolve()
    trials_root = out_dir / TRIALS_DIR
    if trials_root.exists():
        # Leftover workdirs from an earlier run would leak files into this one.
        raise EvaluationError(f"{trials_root} already exists; evaluate into a fresh directory")
    trials_root.mkdir(parents=True)

    agent = adapter.build(candidate, model, agent_settings)

    jobs = [(task, trial) for task in tasks for trial in range(1, trials_per_task + 1)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            (task.id, trial): pool.submit(_run_trial, agent, suite, task, trial, trials_root) for task, trial in jobs
        }
        # Collect in job order, not completion order, so the report is deterministic.
        outcomes = {key: future.result() for key, future in futures.items()}

    report = EvalReport(
        candidate_id=candidate.id,
        suite=suite.name,
        split=split,
        model=model.qualified,
        trials_per_task=trials_per_task,
        tasks=tuple(
            TaskOutcome(
                task_id=task.id,
                trials=tuple(outcomes[(task.id, trial)] for trial in range(1, trials_per_task + 1)),
            )
            for task in tasks
        ),
    )
    _write_json(out_dir / REPORT_FILE, report.to_dict())
    return report


def _check_tasks(tasks: Sequence[Task], suite_name: str, split: Split) -> None:
    if not tasks:
        raise EvaluationError(f"suite {suite_name!r} has no tasks in split {split!r}")
    seen_ids: set[str] = set()
    seen_dirnames: dict[str, str] = {}
    for task in tasks:
        if task.split != split:
            # A task from another split here would breach the split walls (P7).
            raise EvaluationError(f"suite {suite_name!r} returned task {task.id!r} from split {task.split!r} for {split!r}")
        if task.id in seen_ids:
            raise EvaluationError(f"suite {suite_name!r} has duplicate task id {task.id!r} in split {split!r}")
        seen_ids.add(task.id)
        # Case-insensitive filesystems (the macOS default) would merge these trial directories.
        folded = safe_task_dirname(task.id).casefold()
        if folded in seen_dirnames:
            raise EvaluationError(
                f"task ids {seen_dirnames[folded]!r} and {task.id!r} differ only by case and would share a directory"
            )
        seen_dirnames[folded] = task.id


def _run_trial(agent: Agent, suite: Suite, task: Task, trial: int, trials_root: Path) -> TrialOutcome:
    trial_dir = trials_root / safe_task_dirname(task.id) / str(trial)
    workdir = trial_dir / "workdir"
    trace_dir = trial_dir / "trace"
    workdir.mkdir(parents=True)
    trace_dir.mkdir()

    def outcome(status: TrialStatus, reward, error_class, detail, usage=Usage(), wall_clock_s=0.0) -> TrialOutcome:
        return TrialOutcome(
            task_id=task.id,
            trial=trial,
            status=status,
            reward=reward,
            error_class=error_class,
            detail=detail,
            usage=usage,
            wall_clock_s=wall_clock_s,
            trial_dir=str(trial_dir),
        )

    workspace_problem = _materialize_workspace(task.workspace_files, workdir)
    if workspace_problem is not None:
        result = outcome("errored", None, "workspace_error", workspace_problem)
    else:
        result = _run_and_grade(agent, suite, task, workdir, trace_dir, outcome)
    _write_json(trial_dir / OUTCOME_FILE, result.to_dict())
    return result


def _run_and_grade(agent: Agent, suite: Suite, task: Task, workdir: Path, trace_dir: Path, outcome) -> TrialOutcome:
    started = time.monotonic()
    try:
        result = agent.run(task, workdir, trace_dir)
    except Exception as error:  # the contract says run never raises; an adapter bug is still a trial outcome
        return outcome("errored", None, "agent_error", _describe(error), wall_clock_s=time.monotonic() - started)
    if not isinstance(result, AgentResult):
        return outcome(
            "errored",
            None,
            "agent_error",
            f"agent returned {type(result).__name__}, not AgentResult",
            wall_clock_s=time.monotonic() - started,
        )

    if result.status == "errored":
        detail = result.error or ""
        error_class = result.error_class
        if error_class not in AGENT_ERROR_CLASSES:
            # Keep the unknown class visible to readers, but only report classes the rest of HIVE knows.
            if error_class is not None:
                detail = f"[unknown error_class {error_class!r}] {detail}"
            error_class = "agent_error"
        return outcome("errored", None, error_class, detail, result.usage, result.wall_clock_s)
    if result.status != "completed":
        return outcome(
            "errored", None, "agent_error", f"unknown agent status {result.status!r}", result.usage, result.wall_clock_s
        )

    try:
        grade = suite.grade(task, result, workdir)
    except Exception as error:
        return outcome("errored", None, "grader_error", _describe(error), result.usage, result.wall_clock_s)
    problem = _grade_problem(grade)
    if problem is not None:
        return outcome("errored", None, "grader_error", problem, result.usage, result.wall_clock_s)
    return outcome("scored", float(grade.reward), None, grade.detail, result.usage, result.wall_clock_s)


def _materialize_workspace(files: Mapping[str, str], workdir: Path) -> str | None:
    """Write the task's starting files into workdir. Returns a problem description, or None."""
    root = workdir.resolve()
    for relative, text in files.items():
        path = PurePath(relative)
        if path.is_absolute() or ".." in path.parts:
            return f"workspace path {relative!r} is absolute or leaves the workdir"
        target = (root / path).resolve()
        if target == root or not target.is_relative_to(root):
            return f"workspace path {relative!r} does not name a file inside the workdir"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        except (OSError, TypeError) as error:
            return f"could not write workspace file {relative!r}: {_describe(error)}"
    return None


def _grade_problem(grade: object) -> str | None:
    if not isinstance(grade, Grade):
        return f"grader returned {type(grade).__name__}, not Grade"
    reward = grade.reward
    # bool is an int subclass, but True as a reward is a grader bug, not 1.0.
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return f"reward {reward!r} is not a number"
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        return f"reward {reward!r} is outside [0, 1]"
    return None


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
