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

Paths in the report are relative to the evaluation's out_dir ("trials/<task>/<n>"). The
report and its trials are copied into overseer briefs, and an absolute path would tell the
overseer where the live run directory, the HIVE repo and earlier runs' held-out results are.

Spend is reported trial by trial through `on_trial_cost`, not only in the finished
report: an evaluation aborted by Ctrl-C or a crash has still spent money, and the caller's
budget record must include it (P12).

Extensions are checked here, not trusted to the adapter (§5.2, P8). Every extension file
the adapter's ExtensionSpec matches in the candidate must appear loaded in a completed run's
LoadReport, or the trial is `harness_load_error` (an errored run is held to less; see
below); extension code that threw mid-run makes it `extension_error`. Both are errors, so
they score 0 in selection. Precedence:

- `model_mismatch` outranks everything: that run is not part of the experiment at all (P3).
- A report that shows an extension failed to load (marked not loaded, with a reason)
  replaces any other class: the candidate did not run as written. The agent's own account
  is kept after the load detail, since it may name the real cause.
- A missing report, a file missing from it, or a file marked not loaded without a reason
  only matters for a completed run. A run that errored (a timeout kill, a runner crash, a
  framework that never started) may never have reached its load phase, and calling that a
  load failure would point the overseer at the wrong cause. The rule is
  hive.extensions.load_problem's, so adapters that apply it agree with the controller.
- `extension_error` replaces `completed` and `agent_error` (the throw is the likely cause
  of the crash), but not `timeout`, `quota` or `provider_error`, which happen around the
  candidate's code rather than in it.

The load report is recorded in every trial's outcome, so the overseer's evidence shows
what actually loaded.

The model gateway (hive/gateway) is the authority for model use whenever a real provider is
under test. `evaluate(..., gateway=...)` opens a grant per trial, passes its endpoint to
`agent.run(..., endpoint=...)`, and always closes the grant, even when the run raises. The
ledger then decides (P3, P12), before the extension check above:

- Usage and cost are the ledger's. The adapter's own count stays only in its trace.
- Any gateway violation (a refused model, effort or cost parameter) or a call whose
  provider-reported model is not the experiment's makes the trial `model_mismatch`.
- A run that did not complete after hitting the call limit is `agent_error`, with the limit
  stated; a completed run is kept.
- An errored run with an upstream `insufficient_quota` or `rate_limit_exceeded` is `quota`.
- A call the provider reported no usage for, or one still running when the grant closed,
  keeps the result, but the detail says the trial's spend is a lower bound.

The ledger is written to the trial dir as gateway.json (calls, violations, limit_reached: no
token and no key), and a compact summary goes into the trial's outcome.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal

from hive.contracts import (
    AGENT_ERROR_CLASSES,
    Agent,
    AgentAdapter,
    AgentResult,
    ExtensionLoad,
    Grade,
    LoadReport,
    ModelSpec,
    Split,
    Suite,
    Task,
    Usage,
)
from hive.agents.runner_protocol import model_matches
from hive.extensions import expected_extension_files, load_problem

if TYPE_CHECKING:
    from hive.candidate import Candidate
    from hive.gateway import ModelGateway, TrialLedger

TRIAL_ERROR_CLASSES = AGENT_ERROR_CLASSES + ("grader_error", "workspace_error")

TrialStatus = Literal["scored", "errored"]

REPORT_FILE = "report.json"
OUTCOME_FILE = "outcome.json"
GATEWAY_FILE = "gateway.json"
TRIALS_DIR = "trials"

_SAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")
# Long enough to stay readable, short enough to keep the whole trial path well under OS limits.
_MAX_SAFE_ID_PREFIX = 80

# Upstream error codes that mean the provider account, not the candidate, stopped the run.
QUOTA_ERROR_CODES = ("insufficient_quota", "rate_limit_exceeded")
# Gateway calls whose spend the ledger could not count: the trial's cost is a lower bound.
UNBILLED_ERROR_CODES = ("usage_missing", "in_flight_at_close")


class EvaluationError(Exception):
    """The evaluation is misconfigured and no trial should be run."""


@dataclass(frozen=True)
class GatewaySummary:
    """What the model gateway saw of one trial, in brief; the full ledger is in gateway.json."""

    calls: int
    models_reported: tuple[str, ...]  # distinct, sorted
    error_codes: tuple[str, ...]  # distinct, sorted: provider errors and gateway refusals
    violations: tuple[str, ...]
    limit_reached: bool
    spend_lower_bound: bool  # some call's usage is unknown, so the trial's cost undercounts

    @classmethod
    def of(cls, ledger: TrialLedger) -> GatewaySummary:
        return cls(
            calls=len(ledger.calls),
            models_reported=tuple(sorted({call.model_reported for call in ledger.calls if call.model_reported})),
            error_codes=tuple(sorted({call.error_code for call in ledger.calls if call.error_code})),
            violations=tuple(ledger.violations),
            limit_reached=ledger.limit_reached,
            spend_lower_bound=any(call.error_code in UNBILLED_ERROR_CODES for call in ledger.calls),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> GatewaySummary | None:
        if data is None:
            return None
        return cls(calls=data["calls"], models_reported=tuple(data["models_reported"]),
                   error_codes=tuple(data["error_codes"]), violations=tuple(data["violations"]),
                   limit_reached=data["limit_reached"], spend_lower_bound=data["spend_lower_bound"])  # fmt: skip


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
    trial_dir: str  # relative to the evaluation's out_dir ("trials/<task>/<n>"); holds workdir/, trace/, outcome.json
    # What the run reported loading, so the evidence shows which extensions ran. None when the
    # run returned none (no extensions, or no usable result at all).
    load_report: LoadReport | None = None
    # The model gateway's account of the trial; None when no gateway was used (scripted runs).
    gateway: GatewaySummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "usage": asdict(self.usage)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialOutcome:
        # Outcomes written before load reports or the gateway existed have no key; they recorded none.
        report = data.get("load_report")
        return cls(**{**data, "usage": Usage(**data["usage"]), "load_report": _load_report_from_dict(report),
                      "gateway": GatewaySummary.from_dict(data.get("gateway"))})  # fmt: skip


def _load_report_from_dict(data: Mapping[str, Any] | None) -> LoadReport | None:
    if data is None:
        return None
    return LoadReport(
        extensions=tuple(
            ExtensionLoad(path=load["path"], loaded=load["loaded"], error=load["error"],
                          registered=tuple(load["registered"]))  # fmt: skip
            for load in data["extensions"]
        ),
        runtime_errors=tuple(data["runtime_errors"]),
    )


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
    on_trial_cost: Callable[[float], None] | None = None,
    gateway: ModelGateway | None = None,
) -> EvalReport:
    """Run every task in `split` `trials_per_task` times and write report.json under `out_dir`.

    `on_trial_cost` is called in the calling thread with each trial's validated cost as the
    trial finishes, so a caller can keep an exact spend record even if this raises.

    `gateway`, a running ModelGateway, is required for every real provider: each trial gets
    its own grant, and the trial's ledger decides its usage and model checks (see the module
    docstring). Without one, `agent.run` is called without an endpoint.

    Raises EvaluationError for configuration bugs, before any agent is built or run.
    Exceptions from `adapter.build` propagate: a candidate that can't be built is an
    environment or config problem, not a trial outcome. If anything interrupts the
    collection (Ctrl-C, or a trial that raised), queued trials are cancelled, running ones
    finish and are still reported through `on_trial_cost`, and the exception propagates.
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

    # Once per evaluation: the candidate is fixed for its whole evaluation, and the controller,
    # not the adapter, decides which of its files must load (P8).
    expected_extensions = tuple(expected_extension_files(candidate, adapter.extension_spec()))

    agent = adapter.build(candidate, model, agent_settings)

    jobs = [(task, trial) for task in tasks for trial in range(1, trials_per_task + 1)]
    trial_context = _TrialContext(agent, suite, model, expected_extensions, gateway)
    outcomes = _run_all(jobs, trial_context, trials_root, concurrency, on_trial_cost or _ignore_cost)

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


def _ignore_cost(cost: float) -> None:
    pass


@dataclass
class _TrialContext:
    """What every trial of one evaluation shares."""

    agent: Agent
    suite: Suite
    model: ModelSpec
    expected_extensions: Sequence[str]
    gateway: ModelGateway | None
    # Gateway spend of trials that raised out of the run (Ctrl-C): they have no outcome, but
    # the money is spent. Appended from worker threads; list.append is atomic.
    interrupted_costs: list[float] = dataclasses.field(default_factory=list)


def _run_all(
    jobs: Sequence[tuple[Task, int]],
    context: _TrialContext,
    trials_root: Path,
    concurrency: int,
    on_trial_cost: Callable[[float], None],
) -> dict[tuple[str, int], TrialOutcome]:
    """Run every job on a pool and return its outcomes keyed by (task id, trial).

    Outcomes are keyed, and the report built in job order, so completion order never
    changes the report. Costs are reported in completion order, as the money is spent.
    """
    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures: dict[Future[TrialOutcome], tuple[str, int]] = {}
    reported: set[Future[TrialOutcome]] = set()
    outcomes: dict[tuple[str, int], TrialOutcome] = {}

    def report(future: Future[TrialOutcome]) -> None:
        # Marked first, so a cost is never reported twice even if the callback is interrupted.
        reported.add(future)
        outcome = future.result()
        outcomes[futures[future]] = outcome
        on_trial_cost(outcome.usage.cost_usd)

    try:
        for task, trial in jobs:
            future = pool.submit(_run_trial, context, task, trial, trials_root)
            futures[future] = (task.id, trial)
        for future in as_completed(futures):
            report(future)
    except BaseException:
        # A plain `with` would wait for every queued trial, each a paid agent run, before
        # letting the exception out. Cancel the queue, let running trials finish, and
        # still report what they cost before re-raising.
        pool.shutdown(wait=True, cancel_futures=True)
        for future in futures:
            if future in reported or future.cancelled() or future.exception() is not None:
                continue
            reported.add(future)
            on_trial_cost(future.result().usage.cost_usd)
        for cost in context.interrupted_costs:
            on_trial_cost(cost)
        raise
    pool.shutdown(wait=True)
    return outcomes


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


def _run_trial(context: _TrialContext, task: Task, trial: int, trials_root: Path) -> TrialOutcome:
    relative_trial_dir = f"{TRIALS_DIR}/{safe_task_dirname(task.id)}/{trial}"
    trial_dir = trials_root.parent / relative_trial_dir
    workdir = trial_dir / "workdir"
    trace_dir = trial_dir / "trace"
    workdir.mkdir(parents=True)
    trace_dir.mkdir()

    def outcome(
        status: TrialStatus, reward, error_class, detail, usage=Usage(), wall_clock_s=0.0, load_report=None,
        gateway=None,
    ) -> TrialOutcome:
        return TrialOutcome(
            task_id=task.id,
            trial=trial,
            status=status,
            reward=reward,
            error_class=error_class,
            detail=detail,
            usage=usage,
            wall_clock_s=wall_clock_s,
            trial_dir=relative_trial_dir,
            load_report=load_report,
            gateway=gateway,
        )

    workspace_problem = _materialize_workspace(task.workspace_files, workdir)
    if workspace_problem is not None:
        result = outcome("errored", None, "workspace_error", workspace_problem)
    else:
        result = _run_and_grade(context, task, trial_dir, workdir, trace_dir, outcome)
    _write_json(trial_dir / OUTCOME_FILE, result.to_dict())
    return result


@dataclass(frozen=True)
class _Account:
    """A run's account before grading. error_class None means it completed and will be graded."""

    error_class: str | None
    detail: str
    usage: Usage
    wall_clock_s: float
    load_report: LoadReport | None = None
    result: AgentResult | None = None  # set only when the result was well formed
    check_loads: bool = False  # whether the controller's extension check applies


def _run_and_grade(context: _TrialContext, task: Task, trial_dir: Path, workdir: Path, trace_dir: Path,
                   outcome) -> TrialOutcome:  # fmt: skip
    started = time.monotonic()
    ledger = None
    if context.gateway is None:
        returned = _call(lambda: context.agent.run(task, workdir, trace_dir))
    else:
        trial_key, endpoint = context.gateway.open_trial()
        run_returned = False
        try:
            returned = _call(lambda: context.agent.run(task, workdir, trace_dir, endpoint=endpoint))
            run_returned = True
        finally:
            # Always, even on Ctrl-C: the token must stop working and the spend must be counted.
            ledger = context.gateway.close_trial(trial_key)
            _write_json(trial_dir / GATEWAY_FILE, _ledger_to_dict(ledger))
            if not run_returned:
                context.interrupted_costs.append(ledger.usage.cost_usd)

    account = _account_for(returned, time.monotonic() - started)
    note = ""
    if ledger is not None:
        account = _apply_ledger(account, ledger, context.model, context.gateway.max_calls_per_trial)
        if any(call.error_code in UNBILLED_ERROR_CODES for call in ledger.calls):
            note = _lower_bound_note(ledger)
    if account.check_loads:
        error_class, detail = _apply_load_check(account.error_class, account.detail, context.expected_extensions,
                                                account.load_report)  # fmt: skip
        account = dataclasses.replace(account, error_class=error_class, detail=detail)
    summary = GatewaySummary.of(ledger) if ledger is not None else None

    def finish(status: TrialStatus, reward, error_class, detail) -> TrialOutcome:
        detail = f"{detail}; {note}" if detail and note else (detail or note)
        return outcome(status, reward, error_class, detail, account.usage, account.wall_clock_s,
                       account.load_report, summary)  # fmt: skip

    if account.error_class is not None:
        return finish("errored", None, account.error_class, account.detail)
    result = account.result
    try:
        grade = context.suite.grade(task, result, workdir)
    except Exception as error:
        return finish("errored", None, "grader_error", _describe(error))
    problem = _grade_problem(grade)
    if problem is not None:
        return finish("errored", None, "grader_error", problem)
    return finish("scored", float(grade.reward), None, grade.detail)


def _call(run: Callable[[], object]) -> object:
    """What run returned, or the Exception it raised (the contract says run never raises, but an
    adapter bug is still a trial outcome). A BaseException such as Ctrl-C propagates."""
    try:
        return run()
    except Exception as error:
        return error


def _account_for(returned: object, elapsed_s: float) -> _Account:
    """The agent's own account of the run, checked field by field."""
    if isinstance(returned, Exception):
        return _Account("agent_error", _describe(returned), Usage(), elapsed_s)
    if not isinstance(returned, AgentResult):
        return _Account("agent_error", f"agent returned {type(returned).__name__}, not AgentResult", Usage(), elapsed_s)
    result = returned
    usage, usage_problem = _checked_usage(result.usage)
    load_report, report_problem = _checked_load_report(result.load_report)
    field_problems = [problem for problem in (_result_field_problem(result), report_problem) if problem]
    field_problem = "; ".join(field_problems) or None
    if field_problem is not None:
        # Every field ends up in outcome.json and in the overseer's evidence, so a value that
        # can't be serialized or compared would abort the whole evaluation, not just this trial.
        detail = field_problem if usage_problem is None else f"{field_problem}; {usage_problem}"
        return _Account("agent_error", f"agent returned a malformed result: {detail}", usage, elapsed_s)
    if usage_problem is not None:
        # Spend feeds the budget cap (P12): a negative cost would offset real spend and a NaN
        # would make every cap comparison False, so an unusable report can't be trusted at all.
        return _Account("agent_error", f"agent reported invalid usage: {usage_problem}", usage, result.wall_clock_s,
                        load_report)  # fmt: skip
    if result.status not in ("completed", "errored"):
        return _Account("agent_error", f"unknown agent status {result.status!r}", usage, result.wall_clock_s,
                        load_report)  # fmt: skip
    if result.status == "errored":
        error_class, detail = _agent_error(result)
    else:
        error_class, detail = None, ""
    return _Account(error_class, detail, usage, result.wall_clock_s, load_report, result, check_loads=True)


def _apply_ledger(account: _Account, ledger: TrialLedger, model: ModelSpec, max_calls: int) -> _Account:
    """The account after the gateway's ledger, which outranks the agent on usage and model use."""
    account = dataclasses.replace(account, usage=ledger.usage)

    def errored(error_class: str, detail: str) -> _Account:
        if account.error_class is not None and account.detail:
            detail = f"{detail} (the agent reported {account.error_class}: {account.detail})"
        return dataclasses.replace(account, error_class=error_class, detail=detail)

    mismatches = list(ledger.violations)
    mismatches += [
        f"{call.path}: the provider reported model {call.model_reported!r}, not {model.qualified}"
        for call in ledger.calls
        if call.model_reported is not None and not model_matches(call.model_reported, model)
    ]
    if mismatches:
        mismatches = list(dict.fromkeys(mismatches))  # the gateway already lists some of these
        return errored("model_mismatch", "the model gateway recorded: " + "; ".join(mismatches))
    if account.error_class in (None, "model_mismatch"):
        return account
    if ledger.limit_reached:
        return errored("agent_error", f"model call limit ({max_calls}) reached")
    quota_codes = sorted({call.error_code for call in ledger.calls if call.error_code in QUOTA_ERROR_CODES})
    if quota_codes and account.error_class != "quota":
        return errored("quota", f"the model provider refused calls ({', '.join(quota_codes)})")
    return account


def _lower_bound_note(ledger: TrialLedger) -> str:
    counts: dict[str, int] = {}
    for call in ledger.calls:
        if call.error_code in UNBILLED_ERROR_CODES:
            counts[call.error_code] = counts.get(call.error_code, 0) + 1
    listing = ", ".join(f"{count} {code}" for code, count in sorted(counts.items()))
    return f"spend is a lower bound: the gateway could not count the usage of some calls ({listing})"


def _ledger_to_dict(ledger: TrialLedger) -> dict[str, Any]:
    # The ledger holds no token and no key: GatewayCall records paths, models, statuses and usage.
    return {
        "calls": [{**asdict(call), "usage": asdict(call.usage)} for call in ledger.calls],
        "violations": list(ledger.violations),
        "limit_reached": ledger.limit_reached,
    }


def _agent_error(result: AgentResult) -> tuple[str, str]:
    """(error class, detail) of a run the agent itself reported as errored."""
    detail = result.error or ""
    error_class = result.error_class
    if error_class not in AGENT_ERROR_CLASSES:
        # Keep the unknown class visible to readers, but only report classes the rest of HIVE knows.
        if error_class is not None:
            detail = f"[unknown error_class {error_class!r}] {detail}"
        error_class = "agent_error"
    return error_class, detail


def _apply_load_check(
    error_class: str | None, detail: str, expected_extensions: Sequence[str], load_report: LoadReport | None
) -> tuple[str | None, str]:
    """The trial's (error class or None for completed, detail) after the controller's extension check."""
    if error_class == "model_mismatch":  # P3 outranks everything
        return error_class, detail
    # An errored run may have died before its extensions loaded (a timeout kill, a runner
    # crash, a framework that never started), so only an affirmative failure counts there.
    problem = load_problem(expected_extensions, load_report, run_errored=error_class is not None)
    if problem is None:
        return error_class, detail
    problem_class, problem_detail = problem
    # A throw explains a crash, but a timeout, quota or provider error happens around the
    # candidate's code rather than in it, and any other class is already the agent's own
    # account of an extension or model problem.
    if problem_class == "extension_error" and error_class not in (None, "agent_error"):
        return error_class, detail
    if error_class is not None and detail:
        # Keep the agent's own account: it may name a cause the load report does not.
        problem_detail = f"{problem_detail} (the agent reported {error_class}: {detail})"
    return problem_class, problem_detail


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _result_field_problem(result: AgentResult) -> str | None:
    problems = []
    if not isinstance(result.final_text, str):
        problems.append(f"final_text is {type(result.final_text).__name__}, not str")
    for name in ("error", "error_class"):
        value = getattr(result, name)
        if value is not None and not isinstance(value, str):
            problems.append(f"{name} is {type(value).__name__}, not str or None")
    wall = result.wall_clock_s
    if isinstance(wall, bool) or not isinstance(wall, int | float) or not math.isfinite(wall) or wall < 0:
        problems.append(f"wall_clock_s {wall!r} is not a finite number >= 0")
    return "; ".join(problems) or None


def _checked_load_report(report: object) -> tuple[LoadReport | None, str | None]:
    """(load report to record, problem or None). Containers are normalized to tuples.

    Checked field by field because the report is written to outcome.json and its paths are
    used as keys by the load check: one bad value would abort the whole evaluation.
    """
    if report is None:
        return None, None
    if not isinstance(report, LoadReport):
        return None, f"load_report is {type(report).__name__}, not LoadReport or None"
    problems: list[str] = []
    extensions = _as_tuple(report.extensions)
    if extensions is None:
        problems.append(f"load_report.extensions is {type(report.extensions).__name__}, not a tuple")
        extensions = ()
    loads: list[ExtensionLoad] = []
    for index, load in enumerate(extensions):
        where = f"load_report.extensions[{index}]"
        if not isinstance(load, ExtensionLoad):
            problems.append(f"{where} is {type(load).__name__}, not ExtensionLoad")
            continue
        registered = _as_tuple(load.registered)
        if not isinstance(load.path, str):
            problems.append(f"{where}.path is {type(load.path).__name__}, not str")
        if not isinstance(load.loaded, bool):
            problems.append(f"{where}.loaded is {type(load.loaded).__name__}, not bool")
        if load.error is not None and not isinstance(load.error, str):
            problems.append(f"{where}.error is {type(load.error).__name__}, not str or None")
        if registered is None or not all(isinstance(entry, str) for entry in registered):
            problems.append(f"{where}.registered is not a tuple of str")
        else:
            loads.append(ExtensionLoad(path=load.path, loaded=load.loaded, error=load.error, registered=registered))
    runtime_errors = _as_tuple(report.runtime_errors)
    if runtime_errors is None or not all(isinstance(error, str) for error in runtime_errors):
        problems.append("load_report.runtime_errors is not a tuple of str")
    if problems:
        return None, "; ".join(problems)
    return LoadReport(extensions=tuple(loads), runtime_errors=runtime_errors), None


def _as_tuple(value: object) -> tuple[Any, ...] | None:
    # A list is the same data; a str is a sequence too, but never a list of entries.
    return tuple(value) if isinstance(value, tuple | list) else None


def _is_valid_cost(cost: object) -> bool:
    # bool is an int subclass, but True as a cost is an adapter bug, not $1.
    return not isinstance(cost, bool) and isinstance(cost, int | float) and math.isfinite(cost) and cost >= 0


def _checked_usage(usage: object) -> tuple[Usage, str | None]:
    """(usage to record, problem or None). Every field must be a finite number >= 0, tokens ints.

    When only the token counts are bad, a valid cost is kept: that money was still spent,
    and dropping it would understate the run's spend.
    """
    if not isinstance(usage, Usage):
        return Usage(), f"usage is {type(usage).__name__}, not Usage"
    problems = []
    for name in _TOKEN_FIELDS:
        value = getattr(usage, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            problems.append(f"{name} {value!r} is not an integer >= 0")
    cost = usage.cost_usd
    if not _is_valid_cost(cost):
        problems.append(f"cost_usd {cost!r} is not a finite number >= 0")
    if not problems:
        return usage, None
    kept = Usage(cost_usd=float(cost)) if _is_valid_cost(cost) else Usage()
    return kept, "; ".join(problems)


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
            # UTF-8 per the Task contract, never the locale's encoding.
            target.write_text(text, encoding="utf-8")
        # ValueError covers UnicodeEncodeError, e.g. a lone surrogate from surrogateescape decoding.
        except (OSError, TypeError, ValueError) as error:
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
