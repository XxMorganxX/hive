import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from hive.candidate import Candidate
from hive.contracts import AGENT_ERROR_CLASSES, AgentResult, Grade, ModelSpec, Task, Usage
from hive.evaluate import (
    TRIAL_ERROR_CLASSES,
    EvalReport,
    EvaluationError,
    TaskOutcome,
    TrialOutcome,
    evaluate,
)

MODEL = ModelSpec(provider="fake", id="model-1")


# --- test doubles -------------------------------------------------------------------------------

RunBehavior = Callable[[Task, Path, Path], AgentResult]


def completed(text: str = "ok", usage: Usage | None = None, wall_clock_s: float = 0.5) -> AgentResult:
    return AgentResult(status="completed", final_text=text, usage=usage or Usage(), wall_clock_s=wall_clock_s)


class FakeAgent:
    def __init__(self, behavior: RunBehavior):
        self.behavior = behavior
        self.calls: list[tuple[str, Path, Path]] = []
        self._lock = threading.Lock()

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        with self._lock:
            self.calls.append((task.id, workdir, trace_dir))
        return self.behavior(task, workdir, trace_dir)


class FakeAdapter:
    name = "fake"

    def __init__(self, behavior: RunBehavior | None = None, build_error: Exception | None = None):
        self.agent = FakeAgent(behavior or (lambda task, workdir, trace_dir: completed()))
        self.build_error = build_error
        self.build_calls: list[tuple[str, ModelSpec, Mapping[str, Any]]] = []

    def validate(self, candidate: Candidate) -> list[str]:
        return []

    def build(self, candidate: Candidate, model: ModelSpec, settings: Mapping[str, Any]) -> FakeAgent:
        self.build_calls.append((candidate.id, model, settings))
        if self.build_error is not None:
            raise self.build_error
        return self.agent

    def describe_valves(self) -> str:
        return "none"


class FakeSuite:
    name = "fake-suite"

    def __init__(
        self,
        tasks: list[Task],
        grade: Callable[[Task, AgentResult, Path], Grade] | None = None,
    ):
        self._tasks = tasks
        self._grade = grade or (lambda task, result, workdir: Grade(reward=1.0, detail="passed"))
        self.grade_calls: list[str] = []
        self._lock = threading.Lock()

    def describe(self) -> str:
        return "fake"

    def metric(self):
        raise NotImplementedError

    def tasks(self, split):
        return [task for task in self._tasks if task.split == split]

    def grade(self, task: Task, result: AgentResult, workdir: Path) -> Grade:
        with self._lock:
            self.grade_calls.append(task.id)
        return self._grade(task, result, workdir)


class MislabeledSuite(FakeSuite):
    """Returns every task regardless of the split requested."""

    def tasks(self, split):
        return list(self._tasks)


def search_task(task_id: str, files: Mapping[str, str] | None = None) -> Task:
    return Task(id=task_id, split="search", instruction=f"do {task_id}", workspace_files=dict(files or {}))


@pytest.fixture
def candidate(tmp_path) -> Candidate:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "candidate.toml").write_text('schema_version = 1\nagent = "fake"\n')
    return Candidate.load(root)


def run_eval(candidate, adapter, suite, out_dir, *, trials_per_task=1, concurrency=1, settings=None):
    return evaluate(
        candidate,
        adapter,
        suite,
        "search",
        model=MODEL,
        agent_settings=settings or {},
        trials_per_task=trials_per_task,
        out_dir=out_dir,
        concurrency=concurrency,
    )


def only_trial(report: EvalReport) -> TrialOutcome:
    assert len(report.tasks) == 1 and len(report.tasks[0].trials) == 1
    return report.tasks[0].trials[0]


# --- evaluate(): argument checks ----------------------------------------------------------------


def test_trial_error_classes_extend_agent_error_classes():
    assert TRIAL_ERROR_CLASSES == AGENT_ERROR_CLASSES + ("grader_error", "workspace_error")


def test_empty_split_raises_evaluation_error(candidate, tmp_path):
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), FakeSuite([]), tmp_path / "out")


@pytest.mark.parametrize("trials_per_task", [0, -1])
def test_trials_per_task_below_one_raises_evaluation_error(candidate, tmp_path, trials_per_task):
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), tmp_path / "out", trials_per_task=trials_per_task)


@pytest.mark.parametrize("concurrency", [0, -2])
def test_concurrency_below_one_raises_evaluation_error(candidate, tmp_path, concurrency):
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), tmp_path / "out", concurrency=concurrency)


def test_duplicate_task_ids_raise_evaluation_error(candidate, tmp_path):
    suite = FakeSuite([search_task("t1"), search_task("t1")])
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), suite, tmp_path / "out")


def test_task_ids_differing_only_by_case_raise_evaluation_error(candidate, tmp_path):
    # They would share one trial directory on a case-insensitive filesystem (the macOS default).
    suite = FakeSuite([search_task("Task"), search_task("task")])
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), suite, tmp_path / "out")


def test_task_from_another_split_raises_evaluation_error(candidate, tmp_path):
    held_out = Task(id="secret", split="heldout", instruction="x")
    suite = MislabeledSuite([search_task("t1"), held_out])
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), suite, tmp_path / "out")


def test_candidate_owned_by_another_adapter_raises_evaluation_error(tmp_path):
    root = tmp_path / "other"
    root.mkdir()
    (root / "candidate.toml").write_text('schema_version = 1\nagent = "pi"\n')
    with pytest.raises(EvaluationError):
        run_eval(Candidate.load(root), FakeAdapter(), FakeSuite([search_task("t1")]), tmp_path / "out")


def test_out_dir_with_existing_trials_raises_evaluation_error(candidate, tmp_path):
    out_dir = tmp_path / "out"
    run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), out_dir)
    with pytest.raises(EvaluationError):
        run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), out_dir)


def test_build_exception_propagates(candidate, tmp_path):
    adapter = FakeAdapter(build_error=RuntimeError("missing binary"))
    with pytest.raises(RuntimeError, match="missing binary"):
        run_eval(candidate, adapter, FakeSuite([search_task("t1")]), tmp_path / "out")


# --- evaluate(): running trials -----------------------------------------------------------------


def test_agent_is_built_once_with_model_and_settings(candidate, tmp_path):
    adapter = FakeAdapter()
    suite = FakeSuite([search_task("t1"), search_task("t2")])
    run_eval(candidate, adapter, suite, tmp_path / "out", trials_per_task=3, settings={"timeout_s": 5})
    assert adapter.build_calls == [(candidate.id, MODEL, {"timeout_s": 5})]
    assert len(adapter.agent.calls) == 6


def test_report_records_candidate_suite_split_model_and_trials_per_task(candidate, tmp_path):
    report = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), tmp_path / "out", trials_per_task=2)
    assert (report.candidate_id, report.suite, report.split, report.model, report.trials_per_task) == (
        candidate.id,
        "fake-suite",
        "search",
        "fake/model-1",
        2,
    )


def test_scored_trial_records_grade_reward_detail_usage_and_wall_clock(candidate, tmp_path):
    usage = Usage(input_tokens=10, output_tokens=3, cost_usd=0.01)
    adapter = FakeAdapter(lambda task, workdir, trace_dir: completed(usage=usage, wall_clock_s=2.5))
    suite = FakeSuite([search_task("t1")], grade=lambda task, result, workdir: Grade(reward=0.75, detail="3 of 4"))
    trial = only_trial(run_eval(candidate, adapter, suite, tmp_path / "out"))
    assert (trial.status, trial.reward, trial.error_class, trial.detail) == ("scored", 0.75, None, "3 of 4")
    assert (trial.usage, trial.wall_clock_s) == (usage, 2.5)


def test_report_order_is_suite_order_and_trials_ascending_under_concurrency(candidate, tmp_path):
    # Earlier tasks sleep longer, so with a pool they finish last.
    task_ids = ["t1", "t2", "t3", "t4"]

    def slow_first(task, workdir, trace_dir):
        time.sleep(0.02 * (len(task_ids) - task_ids.index(task.id)))
        return completed()

    suite = FakeSuite([search_task(task_id) for task_id in task_ids])
    report = run_eval(candidate, FakeAdapter(slow_first), suite, tmp_path / "out", trials_per_task=3, concurrency=6)
    assert [task.task_id for task in report.tasks] == task_ids
    assert all([trial.trial for trial in task.trials] == [1, 2, 3] for task in report.tasks)
    assert all(trial.task_id == task.task_id for task in report.tasks for trial in task.trials)


def test_trials_run_concurrently_up_to_the_concurrency_limit(candidate, tmp_path):
    active = 0
    peak = 0
    lock = threading.Lock()

    def track(task, workdir, trace_dir):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return completed()

    suite = FakeSuite([search_task(f"t{i}") for i in range(6)])
    run_eval(candidate, FakeAdapter(track), suite, tmp_path / "out", concurrency=3)
    assert peak == 3


def test_each_trial_gets_its_own_workdir_and_trace_dir(candidate, tmp_path):
    out_dir = tmp_path / "out"
    adapter = FakeAdapter()
    report = run_eval(candidate, adapter, FakeSuite([search_task("t1")]), out_dir, trials_per_task=2)
    expected = {
        (out_dir / "trials" / "t1" / "1" / "workdir", out_dir / "trials" / "t1" / "1" / "trace"),
        (out_dir / "trials" / "t1" / "2" / "workdir", out_dir / "trials" / "t1" / "2" / "trace"),
    }
    assert {(workdir, trace_dir) for _, workdir, trace_dir in adapter.agent.calls} == expected
    assert all(workdir.is_dir() and trace_dir.is_dir() for workdir, trace_dir in expected)
    assert [trial.trial_dir for trial in report.tasks[0].trials] == ["trials/t1/1", "trials/t1/2"]


def test_unsafe_task_ids_get_distinct_directories_inside_trials(candidate, tmp_path):
    out_dir = tmp_path / "out"
    suite = FakeSuite([search_task("a/b"), search_task("a_b"), search_task(".."), search_task("a b")])
    report = run_eval(candidate, FakeAdapter(), suite, out_dir)
    trials_root = (out_dir / "trials").resolve()
    dirs = [(out_dir / task.trials[0].trial_dir).resolve() for task in report.tasks]
    assert len(set(dirs)) == len(dirs)
    assert all(directory.parent.parent == trials_root and directory.is_dir() for directory in dirs)


def test_unsafe_task_id_directory_name_is_deterministic(candidate, tmp_path):
    first = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("x/y")]), tmp_path / "one")
    second = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("x/y")]), tmp_path / "two")
    assert Path(only_trial(first).trial_dir).parent.name == Path(only_trial(second).trial_dir).parent.name


def test_very_long_task_id_gets_a_short_directory_name(candidate, tmp_path):
    report = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t" * 500)]), tmp_path / "out")
    assert len(Path(only_trial(report).trial_dir).parent.name) <= 100


def test_workspace_files_are_written_into_workdir_before_the_run(candidate, tmp_path):
    seen = {}

    def read_files(task, workdir, trace_dir):
        seen["stub"] = (workdir / "pkg" / "solution.py").read_text()
        return completed()

    suite = FakeSuite([search_task("t1", files={"pkg/solution.py": "def f(): ..."})])
    run_eval(candidate, FakeAdapter(read_files), suite, tmp_path / "out")
    assert seen == {"stub": "def f(): ..."}


@pytest.mark.parametrize("bad_path", ["../escape.txt", "sub/../../escape.txt"])
def test_workspace_path_escaping_workdir_is_workspace_error(candidate, tmp_path, bad_path):
    out_dir = tmp_path / "out"
    adapter = FakeAdapter()
    suite = FakeSuite([search_task("t1", files={bad_path: "x"})])
    trial = only_trial(run_eval(candidate, adapter, suite, out_dir))
    assert (trial.status, trial.error_class, trial.reward) == ("errored", "workspace_error", None)
    assert not list(tmp_path.rglob("escape.txt"))
    assert adapter.agent.calls == []


def test_absolute_workspace_path_is_workspace_error(candidate, tmp_path):
    target = tmp_path / "absolute.txt"
    suite = FakeSuite([search_task("t1", files={str(target): "x"})])
    trial = only_trial(run_eval(candidate, FakeAdapter(), suite, tmp_path / "out"))
    assert (trial.status, trial.error_class) == ("errored", "workspace_error")
    assert not target.exists()


def test_workspace_path_naming_the_workdir_itself_is_workspace_error(candidate, tmp_path):
    suite = FakeSuite([search_task("t1", files={".": "x"})])
    trial = only_trial(run_eval(candidate, FakeAdapter(), suite, tmp_path / "out"))
    assert (trial.status, trial.error_class) == ("errored", "workspace_error")


def test_agent_exception_becomes_agent_error_with_the_message(candidate, tmp_path):
    def crash(task, workdir, trace_dir):
        raise ValueError("adapter bug")

    suite = FakeSuite([search_task("t1")])
    trial = only_trial(run_eval(candidate, FakeAdapter(crash), suite, tmp_path / "out"))
    assert (trial.status, trial.error_class, trial.reward) == ("errored", "agent_error", None)
    assert "adapter bug" in trial.detail
    assert suite.grade_calls == []


def test_errored_result_keeps_error_class_usage_and_wall_clock(candidate, tmp_path):
    usage = Usage(input_tokens=7, cost_usd=0.002)

    def time_out(task, workdir, trace_dir):
        return AgentResult(
            status="errored", final_text="", usage=usage, wall_clock_s=30.0, error="ran 30s", error_class="timeout"
        )

    trial = only_trial(run_eval(candidate, FakeAdapter(time_out), FakeSuite([search_task("t1")]), tmp_path / "out"))
    assert (trial.status, trial.error_class, trial.detail) == ("errored", "timeout", "ran 30s")
    assert (trial.usage, trial.wall_clock_s) == (usage, 30.0)


def test_errored_result_without_error_class_falls_back_to_agent_error(candidate, tmp_path):
    def errored(task, workdir, trace_dir):
        return AgentResult(status="errored", final_text="", usage=Usage(), wall_clock_s=1.0, error="no answer")

    trial = only_trial(run_eval(candidate, FakeAdapter(errored), FakeSuite([search_task("t1")]), tmp_path / "out"))
    assert (trial.error_class, trial.detail) == ("agent_error", "no answer")


def test_errored_result_with_unknown_error_class_falls_back_to_agent_error(candidate, tmp_path):
    def errored(task, workdir, trace_dir):
        return AgentResult(
            status="errored", final_text="", usage=Usage(), wall_clock_s=1.0, error="boom", error_class="weird"
        )

    trial = only_trial(run_eval(candidate, FakeAdapter(errored), FakeSuite([search_task("t1")]), tmp_path / "out"))
    assert trial.error_class == "agent_error"
    assert "weird" in trial.detail and "boom" in trial.detail


def test_errored_run_is_never_graded(candidate, tmp_path):
    def errored(task, workdir, trace_dir):
        return AgentResult(status="errored", final_text="", usage=Usage(), wall_clock_s=1.0, error_class="quota")

    suite = FakeSuite([search_task("t1")])
    run_eval(candidate, FakeAdapter(errored), suite, tmp_path / "out")
    assert suite.grade_calls == []


def test_agent_returning_a_non_result_is_agent_error(candidate, tmp_path):
    trial = only_trial(
        run_eval(candidate, FakeAdapter(lambda task, workdir, trace_dir: "done"), FakeSuite([search_task("t1")]), tmp_path / "out")
    )
    assert (trial.status, trial.error_class) == ("errored", "agent_error")


def test_grader_exception_is_grader_error(candidate, tmp_path):
    def broken_grader(task, result, workdir):
        raise KeyError("reference missing")

    usage = Usage(output_tokens=4)
    adapter = FakeAdapter(lambda task, workdir, trace_dir: completed(usage=usage))
    trial = only_trial(run_eval(candidate, adapter, FakeSuite([search_task("t1")], grade=broken_grader), tmp_path / "out"))
    assert (trial.status, trial.error_class, trial.reward) == ("errored", "grader_error", None)
    assert "reference missing" in trial.detail
    assert trial.usage == usage


@pytest.mark.parametrize("bad_reward", [-0.1, 1.5, math.nan, math.inf, True, "1.0", None])
def test_invalid_reward_is_grader_error(candidate, tmp_path, bad_reward):
    suite = FakeSuite([search_task("t1")], grade=lambda task, result, workdir: Grade(reward=bad_reward))
    trial = only_trial(run_eval(candidate, FakeAdapter(), suite, tmp_path / "out"))
    assert (trial.status, trial.error_class) == ("errored", "grader_error")


@pytest.mark.parametrize("edge_reward", [0, 0.0, 1, 1.0])
def test_rewards_at_the_bounds_are_scored(candidate, tmp_path, edge_reward):
    suite = FakeSuite([search_task("t1")], grade=lambda task, result, workdir: Grade(reward=edge_reward))
    trial = only_trial(run_eval(candidate, FakeAdapter(), suite, tmp_path / "out"))
    assert (trial.status, trial.reward) == ("scored", float(edge_reward))


def test_grader_returning_a_non_grade_is_grader_error(candidate, tmp_path):
    suite = FakeSuite([search_task("t1")], grade=lambda task, result, workdir: 1.0)
    trial = only_trial(run_eval(candidate, FakeAdapter(), suite, tmp_path / "out"))
    assert (trial.status, trial.error_class) == ("errored", "grader_error")


def test_outcome_json_is_written_for_every_trial(candidate, tmp_path):
    def crash_on_t2(task, workdir, trace_dir):
        if task.id == "t2":
            raise RuntimeError("crash")
        return completed()

    suite = FakeSuite([search_task("t1"), search_task("t2"), search_task("t3", files={"../x": "y"})])
    out_dir = tmp_path / "out"
    report = run_eval(candidate, FakeAdapter(crash_on_t2), suite, out_dir, trials_per_task=2)
    for task in report.tasks:
        for trial in task.trials:
            written = json.loads((out_dir / trial.trial_dir / "outcome.json").read_text())
            assert (written["task_id"], written["trial"], written["status"]) == (trial.task_id, trial.trial, trial.status)


def test_report_json_is_written_and_loads_back_equal(candidate, tmp_path):
    out_dir = tmp_path / "out"
    report = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1"), search_task("t2")]), out_dir, trials_per_task=2)
    assert EvalReport.from_dict(json.loads((out_dir / "report.json").read_text())) == report


# --- report aggregates --------------------------------------------------------------------------


def scored(task_id: str, trial: int, reward: float, usage: Usage | None = None) -> TrialOutcome:
    return TrialOutcome(
        task_id=task_id,
        trial=trial,
        status="scored",
        reward=reward,
        error_class=None,
        detail="",
        usage=usage or Usage(),
        wall_clock_s=1.0,
        trial_dir=f"trials/{task_id}/{trial}",
    )


def errored(task_id: str, trial: int, error_class: str = "timeout", usage: Usage | None = None) -> TrialOutcome:
    return TrialOutcome(
        task_id=task_id,
        trial=trial,
        status="errored",
        reward=None,
        error_class=error_class,
        detail="timed out",
        usage=usage or Usage(),
        wall_clock_s=2.0,
        trial_dir=f"trials/{task_id}/{trial}",
    )


def report_of(*tasks: TaskOutcome, trials_per_task: int = 2) -> EvalReport:
    return EvalReport(
        candidate_id="abc",
        suite="fake-suite",
        split="search",
        model="fake/model-1",
        trials_per_task=trials_per_task,
        tasks=tasks,
    )


def test_errored_trial_counts_as_zero_in_selection_reward():
    task = TaskOutcome(task_id="t1", trials=(scored("t1", 1, 1.0), errored("t1", 2)))
    assert task.selection_reward == 0.5


def test_measured_mean_ignores_errored_trials():
    task = TaskOutcome(task_id="t1", trials=(scored("t1", 1, 1.0), errored("t1", 2)))
    assert task.measured_mean == 1.0


def test_task_measured_mean_is_none_when_no_trial_scored():
    task = TaskOutcome(task_id="t1", trials=(errored("t1", 1), errored("t1", 2)))
    assert task.measured_mean is None


def test_task_counts_scored_and_errored_trials():
    task = TaskOutcome(task_id="t1", trials=(scored("t1", 1, 0.0), errored("t1", 2), errored("t1", 3)))
    assert (task.n_scored, task.n_errored) == (1, 2)


def test_selection_reward_is_zero_for_task_without_trials():
    assert TaskOutcome(task_id="t1", trials=()).selection_reward == 0.0


def test_errored_trial_counts_as_zero_in_selection_score():
    report = report_of(
        TaskOutcome("t1", (scored("t1", 1, 1.0), scored("t1", 2, 1.0))),
        TaskOutcome("t2", (errored("t2", 1), errored("t2", 2))),
    )
    assert report.selection_score == 0.5


def test_selection_score_is_zero_for_report_without_tasks():
    assert report_of().selection_score == 0.0


def test_report_measured_mean_skips_tasks_without_scored_trials():
    report = report_of(
        TaskOutcome("t1", (scored("t1", 1, 1.0), scored("t1", 2, 0.0))),
        TaskOutcome("t2", (errored("t2", 1), errored("t2", 2))),
        TaskOutcome("t3", (scored("t3", 1, 1.0), errored("t3", 2))),
    )
    assert report.measured_mean == 0.75


def test_report_measured_mean_is_none_when_nothing_scored():
    assert report_of(TaskOutcome("t1", (errored("t1", 1),))).measured_mean is None


def test_stderr_is_sample_stdev_of_task_rewards_over_root_n():
    report = report_of(
        TaskOutcome("t1", (scored("t1", 1, 1.0),)),
        TaskOutcome("t2", (scored("t2", 1, 0.0),)),
        TaskOutcome("t3", (scored("t3", 1, 0.5),)),
    )
    # rewards 1.0, 0.0, 0.5: mean 0.5, sample variance 0.25, stdev 0.5.
    assert report.stderr == pytest.approx(0.5 / math.sqrt(3))


def test_stderr_is_zero_with_fewer_than_two_tasks():
    assert report_of(TaskOutcome("t1", (scored("t1", 1, 1.0), scored("t1", 2, 0.0)))).stderr == 0.0


def test_report_counts_trials_and_errors():
    report = report_of(
        TaskOutcome("t1", (scored("t1", 1, 1.0), errored("t1", 2))),
        TaskOutcome("t2", (errored("t2", 1), errored("t2", 2))),
    )
    assert (report.n_trials, report.n_errored) == (4, 3)


def test_report_usage_and_cost_sum_over_every_trial():
    report = report_of(
        TaskOutcome("t1", (scored("t1", 1, 1.0, Usage(input_tokens=5, cost_usd=0.25)),)),
        TaskOutcome("t2", (errored("t2", 1, usage=Usage(input_tokens=3, output_tokens=2, cost_usd=0.5)),)),
    )
    assert report.usage == Usage(input_tokens=8, output_tokens=2, cost_usd=0.75)
    assert report.cost_usd == 0.75


def sample_report() -> EvalReport:
    return report_of(
        TaskOutcome("t1", (scored("t1", 1, 0.1, Usage(1, 2, 3, 4, 0.0123)), errored("t1", 2, "quota"))),
        TaskOutcome("t2", (scored("t2", 1, 1.0), scored("t2", 2, 1 / 3))),
    )


def test_to_dict_is_json_safe_and_includes_aggregates_and_every_trial():
    report = sample_report()
    data = json.loads(json.dumps(report.to_dict()))
    assert data["selection_score"] == report.selection_score
    assert data["measured_mean"] == report.measured_mean
    assert data["stderr"] == report.stderr
    assert data["cost_usd"] == report.cost_usd
    assert (data["n_trials"], data["n_errored"]) == (4, 1)
    assert [len(task["trials"]) for task in data["tasks"]] == [2, 2]


def test_from_dict_round_trips_exactly_through_json():
    report = sample_report()
    assert EvalReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report


def test_from_dict_recomputes_aggregates_instead_of_trusting_them():
    data = sample_report().to_dict()
    data["selection_score"] = 0.99
    data["tasks"][0]["selection_reward"] = 0.99
    assert EvalReport.from_dict(data).selection_score == sample_report().selection_score


# --- spend accounting and aborted evaluations ---------------------------------------------------


class AbortTrial(BaseException):
    """Escapes `except Exception`, as a real interrupt or an evaluator bug would."""


def costly_run(cost: float, counter: list[str], lock: threading.Lock, raise_on: str | None = None) -> RunBehavior:
    def run(task, workdir, trace_dir):
        with lock:
            counter.append(task.id)
        if task.id == raise_on:
            raise AbortTrial("worker blew up")
        time.sleep(0.01)
        return completed(usage=Usage(cost_usd=cost))

    return run


def test_on_trial_cost_is_called_once_per_finished_trial_with_its_cost(candidate, tmp_path):
    seen: list[float] = []
    suite = FakeSuite([search_task(f"t{i}") for i in range(3)])
    adapter = FakeAdapter(lambda task, workdir, trace_dir: completed(usage=Usage(cost_usd=0.25)))
    report = evaluate(
        candidate, adapter, suite, "search", model=MODEL, agent_settings={}, trials_per_task=2,
        out_dir=tmp_path / "out", concurrency=2, on_trial_cost=seen.append,
    )  # fmt: skip
    assert seen == [0.25] * 6
    assert sum(seen) == report.cost_usd


def test_interrupt_while_collecting_cancels_queued_trials_and_reports_finished_ones(candidate, tmp_path):
    calls: list[str] = []
    seen: list[float] = []

    def interrupt_after_first(cost: float) -> None:
        seen.append(cost)
        if len(seen) == 1:
            raise KeyboardInterrupt  # stands in for Ctrl-C arriving while the main thread waits

    suite = FakeSuite([search_task(f"t{i}") for i in range(12)])
    adapter = FakeAdapter(costly_run(0.1, calls, threading.Lock()))
    with pytest.raises(KeyboardInterrupt):
        evaluate(
            candidate, adapter, suite, "search", model=MODEL, agent_settings={}, trials_per_task=1,
            out_dir=tmp_path / "out", concurrency=1, on_trial_cost=interrupt_after_first,
        )  # fmt: skip
    assert len(calls) < 12, "queued paid trials kept running after the interrupt"
    # Every trial that ran finished (none raised), so every one of them was reported.
    assert len(seen) == len(calls)
    assert sum(seen) == pytest.approx(0.1 * len(calls))


def test_worker_exception_cancels_queued_trials_and_still_reports_finished_ones(candidate, tmp_path):
    calls: list[str] = []
    seen: list[float] = []
    suite = FakeSuite([search_task(f"t{i}") for i in range(12)])
    adapter = FakeAdapter(costly_run(0.1, calls, threading.Lock(), raise_on="t1"))
    with pytest.raises(AbortTrial):
        evaluate(
            candidate, adapter, suite, "search", model=MODEL, agent_settings={}, trials_per_task=1,
            out_dir=tmp_path / "out", concurrency=1, on_trial_cost=seen.append,
        )  # fmt: skip
    assert len(calls) < 12, "queued paid trials kept running after a worker raised"
    finished = len(calls) - 1  # t1 raised, everything else that started finished
    assert len(seen) == finished
    assert sum(seen) == pytest.approx(0.1 * finished)


# --- agent-reported usage -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_usage",
    [
        Usage(cost_usd=-5.0),
        Usage(cost_usd=math.nan),
        Usage(cost_usd=math.inf),
        Usage(cost_usd=True),
        Usage(cost_usd="0.1"),
        Usage(input_tokens=-1),
        Usage(output_tokens=1.5),
        Usage(cache_read_tokens=True),
        None,
        {"cost_usd": 0.1},
    ],
    ids=["negative-cost", "nan-cost", "inf-cost", "bool-cost", "str-cost", "negative-tokens", "float-tokens",
         "bool-tokens", "none", "dict"],
)  # fmt: skip
@pytest.mark.parametrize("status", ["completed", "errored"])
def test_invalid_usage_makes_the_trial_an_agent_error(candidate, tmp_path, bad_usage, status):
    def run(task, workdir, trace_dir):
        return AgentResult(status=status, final_text="ok", usage=bad_usage, wall_clock_s=1.0, error_class="timeout")

    suite = FakeSuite([search_task("t1")])
    report = run_eval(candidate, FakeAdapter(run), suite, tmp_path / "out")
    trial = only_trial(report)
    assert (trial.status, trial.error_class, trial.reward) == ("errored", "agent_error", None)
    assert "usage" in trial.detail
    assert suite.grade_calls == []
    cost = report.cost_usd
    assert math.isfinite(cost) and cost >= 0


def test_invalid_tokens_keep_a_valid_reported_cost(candidate, tmp_path):
    # The money was still spent; dropping it would understate the run's spend.
    usage = Usage(input_tokens=1.5, cost_usd=0.2)
    adapter = FakeAdapter(lambda task, workdir, trace_dir: completed(usage=usage))
    trial = only_trial(run_eval(candidate, adapter, FakeSuite([search_task("t1")]), tmp_path / "out"))
    assert trial.error_class == "agent_error"
    assert trial.usage == Usage(cost_usd=0.2)


# --- workspace encoding -------------------------------------------------------------------------


def test_workspace_text_that_cannot_be_encoded_is_a_workspace_error_for_that_trial(candidate, tmp_path):
    out_dir = tmp_path / "out"
    suite = FakeSuite([search_task("bad", files={"notes.txt": "caf\udce9"}), search_task("good")])
    report = run_eval(candidate, FakeAdapter(), suite, out_dir)
    bad, good = (task.trials[0] for task in report.tasks)
    assert (bad.status, bad.error_class) == ("errored", "workspace_error")
    assert "notes.txt" in bad.detail
    assert good.status == "scored"
    assert (out_dir / "report.json").is_file()


def test_workspace_files_are_written_as_utf8(candidate, tmp_path):
    # The Task contract fixes UTF-8; the locale's encoding must not decide what lands on disk.
    seen = {}

    def read_bytes(task, workdir, trace_dir):
        seen["bytes"] = (workdir / "names.txt").read_bytes()
        return completed()

    suite = FakeSuite([search_task("t1", files={"names.txt": "Zoë → 東京"})])
    run_eval(candidate, FakeAdapter(read_bytes), suite, tmp_path / "out")
    assert seen["bytes"] == "Zoë → 東京".encode("utf-8")


# --- trial paths ------------------------------------------------------------------------------


def test_trial_dir_is_relative_to_the_evaluation_out_dir(candidate, tmp_path):
    out_dir = tmp_path / "out"
    report = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), out_dir)
    trial = only_trial(report)
    assert trial.trial_dir == "trials/t1/1"
    assert str(tmp_path) not in (out_dir / "report.json").read_text()
    assert str(tmp_path) not in (out_dir / "trials" / "t1" / "1" / "outcome.json").read_text()


# --- malformed AgentResult fields ---------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [("error", object()), ("error_class", 7), ("final_text", None), ("wall_clock_s", float("nan")), ("wall_clock_s", -1.0)],
)
def test_malformed_agent_result_field_errors_one_trial_and_keeps_its_cost(candidate, tmp_path, field, value):
    def malformed(task, workdir, trace_dir):
        values = dict(status="errored", final_text="", usage=Usage(cost_usd=0.1), wall_clock_s=1.0, error="x")
        values[field] = value
        return AgentResult(**values)

    report = run_eval(candidate, FakeAdapter(malformed), FakeSuite([search_task("t1")]), tmp_path / "out")
    trial = only_trial(report)
    assert trial.status == "errored"
    assert trial.error_class == "agent_error"
    assert field in trial.detail
    assert trial.usage.cost_usd == pytest.approx(0.1)
    assert (tmp_path / "out" / "report.json").is_file()
