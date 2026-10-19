import http.client
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from fakes.fake_openai_upstream import CHAT_USAGE, FakeOpenAIUpstream
from hive.candidate import Candidate
from hive.contracts import (
    AGENT_ERROR_CLASSES,
    AgentAdapter,
    AgentResult,
    ExtensionLoad,
    ExtensionSpec,
    Grade,
    LoadReport,
    ModelEndpoint,
    ModelSpec,
    Task,
    Usage,
)
from hive.evaluate import (
    GATEWAY_FILE,
    TRIAL_ERROR_CLASSES,
    EvalReport,
    EvaluationError,
    GatewaySummary,
    TaskOutcome,
    TrialOutcome,
    evaluate,
)
from hive.gateway import ModelGateway
from hive.pricing import Prices

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

    def __init__(
        self,
        behavior: RunBehavior | None = None,
        build_error: Exception | None = None,
        spec: ExtensionSpec | None = None,
    ):
        self.agent = FakeAgent(behavior or (lambda task, workdir, trace_dir: completed()))
        self.build_error = build_error
        self.spec = spec
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

    def extension_spec(self) -> ExtensionSpec | None:
        return self.spec


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


# --- extensions: the controller's load check (P8) ----------------------------------------------

EXTENSION_SPEC = ExtensionSpec(
    language="python",
    suffixes=(".py",),
    entry_convention="define `def extension()`",
    framework_version="0.0.0",
    lifecycle=(),
)
EXTENSION_FILES = ("extensions/guard.py", "extensions/retry.py")
SPENT = Usage(input_tokens=11, output_tokens=5, cost_usd=0.03)


def all_loaded(*runtime_errors: str) -> LoadReport:
    return LoadReport(
        extensions=tuple(ExtensionLoad(path=path, loaded=True, registered=("before_tool_call",)) for path in EXTENSION_FILES),
        runtime_errors=runtime_errors,
    )


def guard_failed() -> LoadReport:
    return LoadReport(
        extensions=(
            ExtensionLoad(path="extensions/guard.py", loaded=False, error="SyntaxError: bad"),
            ExtensionLoad(path="extensions/retry.py", loaded=True),
        )
    )


@pytest.fixture
def extended_candidate(tmp_path) -> Candidate:
    root = tmp_path / "extended"
    (root / "extensions").mkdir(parents=True)
    (root / "candidate.toml").write_text('schema_version = 1\nagent = "fake"\n')
    for path in EXTENSION_FILES:
        (root / path).write_text("def extension(): ...\n")
    (root / "extensions" / "README.md").write_text("not an extension: wrong suffix\n")
    return Candidate.load(root)


def returning(result: AgentResult) -> RunBehavior:
    return lambda task, workdir, trace_dir: result


def run_extended(candidate, result: AgentResult, tmp_path, suite: FakeSuite | None = None) -> TrialOutcome:
    adapter = FakeAdapter(returning(result), spec=EXTENSION_SPEC)
    return only_trial(run_eval(candidate, adapter, suite or FakeSuite([search_task("t1")]), tmp_path / "out"))


def errored_result(error_class: str, report: LoadReport | None, error: str = "the agent's own message") -> AgentResult:
    return AgentResult(status="errored", final_text="", usage=SPENT, wall_clock_s=4.0, error=error,
                       error_class=error_class, load_report=report)  # fmt: skip


def completed_with(report: LoadReport | None) -> AgentResult:
    return AgentResult(status="completed", final_text="ok", usage=SPENT, wall_clock_s=4.0, load_report=report)


def test_fake_adapter_satisfies_the_agent_adapter_protocol():
    assert isinstance(FakeAdapter(spec=EXTENSION_SPEC), AgentAdapter)


def test_run_whose_extensions_all_loaded_is_scored_and_records_the_load_report(extended_candidate, tmp_path):
    trial = run_extended(extended_candidate, completed_with(all_loaded()), tmp_path)
    assert (trial.status, trial.reward) == ("scored", 1.0)
    assert trial.load_report == all_loaded()


def test_completed_run_with_an_extension_that_failed_to_load_is_harness_load_error(extended_candidate, tmp_path):
    suite = FakeSuite([search_task("t1")])
    trial = run_extended(extended_candidate, completed_with(guard_failed()), tmp_path, suite)
    assert (trial.status, trial.error_class, trial.reward) == ("errored", "harness_load_error", None)
    assert "extensions/guard.py" in trial.detail and "SyntaxError: bad" in trial.detail
    assert suite.grade_calls == []


def test_harness_load_error_keeps_the_runs_usage_and_wall_clock(extended_candidate, tmp_path):
    # The spend is real even though the run didn't measure the candidate as written (P12).
    trial = run_extended(extended_candidate, completed_with(guard_failed()), tmp_path)
    assert (trial.usage, trial.wall_clock_s) == (SPENT, 4.0)


def test_completed_run_without_a_load_report_is_harness_load_error_when_extensions_are_expected(
    extended_candidate, tmp_path
):
    trial = run_extended(extended_candidate, completed_with(None), tmp_path)
    assert (trial.status, trial.error_class) == ("errored", "harness_load_error")
    assert "no load report" in trial.detail


def test_completed_run_whose_report_leaves_out_an_extension_is_harness_load_error(extended_candidate, tmp_path):
    report = LoadReport(extensions=(ExtensionLoad(path="extensions/guard.py", loaded=True),))
    trial = run_extended(extended_candidate, completed_with(report), tmp_path)
    assert trial.error_class == "harness_load_error"
    assert "extensions/retry.py" in trial.detail


@pytest.mark.parametrize("error_class", ["timeout", "quota", "provider_error", "agent_error", "extension_error"])
def test_a_failed_load_replaces_the_class_of_an_errored_run_and_keeps_its_detail(extended_candidate, tmp_path, error_class):
    trial = run_extended(extended_candidate, errored_result(error_class, guard_failed()), tmp_path)
    assert trial.error_class == "harness_load_error"
    assert trial.detail.startswith("extensions/guard.py: failed to load: SyntaxError: bad")
    assert f"(the agent reported {error_class}: the agent's own message)" in trial.detail
    assert (trial.usage, trial.wall_clock_s) == (SPENT, 4.0)


def test_model_mismatch_outranks_a_failed_load(extended_candidate, tmp_path):
    # P3: a run on another model is not part of the experiment at all, whatever it loaded.
    trial = run_extended(extended_candidate, errored_result("model_mismatch", guard_failed(), "went to gpt-x"), tmp_path)
    assert (trial.error_class, trial.detail) == ("model_mismatch", "went to gpt-x")


def test_model_mismatch_outranks_extension_runtime_errors(extended_candidate, tmp_path):
    trial = run_extended(extended_candidate, errored_result("model_mismatch", all_loaded("guard.py threw")), tmp_path)
    assert trial.error_class == "model_mismatch"


@pytest.mark.parametrize("error_class", ["timeout", "quota", "provider_error", "agent_error"])
def test_an_errored_run_without_a_load_report_keeps_its_own_class(extended_candidate, tmp_path, error_class):
    # A run killed on timeout never wrote its report: the missing report is explained by the
    # error itself, and calling it a load failure would point the overseer at the wrong cause.
    trial = run_extended(extended_candidate, errored_result(error_class, None), tmp_path)
    assert (trial.error_class, trial.detail) == (error_class, "the agent's own message")


def test_extension_code_that_threw_turns_a_completed_run_into_extension_error(extended_candidate, tmp_path):
    suite = FakeSuite([search_task("t1")])
    trial = run_extended(extended_candidate, completed_with(all_loaded("extensions/guard.py: KeyError 'x'")), tmp_path, suite)
    assert (trial.status, trial.error_class, trial.reward) == ("errored", "extension_error", None)
    assert "KeyError 'x'" in trial.detail
    assert (trial.usage, trial.wall_clock_s) == (SPENT, 4.0)
    assert suite.grade_calls == []


def test_extension_code_that_threw_turns_an_agent_error_into_extension_error(extended_candidate, tmp_path):
    trial = run_extended(extended_candidate, errored_result("agent_error", all_loaded("guard.py threw")), tmp_path)
    assert trial.error_class == "extension_error"
    assert "guard.py threw" in trial.detail
    assert "(the agent reported agent_error: the agent's own message)" in trial.detail


def not_reached() -> LoadReport:
    """What a runner that failed before its load phase reports: no entries at all."""
    return LoadReport()


def unattempted() -> LoadReport:
    """Strands-style records written before the load phase: not loaded, no reason."""
    return LoadReport(extensions=tuple(ExtensionLoad(path=path, loaded=False) for path in EXTENSION_FILES))


@pytest.mark.parametrize("error_class", ["timeout", "quota", "provider_error", "agent_error"])
@pytest.mark.parametrize("report", [not_reached(), unattempted()], ids=["no_entries", "no_reasons"])
def test_an_errored_run_whose_report_shows_no_load_failure_keeps_its_own_class(
    extended_candidate, tmp_path, error_class, report
):
    # The runner failed before it loaded anything; its own error is the cause, not the extensions.
    trial = run_extended(extended_candidate, errored_result(error_class, report), tmp_path)
    assert (trial.error_class, trial.detail) == (error_class, "the agent's own message")


def test_a_completed_run_whose_report_has_no_entries_is_harness_load_error(extended_candidate, tmp_path):
    trial = run_extended(extended_candidate, completed_with(not_reached()), tmp_path)
    assert trial.error_class == "harness_load_error"
    assert "absent from the load report" in trial.detail


def test_extension_code_that_threw_turns_an_unknown_error_class_into_extension_error(extended_candidate, tmp_path):
    # An unknown class is recorded as agent_error, so it is treated as one.
    trial = run_extended(extended_candidate, errored_result("weird", all_loaded("guard.py threw")), tmp_path)
    assert trial.error_class == "extension_error"


@pytest.mark.parametrize("error_class", ["timeout", "quota", "provider_error"])
def test_extension_runtime_errors_do_not_replace_environment_failures(extended_candidate, tmp_path, error_class):
    trial = run_extended(extended_candidate, errored_result(error_class, all_loaded("guard.py threw")), tmp_path)
    assert (trial.error_class, trial.detail) == (error_class, "the agent's own message")


def test_runtime_errors_are_checked_even_when_the_adapter_expects_no_extension_files(candidate, tmp_path):
    adapter = FakeAdapter(returning(completed_with(LoadReport(runtime_errors=("hook threw",)))))
    trial = only_trial(run_eval(candidate, adapter, FakeSuite([search_task("t1")]), tmp_path / "out"))
    assert trial.error_class == "extension_error"


def test_extension_files_are_not_expected_from_an_adapter_without_an_extension_spec(extended_candidate, tmp_path):
    adapter = FakeAdapter(returning(completed_with(None)), spec=None)
    trial = only_trial(run_eval(extended_candidate, adapter, FakeSuite([search_task("t1")]), tmp_path / "out"))
    assert trial.status == "scored"
    assert trial.load_report is None


def test_the_load_report_is_written_to_outcome_json(extended_candidate, tmp_path):
    run_extended(extended_candidate, completed_with(guard_failed()), tmp_path)
    written = json.loads((tmp_path / "out" / "trials" / "t1" / "1" / "outcome.json").read_text())
    assert written["load_report"]["extensions"][0] == {
        "path": "extensions/guard.py",
        "loaded": False,
        "error": "SyntaxError: bad",
        "registered": [],
    }


def test_the_load_report_is_kept_when_the_run_errored(extended_candidate, tmp_path):
    trial = run_extended(extended_candidate, errored_result("timeout", all_loaded()), tmp_path)
    assert trial.load_report == all_loaded()


def test_a_report_with_a_load_report_loads_back_equal(extended_candidate, tmp_path):
    adapter = FakeAdapter(returning(completed_with(all_loaded("x threw"))), spec=EXTENSION_SPEC)
    report = run_eval(extended_candidate, adapter, FakeSuite([search_task("t1"), search_task("t2")]), tmp_path / "out")
    assert EvalReport.from_dict(json.loads((tmp_path / "out" / "report.json").read_text())) == report


def test_a_load_report_given_as_lists_is_recorded_as_tuples(extended_candidate, tmp_path):
    loads = [ExtensionLoad(path=path, loaded=True, registered=["tool:search"]) for path in EXTENSION_FILES]
    trial = run_extended(extended_candidate, completed_with(LoadReport(extensions=loads, runtime_errors=[])), tmp_path)
    assert trial.status == "scored"
    assert trial.load_report == LoadReport(
        extensions=tuple(ExtensionLoad(path=path, loaded=True, registered=("tool:search",)) for path in EXTENSION_FILES)
    )


def test_trial_outcome_without_a_load_report_key_loads_as_none():
    # Reports written before load reports existed must still load.
    data = scored("t1", 1, 1.0).to_dict()
    del data["load_report"]
    assert TrialOutcome.from_dict(data).load_report is None


def test_trial_outcome_round_trips_its_load_report_through_json():
    outcome = TrialOutcome(
        task_id="t1", trial=1, status="errored", reward=None, error_class="harness_load_error", detail="x",
        usage=Usage(), wall_clock_s=1.0, trial_dir="trials/t1/1", load_report=guard_failed(),
    )  # fmt: skip
    assert TrialOutcome.from_dict(json.loads(json.dumps(outcome.to_dict()))) == outcome


@pytest.mark.parametrize(
    "bad_report",
    [
        {"extensions": []},
        "loaded",
        LoadReport(extensions=({"path": "extensions/guard.py", "loaded": True},)),
        LoadReport(extensions=(ExtensionLoad(path=Path("extensions/guard.py"), loaded=True),)),
        LoadReport(extensions=(ExtensionLoad(path="extensions/guard.py", loaded="yes"),)),
        LoadReport(extensions=(ExtensionLoad(path="extensions/guard.py", loaded=False, error=ValueError("x")),)),
        LoadReport(extensions=(ExtensionLoad(path="extensions/guard.py", loaded=True, registered=(1,)),)),
        LoadReport(extensions=(ExtensionLoad(path="extensions/guard.py", loaded=True, registered="tool:x"),)),
        LoadReport(runtime_errors=(object(),)),
        LoadReport(runtime_errors="threw"),
        LoadReport(extensions=None),
    ],
    ids=["dict", "str", "dict-load", "path-not-str", "loaded-not-bool", "error-not-str", "registered-not-str",
         "registered-a-str", "runtime-error-not-str", "runtime-errors-a-str", "extensions-none"],
)  # fmt: skip
def test_a_malformed_load_report_makes_the_trial_an_agent_error_and_keeps_its_cost(
    extended_candidate, tmp_path, bad_report
):
    suite = FakeSuite([search_task("t1")])
    trial = run_extended(extended_candidate, completed_with(bad_report), tmp_path, suite)
    assert (trial.status, trial.error_class) == ("errored", "agent_error")
    assert "load_report" in trial.detail
    assert trial.usage == SPENT
    assert trial.load_report is None
    assert suite.grade_calls == []
    assert (tmp_path / "out" / "report.json").is_file()


# --- the model gateway: per-trial endpoints, and the ledger as the authority (P3, P12) -----------

GATEWAY_MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
PRICES = Prices(input=0.05, output=0.40, cache_read=0.005)
# A made-up stand-in for the provider key. Never a real key.
UPSTREAM_KEY = "sk-placeholder-upstream-key-for-evaluate-tests"
CACHED = CHAT_USAGE["prompt_tokens_details"]["cached_tokens"]
CHAT_TOKENS = Usage(input_tokens=CHAT_USAGE["prompt_tokens"] - CACHED,
                    output_tokens=CHAT_USAGE["completion_tokens"], cache_read_tokens=CACHED)  # fmt: skip
CHAT_BILL = Usage(input_tokens=CHAT_TOKENS.input_tokens, output_tokens=CHAT_TOKENS.output_tokens,
                  cache_read_tokens=CACHED, cost_usd=PRICES.cost(CHAT_TOKENS))  # fmt: skip

EndpointBehavior = Callable[[Task, Path, Path, ModelEndpoint], AgentResult]


def model_call(endpoint: ModelEndpoint, *, model: str = "gpt-5-nano", scenario: str = "ok") -> int:
    """One chat completion through the endpoint, sent the way a framework's HTTP client would."""
    parts = urlsplit(endpoint.base_url)
    body = {"model": model, "messages": [{"role": "user", "content": f"hi scenario={scenario}"}]}
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request("POST", parts.path + "/chat/completions", body=json.dumps(body),
                           headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"})  # fmt: skip
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


class EndpointAgent:
    """An agent that sends its model traffic to the endpoint it is given, as every adapter must."""

    def __init__(self, behavior: EndpointBehavior):
        self.behavior = behavior
        self.endpoints: list[ModelEndpoint] = []

    def run(self, task: Task, workdir: Path, trace_dir: Path, endpoint: ModelEndpoint | None = None) -> AgentResult:
        assert endpoint is not None, "with a gateway, evaluate must pass the endpoint"
        self.endpoints.append(endpoint)
        return self.behavior(task, workdir, trace_dir, endpoint)


class EndpointAdapter(FakeAdapter):
    def __init__(self, behavior: EndpointBehavior, spec: ExtensionSpec | None = None):
        super().__init__(spec=spec)
        self.agent = EndpointAgent(behavior)


@pytest.fixture
def upstream():
    with FakeOpenAIUpstream() as fake:
        yield fake


@contextmanager
def make_gateway(upstream: FakeOpenAIUpstream, *, max_calls: int = 50, **options: Any):
    with ModelGateway(model=GATEWAY_MODEL, prices=PRICES, upstream_url=upstream.url, upstream_key=UPSTREAM_KEY,
                      max_calls_per_trial=max_calls, **options) as gateway:  # fmt: skip
        yield gateway


def gateway_eval(candidate, behavior: EndpointBehavior, tmp_path, *, max_calls: int = 50, suite=None,
                 upstream: FakeOpenAIUpstream, on_trial_cost=None, **options: Any) -> EvalReport:  # fmt: skip
    with make_gateway(upstream, max_calls=max_calls, **options) as gateway:
        return evaluate(candidate, EndpointAdapter(behavior), suite or FakeSuite([search_task("t1")]), "search",
                        model=GATEWAY_MODEL, agent_settings={}, trials_per_task=1, out_dir=tmp_path / "out",
                        gateway=gateway, on_trial_cost=on_trial_cost)  # fmt: skip


def calls_then(statuses: list[int], scenarios: list[str] | None = None, result: AgentResult | None = None,
               model: str = "gpt-5-nano") -> EndpointBehavior:  # fmt: skip
    def behavior(task, workdir, trace_dir, endpoint):
        for scenario in scenarios or ["ok"]:
            statuses.append(model_call(endpoint, model=model, scenario=scenario))
        return result or completed(usage=Usage(input_tokens=1, cost_usd=0.0))
    return behavior


def gateway_json(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "out" / "trials" / "t1" / "1" / GATEWAY_FILE).read_text())


def test_without_a_gateway_the_agent_gets_no_endpoint(candidate, tmp_path):
    adapter = FakeAdapter()  # FakeAgent.run takes no endpoint, so passing one would be an agent_error
    assert only_trial(run_eval(candidate, adapter, FakeSuite([search_task("t1")]), tmp_path / "out")).status == "scored"
    assert not (tmp_path / "out" / "trials" / "t1" / "1" / GATEWAY_FILE).exists()


def test_each_trial_gets_its_own_endpoint_token(candidate, tmp_path, upstream):
    adapter = EndpointAdapter(lambda task, workdir, trace_dir, endpoint: completed())
    with make_gateway(upstream) as gateway:
        evaluate(candidate, adapter, FakeSuite([search_task("t1"), search_task("t2")]), "search", model=GATEWAY_MODEL,
                 agent_settings={}, trials_per_task=2, out_dir=tmp_path / "out", gateway=gateway)  # fmt: skip
    tokens = [endpoint.api_key for endpoint in adapter.agent.endpoints]
    assert len(tokens) == len(set(tokens)) == 4
    assert UPSTREAM_KEY not in tokens


def test_usage_and_cost_come_from_the_ledger_not_the_adapter(candidate, tmp_path, upstream):
    statuses: list[int] = []
    claimed = completed(usage=Usage(input_tokens=1, cost_usd=0.0))  # an adapter that under-reports its spend
    costs: list[float] = []
    report = gateway_eval(candidate, calls_then(statuses, ["ok", "ok"], claimed), tmp_path, upstream=upstream,
                          on_trial_cost=costs.append)  # fmt: skip
    trial = only_trial(report)
    assert statuses == [200, 200]
    assert trial.status == "scored"
    assert trial.usage == CHAT_BILL + CHAT_BILL
    assert costs == [pytest.approx(2 * CHAT_BILL.cost_usd)]
    assert trial.gateway == GatewaySummary(calls=2, models_reported=("gpt-5-nano-2025-08-07",), error_codes=(),
                                           violations=(), limit_reached=False, spend_lower_bound=False)  # fmt: skip


def test_a_refused_model_is_model_mismatch_even_when_the_run_completed(candidate, tmp_path, upstream):
    statuses: list[int] = []
    suite = FakeSuite([search_task("t1")])
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, model="gpt-4o"), tmp_path, upstream=upstream,
                                    suite=suite))  # fmt: skip
    assert statuses == [403]
    assert (trial.status, trial.error_class) == ("errored", "model_mismatch")
    assert "gpt-4o" in trial.detail
    assert suite.grade_calls == []
    assert trial.gateway.violations


def test_a_provider_reporting_another_model_is_model_mismatch(candidate, tmp_path, upstream):
    statuses: list[int] = []
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, ["other_model"]), tmp_path, upstream=upstream))
    assert (trial.status, trial.error_class) == ("errored", "model_mismatch")
    assert "gpt-4o-2024-08-06" in trial.detail
    assert trial.usage == CHAT_BILL  # the mismatched call was still billed


def test_model_mismatch_outranks_the_agents_own_error_class(candidate, tmp_path, upstream):
    statuses: list[int] = []
    errored_run = AgentResult(status="errored", final_text="", usage=Usage(), wall_clock_s=1.0, error="boom",
                              error_class="timeout")  # fmt: skip
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, ["other_model", "quota"], errored_run), tmp_path,
                                    upstream=upstream))  # fmt: skip
    assert trial.error_class == "model_mismatch"


def test_the_call_limit_makes_an_unfinished_run_an_agent_error(candidate, tmp_path, upstream):
    statuses: list[int] = []
    errored_run = AgentResult(status="errored", final_text="", usage=Usage(), wall_clock_s=1.0,
                              error="the provider refused the request", error_class="provider_error")  # fmt: skip
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, ["ok", "ok", "ok"], errored_run), tmp_path,
                                    upstream=upstream, max_calls=2))  # fmt: skip
    assert statuses == [200, 200, 403]
    assert (trial.status, trial.error_class) == ("errored", "agent_error")
    assert trial.detail.startswith("model call limit (2) reached")
    assert "provider_error: the provider refused the request" in trial.detail
    assert trial.usage == CHAT_BILL + CHAT_BILL
    assert trial.gateway.limit_reached is True


def test_the_call_limit_leaves_a_completed_run_scored(candidate, tmp_path, upstream):
    statuses: list[int] = []
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, ["ok", "ok"]), tmp_path, upstream=upstream,
                                    max_calls=1))  # fmt: skip
    assert statuses == [200, 403]
    assert trial.status == "scored"
    assert trial.gateway.limit_reached is True


@pytest.mark.parametrize("scenario", ["quota", "rate_limit"])
def test_an_upstream_quota_or_rate_limit_makes_an_errored_run_quota(candidate, tmp_path, upstream, scenario):
    statuses: list[int] = []
    errored_run = AgentResult(status="errored", final_text="", usage=Usage(), wall_clock_s=1.0, error="429",
                              error_class="provider_error")  # fmt: skip
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, [scenario], errored_run), tmp_path,
                                    upstream=upstream))  # fmt: skip
    assert statuses == [429]
    assert (trial.status, trial.error_class) == ("errored", "quota")
    assert ("insufficient_quota" if scenario == "quota" else "rate_limit_exceeded") in trial.detail


def test_an_upstream_quota_error_leaves_a_completed_run_scored(candidate, tmp_path, upstream):
    statuses: list[int] = []
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, ["quota", "ok"]), tmp_path, upstream=upstream))
    assert trial.status == "scored"
    assert trial.gateway.error_codes == ("insufficient_quota",)


def test_a_call_without_usage_keeps_the_result_but_marks_spend_as_a_lower_bound(candidate, tmp_path, upstream):
    statuses: list[int] = []
    trial = only_trial(gateway_eval(candidate, calls_then(statuses, ["no_usage", "ok"]), tmp_path, upstream=upstream))
    assert trial.status == "scored"
    assert trial.usage == CHAT_BILL
    assert "lower bound" in trial.detail
    assert trial.gateway.spend_lower_bound is True


def test_a_call_still_running_at_close_marks_spend_as_a_lower_bound(candidate, tmp_path):
    def leave_a_call_running(task, workdir, trace_dir, endpoint):
        threading.Thread(target=model_call, args=(endpoint,), kwargs={"scenario": "hang"}, daemon=True).start()
        deadline = time.monotonic() + 5
        while not slow.requests and time.monotonic() < deadline:
            time.sleep(0.01)
        return completed()

    with FakeOpenAIUpstream(hang_s=2.0) as slow:
        trial = only_trial(gateway_eval(candidate, leave_a_call_running, tmp_path, upstream=slow, close_timeout_s=0.1))
    assert trial.status == "scored"
    assert "lower bound" in trial.detail
    assert "in_flight_at_close" in trial.gateway.error_codes


def test_the_grant_is_closed_and_billed_when_run_raises(candidate, tmp_path, upstream):
    seen: list[ModelEndpoint] = []

    def call_then_raise(task, workdir, trace_dir, endpoint):
        seen.append(endpoint)
        model_call(endpoint)
        raise RuntimeError("adapter bug")

    trial = only_trial(gateway_eval(candidate, call_then_raise, tmp_path, upstream=upstream))
    assert (trial.status, trial.error_class) == ("errored", "agent_error")
    assert "adapter bug" in trial.detail
    assert trial.usage == CHAT_BILL
    assert gateway_json(tmp_path)["calls"][0]["status"] == 200


def test_the_token_is_revoked_after_the_trial(candidate, tmp_path, upstream):
    seen: list[ModelEndpoint] = []

    def remember(task, workdir, trace_dir, endpoint):
        seen.append(endpoint)
        return completed()

    with make_gateway(upstream) as gateway:
        evaluate(candidate, EndpointAdapter(remember), FakeSuite([search_task("t1")]), "search", model=GATEWAY_MODEL,
                 agent_settings={}, trials_per_task=1, out_dir=tmp_path / "out", gateway=gateway)  # fmt: skip
        assert model_call(seen[0]) == 401


def test_an_interrupted_trial_still_reports_its_gateway_spend(candidate, tmp_path, upstream):
    def call_then_interrupt(task, workdir, trace_dir, endpoint):
        model_call(endpoint)
        raise AbortTrial

    costs: list[float] = []
    with pytest.raises(AbortTrial):
        gateway_eval(candidate, call_then_interrupt, tmp_path, upstream=upstream, on_trial_cost=costs.append)
    assert costs == [pytest.approx(CHAT_BILL.cost_usd)]
    assert gateway_json(tmp_path)["calls"][0]["status"] == 200


def test_gateway_json_records_the_ledger_and_no_secret(candidate, tmp_path, upstream):
    statuses: list[int] = []
    tokens: list[str] = []

    def behavior(task, workdir, trace_dir, endpoint):
        tokens.append(endpoint.api_key)
        return calls_then(statuses, ["ok", "quota"], model="gpt-5-nano")(task, workdir, trace_dir, endpoint)

    gateway_eval(candidate, behavior, tmp_path, upstream=upstream)
    data = gateway_json(tmp_path)
    assert set(data) == {"calls", "violations", "limit_reached"}
    assert [call["status"] for call in data["calls"]] == [200, 429]
    assert data["calls"][0]["usage"] == {**CHAT_BILL.__dict__}
    assert data["calls"][1]["error_code"] == "insufficient_quota"
    assert (data["violations"], data["limit_reached"]) == ([], False)
    for path in (tmp_path / "out").rglob("*"):
        if path.is_file():
            text = path.read_text(errors="replace")
            assert UPSTREAM_KEY not in text and tokens[0] not in text, path


def test_the_gateway_summary_round_trips_through_report_json(candidate, tmp_path, upstream):
    statuses: list[int] = []
    report = gateway_eval(candidate, calls_then(statuses, ["ok", "no_usage"]), tmp_path, upstream=upstream)
    loaded = EvalReport.from_dict(json.loads((tmp_path / "out" / "report.json").read_text()))
    assert loaded == report
    outcome = json.loads((tmp_path / "out" / "trials" / "t1" / "1" / "outcome.json").read_text())
    assert TrialOutcome.from_dict(outcome) == only_trial(report)
    assert only_trial(report).gateway.calls == 2


def test_outcomes_without_a_gateway_summary_still_load(candidate, tmp_path):
    report = run_eval(candidate, FakeAdapter(), FakeSuite([search_task("t1")]), tmp_path / "out")
    data = only_trial(report).to_dict()
    assert data["gateway"] is None
    del data["gateway"]
    assert TrialOutcome.from_dict(data) == only_trial(report)
