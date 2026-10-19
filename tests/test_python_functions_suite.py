"""Tests for the python_functions suite: task loading, the hidden-case grader and its P2 checks."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

import hive.suites.python_functions as python_functions_package
from hive.contracts import SPLITS, AgentResult, Suite, Usage
from hive.suites.python_functions import PythonFunctionsSuite

SHIPPED_TASKS_DIR = Path(python_functions_package.__file__).parent / "tasks"
COMPLETED = AgentResult(status="completed", final_text="done", usage=Usage(), wall_clock_s=1.0)


def all_tasks(suite: PythonFunctionsSuite):
    return [task for split in SPLITS for task in suite.tasks(split)]


def grade_solution(suite: PythonFunctionsSuite, task, solution_text: str, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "solution.py").write_text(solution_text)
    return suite.grade(task, COMPLETED, workdir)


def task_by_id(suite: PythonFunctionsSuite, task_id: str):
    return next(task for task in all_tasks(suite) if task.id == task_id)


@pytest.fixture(scope="module")
def suite() -> PythonFunctionsSuite:
    return PythonFunctionsSuite()


@pytest.fixture(scope="module")
def slugify_task(suite):
    return task_by_id(suite, "slugify")


# ---------------------------------------------------------------- shipped data and contract shape


def test_suite_satisfies_the_suite_protocol(suite):
    assert isinstance(suite, Suite)
    assert suite.name == "python_functions"


def test_ships_four_search_tasks_and_two_heldout_tasks(suite):
    assert len(suite.tasks("search")) == 4
    assert len(suite.tasks("heldout")) == 2
    assert suite.tasks("validation") == []


def test_splits_are_disjoint_by_task_id(suite):
    ids_per_split = [{task.id for task in suite.tasks(split)} for split in SPLITS]
    all_ids = [task.id for task in all_tasks(suite)]
    assert len(all_ids) == len(set(all_ids))
    assert sum(len(ids) for ids in ids_per_split) == len(set(all_ids))


def test_every_task_reports_the_split_it_was_requested_from(suite):
    for split in SPLITS:
        assert all(task.split == split for task in suite.tasks(split))


def test_workspace_holds_only_the_stub_solution_file(suite):
    for task in all_tasks(suite):
        assert set(task.workspace_files) == {"solution.py"}
        assert "NotImplementedError" in task.workspace_files["solution.py"]


def test_instruction_names_the_function_and_the_solution_file(suite, slugify_task):
    assert "slugify" in slugify_task.instruction
    assert "solution.py" in slugify_task.instruction
    assert "standard library" in slugify_task.instruction


def test_instruction_has_no_unrendered_placeholders(suite):
    for task in all_tasks(suite):
        assert "$" not in task.instruction


def test_describe_mentions_no_function_name_or_task_id(suite):
    description = suite.describe()
    assert "solution.py" in description
    for task in all_tasks(suite):
        assert task.id not in description
        stub = task.workspace_files["solution.py"]
        function_name = stub.split("def ", 1)[1].split("(", 1)[0]
        assert function_name not in description


def test_metric_is_the_mean_fraction_of_hidden_cases_passed(suite):
    metric = suite.metric()
    assert metric.name == "mean_fraction_of_hidden_cases_passed"
    assert metric.description.strip()


# ---------------------------------------------------------------- P2 checks on every shipped task


@pytest.mark.parametrize("task_id", sorted(p.name for p in SHIPPED_TASKS_DIR.iterdir() if p.is_dir()))
def test_reference_solution_grades_one(suite, task_id, tmp_path):
    task = task_by_id(suite, task_id)
    reference = (SHIPPED_TASKS_DIR / task_id / "reference.py").read_text()
    grade = grade_solution(suite, task, reference, tmp_path / "work")
    assert grade.reward == 1.0, grade.detail


@pytest.mark.parametrize("task_id", sorted(p.name for p in SHIPPED_TASKS_DIR.iterdir() if p.is_dir()))
def test_untouched_stub_grades_zero(suite, task_id, tmp_path):
    task = task_by_id(suite, task_id)
    grade = grade_solution(suite, task, task.workspace_files["solution.py"], tmp_path / "work")
    assert grade.reward == 0.0
    assert "NotImplementedError" in grade.detail


# ---------------------------------------------------------------- grading behaviour


def test_missing_solution_file_grades_zero(suite, slugify_task, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    grade = suite.grade(slugify_task, COMPLETED, workdir)
    assert grade.reward == 0.0
    assert grade.detail == "solution.py not found"


def test_partially_correct_solution_earns_the_fraction_of_cases_passed(suite, tmp_path):
    task = task_by_id(suite, "roman_to_int")
    # Only handles single-symbol numerals; every other case is a wrong answer or a KeyError.
    solution = "def roman_to_int(s):\n    return {'I': 1}[s] if s == 'I' else 0\n"
    grade = grade_solution(suite, task, solution, tmp_path / "work")
    assert 0.0 < grade.reward < 1.0
    assert grade.detail.startswith("passed 1/")
    assert "wrong answer" in grade.detail


def test_detail_counts_failures_by_kind(suite, tmp_path):
    task = task_by_id(suite, "roman_to_int")
    solution = "def roman_to_int(s):\n    if s == 'I':\n        raise TypeError('no')\n    return 0\n"
    grade = grade_solution(suite, task, solution, tmp_path / "work")
    assert "1 TypeError" in grade.detail
    assert "wrong answer" in grade.detail


def test_detail_never_reveals_expected_values(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "def slugify(text):\n    return 'nope'\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "hello-world" not in grade.detail
    assert "Hello" not in grade.detail


def test_tuple_results_compare_equal_to_json_lists(suite, tmp_path):
    task = task_by_id(suite, "run_length_encode")
    solution = (
        "import itertools\n"
        "def run_length_encode(text):\n"
        "    return [(char, len(list(group))) for char, group in itertools.groupby(text)]\n"
    )
    grade = grade_solution(suite, task, solution, tmp_path / "work")
    assert grade.reward == 1.0, grade.detail


def test_integer_result_does_not_pass_for_a_boolean_expectation(suite, tmp_path):
    task = task_by_id(suite, "is_balanced")
    reference = (SHIPPED_TASKS_DIR / "is_balanced" / "reference.py").read_text()
    as_ints = reference + "\n_original = is_balanced\ndef is_balanced(text):\n    return int(_original(text))\n"
    grade = grade_solution(suite, task, as_ints, tmp_path / "work")
    assert grade.reward == 0.0


def test_unserializable_result_counts_as_wrong_answer(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "def slugify(text):\n    return object()\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "wrong answer" in grade.detail


def test_infinite_loop_times_out_and_grades_zero_within_the_timeout(slugify_task, tmp_path):
    suite = PythonFunctionsSuite({"grade_timeout_s": 1})
    started = time.monotonic()
    grade = grade_solution(suite, slugify_task, "def slugify(text):\n    while True:\n        pass\n", tmp_path / "work")
    elapsed = time.monotonic() - started
    assert grade.reward == 0.0
    assert grade.detail == "grading timed out"
    assert elapsed < 5


def test_infinite_loop_at_import_times_out(slugify_task, tmp_path):
    suite = PythonFunctionsSuite({"grade_timeout_s": 1})
    grade = grade_solution(suite, slugify_task, "while True:\n    pass\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert grade.detail == "grading timed out"


def test_timeout_also_kills_processes_the_solution_spawned(slugify_task, tmp_path):
    suite = PythonFunctionsSuite({"grade_timeout_s": 1})
    pid_file = tmp_path / "child.pid"
    # A surviving child would keep running after grading and could hold the stdout pipe open.
    solution = (
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "while True:\n    pass\n"
    )
    started = time.monotonic()
    grade = grade_solution(suite, slugify_task, solution, tmp_path / "work")
    assert grade.detail == "grading timed out"
    assert time.monotonic() - started < 5
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("the solution's child process survived grading")


def test_printing_solution_still_grades_correctly(suite, slugify_task, tmp_path):
    reference = (SHIPPED_TASKS_DIR / "slugify" / "reference.py").read_text()
    noisy = (
        "import sys\n"
        "print('{\"passed\": 0, \"total\": 1, \"failures\": []}')\n"
        "sys.__stdout__.write('import-time noise\\n')\n"
        + reference
        + "\n_original = slugify\ndef slugify(text):\n    print('called with', text)\n    return _original(text)\n"
    )
    grade = grade_solution(suite, slugify_task, noisy, tmp_path / "work")
    assert grade.reward == 1.0, grade.detail


def test_solution_that_crashes_on_import_grades_zero_with_exception_type(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "raise RuntimeError('boom')\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "RuntimeError" in grade.detail
    assert "boom" not in grade.detail


def test_solution_with_syntax_error_grades_zero(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "def slugify(text)\n    return text\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "SyntaxError" in grade.detail


def test_solution_that_exits_on_import_grades_zero(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "import sys\nsys.exit(0)\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "SystemExit" in grade.detail


def test_solution_that_kills_the_runner_grades_zero(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "import os\nos._exit(0)\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "without a result" in grade.detail


def test_solution_missing_the_function_grades_zero(suite, slugify_task, tmp_path):
    grade = grade_solution(suite, slugify_task, "def something_else(text):\n    return text\n", tmp_path / "work")
    assert grade.reward == 0.0
    assert "not defined" in grade.detail


def test_grading_leaves_the_workdir_untouched_and_never_copies_cases_into_it(suite, slugify_task, tmp_path):
    workdir = tmp_path / "work"
    reference = (SHIPPED_TASKS_DIR / "slugify" / "reference.py").read_text()
    grade_solution(suite, slugify_task, reference, workdir)
    assert sorted(p.relative_to(workdir).as_posix() for p in workdir.rglob("*")) == ["solution.py"]
    assert (workdir / "solution.py").read_text() == reference


def test_grading_an_unknown_task_raises(suite, slugify_task, tmp_path):
    from hive.contracts import Task

    unknown = Task(id="no_such_task", split="search", instruction="x")
    with pytest.raises(ValueError, match="no_such_task"):
        suite.grade(unknown, COMPLETED, tmp_path)


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize("bad_timeout", [0, -1, "10", True, float("nan"), float("inf")])
def test_rejects_a_non_positive_or_non_numeric_timeout(bad_timeout):
    with pytest.raises(ValueError, match="grade_timeout_s"):
        PythonFunctionsSuite({"grade_timeout_s": bad_timeout})


# ---------------------------------------------------------------- task data validation


def write_task(tasks_dir: Path, task_id: str, split: str = "search", cases=None, toml_id: str | None = None):
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        f'id = "{toml_id or task_id}"\nsplit = "{split}"\nfunction = "double"\ndescription = "Return twice x."\n'
    )
    (task_dir / "stub.py").write_text("def double(x):\n    raise NotImplementedError\n")
    (task_dir / "reference.py").write_text("def double(x):\n    return 2 * x\n")
    cases = [{"args": [2], "kwargs": {}, "expected": 4}] if cases is None else cases
    (task_dir / "cases.json").write_text(json.dumps(cases))
    return task_dir


def test_loads_a_custom_tasks_dir(tmp_path):
    write_task(tmp_path, "double")
    suite = PythonFunctionsSuite({"tasks_dir": str(tmp_path)})
    assert [task.id for task in suite.tasks("search")] == ["double"]


def test_rejects_a_task_whose_id_does_not_match_its_directory(tmp_path):
    write_task(tmp_path, "double", toml_id="triple")
    with pytest.raises(ValueError, match="double"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})


def test_rejects_an_invalid_split(tmp_path):
    write_task(tmp_path, "double", split="training")
    with pytest.raises(ValueError, match="double"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})


@pytest.mark.parametrize("missing_file", ["task.toml", "stub.py", "cases.json", "reference.py"])
def test_rejects_a_task_with_a_missing_file(tmp_path, missing_file):
    task_dir = write_task(tmp_path, "double")
    (task_dir / missing_file).unlink()
    with pytest.raises(ValueError, match="double"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})


def test_rejects_a_task_with_no_cases(tmp_path):
    write_task(tmp_path, "double", cases=[])
    with pytest.raises(ValueError, match="double"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})


def test_rejects_a_case_without_an_expected_value(tmp_path):
    write_task(tmp_path, "double", cases=[{"args": [2]}])
    with pytest.raises(ValueError, match="double"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})


def test_rejects_malformed_task_toml(tmp_path):
    task_dir = write_task(tmp_path, "double")
    (task_dir / "task.toml").write_text("id = \n")
    with pytest.raises(ValueError, match="double"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})


def test_rejects_a_tasks_dir_with_no_tasks(tmp_path):
    with pytest.raises(ValueError, match="no tasks"):
        PythonFunctionsSuite({"tasks_dir": str(tmp_path)})
