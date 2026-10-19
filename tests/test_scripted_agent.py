import json
from pathlib import Path

import pytest

from hive.agents.scripted import ScriptedAdapter
from hive.candidate import Candidate, CandidateError
from hive.contracts import Agent, AgentAdapter, ModelSpec, Task, Usage

MODEL = ModelSpec(provider="none", id="none")

RESPONSES = """\
default = "ANSWER: unknown"

[responses]
"task-a" = "ANSWER: 4"

[files."task-b"]
"solution.py" = "def f(): return 1"
"pkg/helper.py" = "X = 2"
"""


def write_candidate(
    root: Path,
    responses: str | None = RESPONSES,
    agent: str = "scripted",
    valves: str = "",
) -> Candidate:
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text(f'schema_version = 1\nagent = "{agent}"\n{valves}')
    if responses is not None:
        (root / "responses.toml").write_text(responses)
    return Candidate.load(root)


def make_task(task_id: str) -> Task:
    return Task(id=task_id, split="search", instruction="whatever")


def run_task(candidate: Candidate, task_id: str, tmp_path: Path):
    agent = ScriptedAdapter().build(candidate, MODEL, {})
    workdir = tmp_path / "work"
    trace_dir = tmp_path / "trace"
    workdir.mkdir()
    trace_dir.mkdir()
    return agent.run(make_task(task_id), workdir, trace_dir), workdir, trace_dir


def assert_one_problem_mentioning(problems: list[str], *words: str) -> None:
    assert len(problems) == 1, problems
    for word in words:
        assert word in problems[0], problems


# --- adapter shape ---------------------------------------------------------


def test_adapter_satisfies_the_agent_adapter_contract():
    adapter = ScriptedAdapter()
    assert isinstance(adapter, AgentAdapter)
    assert adapter.name == "scripted"


def test_adapter_accepts_settings_argument():
    assert ScriptedAdapter({"anything": 1}).name == "scripted"


def test_built_agent_satisfies_the_agent_contract(tmp_path):
    agent = ScriptedAdapter().build(write_candidate(tmp_path / "c"), MODEL, {})
    assert isinstance(agent, Agent)


def test_describe_valves_says_responses_toml_is_the_only_valve_and_it_is_a_test_double():
    text = ScriptedAdapter().describe_valves()
    assert "responses.toml" in text
    assert "test double" in text.lower()


def test_adapter_has_no_extension_spec_because_the_scripted_agent_runs_no_extensions():
    assert ScriptedAdapter().extension_spec() is None


def test_describe_valves_says_extensions_are_not_supported():
    assert "no extensions" in ScriptedAdapter().describe_valves().lower()


# --- validate --------------------------------------------------------------


def test_valid_candidate_has_no_problems(tmp_path):
    assert ScriptedAdapter().validate(write_candidate(tmp_path / "c")) == []


def test_empty_responses_file_is_valid(tmp_path):
    assert ScriptedAdapter().validate(write_candidate(tmp_path / "c", responses="")) == []


def test_wrong_agent_name_is_a_problem(tmp_path):
    problems = ScriptedAdapter().validate(write_candidate(tmp_path / "c", agent="pi"))
    assert_one_problem_mentioning(problems, "pi")


def test_valves_are_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", valves='\n[valves]\nthinking = "low"\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "valves")


def test_missing_responses_file_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses=None)
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "responses.toml")


def test_responses_file_that_is_a_directory_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses=None)
    (candidate.root / "responses.toml").mkdir()
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "responses.toml")


def test_unparseable_responses_file_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses="this is = = not toml")
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "responses.toml")


def test_non_string_default_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses="default = 3\n")
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "default")


def test_responses_that_are_not_a_table_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses='responses = "ANSWER: 4"\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "responses")


def test_non_string_response_value_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses='[responses]\n"task-a" = 4\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "task-a")


def test_files_that_are_not_a_table_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses='files = "solution.py"\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "files")


def test_files_entry_for_a_task_that_is_not_a_table_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses='[files]\n"task-b" = "solution.py"\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "task-b")


def test_non_string_file_content_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses='[files."task-b"]\n"solution.py" = 1\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "solution.py")


def test_unknown_top_level_key_is_a_problem(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses='defualt = "ANSWER: 1"\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "defualt")


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.py", "sub/../../outside.py", "sub/..", "", "."],
)
def test_file_path_that_is_absolute_or_escapes_the_workdir_is_a_problem(tmp_path, path):
    candidate = write_candidate(tmp_path / "c", responses=f'[files."task-b"]\n"{path}" = "x"\n')
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "task-b")


def test_every_problem_is_reported_not_just_the_first(tmp_path):
    candidate = write_candidate(
        tmp_path / "c",
        agent="pi",
        responses='default = 3\n[files."t"]\n"../x" = "y"\n',
    )
    assert len(ScriptedAdapter().validate(candidate)) == 3


def test_validate_does_not_raise_when_the_candidate_directory_vanished(tmp_path):
    candidate = write_candidate(tmp_path / "c")
    (candidate.root / "responses.toml").unlink()
    (candidate.root / "candidate.toml").unlink()
    candidate.root.rmdir()
    assert_one_problem_mentioning(ScriptedAdapter().validate(candidate), "responses.toml")


# --- build -----------------------------------------------------------------


def test_build_rejects_an_invalid_candidate(tmp_path):
    candidate = write_candidate(tmp_path / "c", responses=None)
    with pytest.raises(CandidateError, match="responses.toml"):
        ScriptedAdapter().build(candidate, MODEL, {})


def test_built_agent_is_unaffected_by_later_edits_to_the_candidate(tmp_path):
    candidate = write_candidate(tmp_path / "c")
    agent = ScriptedAdapter().build(candidate, MODEL, {})
    (candidate.root / "responses.toml").write_text('[responses]\n"task-a" = "ANSWER: 5"\n')
    workdir, trace_dir = tmp_path / "w", tmp_path / "t"
    workdir.mkdir()
    trace_dir.mkdir()
    assert agent.run(make_task("task-a"), workdir, trace_dir).final_text == "ANSWER: 4"


# --- run -------------------------------------------------------------------


def test_run_returns_the_scripted_response_for_the_task(tmp_path):
    result, _, _ = run_task(write_candidate(tmp_path / "c"), "task-a", tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 4"
    assert result.error is None
    assert result.error_class is None


def test_run_reports_zero_usage_and_a_measured_wall_clock(tmp_path):
    result, _, _ = run_task(write_candidate(tmp_path / "c"), "task-a", tmp_path)
    assert result.usage == Usage()
    assert result.wall_clock_s >= 0.0


def test_run_falls_back_to_the_default_response_for_an_unscripted_task(tmp_path):
    result, _, _ = run_task(write_candidate(tmp_path / "c"), "task-z", tmp_path)
    assert result.final_text == "ANSWER: unknown"


def test_run_returns_empty_text_when_there_is_no_entry_and_no_default(tmp_path):
    result, _, _ = run_task(write_candidate(tmp_path / "c", responses=""), "task-z", tmp_path)
    assert result.status == "completed"
    assert result.final_text == ""


def test_run_writes_the_tasks_files_into_the_workdir(tmp_path):
    _, workdir, _ = run_task(write_candidate(tmp_path / "c"), "task-b", tmp_path)
    assert (workdir / "solution.py").read_text() == "def f(): return 1"
    assert (workdir / "pkg" / "helper.py").read_text() == "X = 2"


def test_run_overwrites_an_existing_workspace_file(tmp_path):
    candidate = write_candidate(tmp_path / "c")
    agent = ScriptedAdapter().build(candidate, MODEL, {})
    workdir, trace_dir = tmp_path / "w", tmp_path / "t"
    workdir.mkdir()
    trace_dir.mkdir()
    (workdir / "solution.py").write_text("def f(): raise NotImplementedError")
    agent.run(make_task("task-b"), workdir, trace_dir)
    assert (workdir / "solution.py").read_text() == "def f(): return 1"


def test_run_writes_no_files_for_a_task_without_file_entries(tmp_path):
    _, workdir, _ = run_task(write_candidate(tmp_path / "c"), "task-a", tmp_path)
    assert list(workdir.iterdir()) == []


def test_run_writes_a_trace_with_task_id_response_and_files_written(tmp_path):
    _, _, trace_dir = run_task(write_candidate(tmp_path / "c"), "task-b", tmp_path)
    trace = json.loads((trace_dir / "scripted.json").read_text())
    assert trace["task_id"] == "task-b"
    assert trace["response"] == "ANSWER: unknown"
    assert trace["files_written"] == ["pkg/helper.py", "solution.py"]


def test_run_returns_agent_error_instead_of_raising_when_the_workdir_is_unwritable(tmp_path):
    agent = ScriptedAdapter().build(write_candidate(tmp_path / "c"), MODEL, {})
    workdir = tmp_path / "not-a-dir"
    workdir.write_text("a file, so nothing can be written beneath it")
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    result = agent.run(make_task("task-b"), workdir, trace_dir)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert result.error
    assert result.usage == Usage()


def test_run_returns_agent_error_instead_of_raising_when_the_trace_dir_is_missing(tmp_path):
    agent = ScriptedAdapter().build(write_candidate(tmp_path / "c"), MODEL, {})
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = agent.run(make_task("task-a"), workdir, tmp_path / "missing" / "trace")
    assert result.status == "errored"
    assert result.error_class == "agent_error"


def test_run_accepts_a_gateway_endpoint_and_ignores_it(tmp_path):
    from hive.contracts import ModelEndpoint

    candidate = write_candidate(tmp_path / "c")
    agent = ScriptedAdapter().build(candidate, MODEL, {})
    workdir, trace_dir = tmp_path / "work", tmp_path / "trace"
    workdir.mkdir()
    trace_dir.mkdir()
    endpoint = ModelEndpoint(base_url="http://127.0.0.1:9/v1", api_key="trial-token-placeholder")
    with_endpoint = agent.run(make_task("t1"), workdir, trace_dir, endpoint=endpoint)
    assert with_endpoint.status == "completed"
    assert with_endpoint.usage == Usage()
    assert "trial-token-placeholder" not in (trace_dir / "scripted.json").read_text()
