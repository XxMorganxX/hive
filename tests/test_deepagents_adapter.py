"""Layer 1: the Deep Agents adapter's controller side, with no framework installed.

`python_path` points at tests/fakes/deepagents_fake_python.py, which answers the version
check and plays canned runner scenarios in the runner protocol's file formats. So these tests
cover what the controller decides (validation, settings, the launch, classification, process
cleanup) without a venv and without any model call.
"""

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

import hive.agents.deepagents.adapter as da
from hive.agents.deepagents import DeepAgentsAdapter
from hive.candidate import Candidate
from hive.contracts import LIFECYCLE_POINTS, AgentAdapter, ModelSpec, Task
from hive.extensions import expected_extension_files, render_extension_guide

FAKE_PYTHON = Path(__file__).parent / "fakes" / "deepagents_fake_python.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "harnesses" / "deepagents-baseline"
MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
PRICES = {"input": 0.05, "output": 0.40, "cache_read": 0.005}

GOOD_EXTENSION = '''
from langchain.agents.middleware import AgentMiddleware


class Guard(AgentMiddleware):
    def before_agent(self, state, runtime):
        return None


def extension():
    return [Guard()]
'''


def write_candidate(root: Path, valves: str = "", files: dict[str, str] | None = None,
                    agent: str = "deepagents") -> Candidate:  # fmt: skip
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text(
        f'schema_version = 1\nagent = "{agent}"\ndescription = "test"\n\n[valves]\n{valves}'
    )
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Candidate.load(root)


@pytest.fixture
def fake_python(tmp_path) -> Path:
    """A private copy of the fake interpreter, so its sidecar files belong to one test."""
    directory = tmp_path / "venv-bin"
    directory.mkdir()
    python = directory / "python"
    shutil.copy(FAKE_PYTHON, python)
    python.chmod(0o755)
    return python


@pytest.fixture(autouse=True)
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


def settings(fake_python: Path, **overrides) -> dict:
    return {"python_path": str(fake_python), "prices": PRICES, **overrides}


def build(candidate: Candidate, fake_python: Path, model: ModelSpec = MODEL, **overrides):
    return DeepAgentsAdapter().build(candidate, model, settings(fake_python, **overrides))


def run(agent, tmp_path: Path, instruction: str = "SCENARIO completed"):
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(exist_ok=True)
    result = agent.run(Task(id="t1", split="search", instruction=instruction), workdir, trace_dir)
    return result, workdir, trace_dir


def record(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "fake_record.json").read_text())


def invocations(fake_python: Path) -> list[list[str]]:
    log = fake_python.parent / "invocations.jsonl"
    return [json.loads(line)["argv"] for line in log.read_text().splitlines()] if log.is_file() else []


def pid_alive(pid: int) -> bool:
    # A zombie still answers kill(0); it is dead for our purposes, so ask ps for its state.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def wait_dead(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return False


# --- contract ------------------------------------------------------------------------------


def test_adapter_satisfies_the_agent_adapter_contract():
    adapter = DeepAgentsAdapter()
    assert isinstance(adapter, AgentAdapter)
    assert adapter.name == "deepagents"


# --- validate --------------------------------------------------------------------------------


def test_baseline_harness_is_valid():
    assert DeepAgentsAdapter().validate(Candidate.load(BASELINE)) == []


def test_baseline_harness_has_no_extensions():
    candidate = Candidate.load(BASELINE)
    assert expected_extension_files(candidate, DeepAgentsAdapter().extension_spec()) == []


def test_valid_candidate_with_system_prompt_extension_and_skill(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={
        "SYSTEM.md": "Be careful.",
        "extensions/guard.py": GOOD_EXTENSION,
        "skills/plan-first/SKILL.md": "---\nname: plan-first\ndescription: Use when planning.\n---\nSteps.\n",
        "README.md": "notes",
    })  # fmt: skip
    assert DeepAgentsAdapter().validate(candidate) == []


def test_rejects_a_candidate_for_another_agent(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", agent="pi"))
    assert any("expected 'deepagents'" in problem for problem in problems)


def test_rejects_an_unknown_top_level_entry(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"AGENTS.md": "x"}))
    assert any("unknown top-level entry 'AGENTS.md'" in problem for problem in problems)


def test_rejects_an_extension_that_is_not_a_python_file(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/guard.ts": "x"}))
    assert any("extensions/guard.ts" in problem and ".py" in problem for problem in problems)


def test_rejects_an_extension_in_a_subdirectory(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={"extensions/lib/helper.py": GOOD_EXTENSION})
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("directly in extensions/" in problem for problem in problems)


def test_rejects_an_extension_with_a_syntax_error(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={"extensions/bad.py": "def extension(:\n"})
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("extensions/bad.py" in problem and "syntax" in problem.lower() for problem in problems)


def test_rejects_an_extension_without_an_entry_function(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={"extensions/bad.py": "MIDDLEWARE = []\n"})
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("extensions/bad.py" in problem and "def extension()" in problem for problem in problems)


def test_rejects_an_async_entry_function(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={"extensions/bad.py": "async def extension():\n    return []\n"})
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("extensions/bad.py" in problem and "async" in problem for problem in problems)


def test_rejects_an_entry_function_with_required_parameters(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={"extensions/bad.py": "def extension(model):\n    return []\n"})
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("extensions/bad.py" in problem and "no arguments" in problem for problem in problems)


def test_rejects_an_extension_that_builds_its_own_chat_model(tmp_path):
    source = "from langchain_openai import ChatOpenAI\n\ndef extension():\n    ChatOpenAI(model='gpt-5')\n    return []\n"
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/swap.py": source}))
    assert any("extensions/swap.py" in problem and "ChatOpenAI" in problem for problem in problems)


def test_rejects_an_extension_that_overrides_the_model_of_a_request(tmp_path):
    source = (
        "from langchain.agents.middleware import wrap_model_call\n\n"
        "@wrap_model_call\ndef route(request, handler):\n    return handler(request.override(model=None))\n\n"
        "def extension():\n    return [route]\n"
    )
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/route.py": source}))
    assert any("extensions/route.py" in problem and "override(model=" in problem for problem in problems)


def test_validate_never_runs_extension_code(tmp_path):
    marker = tmp_path / "ran.txt"
    source = f"open({str(marker)!r}, 'w').write('ran')\n\ndef extension():\n    return []\n"
    DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/side.py": source}))
    assert not marker.exists()


def test_rejects_an_unknown_valve(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", valves="temperature = 0.2\n"))
    assert any("unknown valve 'temperature'" in problem for problem in problems)


def test_rejects_a_tools_valve_that_is_not_a_list_of_names(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", valves='tools = "ls"\n'))
    assert any("valve 'tools'" in problem for problem in problems)


def test_rejects_a_duplicate_tool_name(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", valves='tools = ["ls", "ls"]\n'))
    assert any("'ls'" in problem and "more than once" in problem for problem in problems)


def test_rejects_the_task_tool_without_the_general_purpose_subagent(tmp_path):
    candidate = write_candidate(tmp_path / "c", valves='tools = ["read_file", "task"]\n')
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("'task'" in problem and "general_purpose_subagent" in problem for problem in problems)


def test_rejects_the_general_purpose_subagent_without_the_task_tool(tmp_path):
    candidate = write_candidate(tmp_path / "c", valves='general_purpose_subagent = true\ntools = ["read_file"]\n')
    problems = DeepAgentsAdapter().validate(candidate)
    assert any("'task'" in problem and "general_purpose_subagent" in problem for problem in problems)


def test_accepts_the_general_purpose_subagent_with_the_task_tool(tmp_path):
    candidate = write_candidate(tmp_path / "c", valves='general_purpose_subagent = true\ntools = ["read_file", "task"]\n')
    assert DeepAgentsAdapter().validate(candidate) == []


def test_rejects_a_non_boolean_general_purpose_subagent_valve(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", valves='general_purpose_subagent = "yes"\n'))
    assert any("general_purpose_subagent" in problem and "true or false" in problem for problem in problems)


def test_rejects_an_empty_system_prompt(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"SYSTEM.md": "  \n"}))
    assert any("SYSTEM.md is empty" in problem for problem in problems)


def test_rejects_a_skill_without_a_description(tmp_path):
    files = {"skills/plan/SKILL.md": "---\nname: plan\n---\nSteps.\n"}
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("skills/plan" in problem and "description" in problem for problem in problems)


def test_rejects_a_skill_whose_name_differs_from_its_directory(tmp_path):
    files = {"skills/plan/SKILL.md": "---\nname: other\ndescription: Use it.\n---\n"}
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("skills/plan" in problem and "name" in problem for problem in problems)


def test_rejects_a_skill_directory_without_skill_md(tmp_path):
    problems = DeepAgentsAdapter().validate(write_candidate(tmp_path / "c", files={"skills/plan/notes.md": "x"}))
    assert any("skills/plan" in problem and "SKILL.md" in problem for problem in problems)


def test_validate_reports_an_unreadable_extension_instead_of_raising(tmp_path):
    candidate = write_candidate(tmp_path / "c")
    (candidate.root / "extensions").mkdir()
    (candidate.root / "extensions" / "bin.py").write_bytes(b"\xff\xfe\x00 not python")
    problems = DeepAgentsAdapter().validate(Candidate.load(candidate.root))
    assert any("extensions/bin.py" in problem for problem in problems)


# --- settings ----------------------------------------------------------------------------------


def test_build_requires_a_python_path(tmp_path):
    with pytest.raises(ValueError, match="python_path"):
        DeepAgentsAdapter().build(Candidate.load(BASELINE), MODEL, {"prices": PRICES})


def test_build_requires_prices(tmp_path, fake_python):
    with pytest.raises(ValueError, match="prices"):
        DeepAgentsAdapter().build(Candidate.load(BASELINE), MODEL, {"python_path": str(fake_python)})


def test_build_rejects_an_unknown_setting(fake_python):
    with pytest.raises(ValueError, match="unknown Deep Agents adapter settings: pi_bin"):
        build(Candidate.load(BASELINE), fake_python, pi_bin="pi")


def test_build_rejects_a_non_positive_timeout(fake_python):
    with pytest.raises(ValueError, match="timeout_s"):
        build(Candidate.load(BASELINE), fake_python, timeout_s=0)


def test_build_rejects_a_non_integer_model_call_limit(fake_python):
    with pytest.raises(ValueError, match="max_model_calls"):
        build(Candidate.load(BASELINE), fake_python, max_model_calls=2.5)


def test_build_rejects_a_pin_other_than_the_version_the_adapter_was_checked_against(fake_python):
    with pytest.raises(ValueError, match="0.7.28"):
        build(Candidate.load(BASELINE), fake_python, deepagents_version="0.7.27")


def test_build_raises_when_the_venv_has_another_deepagents_version(fake_python):
    (fake_python.parent / "fake_version.txt").write_text("0.7.27\n")
    with pytest.raises(da.DeepAgentsVersionMismatch, match="0.7.27"):
        build(Candidate.load(BASELINE), fake_python)


def test_build_checks_the_installed_version_once_and_runs_do_not(tmp_path, fake_python):
    agent = build(Candidate.load(BASELINE), fake_python)
    run(agent, tmp_path)
    run(agent, tmp_path)
    version_checks = [argv for argv in invocations(fake_python) if argv[:2] == ["-I", "-c"]]
    assert len(version_checks) == 1


def test_build_raises_when_the_interpreter_cannot_be_started(tmp_path):
    with pytest.raises(da.DeepAgentsVersionMismatch, match="could not run"):
        DeepAgentsAdapter().build(Candidate.load(BASELINE), MODEL,
                                  {"python_path": str(tmp_path / "missing" / "python"), "prices": PRICES})  # fmt: skip


def test_build_rejects_an_unsupported_provider(fake_python):
    with pytest.raises(ValueError, match="provider"):
        build(Candidate.load(BASELINE), fake_python, model=ModelSpec(provider="anthropic", id="claude"))


def test_build_rejects_an_unknown_thinking_level(fake_python):
    with pytest.raises(ValueError, match="thinking"):
        build(Candidate.load(BASELINE), fake_python, model=ModelSpec(provider="openai", id="gpt-5-nano", thinking="max"))


def test_build_raises_when_the_provider_credential_is_missing(fake_python, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build(Candidate.load(BASELINE), fake_python)


def test_scripted_provider_requires_a_turns_file(fake_python):
    with pytest.raises(ValueError, match="scripted_turns_path"):
        build(Candidate.load(BASELINE), fake_python, model=ModelSpec(provider="scripted", id="gpt-5-nano"))


def test_turns_file_is_rejected_for_a_real_provider(tmp_path, fake_python):
    turns = tmp_path / "turns.json"
    turns.write_text("[]")
    with pytest.raises(ValueError, match="scripted_turns_path"):
        build(Candidate.load(BASELINE), fake_python, scripted_turns_path=str(turns))


def test_build_rejects_an_invalid_candidate(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"AGENTS.md": "x"})
    with pytest.raises(ValueError, match="not a valid Deep Agents candidate"):
        build(candidate, fake_python)


# --- launch: argv, cwd, environment, run spec ----------------------------------------------


def test_runner_is_started_isolated_with_the_run_spec_in_the_trace_dir(tmp_path, fake_python):
    _, workdir, trace_dir = run(build(Candidate.load(BASELINE), fake_python), tmp_path)
    recorded = record(trace_dir)
    assert recorded["argv"][1:] == ["-I", str(da.RUNNER), str(trace_dir / "run_spec.json")]
    assert Path(recorded["cwd"]).resolve() == workdir.resolve()


def test_runner_is_referenced_by_absolute_path_and_exists():
    assert da.RUNNER.is_absolute() and da.RUNNER.is_file()


def test_environment_holds_only_path_home_lang_and_the_provider_key(tmp_path, fake_python, monkeypatch):
    for name, value in {
        "LANGSMITH_API_KEY": "ls-secret", "LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true",
        "AWS_ACCESS_KEY_ID": "AKIA", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel", "ANTHROPIC_API_KEY": "sk-ant",
        "OPENAI_BASE_URL": "http://evil", "PYTHONPATH": "/evil", "VIRTUAL_ENV": "/elsewhere",
    }.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    _, _, trace_dir = run(build(Candidate.load(BASELINE), fake_python), tmp_path)
    env = {name: value for name, value in record(trace_dir)["env"].items() if not name.startswith("__CF")}
    assert set(env) == {"PATH", "HOME", "LANG", "OPENAI_API_KEY"}
    assert env["OPENAI_API_KEY"] == "sk-test-not-a-real-key"


def test_home_is_a_fresh_directory_in_the_trace_dir(tmp_path, fake_python):
    _, _, trace_dir = run(build(Candidate.load(BASELINE), fake_python), tmp_path)
    home = Path(record(trace_dir)["env"]["HOME"])
    assert home.parent == trace_dir and home.is_dir()


def test_the_key_never_appears_in_argv_or_the_run_spec(tmp_path, fake_python):
    _, _, trace_dir = run(build(Candidate.load(BASELINE), fake_python), tmp_path)
    assert not any("sk-test-not-a-real-key" in arg for arg in record(trace_dir)["argv"])
    assert "sk-test-not-a-real-key" not in (trace_dir / "run_spec.json").read_text()


def test_scripted_provider_gets_no_credential(tmp_path, fake_python):
    turns = tmp_path / "turns.json"
    turns.write_text("[]")
    model = ModelSpec(provider="scripted", id="gpt-5-nano")
    _, _, trace_dir = run(build(Candidate.load(BASELINE), fake_python, model=model, scripted_turns_path=str(turns)),
                          tmp_path)  # fmt: skip
    env = {name for name in record(trace_dir)["env"] if not name.startswith("__CF")}
    assert env == {"PATH", "HOME", "LANG"}
    spec = json.loads((trace_dir / "run_spec.json").read_text())
    assert spec["model"]["provider"] == "scripted" and spec["scripted_turns_file"] == str(turns.resolve())


def test_run_spec_carries_the_model_limits_valves_and_output_files(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", valves='tools = ["read_file", "execute"]\n',
                                files={"SYSTEM.md": "Be brief.", "extensions/guard.py": GOOD_EXTENSION})  # fmt: skip
    _, workdir, trace_dir = run(build(candidate, fake_python, max_model_calls=7, recursion_limit=99), tmp_path)
    spec = json.loads((trace_dir / "run_spec.json").read_text())
    assert spec["model"] == {"provider": "openai", "id": "gpt-5-nano", "reasoning_effort": "low"}
    assert spec["limits"]["max_model_calls"] == 7 and spec["limits"]["recursion_limit"] == 99
    assert spec["tools"] == ["read_file", "execute"]
    assert spec["builtin_tools"] == [*da.FILESYSTEM_TOOLS, "task"]
    assert spec["general_purpose_subagent"] is False
    assert spec["system_prompt"] == "Be brief."
    assert spec["workdir"] == str(workdir.resolve())
    assert spec["trace_file"] == str(trace_dir / "trace.jsonl")
    assert spec["result_file"] == str(trace_dir / "result.json")
    assert spec["child_pgids_file"] == str(trace_dir / "child_pgids.txt")
    assert spec["extensions"] == [{"path": "extensions/guard.py", "file": str(candidate.root / "extensions/guard.py")}]


def test_run_spec_lists_exactly_the_files_the_controller_expects_to_load(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/b.py": GOOD_EXTENSION, "extensions/a.py": GOOD_EXTENSION})
    _, _, trace_dir = run(build(candidate, fake_python), tmp_path)
    spec = json.loads((trace_dir / "run_spec.json").read_text())
    expected = expected_extension_files(candidate, DeepAgentsAdapter().extension_spec())
    assert [entry["path"] for entry in spec["extensions"]] == expected == ["extensions/a.py", "extensions/b.py"]


def test_run_spec_tells_the_model_the_absolute_workdir(tmp_path, fake_python):
    _, workdir, trace_dir = run(build(Candidate.load(BASELINE), fake_python), tmp_path)
    spec = json.loads((trace_dir / "run_spec.json").read_text())
    assert str(workdir.resolve()) in spec["workdir_note"]


def test_thinking_defaults_to_low_reasoning_effort(tmp_path, fake_python):
    model = ModelSpec(provider="openai", id="gpt-5-nano")
    _, _, trace_dir = run(build(Candidate.load(BASELINE), fake_python, model=model), tmp_path)
    assert json.loads((trace_dir / "run_spec.json").read_text())["model"]["reasoning_effort"] == "low"


# --- classification (through hive.agents.runner_protocol) ----------------------------------


def test_a_completed_run_returns_its_text_usage_and_cost(tmp_path, fake_python):
    result, _, _ = run(build(Candidate.load(BASELINE), fake_python), tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 42"
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (300, 50, 50)
    assert result.usage.cost_usd == pytest.approx((300 * 0.05 + 50 * 0.40 + 50 * 0.005) / 1_000_000)
    assert result.wall_clock_s > 0


def test_a_completed_run_carries_the_load_report(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build(candidate, fake_python), tmp_path)
    assert [(load.path, load.loaded) for load in result.load_report.extensions] == [("extensions/guard.py", True)]


def test_an_extension_that_fails_to_load_is_a_harness_load_error(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build(candidate, fake_python), tmp_path, "SCENARIO load_error")
    assert (result.status, result.error_class) == ("errored", "harness_load_error")
    assert result.load_report.extensions[0].loaded is False
    assert "SyntaxError" in result.load_report.extensions[0].error


def test_extension_code_that_throws_mid_run_is_an_extension_error(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build(candidate, fake_python), tmp_path, "SCENARIO extension_error")
    assert (result.status, result.error_class) == ("errored", "extension_error")
    assert result.load_report.runtime_errors == ("extensions/a.py:3 in before_model: ValueError: boom",)


def test_a_quota_failure_is_classed_quota(tmp_path, fake_python):
    result, _, _ = run(build(Candidate.load(BASELINE), fake_python), tmp_path, "SCENARIO quota")
    assert (result.status, result.error_class) == ("errored", "quota")


def test_a_reply_from_another_model_is_a_model_mismatch(tmp_path, fake_python):
    result, _, _ = run(build(Candidate.load(BASELINE), fake_python), tmp_path, "SCENARIO wrong_model")
    assert (result.status, result.error_class) == ("errored", "model_mismatch")
    assert "gpt-4o-2024-08-06" in result.error


def test_a_runner_crash_without_a_result_is_an_agent_error_with_the_stderr_tail(tmp_path, fake_python):
    result, _, _ = run(build(Candidate.load(BASELINE), fake_python), tmp_path, "SCENARIO crash")
    assert (result.status, result.error_class) == ("errored", "agent_error")
    assert "exited with code 3" in result.error and "the runner fell over" in result.error


def test_an_unknown_runner_error_class_becomes_an_agent_error(tmp_path, fake_python):
    result, _, _ = run(build(Candidate.load(BASELINE), fake_python), tmp_path, "SCENARIO bogus_class")
    assert result.error_class == "agent_error" and "not_a_class" in result.error


def test_a_runner_that_vanished_after_build_is_an_agent_error(tmp_path, fake_python):
    agent = build(Candidate.load(BASELINE), fake_python)
    fake_python.unlink()
    result, _, _ = run(agent, tmp_path)
    assert (result.status, result.error_class) == ("errored", "agent_error")
    assert "could not start the runner" in result.error


def test_run_never_raises_when_the_trace_dir_cannot_be_written(tmp_path, fake_python):
    agent = build(Candidate.load(BASELINE), fake_python)
    workdir = tmp_path / "work"
    workdir.mkdir()
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    result = agent.run(Task(id="t1", split="search", instruction="hi"), workdir, blocker / "trace")
    assert (result.status, result.error_class) == ("errored", "agent_error")


# --- stopping a run completely --------------------------------------------------------------


def test_a_timed_out_run_is_killed_with_its_detached_children(tmp_path, fake_python):
    agent = build(Candidate.load(BASELINE), fake_python, timeout_s=1.5, term_grace_s=0.5)
    started = time.monotonic()
    result, _, trace_dir = run(agent, tmp_path, "SCENARIO hang")
    assert time.monotonic() - started < 10
    assert (result.status, result.error_class) == ("errored", "timeout")
    assert result.usage.input_tokens == 100  # the spend before the kill still counts (P12)
    sleeper = int((trace_dir / "sleeper.pid").read_text())
    assert wait_dead(sleeper)


def test_detached_children_are_reaped_after_a_normal_run(tmp_path, fake_python):
    result, _, trace_dir = run(build(Candidate.load(BASELINE), fake_python), tmp_path, "SCENARIO orphan")
    assert result.status == "completed"
    sleeper = int((trace_dir / "sleeper.pid").read_text())
    try:
        assert wait_dead(sleeper)
    finally:
        try:
            os.killpg(sleeper, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# --- the extension spec and the valve catalog ------------------------------------------------


def test_extension_spec_maps_every_lifecycle_point_once():
    spec = DeepAgentsAdapter().extension_spec()
    assert [point.id for point in spec.lifecycle] == list(LIFECYCLE_POINTS)


def test_extension_spec_gives_support_native_api_and_effect_for_every_point():
    for point in DeepAgentsAdapter().extension_spec().lifecycle:
        assert point.support in ("supported", "partial", "unsupported"), point.id
        assert point.native_api.strip() and point.can_modify.strip(), point.id


def test_extension_spec_declares_python_files_the_entry_function_and_the_pinned_version():
    spec = DeepAgentsAdapter().extension_spec()
    assert (spec.language, spec.suffixes, spec.framework_version) == ("python", (".py",), "0.7.28")
    assert "def extension()" in spec.entry_convention


def test_extension_spec_forbids_switching_the_model():
    forbidden = " ".join(DeepAgentsAdapter().extension_spec().forbidden)
    assert "override(model=" in forbidden and "ChatOpenAI" in forbidden


def test_extension_spec_api_reference_is_the_pinned_api_file():
    spec = DeepAgentsAdapter().extension_spec()
    assert spec.api_reference == (REPO_ROOT / "hive/agents/deepagents/EXTENSIONS_API.md").read_text()
    assert "0.7.28" in spec.api_reference and "class AgentMiddleware" in spec.api_reference


def test_extension_guide_renders_every_point_for_the_overseer():
    guide = render_extension_guide(DeepAgentsAdapter().extension_spec())
    for point_id in LIFECYCLE_POINTS:
        assert f"`{point_id}`" in guide
    assert "not declared" not in guide


def test_describe_valves_is_the_valves_doc():
    assert DeepAgentsAdapter().describe_valves() == (REPO_ROOT / "hive/agents/deepagents/VALVES.md").read_text()


def test_valves_doc_names_every_valve_and_file_and_fixes_the_model():
    doc = DeepAgentsAdapter().describe_valves()
    for name in ("tools", "general_purpose_subagent", "SYSTEM.md", "extensions/", "skills/", "README.md"):
        assert name in doc, name
    assert "def extension()" in doc
    assert "fixed by the experiment" in doc
