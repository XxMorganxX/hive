"""Layer 1 for the Strands adapter: the controller side only, with no framework installed.

`python_path` points at a wrapper around tests/fakes/strands_fake_python.py, which answers
the version check and plays canned runner scenarios (trace.jsonl, result.json, child pgids).
The real runner is covered by tests/test_strands_runner.py.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

import hive.agents.strands.adapter as strands_adapter
from hive.agents.strands import StrandsAdapter
from hive.agents.strands.adapter import RUNNER, StrandsAgent, StrandsVersionMismatch, check_strands_version
from hive.candidate import Candidate
from hive.contracts import LIFECYCLE_POINTS, Agent, AgentAdapter, ExtensionSpec, ModelSpec, Task
from hive.extensions import render_extension_guide

FAKE_PYTHON = Path(__file__).parent / "fakes" / "strands_fake_python.py"
MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
TASK = Task(id="t1", split="search", instruction="What is 6 * 7? End with ANSWER: <x>.")
PRICES = {"input": 0.05, "output": 0.40, "cache_read": 0.005}
# Variables a shell or the interpreter adds on its own, not the adapter.
INTERPRETER_NOISE = {"PWD", "SHLVL", "_", "OLDPWD", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}


def write_candidate(root: Path, valves: str = "", files: dict[str, str] | None = None, agent: str = "strands") -> Candidate:
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text(
        f'schema_version = 1\nagent = "{agent}"\ndescription = "test"\n\n[valves]\n{valves}'
    )
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Candidate.load(root)


GOOD_EXTENSION = '''
from strands.hooks import BeforeToolCallEvent
from strands.plugins import Plugin, hook


class Guard(Plugin):
    name = "guard"

    @hook
    def check(self, event: BeforeToolCallEvent) -> None:
        pass


def extension():
    return Guard()
'''


def skill_md(name: str | None = "plan", description: str | None = "Use when planning.") -> str:
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    return "\n".join([*lines, "---", "", "# Steps"]) + "\n"


@pytest.fixture
def fake_python(tmp_path):
    """Returns a function making a fake interpreter that plays one scenario."""

    def make(scenario: str = "completed", version: str = "1.61.1") -> Path:
        wrapper = tmp_path / f"python-{scenario}-{version}"
        wrapper.write_text(
            "#!/bin/sh\n"
            f'exec "{sys.executable}" "{FAKE_PYTHON}" --fake-scenario {scenario} --fake-version {version} "$@"\n'
        )
        wrapper.chmod(0o755)
        return wrapper

    return make


def settings(python_path: Path, **overrides) -> dict:
    return {"python_path": str(python_path), "strands_version": "1.61.1", "prices": PRICES, "timeout_s": 30,
            **overrides}  # fmt: skip


def build_agent(candidate: Candidate, python_path: Path, model: ModelSpec = MODEL, **overrides) -> StrandsAgent:
    return StrandsAdapter().build(candidate, model, settings(python_path, **overrides))


def run(agent: StrandsAgent, tmp_path: Path, task: Task = TASK):
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(exist_ok=True)
    return agent.run(task, workdir, trace_dir), workdir, trace_dir


def record(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "fake_record.json").read_text())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(status) and not status.startswith("Z")


def wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


# --- protocol conformance ---------------------------------------------------------------


def test_adapter_satisfies_agent_adapter_protocol():
    adapter = StrandsAdapter()
    assert isinstance(adapter, AgentAdapter)
    assert adapter.name == "strands"


def test_built_agent_satisfies_agent_protocol(tmp_path, fake_python):
    assert isinstance(build_agent(write_candidate(tmp_path / "c"), fake_python()), Agent)


def test_adapter_accepts_settings_in_constructor(tmp_path, fake_python):
    adapter = StrandsAdapter(settings(fake_python()))
    agent = adapter.build(write_candidate(tmp_path / "c"), MODEL, {})
    assert isinstance(agent, StrandsAgent)


def test_the_controller_side_imports_no_framework():
    source = Path(strands_adapter.__file__).read_text()
    assert "import strands" not in source and "from strands" not in source


def test_the_runner_never_imports_hive():
    source = RUNNER.read_text()
    assert "import hive" not in source and "from hive" not in source


# --- validate ---------------------------------------------------------------------------


def test_minimal_candidate_is_valid(tmp_path):
    assert StrandsAdapter().validate(write_candidate(tmp_path / "c")) == []


def test_full_candidate_is_valid(tmp_path):
    candidate = write_candidate(
        tmp_path / "c",
        valves='tools = ["shell", "file_editor", "python_repl"]\nconversation_window = 20\n',
        files={
            "SYSTEM.md": "You are careful.",
            "README.md": "notes",
            "extensions/guard.py": GOOD_EXTENSION,
            "skills/plan/SKILL.md": skill_md(),
        },
    )
    assert StrandsAdapter().validate(candidate) == []


def test_validate_rejects_another_adapters_candidate(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", agent="pi"))
    assert any("expected 'strands'" in problem for problem in problems)


def test_validate_rejects_unknown_top_level_entry(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"tools/x.py": "pass"}))
    assert any("unknown top-level entry 'tools'" in problem for problem in problems)


def test_validate_rejects_unknown_valve(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", valves="temperature = 0.2\n"))
    assert any("unknown valve 'temperature'" in problem for problem in problems)


def test_validate_rejects_tools_that_are_not_a_list(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", valves='tools = "shell"\n'))
    assert any("valve 'tools'" in problem for problem in problems)


def test_validate_rejects_a_blank_tool_name(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", valves='tools = ["shell", " "]\n'))
    assert any("invalid tool name" in problem for problem in problems)


def test_validate_rejects_a_non_positive_conversation_window(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", valves="conversation_window = 0\n"))
    assert any("conversation_window" in problem for problem in problems)


def test_validate_rejects_an_empty_system_prompt(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"SYSTEM.md": "  \n"}))
    assert any("SYSTEM.md is empty" in problem for problem in problems)


def test_validate_rejects_an_extension_in_a_subdirectory(tmp_path):
    files = {"extensions/sub/guard.py": GOOD_EXTENSION}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("directly in extensions/" in problem for problem in problems)


def test_validate_rejects_a_non_python_file_in_extensions(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/guard.ts": "x"}))
    assert any("must end in .py" in problem for problem in problems)


def test_validate_rejects_an_extension_with_a_syntax_error(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/guard.py": "def (:\n"}))
    assert any("extensions/guard.py" in problem and "syntax error" in problem for problem in problems)


def test_validate_rejects_an_extension_without_an_entry_function(tmp_path):
    files = {"extensions/guard.py": "x = 1\n"}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("top-level function extension()" in problem for problem in problems)


def test_validate_rejects_an_async_entry_function(tmp_path):
    files = {"extensions/guard.py": "async def extension():\n    return None\n"}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("must not be async" in problem for problem in problems)


def test_validate_rejects_an_entry_function_with_required_arguments(tmp_path):
    files = {"extensions/guard.py": "def extension(agent):\n    return None\n"}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("no arguments" in problem for problem in problems)


def test_validate_never_runs_extension_code(tmp_path):
    marker = tmp_path / "ran"
    source = f"open({str(marker)!r}, 'w').write('x')\n\ndef extension():\n    return None\n"
    StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/guard.py": source}))
    assert not marker.exists()


@pytest.mark.parametrize(
    "source, needle",
    [
        ("from strands.models.bedrock import BedrockModel\n", "strands.models"),
        ("import strands._middleware\n", "strands._middleware"),
        ("from strands.models.routing import ModelRouter\n", "strands.models"),
        ("import openai\n", "openai"),
        ("def f(event):\n    event.agent.model = None\n", ".model"),
        ("def f(agent):\n    agent.model.update_config(model_id='x')\n", "update_config"),
    ],
)
def test_validate_rejects_extension_code_that_could_switch_the_model(tmp_path, source, needle):
    files = {"extensions/guard.py": source + "\ndef extension():\n    return None\n"}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("extensions/guard.py" in problem and needle in problem and "P3" in problem for problem in problems)


def test_validate_allows_reading_the_model(tmp_path):
    source = "def f(event):\n    return event.agent.model.get_config()\n\ndef extension():\n    return None\n"
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/guard.py": source}))
    assert problems == []


def test_validate_rejects_a_skill_directory_without_skill_md(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"skills/plan/notes.md": "x"}))
    assert any("'skills/plan' has no SKILL.md" in problem for problem in problems)


def test_validate_rejects_a_skill_without_a_description(tmp_path):
    files = {"skills/plan/SKILL.md": skill_md(description=None)}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("description" in problem for problem in problems)


def test_validate_rejects_a_skill_whose_name_differs_from_its_directory(tmp_path):
    files = {"skills/plan/SKILL.md": skill_md(name="planner")}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("must equal its directory name" in problem for problem in problems)


def test_validate_rejects_a_skill_name_strands_would_refuse(tmp_path):
    files = {"skills/Plan_It/SKILL.md": skill_md(name="Plan_It")}
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("lowercase" in problem for problem in problems)


def test_validate_rejects_loose_files_in_skills(tmp_path):
    problems = StrandsAdapter().validate(write_candidate(tmp_path / "c", files={"skills/notes.md": "x"}))
    assert any("skills/ may only contain skill directories" in problem for problem in problems)


def test_validate_never_raises_on_unreadable_candidate(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    (tmp_path / "c" / "extensions" / "guard.py").write_bytes(b"\xff\xfe\x00bad")
    problems = StrandsAdapter().validate(candidate)
    assert problems  # reported, not raised


# --- settings ---------------------------------------------------------------------------


def test_build_rejects_unknown_settings(tmp_path, fake_python):
    with pytest.raises(ValueError, match="unknown Strands adapter settings: recursion_limit"):
        build_agent(write_candidate(tmp_path / "c"), fake_python(), recursion_limit=5)


def test_build_requires_python_path(tmp_path):
    with pytest.raises(ValueError, match="python_path"):
        StrandsAdapter().build(write_candidate(tmp_path / "c"), MODEL, {"strands_version": "1.61.1", "prices": PRICES})


def test_build_requires_a_strands_version_pin(tmp_path, fake_python):
    with pytest.raises(ValueError, match="strands_version"):
        StrandsAdapter().build(write_candidate(tmp_path / "c"), MODEL, {"python_path": str(fake_python()), "prices": PRICES})


def test_build_requires_prices(tmp_path, fake_python):
    with pytest.raises(ValueError, match="prices"):
        StrandsAdapter().build(
            write_candidate(tmp_path / "c"), MODEL, {"python_path": str(fake_python()), "strands_version": "1.61.1"}
        )


@pytest.mark.parametrize("key, value", [("timeout_s", 0), ("timeout_s", True), ("term_grace_s", -1),
                                        ("max_turns", 0), ("max_turns", 2.5), ("max_model_attempts", 0)])  # fmt: skip
def test_build_rejects_bad_numeric_settings(tmp_path, fake_python, key, value):
    with pytest.raises(ValueError, match=key):
        build_agent(write_candidate(tmp_path / "c"), fake_python(), **{key: value})


def test_build_rejects_an_invalid_candidate(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", valves="bogus = 1\n")
    with pytest.raises(ValueError, match="not a valid Strands candidate"):
        build_agent(candidate, fake_python())


def test_build_rejects_an_unsupported_provider(tmp_path, fake_python):
    with pytest.raises(ValueError, match="provider"):
        build_agent(write_candidate(tmp_path / "c"), fake_python(), model=ModelSpec(provider="bedrock", id="claude"))


def test_build_rejects_an_unknown_thinking_level(tmp_path, fake_python):
    model = ModelSpec(provider="openai", id="gpt-5-nano", thinking="extreme")
    with pytest.raises(ValueError, match="thinking"):
        build_agent(write_candidate(tmp_path / "c"), fake_python(), model=model)


def test_build_requires_a_script_for_the_scripted_provider(tmp_path, fake_python):
    with pytest.raises(ValueError, match="script_file"):
        build_agent(write_candidate(tmp_path / "c"), fake_python(), model=ModelSpec(provider="scripted", id="s"))


def test_build_rejects_a_script_for_a_real_provider(tmp_path, fake_python):
    with pytest.raises(ValueError, match="script_file"):
        build_agent(write_candidate(tmp_path / "c"), fake_python(), script_file=str(tmp_path / "s.json"))


# --- version pin ------------------------------------------------------------------------


def test_build_accepts_the_pinned_version(tmp_path, fake_python):
    build_agent(write_candidate(tmp_path / "c"), fake_python(version="1.61.1"))


def test_build_raises_on_a_version_mismatch(tmp_path, fake_python):
    with pytest.raises(StrandsVersionMismatch, match="1.60.0 is installed but 1.61.1 is pinned"):
        build_agent(write_candidate(tmp_path / "c"), fake_python(version="1.60.0"))


def test_version_check_reports_a_missing_package(tmp_path, fake_python):
    with pytest.raises(StrandsVersionMismatch, match="PackageNotFoundError"):
        check_strands_version(str(fake_python(version="missing")), "1.61.1")


def test_version_check_reports_a_missing_interpreter(tmp_path):
    with pytest.raises(StrandsVersionMismatch, match="could not run"):
        check_strands_version(str(tmp_path / "no-such-python"), "1.61.1")


# --- argv and environment ---------------------------------------------------------------


def test_runs_the_runner_isolated_with_the_run_spec(tmp_path, fake_python):
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert record(trace_dir)["argv"] == ["-I", str(RUNNER), str(trace_dir / "run_spec.json")]


def test_runs_in_the_trial_workdir(tmp_path, fake_python):
    _, workdir, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert Path(record(trace_dir)["cwd"]).resolve() == workdir.resolve()


def test_environment_holds_only_what_the_run_needs(tmp_path, fake_python, monkeypatch):
    for name, value in {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_PROFILE": "prod", "SECRET_THING": "x",
                        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector", "OPENAI_BASE_URL": "http://evil",
                        "PYTHONPATH": "/evil", "OPENAI_API_KEY": "sk-test-key"}.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    env = record(trace_dir)["env"]
    allowed = {"PATH", "HOME", "LANG", "OPENAI_API_KEY", "AWS_EC2_METADATA_DISABLED", "AWS_CONFIG_FILE",
               "AWS_SHARED_CREDENTIALS_FILE", "OTEL_SDK_DISABLED"}  # fmt: skip
    assert set(env) - INTERPRETER_NOISE <= allowed


def test_environment_passes_the_openai_key_for_openai(tmp_path, fake_python, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert record(trace_dir)["env"]["OPENAI_API_KEY"] == "sk-test-key"


def test_environment_blocks_aws_credential_discovery(tmp_path, fake_python):
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    env = record(trace_dir)["env"]
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"
    assert env["AWS_CONFIG_FILE"] == os.devnull
    assert env["AWS_SHARED_CREDENTIALS_FILE"] == os.devnull


def test_the_scripted_provider_gets_no_api_key(tmp_path, fake_python, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    script = tmp_path / "script.json"
    script.write_text("{}")
    agent = build_agent(write_candidate(tmp_path / "c"), fake_python(), model=ModelSpec(provider="scripted", id="s"),
                        script_file=str(script))  # fmt: skip
    _, _, trace_dir = run(agent, tmp_path)
    assert "OPENAI_API_KEY" not in record(trace_dir)["env"]


def test_the_api_key_never_reaches_argv_or_the_run_spec(tmp_path, fake_python, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert "sk-test-key-123" not in json.dumps(record(trace_dir)["argv"])
    assert "sk-test-key-123" not in (trace_dir / "run_spec.json").read_text()


# --- the run spec -----------------------------------------------------------------------


def spec_of(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "run_spec.json").read_text())


def test_run_spec_carries_the_model_and_its_reasoning_effort(tmp_path, fake_python):
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert spec_of(trace_dir)["model"] == {"provider": "openai", "id": "gpt-5-nano", "reasoning_effort": "low"}


def test_reasoning_effort_defaults_to_low(tmp_path, fake_python):
    agent = build_agent(write_candidate(tmp_path / "c"), fake_python(), model=ModelSpec(provider="openai", id="gpt-5-nano"))
    _, _, trace_dir = run(agent, tmp_path)
    assert spec_of(trace_dir)["model"]["reasoning_effort"] == "low"


def test_run_spec_lists_every_extension_file(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION,
                                                       "extensions/other.py": GOOD_EXTENSION.replace("guard", "other")})  # fmt: skip
    _, _, trace_dir = run(build_agent(candidate, fake_python()), tmp_path)
    extensions = spec_of(trace_dir)["extensions"]
    assert [entry["path"] for entry in extensions] == ["extensions/guard.py", "extensions/other.py"]
    assert extensions[0]["file"] == str(candidate.root / "extensions" / "guard.py")


def test_run_spec_carries_the_tools_valve_with_default(tmp_path, fake_python):
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert spec_of(trace_dir)["tools"] == ["shell", "file_editor"]


def test_system_prompt_is_the_candidate_prompt_plus_the_absolute_workdir(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"SYSTEM.md": "You are careful."})
    _, workdir, trace_dir = run(build_agent(candidate, fake_python()), tmp_path)
    prompt = spec_of(trace_dir)["system_prompt"]
    assert prompt.startswith("You are careful.")
    assert str(workdir.resolve()) in prompt


def test_system_prompt_without_system_md_still_names_the_workdir(tmp_path, fake_python):
    _, workdir, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python()), tmp_path)
    assert str(workdir.resolve()) in spec_of(trace_dir)["system_prompt"]


def test_run_spec_carries_limits(tmp_path, fake_python):
    agent = build_agent(write_candidate(tmp_path / "c"), fake_python(), max_turns=7, max_model_attempts=2)
    _, _, trace_dir = run(agent, tmp_path)
    spec = spec_of(trace_dir)
    assert spec["max_turns"] == 7
    assert spec["retry"]["max_attempts"] == 2


# --- classification through the runner protocol ----------------------------------------


def test_completed_run_reports_text_and_priced_usage(tmp_path, fake_python):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c"), fake_python("completed")), tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 42"
    assert result.usage.input_tokens == 300
    assert result.usage.output_tokens == 60
    assert result.usage.cache_read_tokens == 30
    assert result.usage.cost_usd == pytest.approx((300 * 0.05 + 60 * 0.40 + 30 * 0.005) / 1e6)


def test_completed_run_carries_the_load_report(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build_agent(candidate, fake_python("completed")), tmp_path)
    assert [(load.path, load.loaded) for load in result.load_report.extensions] == [("extensions/guard.py", True)]
    assert result.load_report.extensions[0].registered == ("before_tool_call", "tool:word_count")


def test_a_dated_model_snapshot_counts_as_the_model(tmp_path, fake_python):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c"), fake_python("completed_snapshot")), tmp_path)
    assert result.status == "completed"


def test_another_reported_model_is_model_mismatch(tmp_path, fake_python):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c"), fake_python("model_mismatch")), tmp_path)
    assert result.error_class == "model_mismatch"


@pytest.mark.parametrize(
    "scenario, error_class",
    [("harness_load_error", "harness_load_error"), ("extension_error", "extension_error"), ("quota", "quota"),
     ("provider_error", "provider_error"), ("step_limit", "agent_error"), ("unknown_error_class", "agent_error"),
     ("missing_final_text", "agent_error")],
)
def test_runner_errors_are_classified(tmp_path, fake_python, scenario, error_class):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build_agent(candidate, fake_python(scenario)), tmp_path)
    assert result.status == "errored"
    assert result.error_class == error_class


def test_a_load_failure_reports_which_file_failed(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build_agent(candidate, fake_python("harness_load_error")), tmp_path)
    load = result.load_report.extensions[0]
    assert (load.path, load.loaded) == ("extensions/guard.py", False)
    assert "NameError" in load.error


def test_a_runtime_throw_is_reported_with_its_path(tmp_path, fake_python):
    candidate = write_candidate(tmp_path / "c", files={"extensions/guard.py": GOOD_EXTENSION})
    result, _, _ = run(build_agent(candidate, fake_python("extension_error")), tmp_path)
    assert any("extensions/guard.py" in error for error in result.load_report.runtime_errors)


def test_a_runner_crash_is_agent_error_with_stderr(tmp_path, fake_python):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c"), fake_python("crash")), tmp_path)
    assert result.error_class == "agent_error"
    assert "runner bug" in result.error
    assert result.usage.input_tokens == 100  # spend before the crash is still counted


def test_a_killed_runner_is_agent_error(tmp_path, fake_python):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c"), fake_python("killed")), tmp_path)
    assert result.error_class == "agent_error"
    assert "SIGKILL" in result.error


def test_a_missing_interpreter_is_agent_error_at_run_time(tmp_path, fake_python):
    agent = build_agent(write_candidate(tmp_path / "c"), fake_python())
    broken = StrandsAgent(**{**agent.__dict__, "python_path": str(tmp_path / "gone")})
    result, _, _ = run(broken, tmp_path)
    assert result.error_class == "agent_error"
    assert "could not start" in result.error


# --- timeout and child reaping ----------------------------------------------------------


def test_timeout_stops_the_run_and_keeps_its_spend(tmp_path, fake_python):
    agent = build_agent(write_candidate(tmp_path / "c"), fake_python("hang"), timeout_s=1, term_grace_s=0.5)
    started = time.monotonic()
    result, _, _ = run(agent, tmp_path)
    assert time.monotonic() - started < 15
    assert result.error_class == "timeout"
    assert result.usage.input_tokens == 100


def test_listed_child_process_groups_are_killed_after_the_run(tmp_path, fake_python):
    result, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), fake_python("orphan")), tmp_path)
    assert result.status == "completed"
    pid = int((trace_dir / "orphan.pid").read_text())
    try:
        assert wait_dead(pid)
    finally:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_stale_result_files_from_an_earlier_run_are_removed(tmp_path, fake_python):
    agent = build_agent(write_candidate(tmp_path / "c"), fake_python("crash"))
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "result.json").write_text(json.dumps({"status": "completed", "final_text": "stale"}))
    result, _, _ = run(agent, tmp_path)
    assert result.status == "errored"


# --- extension spec ---------------------------------------------------------------------


def test_extension_spec_is_python_files_with_the_pinned_version():
    spec = StrandsAdapter().extension_spec()
    assert isinstance(spec, ExtensionSpec)
    assert spec.language == "python"
    assert spec.suffixes == (".py",)
    assert spec.framework_version == "strands-agents 1.61.1"


def test_extension_spec_states_the_entry_convention():
    assert "def extension()" in StrandsAdapter().extension_spec().entry_convention
    assert "Plugin" in StrandsAdapter().extension_spec().entry_convention


def test_extension_spec_covers_every_lifecycle_point_once():
    lifecycle = StrandsAdapter().extension_spec().lifecycle
    assert sorted(point.id for point in lifecycle) == sorted(LIFECYCLE_POINTS)


def test_every_lifecycle_point_names_its_native_api():
    for point in StrandsAdapter().extension_spec().lifecycle:
        assert point.support in ("supported", "partial", "unsupported")
        assert point.native_api and point.can_modify


def test_wrapping_the_model_call_is_unsupported():
    points = {point.id: point for point in StrandsAdapter().extension_spec().lifecycle}
    assert points["wrap_model_call"].support == "unsupported"


def test_extension_spec_forbids_switching_the_model():
    forbidden = " ".join(StrandsAdapter().extension_spec().forbidden)
    for api in ("agent.model", "strands._middleware", "ModelRouter", "BedrockModel"):
        assert api in forbidden


def test_extension_spec_api_reference_is_the_pinned_reference_file():
    spec = StrandsAdapter().extension_spec()
    reference = (Path(strands_adapter.__file__).parent / "EXTENSIONS_API.md").read_text()
    assert spec.api_reference == reference
    assert "1.61.1" in reference
    for name in ("class Plugin", "BeforeToolCallEvent", "AfterInvocationEvent", "def tool"):
        assert name in reference


def test_extension_spec_renders_into_a_guide_with_every_point():
    guide = render_extension_guide(StrandsAdapter().extension_spec())
    assert "not declared" not in guide
    for point_id in LIFECYCLE_POINTS:
        assert f"`{point_id}`" in guide


# --- describe_valves --------------------------------------------------------------------


def test_describe_valves_is_valves_md():
    assert StrandsAdapter().describe_valves() == (Path(strands_adapter.__file__).parent / "VALVES.md").read_text()


def test_describe_valves_covers_every_valve_and_file():
    text = StrandsAdapter().describe_valves()
    for name in ("tools", "conversation_window", "SYSTEM.md", "extensions/", "skills/", "README.md"):
        assert name in text


def test_describe_valves_says_the_model_and_thinking_are_fixed():
    text = StrandsAdapter().describe_valves()
    assert "model" in text and "thinking" in text and "model_mismatch" in text


def test_describe_valves_points_to_the_extension_guide():
    text = StrandsAdapter().describe_valves()
    assert "def extension()" in text
    assert "extension guide" in text.lower()


# --- shipped seed -----------------------------------------------------------------------


def test_the_baseline_seed_is_valid():
    repo_root = Path(__file__).resolve().parents[1]
    seed = Candidate.load(repo_root / "harnesses" / "strands-baseline")
    assert seed.agent == "strands"
    assert StrandsAdapter().validate(seed) == []
    assert not (seed.root / "extensions").exists()
