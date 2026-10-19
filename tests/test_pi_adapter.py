import json
import os
import shutil
import signal
import time
from pathlib import Path

import pytest

from fakes.fake_openai_upstream import FakeOpenAIUpstream
from hive.agents.pi import PiAdapter
import hive.agents.pi.adapter as pi_adapter
from hive.agents.pi.adapter import PiAgent, PiVersionMismatch, check_pi_version
from hive.candidate import Candidate
from hive.contracts import (
    LIFECYCLE_POINTS, Agent, AgentAdapter, ExtensionSpec, Grade, LoadReport, MetricSpec, ModelEndpoint, ModelSpec,
    Task, Usage,
)  # fmt: skip
from hive.evaluate import evaluate
from hive.extensions import expected_extension_files, load_problem, render_extension_guide
from hive.gateway import ModelGateway
from hive.pricing import Prices

FAKE_PI = Path(__file__).parent / "fakes" / "fake_pi.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = ModelSpec(provider="openai", id="gpt-5-nano")
TASK = Task(id="t1", split="search", instruction="What is 6 * 7? End with ANSWER: <x>.")


def skill_md(name: str | None = None, description: str | None = "Use when planning.") -> str:
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    return "\n".join([*lines, "---", "", "# Steps"]) + "\n"


def write_candidate(root: Path, valves: str = "", files: dict[str, str] | None = None, agent: str = "pi") -> Candidate:
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
def user_agent_dir(tmp_path, monkeypatch):
    """A stand-in for the user's real Pi agent dir, so tests never touch ~/.pi/agent."""
    agent_dir = tmp_path / "user-pi-agent"
    (agent_dir / "bin").mkdir(parents=True)
    (agent_dir / "bin" / "rg").write_text("#!/bin/sh\n")
    (agent_dir / "auth.json").write_text('{"openai": {"type": "api_key", "key": "sk-test"}}')
    (agent_dir / "settings.json").write_text('{"defaultModel": "gpt-5.4", "retry": {"enabled": false}}')
    (agent_dir / "models.json").write_text('{"providers": {"openai": {"baseUrl": "http://evil"}}}')
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    return agent_dir


@pytest.fixture
def fake_pi(tmp_path, monkeypatch, user_agent_dir):
    """Point runs at the fake pi; returns a function that reads what the fake recorded."""
    record = tmp_path / "fake_pi_record.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    monkeypatch.delenv("FAKE_PI_SCENARIO", raising=False)
    monkeypatch.delenv("FAKE_PI_VERSION", raising=False)
    monkeypatch.delenv("FAKE_PI_FAILING_EXTENSION", raising=False)
    return lambda: json.loads(record.read_text())


def build_agent(candidate: Candidate, model: ModelSpec = MODEL, **settings) -> PiAgent:
    return PiAdapter().build(candidate, model, {"pi_bin": str(FAKE_PI), **settings})


def run(agent: PiAgent, tmp_path: Path, task: Task = TASK):
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    trace_dir = tmp_path / "trace"
    return agent.run(task, workdir, trace_dir), workdir, trace_dir


def flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == flag]


# --- protocol conformance ---------------------------------------------------------------


def test_adapter_satisfies_agent_adapter_protocol():
    adapter = PiAdapter()
    assert isinstance(adapter, AgentAdapter)
    assert adapter.name == "pi"


def test_built_agent_satisfies_agent_protocol(tmp_path):
    assert isinstance(build_agent(write_candidate(tmp_path / "c")), Agent)


def test_adapter_accepts_settings_in_constructor():
    assert PiAdapter({"pi_bin": "pi"}).name == "pi"


# --- argv -------------------------------------------------------------------------------


def test_argv_isolates_pi_from_user_global_config(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    argv = fake_pi()["argv"]
    for flag in ("--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files"):
        assert flag in argv


def test_argv_requests_json_print_mode(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    argv = fake_pi()["argv"]
    assert flag_values(argv, "--mode") == ["json"]
    assert "-p" in argv


def test_argv_is_offline_by_default(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert "--offline" in fake_pi()["argv"]


def test_argv_omits_offline_when_disabled(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c"), offline=False), tmp_path)
    assert "--offline" not in fake_pi()["argv"]


def test_argv_passes_qualified_model(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert flag_values(fake_pi()["argv"], "--model") == ["openai/gpt-5-nano"]


def test_argv_passes_default_tools(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert flag_values(fake_pi()["argv"], "--tools") == ["read,bash,edit,write"]


def test_argv_passes_tools_valve(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c", valves='tools = ["read", "grep"]\n')), tmp_path)
    assert flag_values(fake_pi()["argv"], "--tools") == ["read,grep"]


def test_thinking_defaults_to_low_when_the_model_spec_leaves_it_unset(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert flag_values(fake_pi()["argv"], "--thinking") == ["low"]


def test_thinking_comes_from_the_model_spec(tmp_path, fake_pi):
    model = ModelSpec(provider="openai", id="gpt-5-nano", thinking="high")
    run(build_agent(write_candidate(tmp_path / "c"), model), tmp_path)
    assert flag_values(fake_pi()["argv"], "--thinking") == ["high"]


def test_build_rejects_unknown_model_thinking_level(tmp_path):
    with pytest.raises(ValueError, match="thinking"):
        build_agent(write_candidate(tmp_path / "c"), ModelSpec(provider="openai", id="gpt-5-nano", thinking="extreme"))


def test_empty_tools_valve_disables_all_tools(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c", valves="tools = []\n")), tmp_path)
    argv = fake_pi()["argv"]
    assert "--no-tools" in argv
    assert "--tools" not in argv


def test_argv_writes_session_into_trace_dir(tmp_path, fake_pi):
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert flag_values(fake_pi()["argv"], "--session-dir") == [str(trace_dir / "session")]


def test_pi_runs_with_workdir_as_cwd(tmp_path, fake_pi):
    _, workdir, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert Path(fake_pi()["cwd"]).resolve() == workdir.resolve()


def test_argv_passes_each_extension_with_e(tmp_path, fake_pi):
    candidate = write_candidate(
        tmp_path / "c", files={"extensions/a.ts": "export default () => {}", "extensions/b.js": "x"}
    )
    run(build_agent(candidate), tmp_path)
    assert flag_values(fake_pi()["argv"], "-e") == [
        str(candidate.root / "extensions" / "a.ts"),
        str(candidate.root / "extensions" / "b.js"),
    ]


def test_argv_passes_each_skill_directory(tmp_path, fake_pi):
    candidate = write_candidate(
        tmp_path / "c", files={"skills/alpha/SKILL.md": skill_md(), "skills/beta/SKILL.md": skill_md()}
    )
    run(build_agent(candidate), tmp_path)
    assert flag_values(fake_pi()["argv"], "--skill") == [
        str(candidate.root / "skills" / "alpha"),
        str(candidate.root / "skills" / "beta"),
    ]


def test_argv_passes_system_prompt_files_by_absolute_path(tmp_path, fake_pi):
    candidate = write_candidate(tmp_path / "c", files={"SYSTEM.md": "be terse", "APPEND_SYSTEM.md": "check work"})
    run(build_agent(candidate), tmp_path)
    argv = fake_pi()["argv"]
    assert flag_values(argv, "--system-prompt") == [str(candidate.root / "SYSTEM.md")]
    assert flag_values(argv, "--append-system-prompt") == [str(candidate.root / "APPEND_SYSTEM.md")]


def test_missing_prompt_files_pass_empty_values_to_block_global_discovery(tmp_path, fake_pi):
    # An empty value keeps Pi's stock preamble but stops it reading ~/.pi/agent/SYSTEM.md.
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    argv = fake_pi()["argv"]
    assert flag_values(argv, "--system-prompt") == [""]
    assert flag_values(argv, "--append-system-prompt") == [""]


def test_instruction_is_last_argument(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert fake_pi()["argv"][-1] == TASK.instruction


def test_instruction_that_looks_like_a_flag_is_sent_on_stdin(tmp_path, fake_pi):
    task = Task(id="t2", split="search", instruction="--help me add two numbers")
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path, task)
    record = fake_pi()
    assert task.instruction not in record["argv"]
    assert record["stdin"] == task.instruction


# --- result parsing ---------------------------------------------------------------------


def test_success_is_completed_with_last_assistant_text(tmp_path, fake_pi):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 42"
    assert result.error is None and result.error_class is None


def test_usage_is_summed_over_every_assistant_message(tmp_path, fake_pi):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.usage.input_tokens == 300
    assert result.usage.output_tokens == 50
    assert result.usage.cache_read_tokens == 12
    assert result.usage.cache_write_tokens == 3
    assert result.usage.cost_usd == pytest.approx(0.03)


def test_wall_clock_is_measured(tmp_path, fake_pi):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.wall_clock_s > 0


def test_raw_stdout_and_stderr_are_saved_to_trace_dir(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_runtime_error")
    _, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    lines = (trace_dir / "events.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["type"] == "session"
    assert "Extension error" in (trace_dir / "stderr.txt").read_text()


def test_provider_error_is_errored_provider_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "provider_error")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "provider_error"
    assert "500 Internal Server Error" in result.error


def test_provider_error_still_reports_usage(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "provider_error")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.usage.input_tokens == 10


def test_aborted_run_is_errored_provider_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "aborted")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "provider_error"
    assert result.usage == Usage()


def test_quota_error_is_errored_quota(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "quota")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "quota"


def test_no_assistant_message_is_errored_agent_error_with_stderr_tail(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "no_assistant")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert "No API key found" in result.error


def test_malformed_lines_are_skipped(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "malformed")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 42"


def test_extension_load_failure_is_harness_load_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_load_error")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c", files={"extensions/bad.ts": "x"})), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "harness_load_error"
    assert "bad.ts" in result.error


def test_load_failure_pi_names_under_an_unexpected_path_still_fails_every_file(tmp_path, fake_pi, monkeypatch):
    # Under --no-extensions only -e files load, so any load failure is the candidate's.
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_load_error")
    monkeypatch.setenv("FAKE_PI_FAILING_EXTENSION", "/elsewhere/extensions/bad.ts")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c", files={"extensions/guard.ts": "x"})), tmp_path)
    assert result.error_class == "harness_load_error"
    (guard,) = result.load_report.extensions
    assert guard.loaded is False
    assert "boom in bad.ts" in guard.error


def test_extension_runtime_error_is_extension_error_even_if_run_finished(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_runtime_error")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "extension_error"


def test_timeout_is_errored_timeout_and_keeps_partial_output(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "timeout")
    started = time.monotonic()
    result, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c"), timeout_s=1), tmp_path)
    assert time.monotonic() - started < 15
    assert result.status == "errored"
    assert result.error_class == "timeout"
    assert json.loads((trace_dir / "events.jsonl").read_text().splitlines()[0])["type"] == "session"
    assert "partial stderr" in (trace_dir / "stderr.txt").read_text()


def test_timeout_kills_the_whole_process_group(tmp_path, fake_pi, monkeypatch):
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_PI_SCENARIO", "timeout")
    monkeypatch.setenv("FAKE_PI_GRANDCHILD_PID", str(pid_file))
    run(build_agent(write_candidate(tmp_path / "c"), timeout_s=1), tmp_path)
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} survived the timeout kill")


def test_missing_pi_binary_is_errored_agent_error(tmp_path):
    agent = PiAdapter().build(write_candidate(tmp_path / "c"), MODEL, {"pi_bin": str(tmp_path / "no-such-pi")})
    result, _, trace_dir = run(agent, tmp_path)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert (trace_dir / "events.jsonl").exists()


# --- build ------------------------------------------------------------------------------


def test_version_pin_mismatch_raises(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_VERSION", "0.80.0")
    with pytest.raises(PiVersionMismatch, match="0.80.0"):
        build_agent(write_candidate(tmp_path / "c"), pi_version="0.75.3")


def test_version_pin_match_builds(tmp_path, fake_pi):
    assert isinstance(build_agent(write_candidate(tmp_path / "c"), pi_version="0.75.3"), PiAgent)


def test_version_pin_with_missing_binary_raises(tmp_path):
    with pytest.raises(PiVersionMismatch):
        PiAdapter().build(
            write_candidate(tmp_path / "c"), MODEL, {"pi_bin": str(tmp_path / "no-such-pi"), "pi_version": "0.75.3"}
        )


def test_build_rejects_invalid_candidate(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        build_agent(write_candidate(tmp_path / "c", valves="temperature = 0.2\n"))


def test_build_rejects_unknown_setting(tmp_path):
    with pytest.raises(ValueError, match="pi_binary"):
        PiAdapter().build(write_candidate(tmp_path / "c"), MODEL, {"pi_binary": "pi"})


def test_build_rejects_non_positive_timeout(tmp_path):
    with pytest.raises(ValueError, match="timeout_s"):
        build_agent(write_candidate(tmp_path / "c"), timeout_s=0)


def test_constructor_settings_are_defaults_for_build(tmp_path, fake_pi):
    agent = PiAdapter({"pi_bin": str(FAKE_PI), "offline": False}).build(write_candidate(tmp_path / "c"), MODEL, {})
    run(agent, tmp_path)
    assert "--offline" not in fake_pi()["argv"]


# --- validate ---------------------------------------------------------------------------


def test_minimal_candidate_is_valid(tmp_path):
    assert PiAdapter().validate(write_candidate(tmp_path / "c")) == []


def test_full_candidate_is_valid(tmp_path):
    candidate = write_candidate(
        tmp_path / "c",
        valves='tools = ["read", "bash"]\n',
        files={
            "README.md": "notes",
            "SYSTEM.md": "be terse",
            "APPEND_SYSTEM.md": "check work",
            "extensions/guard.ts": "export default () => {}",
            "skills/plan/SKILL.md": skill_md(),
            "skills/plan/reference.md": "details",
        },
    )
    assert PiAdapter().validate(candidate) == []


def test_validate_rejects_other_agent(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", agent="strands"))
    assert any("agent" in p for p in problems)


def test_validate_rejects_unknown_valve(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves="temperature = 0.2\n"))
    assert any("temperature" in p for p in problems)


def test_validate_rejects_thinking_valve_because_the_model_spec_fixes_it(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves='thinking = "high"\n'))
    assert any("unknown valve 'thinking'" in p for p in problems)


def test_validate_rejects_tools_that_are_not_a_list(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves='tools = "read,bash"\n'))
    assert any("tools" in p for p in problems)


def test_validate_rejects_non_string_tool_names(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves='tools = ["read", 3]\n'))
    assert any("tools" in p for p in problems)


def test_validate_rejects_tool_names_with_commas_or_blanks(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves='tools = ["read,bash", " "]\n'))
    assert len([p for p in problems if "tools" in p]) == 2


def test_validate_rejects_empty_system_prompt(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"SYSTEM.md": "  \n"}))
    assert any("SYSTEM.md" in p for p in problems)


def test_validate_rejects_extension_with_other_suffix(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/notes.md": "x"}))
    assert any("extensions/notes.md" in p for p in problems)


def test_validate_rejects_nested_extension_files(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"extensions/lib/util.ts": "x"}))
    assert any("extensions/lib/util.ts" in p for p in problems)


def test_validate_rejects_skill_without_skill_md(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"skills/plan/notes.md": "x"}))
    assert any("skills/plan" in p and "SKILL.md" in p for p in problems)


def test_validate_rejects_empty_skill_directory(tmp_path):
    candidate = write_candidate(tmp_path / "c")
    (candidate.root / "skills" / "plan").mkdir(parents=True)
    problems = PiAdapter().validate(candidate)
    assert any("skills/plan" in p for p in problems)


def test_validate_rejects_loose_file_in_skills(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"skills/SKILL.md": "x"}))
    assert any("skills/SKILL.md" in p for p in problems)


def test_validate_rejects_unknown_top_level_entry(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"notes.txt": "x"}))
    assert any("notes.txt" in p for p in problems)


def test_validate_rejects_prompt_file_that_is_a_directory(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"SYSTEM.md/part.md": "x"}))
    assert any("SYSTEM.md" in p for p in problems)


def test_validate_ignores_lineage_and_dotfiles(tmp_path):
    candidate = write_candidate(tmp_path / "c", files={".DS_Store": "x"})
    candidate.write_lineage(parents=[], origin="seed", rationale="")
    assert PiAdapter().validate(candidate) == []


def test_validate_never_raises_on_unreadable_candidate(tmp_path):
    candidate = write_candidate(tmp_path / "c")
    (candidate.root / "candidate.toml").unlink()
    problems = PiAdapter().validate(candidate)
    assert isinstance(problems, list)


# --- describe_valves and the seed -------------------------------------------------------


def test_describe_valves_is_the_valves_document():
    text = PiAdapter().describe_valves()
    assert text == (REPO_ROOT / "hive" / "agents" / "pi" / "VALVES.md").read_text()


def test_describe_valves_names_every_valve_and_file():
    text = PiAdapter().describe_valves()
    for name in ("tools", "SYSTEM.md", "APPEND_SYSTEM.md", "extensions/", "skills/", "model"):
        assert name in text


def test_describe_valves_offers_no_thinking_valve():
    text = PiAdapter().describe_valves()
    assert "| `thinking`" not in text
    assert "thinking level" in text  # listed as fixed by the experiment instead


def test_seed_harness_is_valid():
    seed = Candidate.load(REPO_ROOT / "harnesses" / "pi-baseline")
    assert seed.agent == "pi"
    assert PiAdapter().validate(seed) == []


# --- pinned version (P13) ---------------------------------------------------------------


def test_version_check_reads_the_version_pi_prints_to_stderr_without_a_tty(fake_pi):
    # Real Pi 0.75.3 sends `--version` to stderr when stdin is not a TTY.
    check_pi_version(str(FAKE_PI), "0.75.3")


def test_version_check_still_reports_a_mismatch_from_stderr(fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_VERSION", "0.80.0")
    with pytest.raises(PiVersionMismatch, match="0.80.0 is installed"):
        check_pi_version(str(FAKE_PI), "0.75.3")


# --- the model actually used (P3) -------------------------------------------------------


def test_matching_model_on_every_message_completes(tmp_path, fake_pi):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"


@pytest.mark.parametrize("scenario", ["model_switch", "provider_switch"])
def test_assistant_message_from_another_model_is_model_mismatch(tmp_path, fake_pi, monkeypatch, scenario):
    monkeypatch.setenv("FAKE_PI_SCENARIO", scenario)
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "model_mismatch"
    assert "openai/gpt-5-nano" in result.error


def test_model_mismatch_still_reports_usage(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "model_switch")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.usage.cost_usd == pytest.approx(0.03)


def test_mid_run_thinking_level_change_is_model_mismatch(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "thinking_switch")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.error_class == "model_mismatch"
    assert "thinking" in result.error


def test_model_change_recorded_only_in_the_session_file_is_model_mismatch(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "session_model_change")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.error_class == "model_mismatch"
    assert "gpt-5.5-pro" in result.error


def test_the_session_files_own_starting_entries_are_not_a_mismatch(tmp_path, fake_pi):
    # Every fresh Pi session records its starting model and thinking level once.
    result, _, trace_dir = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert list((trace_dir / "session").glob("*.jsonl"))
    assert result.status == "completed"


# --- end-of-run check (P8) --------------------------------------------------------------


def test_run_killed_by_a_signal_is_errored_agent_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "crash_after_tool_use")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert "SIGKILL" in result.error
    assert "heap out of memory" in result.error
    assert result.usage.input_tokens == 100


def test_run_without_agent_end_after_the_last_answer_is_errored(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "exit_without_agent_end")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert "agent_end" in result.error
    assert "TypeError" in result.error


def test_run_ending_on_a_tool_call_is_errored(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "tool_use_last")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert "toolUse" in result.error


def test_run_stopped_at_the_length_limit_is_completed(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "length")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 42"


# --- timeout kill -----------------------------------------------------------------------


def wait_until_dead(pid: int, seconds: float = 5) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_lets_pi_kill_its_detached_bash_commands(tmp_path, fake_pi, monkeypatch):
    # Pi runs bash tool commands in their own session, out of reach of a group kill;
    # only Pi's own SIGTERM handler kills them.
    pid_file = tmp_path / "detached.pid"
    monkeypatch.setenv("FAKE_PI_SCENARIO", "timeout")
    monkeypatch.setenv("FAKE_PI_DETACHED_PID", str(pid_file))
    run(build_agent(write_candidate(tmp_path / "c"), timeout_s=1), tmp_path)
    detached = int(pid_file.read_text())
    if not wait_until_dead(detached):
        os.killpg(detached, signal.SIGKILL)
        pytest.fail(f"detached bash command {detached} survived the timeout kill")


def test_timeout_falls_back_to_sigkill_when_pi_ignores_sigterm(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setattr(pi_adapter, "TERM_GRACE_S", 0.5)
    monkeypatch.setenv("FAKE_PI_SCENARIO", "timeout_ignores_sigterm")
    started = time.monotonic()
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c"), timeout_s=1), tmp_path)
    assert time.monotonic() - started < 10
    assert result.error_class == "timeout"
    assert wait_until_dead(fake_pi()["pid"], 1)


# --- isolation from the user's global Pi config -----------------------------------------


def test_pi_runs_with_a_fresh_agent_dir_outside_the_repo(tmp_path, fake_pi, user_agent_dir):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    agent_dir = Path(fake_pi()["agent_dir"])
    assert agent_dir.resolve() != user_agent_dir.resolve()
    assert not agent_dir.resolve().is_relative_to(REPO_ROOT)


def test_agent_dir_links_auth_and_bin_and_never_copies_them(tmp_path, fake_pi, user_agent_dir):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    entries = fake_pi()["agent_dir_entries"]
    assert entries["auth.json"]["symlink"] == str(user_agent_dir / "auth.json")
    assert entries["bin"]["symlink"] == str(user_agent_dir / "bin")


def test_agent_dir_pins_settings_and_drops_models_json(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    entries = fake_pi()["agent_dir_entries"]
    assert entries["settings.json"]["symlink"] is None
    assert json.loads(entries["settings.json"]["text"]) == pi_adapter.PINNED_SETTINGS
    assert "models.json" not in entries


def test_agent_dir_is_removed_after_the_run_and_the_real_one_is_untouched(tmp_path, fake_pi, user_agent_dir):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert not Path(fake_pi()["agent_dir"]).exists()
    assert (user_agent_dir / "auth.json").read_text().startswith('{"openai"')
    assert (user_agent_dir / "bin" / "rg").exists()


def test_agent_dir_skips_auth_and_bin_the_user_does_not_have(tmp_path, fake_pi, user_agent_dir):
    (user_agent_dir / "auth.json").unlink()
    (user_agent_dir / "bin" / "rg").unlink()
    (user_agent_dir / "bin").rmdir()
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert set(fake_pi()["agent_dir_entries"]) == {"settings.json"}


# --- the model gateway: every request goes to the endpoint, with the per-trial token --------

# Placeholders only: no real key is ever read or sent by these tests.
GATEWAY_TOKEN = "trial-token-placeholder-7f3a"
ENDPOINT = ModelEndpoint(base_url="http://127.0.0.1:9/v1", api_key=GATEWAY_TOKEN)


def run_via_gateway(agent: PiAgent, tmp_path: Path, endpoint: ModelEndpoint = ENDPOINT, task: Task = TASK):
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    trace_dir = tmp_path / "trace"
    return agent.run(task, workdir, trace_dir, endpoint=endpoint), workdir, trace_dir


def test_gateway_run_points_the_experiments_provider_at_the_endpoint_with_the_token(tmp_path, fake_pi):
    result, _, _ = run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"
    models_json = fake_pi()["agent_dir_entries"]["models.json"]
    assert models_json["symlink"] is None
    assert json.loads(models_json["text"]) == {
        "providers": {"openai": {"baseUrl": "http://127.0.0.1:9/v1", "apiKey": GATEWAY_TOKEN}}
    }


def test_gateway_run_keeps_the_model_id_so_the_model_check_and_gateway_policy_match(tmp_path, fake_pi):
    run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert flag_values(fake_pi()["argv"], "--model") == ["openai/gpt-5-nano"]


def test_gateway_run_never_links_the_users_credentials(tmp_path, fake_pi, user_agent_dir):
    run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    entries = fake_pi()["agent_dir_entries"]
    assert "auth.json" not in entries
    assert entries["bin"]["symlink"] == str(user_agent_dir / "bin")
    assert json.loads(entries["settings.json"]["text"]) == pi_adapter.PINNED_SETTINGS


def test_gateway_run_drops_provider_keys_from_pis_environment(tmp_path, fake_pi, monkeypatch):
    # Pi's auth storage prefers OPENAI_API_KEY from the environment over models.json's apiKey.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-placeholder")
    monkeypatch.setenv("HF_TOKEN", "hf-placeholder")
    run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    names = set(fake_pi()["env_names"])
    assert not names & {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"}
    assert {"PATH", "HOME", "PI_CODING_AGENT_DIR"} <= names


def test_run_without_an_endpoint_keeps_the_environment_and_the_credential_link(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-not-a-real-key")
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    record = fake_pi()
    assert "OPENAI_API_KEY" in record["env_names"]
    assert "auth.json" in record["agent_dir_entries"]
    assert "models.json" not in record["agent_dir_entries"]


def test_gateway_token_appears_in_no_trace_file(tmp_path, fake_pi):
    result, _, trace_dir = run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    files = [path for path in trace_dir.rglob("*") if path.is_file()]
    assert files  # events, stderr and the session file
    for path in files:
        assert GATEWAY_TOKEN not in path.read_text(errors="replace"), path
    assert GATEWAY_TOKEN not in repr(result)


def test_gateway_agent_dir_and_its_token_are_removed_after_the_run(tmp_path, fake_pi):
    run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert not Path(fake_pi()["agent_dir"]).exists()


def test_gateway_run_with_fixed_thinking_sends_that_effort(tmp_path, fake_pi):
    model = ModelSpec(provider="openai", id="gpt-5-nano", thinking="high")
    run_via_gateway(build_agent(write_candidate(tmp_path / "c"), model), tmp_path)
    assert flag_values(fake_pi()["argv"], "--thinking") == ["high"]


def test_gateway_run_refuses_a_token_pi_would_run_as_a_shell_command(tmp_path, fake_pi):
    # Pi resolves a models.json apiKey starting with "!" by running it (resolve-config-value.js).
    endpoint = ModelEndpoint(base_url="http://127.0.0.1:9/v1", api_key="!echo pwned")
    result, _, _ = run_via_gateway(build_agent(write_candidate(tmp_path / "c")), tmp_path, endpoint)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert "pwned" not in (result.error or "")
    assert not (tmp_path / "fake_pi_record.json").exists()


def require_real_pi() -> None:
    if shutil.which("pi") is None:
        pytest.skip("needs the real pi binary")
    try:
        check_pi_version("pi", "0.75.3")
    except (PiVersionMismatch, OSError) as error:
        pytest.skip(f"needs pi 0.75.3: {error}")


def test_real_pi_sends_every_request_to_the_gateway_with_the_token(tmp_path, user_agent_dir, monkeypatch):
    """The real binary, with no network and no paid call: Pi -> ModelGateway -> a fake upstream.
    It proves Pi honours the models.json override (base URL and token), sends the experiment's
    effort, and uses neither the user's auth.json nor a provider key in its environment: either
    would reach the gateway as an unknown token and be refused with 401."""
    require_real_pi()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-must-never-reach-pi")
    model = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
    agent = PiAdapter().build(
        write_candidate(tmp_path / "c", valves="tools = []\n"), model, {"pi_bin": "pi", "timeout_s": 120}
    )
    upstream_key = "sk-fake-upstream-placeholder"
    with FakeOpenAIUpstream() as upstream, ModelGateway(
        model=model, prices=Prices(input=0.05, output=0.40), upstream_url=upstream.url,
        upstream_key=upstream_key, max_calls_per_trial=5,
    ) as gateway:  # fmt: skip
        trial_key, endpoint = gateway.open_trial()
        result, _, trace_dir = run_via_gateway(agent, tmp_path, endpoint, Task(id="t", split="search", instruction="Say hi."))
        ledger = gateway.close_trial(trial_key)
        received = list(upstream.requests)

    assert result.status == "completed", result.error
    assert ledger.calls and ledger.violations == ()
    assert all(call.status == 200 and call.model_requested == "gpt-5-nano" for call in ledger.calls)
    assert len(received) == len(ledger.calls)
    for request in received:
        assert request.headers["authorization"] == f"Bearer {upstream_key}"
        body = json.loads(request.body)
        assert body["model"] == "gpt-5-nano"
        assert body["reasoning"]["effort"] == "low"
    for path in (path for path in trace_dir.rglob("*") if path.is_file()):
        assert endpoint.api_key not in path.read_text(errors="replace"), path


def test_real_agent_dir_defaults_to_home_pi_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pi_adapter.user_agent_dir() == tmp_path / ".pi" / "agent"


def test_real_agent_dir_expands_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/custom-agent")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pi_adapter.user_agent_dir() == tmp_path / "custom-agent"


# --- extension crashes ------------------------------------------------------------------


def test_stack_trace_naming_an_extension_before_the_run_started_is_harness_load_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_crash_at_start")
    candidate = write_candidate(tmp_path / "c", files={"extensions/crash.ts": "export default () => {}"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "harness_load_error"
    assert "boom from candidate extension" in result.error


def test_stack_trace_naming_an_extension_mid_run_is_extension_error(tmp_path, fake_pi, monkeypatch):
    # The file loaded and the run started; its code then threw outside any handler Pi catches.
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_crash_mid_run")
    candidate = write_candidate(tmp_path / "c", files={"extensions/crash.ts": "export default () => {}"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "extension_error"
    assert "boom from candidate extension" in result.error


def test_crash_that_names_no_candidate_extension_is_agent_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "unrelated_crash")
    candidate = write_candidate(tmp_path / "c", files={"extensions/fine.ts": "export default () => {}"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.error_class == "agent_error"


def test_valves_doc_says_a_throwing_tool_is_seen_by_the_model_not_marked():
    text = PiAdapter().describe_valves()
    assert "execute()" in text
    assert "stack trace" in text


# --- skill frontmatter ------------------------------------------------------------------


def test_validate_accepts_skill_with_description(tmp_path):
    assert PiAdapter().validate(write_candidate(tmp_path / "c", files={"skills/plan/SKILL.md": skill_md()})) == []


@pytest.mark.parametrize(
    "text",
    [
        "# plan\n\n1. Read the task.\n",  # no frontmatter at all
        "---\nname: plan\n---\n# plan\n",  # frontmatter without description
        "---\ndescription:   \n---\n# plan\n",  # blank description
        '---\ndescription: ""\n---\n# plan\n',  # empty quoted description
    ],
    ids=["no-frontmatter", "no-description", "blank", "empty-quoted"],
)
def test_validate_rejects_skill_without_description(tmp_path, text):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files={"skills/plan/SKILL.md": text}))
    assert any("skills/plan/SKILL.md" in p and "description" in p for p in problems)


@pytest.mark.parametrize(
    "text",
    [
        "---\ndescription: 'Use: when planning'\n---\n",
        "---\nname: plan\ndescription: >-\n  Use when the task\n  needs a plan.\n---\n",
        "---\ndescription: |\n  Line one.\n---\n",
        "---\r\ndescription: Windows line endings\r\n---\r\n",
    ],
    ids=["quoted", "folded-block", "literal-block", "crlf"],
)
def test_validate_accepts_description_forms_yaml_allows(tmp_path, text):
    assert PiAdapter().validate(write_candidate(tmp_path / "c", files={"skills/plan/SKILL.md": text})) == []


def test_validate_rejects_skills_with_the_same_frontmatter_name(tmp_path):
    files = {"skills/one/SKILL.md": skill_md(name="same"), "skills/two/SKILL.md": skill_md(name="same")}
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("'same'" in p and "skills/two" in p for p in problems)


def test_validate_rejects_frontmatter_name_that_matches_another_skill_directory(tmp_path):
    files = {"skills/plan/SKILL.md": skill_md(), "skills/other/SKILL.md": skill_md(name="plan")}
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", files=files))
    assert any("'plan'" in p for p in problems)


# --- extension spec (docs/FOUNDING_PRINCIPLES.md §5.2) ----------------------------------


def spec() -> ExtensionSpec:
    result = PiAdapter().extension_spec()
    assert result is not None
    return result


def lifecycle_point(point_id: str):
    return next(point for point in spec().lifecycle if point.id == point_id)


def test_extension_spec_declares_typescript_and_javascript_files():
    assert spec().language == "typescript"
    assert spec().suffixes == (".ts", ".js")


def test_extension_spec_suffixes_are_the_ones_validate_accepts():
    assert spec().suffixes == pi_adapter.EXTENSION_SUFFIXES


def test_extension_spec_entry_is_a_default_exported_factory():
    entry = spec().entry_convention
    assert "default export" in entry
    assert "(pi: ExtensionAPI) => void | Promise<void>" in entry


def test_extension_spec_pins_pi_0_75_3():
    assert spec().framework_version == "0.75.3"


def test_extension_spec_declares_every_lifecycle_point_exactly_once():
    assert sorted(point.id for point in spec().lifecycle) == sorted(LIFECYCLE_POINTS)


def test_extension_spec_gives_every_point_a_native_api_and_what_it_can_modify():
    for point in spec().lifecycle:
        assert point.support in ("supported", "partial", "unsupported")
        assert point.native_api and point.can_modify, point.id


@pytest.mark.parametrize(
    "point_id, native",
    [
        ("before_run", 'pi.on("before_agent_start")'),
        ("system_prompt", 'pi.on("before_agent_start")'),
        ("before_model_call", 'pi.on("context")'),
        ("after_model_call", 'pi.on("message_end")'),
        ("wrap_model_call", 'pi.on("before_provider_request")'),
        ("before_tool_call", 'pi.on("tool_call")'),
        ("after_tool_call", 'pi.on("tool_result")'),
        ("register_tool", "pi.registerTool"),
        ("compaction", 'pi.on("session_before_compact")'),
        ("after_run", 'pi.on("agent_end")'),
    ],
)
def test_extension_spec_maps_each_point_to_its_exact_pi_api(point_id, native):
    assert native in lifecycle_point(point_id).native_api


def test_extension_spec_marks_points_pi_only_partly_supports():
    for point_id in ("after_model_call", "after_run"):
        assert lifecycle_point(point_id).support == "partial"


def test_context_edits_are_documented_as_applying_to_one_call_only():
    assert "one call" in lifecycle_point("before_model_call").caveats


def test_tool_call_argument_rewrites_are_documented_as_in_place_mutation():
    point = lifecycle_point("before_tool_call")
    assert "event.input" in point.can_modify
    assert "in place" in point.can_modify


def test_tool_call_handler_throws_are_documented_as_unreported():
    # Pi turns a throwing tool_call handler into an error tool result, with nothing on stderr.
    assert "not reported" in lifecycle_point("before_tool_call").caveats


def test_registered_tools_are_documented_as_gated_by_the_tools_valve():
    assert "[valves].tools" in lifecycle_point("register_tool").caveats


def test_extension_spec_forbids_apis_that_change_the_model():
    forbidden = " ".join(spec().forbidden)
    for api in ("pi.setModel", "pi.setThinkingLevel", "pi.registerProvider", "before_provider_request"):
        assert api in forbidden


def test_extension_spec_api_reference_is_the_pinned_excerpt():
    excerpt = (REPO_ROOT / "hive" / "agents" / "pi" / "EXTENSIONS_API.md").read_text()
    assert spec().api_reference == excerpt


def test_api_reference_says_where_it_was_excerpted_from_and_stays_short():
    reference = spec().api_reference
    assert "0.75.3" in reference
    assert "dist/core/extensions/types.d.ts" in reference
    assert "interface ExtensionAPI" in reference
    assert len(reference.splitlines()) <= 420


def test_rendered_guide_declares_every_point():
    assert "not declared" not in render_extension_guide(spec())


# --- load report ------------------------------------------------------------------------

GUARD_TS = """import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
export default function (pi: ExtensionAPI) {
    pi.on("tool_call", async (event) => { if (event.toolName === "bash") return { block: true }; });
    pi.on("context", async (event) => ({ messages: event.messages }));
}
"""
CALC_TS = """import { Type } from "typebox";
export default (pi) => {
    pi.registerTool({
        name: "calc",
        label: "Calc",
        description: "Add numbers",
        parameters: Type.Object({ a: Type.Number() }),
        async execute(id, params) { return { content: [], details: {} }; },
    });
};
"""


def candidate_with(tmp_path: Path, files: dict[str, str], valves: str = "") -> Candidate:
    return write_candidate(tmp_path / "c", valves=valves, files=files)


def loads_by_path(result) -> dict:
    return {load.path: load for load in result.load_report.extensions}


def test_successful_run_reports_every_extension_loaded(tmp_path, fake_pi):
    candidate = candidate_with(tmp_path, {"extensions/a.ts": GUARD_TS, "extensions/b.js": "export default () => {}"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.status == "completed"
    assert [(load.path, load.loaded, load.error) for load in result.load_report.extensions] == [
        ("extensions/a.ts", True, None),
        ("extensions/b.js", True, None),
    ]
    assert result.load_report.runtime_errors == ()


def test_load_report_paths_are_the_ones_the_controller_expects(tmp_path, fake_pi):
    candidate = candidate_with(tmp_path, {"extensions/a.ts": GUARD_TS, "extensions/b.js": "export default () => {}"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    expected = expected_extension_files(candidate, spec())
    assert sorted(loads_by_path(result)) == expected
    assert load_problem(expected, result.load_report) is None


def test_load_report_lists_what_each_extension_registers(tmp_path, fake_pi):
    candidate = candidate_with(
        tmp_path, {"extensions/guard.ts": GUARD_TS, "extensions/calc.ts": CALC_TS}, valves='tools = ["read", "calc"]\n'
    )
    result, _, _ = run(build_agent(candidate), tmp_path)
    loads = loads_by_path(result)
    assert loads["extensions/guard.ts"].registered == ("before_model_call", "before_tool_call")
    assert loads["extensions/calc.ts"].registered == ("register_tool", "tool:calc")


def test_candidate_without_extensions_gets_an_empty_load_report(tmp_path, fake_pi):
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.load_report == LoadReport()


def test_startup_load_failure_marks_the_named_file_not_loaded_with_its_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_load_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": "x", "extensions/good.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    bad = loads_by_path(result)["extensions/bad.ts"]
    assert bad.loaded is False
    assert "boom in bad.ts" in bad.error
    assert bad.registered == ()


def test_startup_load_failure_marks_the_other_files_unknown_and_not_loaded(tmp_path, fake_pi, monkeypatch):
    # Pi exits at startup when any -e file fails, so the others never ran in a trial.
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_load_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": "x", "extensions/good.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    good = loads_by_path(result)["extensions/good.ts"]
    assert good.loaded is False
    assert "unknown" in good.error


def test_each_failing_file_keeps_its_own_multi_line_load_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_load_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": "x", "extensions/bad2.ts": "y"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    loads = loads_by_path(result)
    assert "boom in bad.ts" in loads["extensions/bad.ts"].error
    assert "bad.ts:3:1" in loads["extensions/bad.ts"].error  # the second line of its message
    assert "bad2.ts" not in loads["extensions/bad.ts"].error
    assert "boom in bad2.ts" in loads["extensions/bad2.ts"].error


def test_startup_load_failure_is_harness_load_error_naming_the_file(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_load_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": "x"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.error_class == "harness_load_error"
    assert "extensions/bad.ts" in result.error


def test_runtime_extension_error_is_attributed_to_the_extension(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_runtime_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": GUARD_TS, "extensions/good.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.load_report.runtime_errors == ("extensions/bad.ts: handler threw",)


def test_runtime_extension_error_leaves_every_file_loaded(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_runtime_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": GUARD_TS, "extensions/good.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert all(load.loaded for load in result.load_report.extensions)


def test_runtime_extension_error_is_extension_error_not_harness_load_error(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_runtime_error")
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "extension_error"
    assert "extensions/bad.ts: handler threw" in result.error


def test_crash_mid_run_is_a_runtime_error_of_the_named_file(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_crash_mid_run")
    candidate = candidate_with(tmp_path, {"extensions/crash.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert loads_by_path(result)["extensions/crash.ts"].loaded is True
    (runtime_error,) = result.load_report.runtime_errors
    assert runtime_error.startswith("extensions/crash.ts: ")
    assert "boom from candidate extension" in runtime_error


def test_crash_at_start_marks_the_named_file_not_loaded(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_crash_at_start")
    candidate = candidate_with(tmp_path, {"extensions/crash.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    crash = loads_by_path(result)["extensions/crash.ts"]
    assert crash.loaded is False
    assert "boom from candidate extension" in crash.error


def test_startup_failure_unrelated_to_extensions_is_agent_error_with_no_load_report(tmp_path, fake_pi, monkeypatch):
    # Pi stopped before loading anything, for a reason that names no extension: there is no
    # evidence about the extensions, so there is no report to hold against the candidate.
    monkeypatch.setenv("FAKE_PI_SCENARIO", "startup_failure")
    candidate = candidate_with(tmp_path, {"extensions/guard.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.error_class == "agent_error"
    assert "No models available" in result.error
    assert result.load_report is None


def test_launch_failure_has_no_load_report(tmp_path):
    candidate = candidate_with(tmp_path, {"extensions/guard.ts": GUARD_TS})
    agent = PiAdapter().build(candidate, MODEL, {"pi_bin": str(tmp_path / "no-such-pi")})
    result, _, _ = run(agent, tmp_path)
    assert result.error_class == "agent_error"
    assert result.load_report is None


def test_timeout_before_the_session_header_has_no_load_report(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "hang_before_header")
    candidate = candidate_with(tmp_path, {"extensions/guard.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate, timeout_s=1), tmp_path)
    assert result.error_class == "timeout"
    assert result.load_report is None


def test_timeout_after_the_run_started_reports_extensions_loaded(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "timeout")
    candidate = candidate_with(tmp_path, {"extensions/guard.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate, timeout_s=1), tmp_path)
    assert result.error_class == "timeout"
    assert [load.loaded for load in result.load_report.extensions] == [True]


def test_model_mismatch_still_carries_the_load_report(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "model_switch")
    candidate = candidate_with(tmp_path, {"extensions/guard.ts": GUARD_TS})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.error_class == "model_mismatch"
    assert [load.loaded for load in result.load_report.extensions] == [True]


class OneTaskSuite:
    name = "one-task"

    def describe(self) -> str:
        return ""

    def metric(self) -> MetricSpec:
        return MetricSpec("reward", "1 for any answer")

    def tasks(self, split):
        return [Task("t1", split, TASK.instruction)]

    def grade(self, task, result, workdir) -> Grade:
        return Grade(1.0)


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "extension_load_error",
        "extension_runtime_error",
        "extension_crash_at_start",
        "extension_crash_mid_run",
        "startup_failure",
        "hang_before_header",
        "timeout",
        "launch_failure",
    ],
)
def test_adapter_classification_agrees_with_the_controller(tmp_path, fake_pi, monkeypatch, scenario):
    # The controller re-checks every load report (hive.evaluate); it must never relabel a run
    # the adapter classified, least of all a Pi that never started as a candidate's load failure.
    monkeypatch.setenv("FAKE_PI_SCENARIO", scenario)
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": GUARD_TS, "extensions/good.ts": GUARD_TS})
    pi_bin = str(tmp_path / "no-such-pi") if scenario == "launch_failure" else str(FAKE_PI)
    settings = {"pi_bin": pi_bin, "timeout_s": 1}
    agent = PiAdapter().build(candidate, MODEL, settings)
    result, _, _ = run(agent, tmp_path)
    report = evaluate(candidate, PiAdapter(), OneTaskSuite(), "search", model=MODEL, agent_settings=settings,
                      trials_per_task=1, out_dir=tmp_path / "eval")  # fmt: skip
    trial = report.tasks[0].trials[0]
    assert trial.error_class == result.error_class


# --- static registration analysis -------------------------------------------------------


def test_registered_maps_pi_events_to_lifecycle_points():
    source = """export default (pi) => {
        pi.on("before_agent_start", () => {});
        pi.on("tool_result", () => {});
        pi.on("session_before_compact", () => {});
        pi.on('agent_end', () => {});
    };"""
    assert pi_adapter.registered_by(source) == (
        "after_run", "after_tool_call", "before_run", "compaction", "system_prompt",
    )  # fmt: skip


def test_registered_ignores_events_outside_the_catalog():
    assert pi_adapter.registered_by('export default (pi) => { pi.on("model_select", () => {}); }') == ()


def test_registered_works_whatever_the_api_parameter_is_named():
    source = 'export default function (api: ExtensionAPI) { api.on("tool_call", () => {}); }'
    assert pi_adapter.registered_by(source) == ("before_tool_call",)


def test_registered_ignores_commented_out_calls():
    source = """export default (pi) => {
        // pi.on("tool_call", () => {});
        /* pi.registerTool({ name: "calc" }); */
        pi.on("context", () => {}); // a URL like "http://x" in a string is not a comment
    };"""
    assert pi_adapter.registered_by(source) == ("before_model_call",)


def test_registered_reads_the_tool_name_even_after_nested_name_keys():
    source = """pi.registerTool({
        parameters: Type.Object({ name: Type.String() }),
        label: "Lookup",
        name: `lookup`,
    });"""
    assert pi_adapter.registered_by(source) == ("register_tool", "tool:lookup")


def test_registered_records_a_tool_whose_name_is_not_a_literal_without_naming_it():
    assert pi_adapter.registered_by("const tool = makeTool(); pi.registerTool(tool);") == ("register_tool",)


def test_registered_ignores_calls_that_are_only_text_inside_a_string():
    source = """export default (pi) => {
        const help = `Tools come from pi.registerTool({ name: "ghost" }) and pi.on("context", f).`;
        pi.on("before_agent_start", (event) => ({ systemPrompt: event.systemPrompt + help }));
    };"""
    assert pi_adapter.registered_by(source) == ("before_run", "system_prompt")


def test_registered_reads_an_event_name_after_a_type_argument():
    source = 'export default (pi) => { pi.on<"tool_result">("tool_result", () => {}); };'
    assert pi_adapter.registered_by(source) == ("after_tool_call",)


def test_registered_does_not_treat_an_interpolated_template_as_a_literal_name():
    assert pi_adapter.literal_tool_names("pi.registerTool({ name: `calc_${suffix}` });") == ()


# --- extension tools must be in the tools valve -----------------------------------------


def test_validate_rejects_extension_tool_missing_from_tools_valve(tmp_path):
    candidate = candidate_with(tmp_path, {"extensions/calc.ts": CALC_TS}, valves='tools = ["read"]\n')
    problems = PiAdapter().validate(candidate)
    assert any("extensions/calc.ts" in p and "'calc'" in p and "tools" in p for p in problems)


def test_validate_accepts_extension_tool_listed_in_tools_valve(tmp_path):
    candidate = candidate_with(tmp_path, {"extensions/calc.ts": CALC_TS}, valves='tools = ["read", "calc"]\n')
    assert PiAdapter().validate(candidate) == []


def test_validate_checks_extension_tools_against_the_default_tools(tmp_path):
    candidate = candidate_with(tmp_path, {"extensions/calc.ts": CALC_TS})
    assert any("'calc'" in p for p in PiAdapter().validate(candidate))


def test_validate_accepts_an_extension_overriding_a_listed_built_in_tool(tmp_path):
    override = 'export default (pi) => { pi.registerTool({ name: "read", label: "r", description: "d" }); };'
    assert PiAdapter().validate(candidate_with(tmp_path, {"extensions/read.ts": override})) == []


def test_validate_accepts_an_extension_whose_string_mentions_register_tool(tmp_path):
    tip = """export default (pi) => {
        pi.on("before_agent_start", (event) => ({
            systemPrompt: event.systemPrompt + 'Extensions add tools with pi.registerTool({ name: "ghost" }).',
        }));
    };"""
    assert PiAdapter().validate(candidate_with(tmp_path, {"extensions/tip.ts": tip})) == []


def test_validate_ignores_tool_names_it_cannot_read_statically(tmp_path):
    dynamic = "export default (pi) => { pi.registerTool(makeTool()); };"
    assert PiAdapter().validate(candidate_with(tmp_path, {"extensions/dyn.ts": dynamic})) == []


# --- before_provider_request is forbidden (P3) -----------------------------------------
#
# Pi fills each assistant message's provider and model from the configured model, never from
# the response, so a payload whose model was rewritten would pass the run's model check.

PAYLOAD_REWRITE_TS = """export default (pi) => {
    pi.on("before_provider_request", (event) => ({ ...event.payload, model: "gpt-5" }));
};"""


def test_validate_rejects_an_extension_that_hooks_before_provider_request(tmp_path):
    problems = PiAdapter().validate(candidate_with(tmp_path, {"extensions/swap.ts": PAYLOAD_REWRITE_TS}))
    assert any("extensions/swap.ts" in p and "before_provider_request" in p for p in problems)


def test_validate_rejects_before_provider_request_named_through_a_constant(tmp_path):
    source = 'const EVENT = "before_provider_request";\nexport default (pi) => { pi.on(EVENT, (e) => e.payload); };'
    problems = PiAdapter().validate(candidate_with(tmp_path, {"extensions/swap.ts": source}))
    assert any("before_provider_request" in p for p in problems)


def test_validate_does_not_reject_before_provider_request_mentioned_in_a_comment(tmp_path):
    source = '// never use pi.on("before_provider_request")\nexport default (pi) => {};'
    assert PiAdapter().validate(candidate_with(tmp_path, {"extensions/ok.ts": source})) == []


def test_wrap_model_call_is_unsupported_because_its_only_hook_is_forbidden():
    point = lifecycle_point("wrap_model_call")
    assert point.support == "unsupported"
    assert "reject" in point.can_modify


def test_forbidden_entries_render_as_code_in_the_guide():
    guide = render_extension_guide(spec())
    assert "\n- `pi.setModel`" in guide
    assert "\n- `pi.on(\"before_provider_request\")`" in guide


# --- compaction usage (P12) ---------------------------------------------------------------
#
# Pi's summary call emits no message_end and Pi records no usage for it; the session file's
# compaction entry is the only trace, so the adapter adds an upper-bound estimate.

# The fake's two assistant messages: 300 input tokens for $0.015, 50 output tokens for $0.015.
FAKE_INPUT_RATE = 0.015 / 300
FAKE_OUTPUT_RATE = 0.015 / 50


def test_a_compaction_summary_call_is_counted_from_the_session_entry(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "compaction")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"
    summary_output = pi_adapter.COMPACTION_SUMMARY_MAX_OUTPUT_TOKENS
    assert result.usage.input_tokens == 300 + 100000
    assert result.usage.output_tokens == 50 + summary_output
    assert result.usage.cost_usd == pytest.approx(0.03 + 100000 * FAKE_INPUT_RATE + summary_output * FAKE_OUTPUT_RATE)


def test_a_compaction_summary_an_extension_supplied_costs_nothing(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "compaction")
    monkeypatch.setenv("FAKE_PI_COMPACTION_FROM_HOOK", "1")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (300, 50)
    assert result.usage.cost_usd == pytest.approx(0.03)


def test_a_split_turn_compaction_counts_both_summary_calls():
    entry = {"type": "compaction", "summary": "history\n\n---\n\n**Turn Context (split turn):**\n\nprefix",
             "tokensBefore": 1000, "fromHook": False}  # fmt: skip
    usage = pi_adapter.compaction_usage([entry], input_rate=0.0, output_rate=0.0)
    assert usage.output_tokens == (
        pi_adapter.COMPACTION_SUMMARY_MAX_OUTPUT_TOKENS + pi_adapter.COMPACTION_TURN_PREFIX_MAX_OUTPUT_TOKENS
    )


def test_compaction_caveat_says_how_its_cost_is_counted():
    assert "estimate" in lifecycle_point("compaction").caveats


# --- processes Pi's commands and extensions leave behind --------------------------------


def test_daemons_left_by_a_run_are_killed_after_it(tmp_path, fake_pi, monkeypatch):
    daemon_pid, sleeper_pid = tmp_path / "daemon.pid", tmp_path / "sleeper.pid"
    monkeypatch.setenv("FAKE_PI_SCENARIO", "daemons_left_running")
    monkeypatch.setenv("FAKE_PI_DAEMON_PID", str(daemon_pid))
    monkeypatch.setenv("FAKE_PI_SLEEPER_PID", str(sleeper_pid))
    started = time.monotonic()
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "completed"
    assert time.monotonic() - started < 30  # the run did not wait for them
    for pid_file in (daemon_pid, sleeper_pid):
        pid = int(pid_file.read_text())
        if not wait_until_dead(pid):
            os.kill(pid, signal.SIGKILL)
            pytest.fail(f"{pid_file.name} {pid} outlived the trial")


# --- VALVES.md and extensions -----------------------------------------------------------


def test_valves_doc_points_to_the_controller_rendered_extension_guide():
    assert "extension guide" in PiAdapter().describe_valves().lower()


def test_valves_doc_states_the_registered_tool_rule():
    text = PiAdapter().describe_valves()
    assert "registerTool" in text
    assert "string literal" in text


def test_valves_doc_names_the_extension_error_class():
    assert "extension_error" in PiAdapter().describe_valves()


# Recorded from the real Pi 0.75.3 (an extension with a syntax error, run offline with no
# credentials); {path} stands for the extension's absolute path.
REAL_LOAD_FAILURE_STDERR = (
    'Error: Failed to load extension "{path}": Failed to load extension: ParseError: Unexpected token, '
    'expected ","  \n {path}:3:1\n'
)


def test_real_pi_load_failure_output_marks_the_file_not_loaded(tmp_path):
    candidate = candidate_with(tmp_path, {"extensions/bad.ts": "x", "extensions/good.ts": GUARD_TS})
    agent = build_agent(candidate)
    bad_path = candidate.root / "extensions" / "bad.ts"
    raw = pi_adapter.RawRun(
        stdout="",
        stderr=REAL_LOAD_FAILURE_STDERR.format(path=bad_path),
        session_entries=(),
        returncode=1,
        timed_out=False,
        wall_clock_s=0.5,
    )
    result = pi_adapter.interpret_run(raw, model=MODEL, extensions=agent.extensions, timeout_s=60)
    assert result.error_class == "harness_load_error"
    bad = loads_by_path(result)["extensions/bad.ts"]
    assert "ParseError: Unexpected token" in bad.error
    assert f"{bad_path}:3:1" in bad.error
