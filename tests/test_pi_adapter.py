import json
import os
import signal
import time
from pathlib import Path

import pytest

from hive.agents.pi import PiAdapter
import hive.agents.pi.adapter as pi_adapter
from hive.agents.pi.adapter import PiAgent, PiVersionMismatch, check_pi_version
from hive.candidate import Candidate
from hive.contracts import Agent, AgentAdapter, ModelSpec, Task, Usage

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
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "harness_load_error"
    assert "bad.ts" in result.error


def test_extension_runtime_error_is_harness_load_error_even_if_run_finished(tmp_path, fake_pi, monkeypatch):
    monkeypatch.setenv("FAKE_PI_SCENARIO", "extension_runtime_error")
    result, _, _ = run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "harness_load_error"


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


def test_real_agent_dir_defaults_to_home_pi_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pi_adapter.user_agent_dir() == tmp_path / ".pi" / "agent"


def test_real_agent_dir_expands_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/custom-agent")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pi_adapter.user_agent_dir() == tmp_path / "custom-agent"


# --- extension crashes ------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["extension_crash_at_start", "extension_crash_mid_run"])
def test_stack_trace_naming_a_candidate_extension_is_harness_load_error(tmp_path, fake_pi, monkeypatch, scenario):
    monkeypatch.setenv("FAKE_PI_SCENARIO", scenario)
    candidate = write_candidate(tmp_path / "c", files={"extensions/crash.ts": "export default () => {}"})
    result, _, _ = run(build_agent(candidate), tmp_path)
    assert result.status == "errored"
    assert result.error_class == "harness_load_error"
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
