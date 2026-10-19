import json
import os
import time
from pathlib import Path

import pytest

from hive.agents.pi import PiAdapter
from hive.agents.pi.adapter import PiAgent, PiVersionMismatch
from hive.candidate import Candidate
from hive.contracts import Agent, AgentAdapter, ModelSpec, Task, Usage

FAKE_PI = Path(__file__).parent / "fakes" / "fake_pi.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = ModelSpec(provider="openai", id="gpt-5-nano")
TASK = Task(id="t1", split="search", instruction="What is 6 * 7? End with ANSWER: <x>.")


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
def fake_pi(tmp_path, monkeypatch):
    """Point runs at the fake pi; returns a function that reads what the fake recorded."""
    record = tmp_path / "fake_pi_record.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    monkeypatch.delenv("FAKE_PI_SCENARIO", raising=False)
    monkeypatch.delenv("FAKE_PI_VERSION", raising=False)
    return lambda: json.loads(record.read_text())


def build_agent(candidate: Candidate, **settings) -> PiAgent:
    return PiAdapter().build(candidate, MODEL, {"pi_bin": str(FAKE_PI), **settings})


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


def test_argv_passes_default_thinking_and_tools(tmp_path, fake_pi):
    run(build_agent(write_candidate(tmp_path / "c")), tmp_path)
    argv = fake_pi()["argv"]
    assert flag_values(argv, "--thinking") == ["low"]
    assert flag_values(argv, "--tools") == ["read,bash,edit,write"]


def test_argv_passes_thinking_and_tools_valves(tmp_path, fake_pi):
    candidate = write_candidate(tmp_path / "c", valves='thinking = "high"\ntools = ["read", "grep"]\n')
    run(build_agent(candidate), tmp_path)
    argv = fake_pi()["argv"]
    assert flag_values(argv, "--thinking") == ["high"]
    assert flag_values(argv, "--tools") == ["read,grep"]


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
        tmp_path / "c", files={"skills/alpha/SKILL.md": "# alpha", "skills/beta/SKILL.md": "# beta"}
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
    with pytest.raises(ValueError, match="thinking"):
        build_agent(write_candidate(tmp_path / "c", valves='thinking = "extreme"\n'))


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
        valves='thinking = "medium"\ntools = ["read", "bash"]\n',
        files={
            "README.md": "notes",
            "SYSTEM.md": "be terse",
            "APPEND_SYSTEM.md": "check work",
            "extensions/guard.ts": "export default () => {}",
            "skills/plan/SKILL.md": "# plan",
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


def test_validate_rejects_unknown_thinking_level(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves='thinking = "extreme"\n'))
    assert any("thinking" in p for p in problems)


def test_validate_rejects_non_string_thinking(tmp_path):
    problems = PiAdapter().validate(write_candidate(tmp_path / "c", valves="thinking = 3\n"))
    assert any("thinking" in p for p in problems)


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
    for name in ("thinking", "tools", "SYSTEM.md", "APPEND_SYSTEM.md", "extensions/", "skills/", "model"):
        assert name in text


def test_seed_harness_is_valid():
    seed = Candidate.load(REPO_ROOT / "harnesses" / "pi-baseline")
    assert seed.agent == "pi"
    assert PiAdapter().validate(seed) == []
