import json
import shlex
import sys
from pathlib import Path

import pytest

from hive.candidate import Candidate
from hive.contracts import MetricSpec, Overseer, OverseerBrief, Proposal
from hive.overseers.claude_code import ClaudeCodeOverseer

FAKE_CLAUDE = Path(__file__).parent / "fakes" / "fake_claude.py"

MANIFEST_TEXT = 'schema_version = 1\nagent = "scripted"\ndescription = "seed"\n\n[valves]\nthinking = "low"\n'
SUITE_DESCRIPTION = "Short arithmetic questions. The agent must end with ANSWER: <x>."
METRIC = MetricSpec(name="exact_match", description="1 if the normalized answer matches, else 0.")
VALVES = "## Valves\n- APPEND_SYSTEM.md: text appended to the system prompt."
SELECTION_RULE = "Promote iff the child's selection score beats the incumbent's by more than 0.05."
HISTORY = [{"generation": 1, "candidate": "abc123", "score": 0.4, "decision": "rejected"}]
REPORT = {"candidate": "seed", "split": "search", "selection_score": 0.5}


@pytest.fixture
def fake_claude_bin(tmp_path) -> str:
    # exec keeps the wrapper's pid, so a timeout kill reaches the fake itself.
    wrapper = tmp_path / "bin" / "claude"
    wrapper.parent.mkdir()
    wrapper.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_CLAUDE))} \"$@\"\n")
    wrapper.chmod(0o755)
    return str(wrapper)


@pytest.fixture
def record_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(path))
    return path


@pytest.fixture
def brief(tmp_path) -> OverseerBrief:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "candidate.toml").write_text(MANIFEST_TEXT)
    (seed / "SYSTEM.md").write_text("You are a careful assistant.\n")

    report = tmp_path / "report.json"
    report.write_text(json.dumps(REPORT))

    evidence = tmp_path / "evidence"
    trace_file = evidence / "q1" / "0" / "trace" / "session.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text('{"event": "message_end"}\n')

    return OverseerBrief(
        incumbent=Candidate.load(seed),
        metric=METRIC,
        suite_description=SUITE_DESCRIPTION,
        valves=VALVES,
        selection_rule=SELECTION_RULE,
        generation=3,
        incumbent_report=report,
        evidence_dir=evidence,
        history=HISTORY,
    )


def make_overseer(fake_claude_bin: str, **settings) -> ClaudeCodeOverseer:
    return ClaudeCodeOverseer({"claude_bin": fake_claude_bin, **settings})


def run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="edit", **settings) -> tuple[Proposal, Path]:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    workspace = tmp_path / "workspace"
    proposal = make_overseer(fake_claude_bin, **settings).propose(brief, workspace)
    return proposal, workspace


def recorded_argv(record_path: Path) -> list[str]:
    return json.loads(record_path.read_text())["argv"]


def flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --- identity -------------------------------------------------------------------------------


def test_overseer_satisfies_the_overseer_protocol():
    overseer = ClaudeCodeOverseer()
    assert isinstance(overseer, Overseer)
    assert overseer.name == "claude_code"
    assert overseer.proposes_changes is True


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="max_usd"):
        ClaudeCodeOverseer({"max_usd": 2.0})


def test_non_positive_budget_is_rejected():
    with pytest.raises(ValueError, match="max_usd_per_proposal"):
        ClaudeCodeOverseer({"max_usd_per_proposal": 0})


# --- workspace layout -----------------------------------------------------------------------


def test_workspace_candidate_is_a_loadable_copy_of_the_incumbent(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    assert Candidate.load(workspace / "candidate").id == brief.incumbent.id


def test_context_holds_the_suite_description(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    assert SUITE_DESCRIPTION in (workspace / "context" / "suite.md").read_text()


def test_context_holds_the_metric_name_and_description(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    metric_text = (workspace / "context" / "metric.md").read_text()
    assert METRIC.name in metric_text
    assert METRIC.description in metric_text


def test_context_holds_the_valve_catalog(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    assert VALVES in (workspace / "context" / "valves.md").read_text()


def test_context_holds_the_selection_rule(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    assert SELECTION_RULE in (workspace / "context" / "selection_rule.md").read_text()


def test_context_holds_the_history_as_json(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    assert json.loads((workspace / "context" / "history.json").read_text()) == HISTORY


def test_context_holds_a_copy_of_the_incumbent_report(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    assert json.loads((workspace / "context" / "report.json").read_text()) == REPORT


def test_context_holds_a_copy_of_the_evidence(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="noop")
    copied = workspace / "context" / "evidence" / "q1" / "0" / "trace" / "session.jsonl"
    assert copied.read_text() == '{"event": "message_end"}\n'


def test_missing_evidence_dir_is_skipped_without_error(monkeypatch, fake_claude_bin, brief, tmp_path):
    brief_without_evidence = OverseerBrief(**{**brief.__dict__, "evidence_dir": tmp_path / "does-not-exist"})
    proposal, workspace = run(monkeypatch, fake_claude_bin, brief_without_evidence, tmp_path)
    assert proposal.error is None
    assert not (workspace / "context" / "evidence").exists()


# --- prompt ---------------------------------------------------------------------------------


def test_prompt_contains_the_candidate_toml_text(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert MANIFEST_TEXT in flag_value(recorded_argv(record_path), "-p")


def test_prompt_contains_the_metric(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    prompt = flag_value(recorded_argv(record_path), "-p")
    assert METRIC.name in prompt
    assert METRIC.description in prompt


def test_prompt_contains_the_suite_description(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert SUITE_DESCRIPTION in flag_value(recorded_argv(record_path), "-p")


def test_prompt_contains_the_selection_rule_and_generation(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    prompt = flag_value(recorded_argv(record_path), "-p")
    assert SELECTION_RULE in prompt
    assert "generation 3" in prompt.lower()


def test_prompt_points_at_every_context_file(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    prompt = flag_value(recorded_argv(record_path), "-p")
    for name in ("valves.md", "report.json", "history.json", "evidence/", "RATIONALE.md"):
        assert name in prompt


# --- invocation -----------------------------------------------------------------------------


def test_claude_runs_with_the_workspace_as_cwd(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert Path(json.loads(record_path.read_text())["cwd"]).resolve() == workspace.resolve()


def test_argv_has_every_required_flag(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    argv = recorded_argv(record_path)
    assert flag_value(argv, "--output-format") == "json"
    assert flag_value(argv, "--model") == "sonnet"
    assert flag_value(argv, "--permission-mode") == "acceptEdits"
    assert flag_value(argv, "--tools") == "Read,Grep,Glob,Edit,Write"
    assert flag_value(argv, "--allowedTools") == "Read,Grep,Glob,Edit,Write"
    assert flag_value(argv, "--setting-sources") == "project,local"
    assert "--no-session-persistence" in argv
    assert float(flag_value(argv, "--max-budget-usd")) == 1.0


def test_tools_never_include_bash(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    argv = recorded_argv(record_path)
    assert "Bash" not in flag_value(argv, "--tools").split(",")
    assert "Bash" not in flag_value(argv, "--allowedTools").split(",")


def test_settings_override_model_and_budget(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path, model="opus", max_usd_per_proposal=2.5)
    argv = recorded_argv(record_path)
    assert flag_value(argv, "--model") == "opus"
    assert float(flag_value(argv, "--max-budget-usd")) == 2.5


def test_stdout_and_stderr_are_saved_in_the_workspace(monkeypatch, fake_claude_bin, brief, tmp_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert json.loads((workspace / "overseer_output.json").read_text())["type"] == "result"
    assert "fake claude stderr line" in (workspace / "overseer_stderr.txt").read_text()


# --- successful proposals -------------------------------------------------------------------


def test_successful_run_returns_the_edited_candidate_dir(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert proposal.error is None
    assert proposal.candidate_dir == workspace / "candidate"
    assert (proposal.candidate_dir / "APPEND_SYSTEM.md").is_file()
    assert Candidate.load(proposal.candidate_dir).id != brief.incumbent.id


def test_cost_is_parsed_from_total_cost_usd(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert proposal.cost_usd == pytest.approx(0.42)


def test_cost_defaults_to_zero_when_absent(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="no_cost")
    assert proposal.error is None
    assert proposal.cost_usd == 0.0


def test_rationale_is_read_from_rationale_md(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert "## Observed" in proposal.rationale
    assert "## Why it should transfer" in proposal.rationale


def test_rationale_falls_back_to_result_text_when_rationale_md_is_missing(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="no_rationale")
    assert proposal.error is None
    assert proposal.rationale == "I appended a reminder to the system prompt."


# --- failures never raise -------------------------------------------------------------------


def test_non_json_output_yields_no_candidate(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="not_json")
    assert proposal.candidate_dir is None
    assert proposal.cost_usd == 0.0
    assert "JSON" in proposal.error


def test_is_error_result_yields_no_candidate_but_keeps_cost(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="is_error")
    assert proposal.candidate_dir is None
    assert proposal.cost_usd == pytest.approx(0.42)
    assert "error_max_budget_usd" in proposal.error


def test_nonzero_exit_yields_no_candidate_but_keeps_cost(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="exit_nonzero")
    assert proposal.candidate_dir is None
    assert proposal.cost_usd == pytest.approx(0.1)
    assert "exit" in proposal.error and "3" in proposal.error


def test_timeout_yields_no_candidate(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="sleep", timeout_s=0.5)
    assert proposal.candidate_dir is None
    assert "timed out" in proposal.error


def test_missing_binary_yields_no_candidate(brief, tmp_path):
    overseer = ClaudeCodeOverseer({"claude_bin": str(tmp_path / "no-such-claude")})
    proposal = overseer.propose(brief, tmp_path / "workspace")
    assert proposal.candidate_dir is None
    assert "no-such-claude" in proposal.error


def test_broken_candidate_yields_no_candidate(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="break_candidate")
    assert proposal.candidate_dir is None
    assert proposal.cost_usd == pytest.approx(0.42)
    assert "candidate.toml" in proposal.error


def test_changed_agent_field_yields_no_candidate(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="change_agent")
    assert proposal.candidate_dir is None
    assert "agent" in proposal.error


def test_existing_workspace_candidate_yields_error_instead_of_raising(monkeypatch, fake_claude_bin, brief, tmp_path):
    (tmp_path / "workspace" / "candidate").mkdir(parents=True)
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert proposal.candidate_dir is None
    assert "workspace" in proposal.error
