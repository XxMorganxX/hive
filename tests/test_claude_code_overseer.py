import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from hive.candidate import Candidate
from hive.contracts import MetricSpec, Overseer, OverseerBrief, Proposal
from hive.overseers.claude_code import ClaudeCodeOverseer
from hive.overseers.claude_code.overseer import build_workspace

FAKE_CLAUDE = Path(__file__).parent / "fakes" / "fake_claude.py"

MANIFEST_TEXT = 'schema_version = 1\nagent = "scripted"\ndescription = "seed"\n\n[valves]\nthinking = "low"\n'
SUITE_DESCRIPTION = "Short arithmetic questions. The agent must end with ANSWER: <x>."
METRIC = MetricSpec(name="exact_match", description="1 if the normalized answer matches, else 0.")
VALVES = "## Valves\n- APPEND_SYSTEM.md: text appended to the system prompt."
SELECTION_RULE = "Promote iff the child's selection score beats the incumbent's by more than 0.05."
HISTORY = [{"generation": 1, "candidate": "abc123", "score": 0.4, "decision": "rejected"}]
REPORT = {"candidate": "seed", "split": "search", "selection_score": 0.5}
EXTENSION_GUIDE = "# Extensions\n\nExtensions are python files in `extensions/`.\n| `before_tool_call` | supported |\n"


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


def with_extension_guide(brief: OverseerBrief, guide: str = EXTENSION_GUIDE) -> OverseerBrief:
    return OverseerBrief(**{**brief.__dict__, "extension_guide": guide})


def test_context_holds_the_extension_guide_when_the_framework_has_one(brief, tmp_path):
    context = built_context(with_extension_guide(brief), tmp_path)
    assert (context / "extensions.md").read_text() == EXTENSION_GUIDE


def test_context_has_no_extension_guide_when_the_framework_has_none(brief, tmp_path):
    context = built_context(brief, tmp_path)
    assert not (context / "extensions.md").exists()


def test_missing_evidence_dir_is_skipped_without_error(monkeypatch, fake_claude_bin, brief, tmp_path):
    brief_without_evidence = OverseerBrief(**{**brief.__dict__, "evidence_dir": tmp_path / "does-not-exist"})
    proposal, workspace = run(monkeypatch, fake_claude_bin, brief_without_evidence, tmp_path)
    assert proposal.error is None
    assert not (workspace / "context" / "evidence").exists()


# --- no path in the workspace leads outside it -------------------------------------------------


@pytest.fixture
def run_tree_brief(tmp_path) -> tuple[OverseerBrief, Path]:
    """A brief shaped like the loop's: evidence copied out of a run dir whose path has a space
    and a non-ASCII letter, with outcome.json, report.json and a trace naming that run dir."""
    run_dir = tmp_path / "CS 4701 café" / "experience" / "exp" / "r1"
    trials_root = run_dir / "evals" / "gen-000" / "trials"
    snapshot = run_dir / "candidates" / "abc123"
    bundle = tmp_path / "bundle"

    seed = bundle / "incumbent"
    seed.mkdir(parents=True)
    (seed / "candidate.toml").write_text(MANIFEST_TEXT)

    # The evaluator records trial_dir relative to the evaluation's directory (trials/<task>/<n>);
    # the loop says where the trials were recorded through brief.evidence_origin.
    trial = {"task_id": "q1", "trial": 1, "status": "scored", "trial_dir": "trials/q1/1"}
    # json.dumps escapes the non-ASCII letter, so the path is spelled differently here than in the trace.
    (bundle / "report.json").write_text(json.dumps({"tasks": [{"task_id": "q1", "trials": [trial]}]}, indent=2))

    trial_copy = bundle / "trials" / "q1" / "1"
    (trial_copy / "workdir").mkdir(parents=True)
    (trial_copy / "outcome.json").write_text(json.dumps(trial, indent=2))
    (trial_copy / "trace").mkdir()
    (trial_copy / "trace" / "session.jsonl").write_text(
        json.dumps({"type": "session", "cwd": str(trials_root / "q1" / "1" / "workdir")}, ensure_ascii=False)
        + "\n"
        + f'Failed to load extension "{snapshot / "extensions" / "broken.ts"}": boom\n'
    )
    (trial_copy / "workdir" / "outside-link").symlink_to(run_dir / "secret.txt")
    (trial_copy / "workdir" / "inside-link").symlink_to("../outcome.json")

    history = [{"generation": 1, "error": f"could not read {run_dir / 'heldout' / 'report.json'}"}]
    brief = OverseerBrief(
        incumbent=Candidate.load(seed),
        metric=METRIC,
        suite_description=SUITE_DESCRIPTION,
        valves=VALVES,
        selection_rule=SELECTION_RULE,
        generation=2,
        incumbent_report=bundle / "report.json",
        evidence_dir=bundle / "trials",
        history=history,
        evidence_origin=trials_root,
    )
    return brief, run_dir


def built_context(brief: OverseerBrief, tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    build_workspace(brief, workspace)
    return workspace / "context"


def test_report_trial_dirs_are_relative_to_context_evidence(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    report = json.loads((context / "report.json").read_text())
    trial_dir = report["tasks"][0]["trials"][0]["trial_dir"]
    assert trial_dir == "q1/1"
    assert (context / "evidence" / trial_dir / "outcome.json").is_file()


def test_evidence_outcome_trial_dir_is_relative_to_context_evidence(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    assert json.loads((context / "evidence" / "q1" / "1" / "outcome.json").read_text())["trial_dir"] == "q1/1"


def test_trace_paths_into_the_trial_become_relative(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    first_line = (context / "evidence" / "q1" / "1" / "trace" / "session.jsonl").read_text().splitlines()[0]
    assert json.loads(first_line)["cwd"] == "q1/1/workdir"


def test_paths_elsewhere_in_the_run_dir_are_redacted_but_keep_their_file_name(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    trace = (context / "evidence" / "q1" / "1" / "trace" / "session.jsonl").read_text()
    assert "<outside the workspace>/broken.ts" in trace


def test_no_file_in_context_names_a_location_outside_the_workspace(run_tree_brief, tmp_path):
    brief, run_dir = run_tree_brief
    context = built_context(brief, tmp_path)
    forbidden = {str(tmp_path), str(tmp_path.resolve()), json.dumps(str(run_dir))[1:-1], "café"}
    for path in context.rglob("*"):
        if path.is_file() and not path.is_symlink():
            text = path.read_text()
            for spelling in forbidden:
                assert spelling not in text, f"{spelling!r} in {path}"


def test_symlink_leading_outside_is_replaced_by_a_note(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    replaced = context / "evidence" / "q1" / "1" / "workdir" / "outside-link"
    assert not replaced.is_symlink()
    assert "outside the workspace" in replaced.read_text()


def test_symlink_staying_inside_the_evidence_is_kept(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    kept = context / "evidence" / "q1" / "1" / "workdir" / "inside-link"
    assert kept.is_symlink()
    assert os.readlink(kept) == "../outcome.json"


def test_read_only_evidence_is_still_rewritten(run_tree_brief, tmp_path):
    # An agent can leave read-only files and directories in its workdir; the copy keeps the modes.
    brief, run_dir = run_tree_brief
    workdir = Path(brief.evidence_dir) / "q1" / "1" / "workdir"
    note = workdir / "notes.txt"
    note.write_text(f"saw {run_dir}/secret.txt\n")
    note.chmod(0o444)
    workdir.chmod(0o555)
    try:
        context = built_context(brief, tmp_path)
    finally:
        workdir.chmod(0o755)
    copied = context / "evidence" / "q1" / "1" / "workdir"
    assert (copied / "notes.txt").read_text() == "saw <outside the workspace>/secret.txt\n"
    assert not (copied / "outside-link").is_symlink()


def test_history_paths_are_redacted(run_tree_brief, tmp_path):
    brief, _ = run_tree_brief
    context = built_context(brief, tmp_path)
    assert json.loads((context / "history.json").read_text())[0]["error"] == (
        "could not read <outside the workspace>/report.json"
    )


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


def test_prompt_says_to_read_and_write_only_inside_the_workspace(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert "only read and write inside this directory" in flag_value(recorded_argv(record_path), "-p").lower()


def test_prompt_points_at_every_context_file(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    prompt = flag_value(recorded_argv(record_path), "-p")
    for name in ("valves.md", "report.json", "history.json", "evidence/", "RATIONALE.md"):
        assert name in prompt


def test_prompt_points_at_the_extension_guide_when_there_is_one(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, with_extension_guide(brief), tmp_path)
    assert "context/extensions.md" in flag_value(recorded_argv(record_path), "-p")


def test_prompt_does_not_mention_extensions_when_the_framework_has_none(
    monkeypatch, fake_claude_bin, brief, tmp_path, record_path
):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert "extensions.md" not in flag_value(recorded_argv(record_path), "-p")


def test_prompt_invites_an_extension_for_a_behavioural_fix(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, with_extension_guide(brief), tmp_path)
    prompt = flag_value(recorded_argv(record_path), "-p").lower()
    assert "candidate/extensions/" in prompt
    for kind in ("gate", "rewrite", "retry", "verify"):
        assert kind in prompt


def test_prompt_keeps_one_isolated_change_per_candidate_when_extensions_are_offered(
    monkeypatch, fake_claude_bin, brief, tmp_path, record_path
):
    run(monkeypatch, fake_claude_bin, with_extension_guide(brief), tmp_path)
    assert "ONE isolated change" in flag_value(recorded_argv(record_path), "-p")


# --- invocation -----------------------------------------------------------------------------


def test_claude_runs_with_the_workspace_as_cwd(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    _, workspace = run(monkeypatch, fake_claude_bin, brief, tmp_path)
    assert Path(json.loads(record_path.read_text())["cwd"]).resolve() == workspace.resolve()


def test_argv_has_every_required_flag(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    argv = recorded_argv(record_path)
    assert flag_value(argv, "--output-format") == "json"
    assert flag_value(argv, "--model") == "sonnet"
    assert flag_value(argv, "--tools") == "Read,Grep,Glob,Edit,Write"
    assert "--no-session-persistence" in argv
    assert float(flag_value(argv, "--max-budget-usd")) == 1.0


def test_the_argv_that_runs_is_build_command(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    argv = recorded_argv(record_path)
    overseer = make_overseer(fake_claude_bin)
    assert [fake_claude_bin, *argv] == overseer.build_command(flag_value(argv, "-p"))


def test_tools_never_include_bash(monkeypatch, fake_claude_bin, brief, tmp_path, record_path):
    run(monkeypatch, fake_claude_bin, brief, tmp_path)
    argv = recorded_argv(record_path)
    assert "Bash" not in flag_value(argv, "--tools").split(",")
    assert not any(rule.startswith("Bash") for rule in allow_rules(argv))


# --- the read and write fence ---------------------------------------------------------------


def allow_rules(argv: list[str]) -> list[str]:
    return [rule for rule in flag_value(argv, "--allowedTools").split(",") if rule]


def test_no_allow_rule_is_a_bare_tool_name():
    # A bare rule such as "Edit" or "Read" allows that tool on every path on the host.
    argv = ClaudeCodeOverseer().build_command("prompt")
    for rule in allow_rules(argv):
        assert "(" in rule and rule.endswith(")"), rule


def test_edits_are_allowed_only_under_candidate_and_on_rationale_md():
    argv = ClaudeCodeOverseer().build_command("prompt")
    assert sorted(allow_rules(argv)) == ["Edit(/RATIONALE.md)", "Edit(/candidate/**)"]


def test_file_tools_are_confined_to_the_workspace_by_restricted_mode():
    assert "--restricted" in ClaudeCodeOverseer().build_command("prompt")


def test_anything_not_pre_allowed_is_denied_without_a_prompt():
    argv = ClaudeCodeOverseer().build_command("prompt")
    # acceptEdits would auto-approve writes anywhere in the workspace, context/ included.
    assert flag_value(argv, "--permission-mode") == "dontAsk"
    assert flag_value(argv, "--permission-prompts") == "none"


def test_user_mcp_servers_and_customizations_are_not_loaded():
    argv = ClaudeCodeOverseer().build_command("prompt")
    assert "--strict-mcp-config" in argv
    assert "--safe-mode" in argv


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


def test_missing_cost_is_charged_as_the_per_proposal_limit(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="no_cost", max_usd_per_proposal=2.5)
    assert proposal.error is None
    assert proposal.cost_usd == 2.5


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
    assert "JSON" in proposal.error


def test_non_json_output_charges_the_per_proposal_limit_as_an_upper_bound(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(monkeypatch, fake_claude_bin, brief, tmp_path, mode="not_json", max_usd_per_proposal=2.5)
    assert proposal.cost_usd == 2.5
    assert "upper bound" in proposal.error


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


def test_timeout_charges_the_per_proposal_limit_as_an_upper_bound(monkeypatch, fake_claude_bin, brief, tmp_path):
    proposal, _ = run(
        monkeypatch, fake_claude_bin, brief, tmp_path, mode="sleep", timeout_s=0.5, max_usd_per_proposal=2.5
    )
    assert proposal.cost_usd == 2.5
    assert "upper bound" in proposal.error


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
