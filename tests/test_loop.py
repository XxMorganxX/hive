import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import loop_fixtures
from hive.candidate import Candidate
from hive.config import ExperimentConfig
from hive.contracts import ModelSpec
from hive.loop import ExperimentSummary, LoopError, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_SEED = REPO_ROOT / "harnesses" / "scripted-exact-answer"
EXACT_HELDOUT_IDS = ("fill-rate", "count-letter-a", "unit-price", "date-offset")
SEED_SCORE = 0.5  # the scripted exact_answer seed answers 3 of 6 search tasks correctly


@pytest.fixture(autouse=True)
def reset_fixture_state():
    loop_fixtures.RecordingOverseer.reset()
    loop_fixtures.CrashingOverseer.workspaces = []
    yield


def make_config(tmp_path: Path, **overrides: Any) -> ExperimentConfig:
    seed = tmp_path / "seed"
    if not seed.exists():
        shutil.copytree(EXACT_SEED, seed)
    values: dict[str, Any] = dict(
        name="loop-test",
        config_path=tmp_path / "experiment.toml",
        seed_candidate=seed,
        output_dir=tmp_path / "out",
        model=ModelSpec(provider="scripted", id="none"),
        agent_settings={},
        suite_name="exact_answer",
        suite_settings={},
        trials_per_task=1,
        concurrency=2,
        generations=1,
        promotion_margin=0.05,
        max_usd=1.0,
        overseer_kind="null",
        overseer_settings={},
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def fixture(name: str) -> str:
    return f"loop_fixtures:{name}"


def rows(summary: ExperimentSummary) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (summary.run_dir / "generations.jsonl").read_text().splitlines()]


def run(tmp_path: Path, **overrides: Any) -> ExperimentSummary:
    return run_experiment(make_config(tmp_path, **overrides), run_id="r1")


# --- promotion -------------------------------------------------------------------------------


def test_child_that_beats_the_incumbent_by_more_than_the_margin_is_promoted(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"), promotion_margin=0.05)
    [row] = summary.generations
    assert row["status"] == "evaluated"
    assert row["promoted"] is True
    assert row["child_selection_score"] == pytest.approx(4 / 6)
    assert row["incumbent_score_before"] == pytest.approx(SEED_SCORE)
    assert summary.final_id == row["child_id"] != summary.seed_id
    assert summary.final_search_score == pytest.approx(4 / 6)


def test_promoted_child_becomes_the_incumbent_for_the_next_generation(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"), generations=2)
    first, second = summary.generations
    assert second["incumbent_before"] == first["child_id"]
    assert second["incumbent_score_before"] == pytest.approx(4 / 6)
    assert second["promoted"] is True
    assert summary.final_id == second["child_id"]
    assert summary.final_search_score == pytest.approx(5 / 6)


def test_child_within_the_margin_is_not_promoted(tmp_path):
    # One more correct task is +1/6 = 0.1667, which does not clear a 0.2 margin.
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"), promotion_margin=0.2)
    [row] = summary.generations
    assert row["status"] == "evaluated"
    assert row["promoted"] is False
    assert "margin" in row["reason"]
    assert summary.final_id == summary.seed_id


def test_worse_child_is_rejected(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("WorseningOverseer"))
    [row] = summary.generations
    assert row["status"] == "evaluated"
    assert row["promoted"] is False
    assert row["child_selection_score"] == pytest.approx(2 / 6)
    assert summary.final_id == summary.seed_id


def test_decisions_compare_against_the_seed_report_not_the_incumbent(tmp_path):
    # The seed floor can't bind in this loop (an incumbent is never below the seed), so
    # check the wiring instead: after a promotion, the floor still quotes the seed's score.
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"), generations=2)
    second = summary.generations[1]
    assert f"seed {SEED_SCORE:.4f}" in second["reason"]
    assert f"incumbent {4 / 6:.4f}" in second["reason"]


# --- proposal statuses -----------------------------------------------------------------------


def test_missing_proposal_is_recorded_without_any_evaluation(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("NoProposalOverseer"))
    [row] = summary.generations
    assert row["status"] == "no_proposal"
    assert row["promoted"] is False
    assert row["error"] == "could not think of anything"
    assert row["child_id"] is None
    assert row["overseer_cost_usd"] == pytest.approx(0.25)
    assert row["eval_cost_usd"] == 0.0
    assert not (summary.run_dir / "evals" / "gen-001").exists()


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("no_manifest", "candidate.toml"),
        ("other_agent", "agent"),
        ("bad_responses", "responses.toml"),
        ("outside_workspace", "outside the overseer workspace"),
        ("symlink", "symlink"),
    ],
)
def test_invalid_proposal_is_recorded_with_reasons_and_never_evaluated(tmp_path, mode, expected):
    summary = run(tmp_path, overseer_kind=fixture("InvalidOverseer"), overseer_settings={"mode": mode})
    [row] = summary.generations
    assert row["status"] == "invalid"
    assert row["promoted"] is False
    assert expected in row["reason"]
    assert not (summary.run_dir / "evals" / "gen-001").exists()
    assert summary.final_id == summary.seed_id


def test_unchanged_copy_from_an_overseer_that_claims_changes_is_not_evaluated(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("UnchangedClaimingOverseer"))
    [row] = summary.generations
    assert row["status"] == "unchanged"
    assert row["child_id"] == summary.seed_id
    assert not (summary.run_dir / "evals" / "gen-001").exists()


def test_repeat_of_an_already_evaluated_child_is_a_duplicate_with_the_earlier_score(tmp_path):
    # The worsening overseer is deterministic and the incumbent never changes, so
    # generation 2 proposes exactly the generation-1 child again.
    summary = run(tmp_path, overseer_kind=fixture("WorseningOverseer"), generations=2)
    first, second = summary.generations
    assert first["status"] == "evaluated"
    assert second["status"] == "duplicate"
    assert second["child_id"] == first["child_id"]
    assert second["child_selection_score"] == pytest.approx(first["child_selection_score"])
    assert second["promoted"] is False
    assert second["eval_cost_usd"] == 0.0
    assert not (summary.run_dir / "evals" / "gen-002").exists()


def test_null_overseer_re_evaluates_the_same_candidate_every_generation(tmp_path):
    summary = run(tmp_path, overseer_kind="null", generations=3)
    assert [row["status"] for row in summary.generations] == ["evaluated"] * 3
    assert {row["child_id"] for row in summary.generations} == {summary.seed_id}
    for generation in (1, 2, 3):
        report = json.loads((summary.run_dir / "evals" / f"gen-{generation:03d}" / "report.json").read_text())
        assert report["candidate_id"] == summary.seed_id
    # Identical scores never clear a positive margin.
    assert not any(row["promoted"] for row in summary.generations)


# --- budget ----------------------------------------------------------------------------------


def test_spend_at_or_above_the_cap_stops_the_loop(tmp_path):
    summary = run(
        tmp_path,
        overseer_kind=fixture("ExpensiveOverseer"),
        overseer_settings={"cost": 5.0},
        generations=3,
        max_usd=1.0,
    )
    assert summary.stopped_reason == "budget"
    assert len(summary.generations) == 1
    assert summary.total_cost_usd == pytest.approx(5.0)
    assert summary.generations[0]["spent_usd"] == pytest.approx(5.0)


def test_held_out_is_skipped_with_a_reason_when_the_budget_is_gone(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ExpensiveOverseer"), generations=1, max_usd=1.0)
    assert summary.heldout["seed"] is None
    assert summary.heldout["final"] is None
    assert "budget" in summary.heldout["skipped_reason"]
    assert not (summary.run_dir / "heldout").exists()


def test_completed_run_reports_completed(tmp_path):
    summary = run(tmp_path, generations=2)
    assert summary.stopped_reason == "completed"
    assert len(summary.generations) == 2


def test_zero_generations_evaluates_only_the_seed(tmp_path):
    summary = run(tmp_path, generations=0)
    assert summary.generations == ()
    assert summary.final_id == summary.seed_id
    assert summary.stopped_reason == "completed"


# --- held-out --------------------------------------------------------------------------------


def test_held_out_scores_seed_and_final_once_after_the_search(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"))
    seed_report = json.loads((summary.run_dir / "heldout" / "seed" / "report.json").read_text())
    final_report = json.loads((summary.run_dir / "heldout" / "final" / "report.json").read_text())
    assert seed_report["split"] == final_report["split"] == "heldout"
    assert seed_report["candidate_id"] == summary.seed_id
    assert final_report["candidate_id"] == summary.final_id
    assert {task["task_id"] for task in seed_report["tasks"]} == set(EXACT_HELDOUT_IDS)
    assert summary.heldout["seed"]["selection_score"] == pytest.approx(0.0)
    assert summary.heldout["final"]["candidate_id"] == summary.final_id
    assert summary.heldout["skipped_reason"] is None


def test_held_out_is_evaluated_once_when_the_final_incumbent_is_the_seed(tmp_path):
    summary = run(tmp_path, overseer_kind="null")
    assert (summary.run_dir / "heldout" / "seed" / "report.json").is_file()
    assert not (summary.run_dir / "heldout" / "final").exists()
    assert summary.heldout["final"]["same_report_as_seed"] is True
    assert summary.heldout["final"]["report"] == summary.heldout["seed"]["report"]
    assert summary.heldout["final"]["selection_score"] == summary.heldout["seed"]["selection_score"]


def test_held_out_is_skipped_with_a_reason_when_the_suite_has_none(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "candidate.toml").write_text('schema_version = 1\nagent = "scripted"\n')
    (seed / "responses.toml").write_text('default = "ok"\n')
    summary = run(tmp_path, suite_name=fixture("NoHeldoutSuite"))
    assert summary.heldout["seed"] is None
    assert "no held-out tasks" in summary.heldout["skipped_reason"]


def test_held_out_data_never_reaches_an_overseer(tmp_path):
    run_dir = tmp_path / "out" / "loop-test" / "r1"
    summary = run(
        tmp_path, overseer_kind=fixture("RecordingOverseer"), overseer_settings={"run_dir": str(run_dir)}, generations=2
    )
    briefs = loop_fixtures.RecordingOverseer.briefs
    assert len(briefs) == 2
    assert loop_fixtures.RecordingOverseer.heldout_existed == [False, False]

    for brief in briefs:
        for path in (brief.incumbent_report, brief.evidence_dir, brief.incumbent.root):
            assert "heldout" not in Path(path).parts
        text = json.dumps(
            [brief.suite_description, brief.valves, brief.selection_rule, brief.metric.description, list(brief.history)]
        )
        for task_id in EXACT_HELDOUT_IDS:
            assert task_id not in text

    archived = sorted((summary.run_dir / "overseer").glob("gen-*/workspace"))
    assert len(archived) == 2
    for workspace in archived:
        assert (workspace / "context" / "report.json").is_file()  # the evidence really was copied
        for path in workspace.rglob("*"):
            assert "heldout" not in path.relative_to(workspace).parts
            if path.is_file():
                content = path.read_text(errors="replace")
                for task_id in EXACT_HELDOUT_IDS:
                    assert task_id not in content, f"{task_id} leaked into {path}"


def test_brief_carries_the_incumbent_snapshot_metric_rule_and_history(tmp_path):
    run(tmp_path, overseer_kind=fixture("RecordingOverseer"), generations=2, promotion_margin=0.05)
    first, second = loop_fixtures.RecordingOverseer.briefs
    run_dir = tmp_path / "out" / "loop-test" / "r1"
    assert first.generation == 1 and second.generation == 2
    # The brief holds copies of the incumbent's snapshot, report and evidence (see the integrity tests).
    assert first.incumbent.id == Candidate.load(run_dir / "candidates" / first.incumbent.id).id
    assert loop_fixtures.RecordingOverseer.reports[0] == json.loads((run_dir / "evals" / "gen-000" / "report.json").read_text())
    assert first.metric.name == "accuracy"
    assert "0.0500" in first.selection_rule
    assert "Scripted agent valves" in first.valves
    assert list(first.history) == []
    assert [row["generation"] for row in second.history] == [1]
    json.dumps(list(second.history))  # JSON-safe


# --- overseer workspace ----------------------------------------------------------------------


def test_overseer_workspace_is_a_temp_dir_outside_the_run_dir_and_the_repo(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("RecordingOverseer"))
    [workspace] = loop_fixtures.RecordingOverseer.workspaces
    resolved = workspace.resolve()
    assert workspace.name.startswith("hive-overseer-")
    assert not resolved.is_relative_to(summary.run_dir.resolve())
    assert not resolved.is_relative_to(tmp_path.resolve())
    assert not resolved.is_relative_to(REPO_ROOT)
    assert not workspace.exists()  # removed once archived
    archived = summary.run_dir / "overseer" / "gen-001"
    assert (archived / "workspace" / "candidate" / "candidate.toml").is_file()
    proposal = json.loads((archived / "proposal.json").read_text())
    assert proposal["rationale"] == "recorded"
    assert proposal["error"] is None


def test_overseer_workspace_archive_preserves_symlinks(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("InvalidOverseer"), overseer_settings={"mode": "symlink"})
    link = summary.run_dir / "overseer" / "gen-001" / "workspace" / "candidate" / "linked.txt"
    assert link.is_symlink()


# --- experience store ------------------------------------------------------------------------


def test_run_dir_layout(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"))
    run_dir = tmp_path / "out" / "loop-test" / "r1"
    assert summary.run_dir == run_dir
    for relative in (
        "run_meta.json",
        "generations.jsonl",
        "summary.json",
        "evals/gen-000/report.json",
        "evals/gen-000/trials",
        "evals/gen-001/report.json",
        "overseer/gen-001/proposal.json",
        "overseer/gen-001/workspace",
        "heldout/seed/report.json",
        "heldout/final/report.json",
    ):
        assert (run_dir / relative).exists(), relative
    assert {path.name for path in (run_dir / "candidates").iterdir()} == {summary.seed_id, summary.final_id}


def test_seed_snapshot_gets_seed_lineage_and_child_snapshot_gets_overseer_lineage(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"))
    seed = Candidate.load(summary.run_dir / "candidates" / summary.seed_id)
    child = Candidate.load(summary.run_dir / "candidates" / summary.final_id)
    assert seed.id == summary.seed_id
    assert seed.lineage()["origin"] == "seed"
    assert seed.lineage()["parents"] == []
    assert child.lineage() == {"parents": [summary.seed_id], "origin": "improving", "rationale": "fixed count-letter-e"}


def test_seed_with_existing_lineage_keeps_it(tmp_path):
    config = make_config(tmp_path)
    Candidate.load(config.seed_candidate).write_lineage(["older"], "hand-written", "kept")
    summary = run_experiment(config, run_id="r1")
    seed = Candidate.load(summary.run_dir / "candidates" / summary.seed_id)
    assert seed.lineage() == {"parents": ["older"], "origin": "hand-written", "rationale": "kept"}


def test_existing_snapshot_lineage_is_never_overwritten(tmp_path):
    # The null overseer re-proposes the seed; its snapshot must keep origin "seed".
    summary = run(tmp_path, overseer_kind="null", generations=2)
    seed = Candidate.load(summary.run_dir / "candidates" / summary.seed_id)
    assert seed.lineage()["origin"] == "seed"


def test_generation_rows_have_every_field_and_are_appended_in_order(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"), generations=2)
    written = rows(summary)
    assert written == list(summary.generations)
    expected_fields = {
        "generation", "overseer", "status", "child_id", "child_selection_score", "child_measured_mean",
        "incumbent_before", "incumbent_score_before", "promoted", "reason", "rationale", "error",
        "overseer_cost_usd", "eval_cost_usd", "spent_usd",
    }  # fmt: skip
    for number, row in enumerate(written, start=1):
        assert set(row) == expected_fields
        assert row["generation"] == number
        assert row["overseer"] == "improving"
    assert written[0]["rationale"] == "fixed count-letter-e"
    assert written[0]["child_measured_mean"] == pytest.approx(4 / 6)


def test_summary_json_matches_the_returned_summary(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("ImprovingOverseer"))
    on_disk = json.loads((summary.run_dir / "summary.json").read_text())
    assert on_disk == summary.to_dict()
    assert on_disk["run_dir"] == str(summary.run_dir)
    assert on_disk["seed_id"] == summary.seed_id
    assert on_disk["final_id"] == summary.final_id
    assert on_disk["seed_search_score"] == pytest.approx(SEED_SCORE)
    assert on_disk["final_search_score"] == pytest.approx(4 / 6)
    assert on_disk["stopped_reason"] == "completed"
    assert on_disk["total_cost_usd"] == 0.0
    assert len(on_disk["generations"]) == 1
    assert set(on_disk["heldout"]) == {"seed", "final", "skipped_reason"}


def test_run_meta_records_config_versions_and_outcome(tmp_path):
    summary = run(tmp_path)
    meta = json.loads((summary.run_dir / "run_meta.json").read_text())
    assert meta["config"]["name"] == "loop-test"
    assert meta["config"]["model"] == {"provider": "scripted", "id": "none"}
    assert meta["run_id"] == "r1"
    assert meta["stopped_reason"] == "completed"
    assert meta["error"] is None
    assert meta["spent_usd"] == 0.0
    assert meta["started_at"] <= meta["ended_at"]
    assert meta["python_version"]
    assert meta["platform"]
    assert "git_commit" in meta
    assert meta["seed_id"] == summary.seed_id


def test_run_meta_is_written_even_when_the_run_crashes(tmp_path):
    config = make_config(tmp_path, overseer_kind=fixture("CrashingOverseer"))
    with pytest.raises(RuntimeError, match="overseer exploded"):
        run_experiment(config, run_id="r1")
    run_dir = tmp_path / "out" / "loop-test" / "r1"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert "overseer exploded" in meta["error"]
    assert meta["stopped_reason"] is None
    assert meta["ended_at"]
    # The temp workspace is still cleaned up, and what the overseer wrote is kept for the record.
    [workspace] = loop_fixtures.CrashingOverseer.workspaces
    assert not workspace.exists()
    assert (run_dir / "overseer" / "gen-001" / "workspace" / "partial.txt").is_file()


def test_default_run_id_is_a_utc_timestamp(tmp_path):
    summary = run_experiment(make_config(tmp_path, generations=0))
    assert len(summary.run_dir.name) == len("20261019-120000")
    assert summary.run_dir.name[8] == "-"
    assert summary.run_dir.name.replace("-", "").isdigit()


# --- configuration errors, raised before anything is spent ------------------------------------


def assert_nothing_evaluated(tmp_path: Path) -> None:
    assert not list((tmp_path / "out").rglob("evals"))
    assert not list((tmp_path / "out").rglob("report.json"))


def test_overlapping_splits_raise_before_any_evaluation(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "candidate.toml").write_text('schema_version = 1\nagent = "scripted"\n')
    (seed / "responses.toml").write_text('default = "ok"\n')
    with pytest.raises(LoopError, match="shared"):
        run(tmp_path, suite_name=fixture("OverlappingSuite"))
    assert_nothing_evaluated(tmp_path)


def test_empty_search_split_raises(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "candidate.toml").write_text('schema_version = 1\nagent = "scripted"\n')
    (seed / "responses.toml").write_text('default = "ok"\n')
    with pytest.raises(LoopError, match="search"):
        run(tmp_path, suite_name=fixture("EmptySearchSuite"))
    assert_nothing_evaluated(tmp_path)


def test_invalid_seed_raises_with_its_problems(tmp_path):
    config = make_config(tmp_path)
    (config.seed_candidate / "responses.toml").write_text("not = = toml\n")
    with pytest.raises(LoopError, match="responses.toml"):
        run_experiment(config, run_id="r1")
    assert_nothing_evaluated(tmp_path)


def test_existing_run_dir_raises_so_runs_never_mix(tmp_path):
    config = make_config(tmp_path, generations=0)
    run_experiment(config, run_id="same")
    with pytest.raises(LoopError, match="already exists"):
        run_experiment(config, run_id="same")


@pytest.mark.parametrize("run_id", ["", "..", "a/b", "../escape"])
def test_unsafe_run_id_raises(tmp_path, run_id):
    with pytest.raises(LoopError, match="run id"):
        run_experiment(make_config(tmp_path), run_id=run_id)
    assert not (tmp_path / "out").exists()


# --- snapshot integrity ----------------------------------------------------------------------


def test_overseer_is_briefed_with_a_copy_so_it_cannot_alter_the_incumbent_snapshot(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("TamperingOverseer"))
    snapshot = Candidate.load(summary.run_dir / "candidates" / summary.seed_id)
    assert snapshot.id == summary.seed_id
    # The untouched scripted seed answers no held-out task; forged answers would score 1.0.
    assert summary.heldout["seed"]["selection_score"] == 0.0


def test_brief_points_outside_the_run_dir_at_copies_of_the_incumbent_and_its_evidence(tmp_path):
    summary = run(tmp_path, overseer_kind=fixture("RecordingOverseer"))
    [brief] = loop_fixtures.RecordingOverseer.briefs
    for path in (brief.incumbent.root, brief.incumbent_report, brief.evidence_dir):
        assert not Path(path).resolve().is_relative_to(summary.run_dir)
    assert brief.incumbent.id == summary.seed_id


def test_a_snapshot_edited_on_disk_is_refused_before_it_is_evaluated_again(tmp_path):
    seed_id = Candidate.load(EXACT_SEED).id
    target = tmp_path / "out" / "loop-test" / "r1" / "candidates" / seed_id
    with pytest.raises(LoopError, match="changed on disk"):
        run(tmp_path, overseer_kind=fixture("TargetedTamperOverseer"), overseer_settings={"target": str(target)})


# --- budget estimates ------------------------------------------------------------------------

COST_PER_TRIAL = 0.01
SEARCH_EVAL_COST = 6 * COST_PER_TRIAL  # six search tasks, one trial each
HELDOUT_EVAL_COST = 4 * COST_PER_TRIAL


def costly_config(tmp_path: Path, **overrides: Any) -> ExperimentConfig:
    seed = tmp_path / "seed"
    shutil.copytree(EXACT_SEED, seed)
    manifest = seed / "candidate.toml"
    manifest.write_text(manifest.read_text().replace('agent = "scripted"', 'agent = "loop_fixtures:CostlyScriptedAdapter"'))
    overrides.setdefault("agent_settings", {"cost_per_trial": COST_PER_TRIAL})
    return make_config(tmp_path, **overrides)


def test_a_generation_is_not_started_when_its_estimated_cost_would_break_the_cap(tmp_path):
    config = costly_config(tmp_path, max_usd=SEARCH_EVAL_COST * 1.5, generations=3)
    summary = run_experiment(config, run_id="r1")
    assert summary.stopped_reason == "budget"
    assert summary.generations == ()
    assert not (summary.run_dir / "overseer").exists()
    assert summary.total_cost_usd <= config.max_usd


def test_a_child_is_not_evaluated_when_the_overseer_left_too_little_budget(tmp_path):
    # Room for the seed and one more search evaluation, but the overseer spends most of it.
    config = costly_config(
        tmp_path,
        max_usd=SEARCH_EVAL_COST * 2 + 0.01,
        overseer_kind=fixture("ExpensiveOverseer"),
        overseer_settings={"cost": 0.05},
        generations=3,
    )
    summary = run_experiment(config, run_id="r1")
    [row] = summary.generations
    assert row["status"] == "over_budget"
    assert row["eval_cost_usd"] == 0.0
    assert summary.stopped_reason == "budget"
    assert not (summary.run_dir / "evals" / "gen-001").exists()


def test_held_out_is_skipped_when_its_estimated_cost_would_break_the_cap(tmp_path):
    config = costly_config(tmp_path, max_usd=SEARCH_EVAL_COST + HELDOUT_EVAL_COST / 2, generations=0)
    summary = run_experiment(config, run_id="r1")
    assert summary.heldout["seed"] is None
    assert "budget" in summary.heldout["skipped_reason"]
    assert summary.total_cost_usd <= config.max_usd


def test_held_out_runs_when_the_estimate_fits(tmp_path):
    config = costly_config(tmp_path, max_usd=SEARCH_EVAL_COST + HELDOUT_EVAL_COST + 0.01, generations=0)
    summary = run_experiment(config, run_id="r1")
    assert summary.heldout["seed"] is not None
    assert summary.total_cost_usd == pytest.approx(SEARCH_EVAL_COST + HELDOUT_EVAL_COST)


# --- temp dir guard --------------------------------------------------------------------------


def test_a_temp_dir_inside_a_git_checkout_is_refused_before_anything_runs(tmp_path, monkeypatch):
    repo = tmp_path / "some-repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "tmp").mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(repo / "tmp"))
    with pytest.raises(LoopError, match="git checkout"):
        run(tmp_path)
    assert_nothing_evaluated(tmp_path)


def test_run_meta_records_whether_the_checkout_had_uncommitted_changes(tmp_path):
    summary = run(tmp_path)
    meta = json.loads((summary.run_dir / "run_meta.json").read_text())
    assert meta["git_dirty"] in (True, False, None)  # None only when git is unavailable
