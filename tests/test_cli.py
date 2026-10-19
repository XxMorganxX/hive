import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hive.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
DRYRUN_EXACT = REPO_ROOT / "experiments" / "dryrun-exact-answer.toml"
DRYRUN_PYTHON = REPO_ROOT / "experiments" / "dryrun-python-functions.toml"
EXACT_SEED = REPO_ROOT / "harnesses" / "scripted-exact-answer"


def one_error_line(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.strip() and line.startswith("hive: error")]
    assert len(lines) == 1, stderr
    assert "Traceback" not in stderr
    return lines[0]


# --- hive run --------------------------------------------------------------------------------


def test_run_dry_run_returns_0_and_prints_a_summary(tmp_path, capsys):
    code = main(["run", str(DRYRUN_EXACT), "--output-dir", str(tmp_path), "--run-id", "t"])
    out = capsys.readouterr().out
    assert code == 0
    run_dir = tmp_path / "dryrun-exact-answer" / "t"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stopped_reason"] == "completed"
    assert str(run_dir) in out
    assert "seed" in out and "final" in out
    assert f"{summary['seed_search_score']:.4f}" in out
    assert "held-out" in out
    assert "promotions: 0 of 2" in out
    assert "$0.00" in out


def test_run_python_functions_dry_run(tmp_path, capsys):
    code = main(["run", str(DRYRUN_PYTHON), "--output-dir", str(tmp_path), "--run-id", "t"])
    assert code == 0
    summary = json.loads((tmp_path / "dryrun-python-functions" / "t" / "summary.json").read_text())
    assert 0.0 < summary["seed_search_score"] < 1.0
    assert [row["status"] for row in summary["generations"]] == ["evaluated", "evaluated"]


def test_run_logs_progress_to_stderr(tmp_path, capsys):
    main(["run", str(DRYRUN_EXACT), "--output-dir", str(tmp_path), "--run-id", "t"])
    err = capsys.readouterr().err
    assert "generation 1" in err
    assert "held-out" in err


def test_run_into_an_existing_run_dir_is_a_user_error(tmp_path, capsys):
    assert main(["run", str(DRYRUN_EXACT), "--output-dir", str(tmp_path), "--run-id", "t"]) == 0
    capsys.readouterr()
    assert main(["run", str(DRYRUN_EXACT), "--output-dir", str(tmp_path), "--run-id", "t"]) == 2
    assert "already exists" in one_error_line(capsys.readouterr().err)


# --- hive eval -------------------------------------------------------------------------------


def test_eval_scores_the_seed_and_prints_a_per_task_table(tmp_path, capsys):
    code = main(["eval", str(DRYRUN_EXACT), "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    [eval_dir] = (tmp_path / "dryrun-exact-answer").glob("eval-*")
    report = json.loads((eval_dir / "report.json").read_text())
    assert report["split"] == "search"
    assert report["trials_per_task"] == 2  # from the config
    for task in report["tasks"]:
        assert task["task_id"] in out
    assert "selection_score" in out
    assert f"{report['selection_score']:.4f}" in out
    assert str(eval_dir / "report.json") in out


def test_eval_accepts_candidate_split_and_trials(tmp_path, capsys):
    candidate = tmp_path / "other"
    shutil.copytree(EXACT_SEED, candidate)
    code = main(
        ["eval", str(DRYRUN_EXACT), "--candidate", str(candidate), "--split", "heldout", "--trials", "1",
         "--output-dir", str(tmp_path / "out")]
    )  # fmt: skip
    assert code == 0
    [eval_dir] = (tmp_path / "out" / "dryrun-exact-answer").glob("eval-*")
    report = json.loads((eval_dir / "report.json").read_text())
    assert report["split"] == "heldout"
    assert report["trials_per_task"] == 1
    assert "fill-rate" in capsys.readouterr().out


def test_eval_of_an_invalid_candidate_is_a_user_error(tmp_path, capsys):
    candidate = tmp_path / "broken"
    shutil.copytree(EXACT_SEED, candidate)
    (candidate / "responses.toml").write_text("= nope\n")
    code = main(["eval", str(DRYRUN_EXACT), "--candidate", str(candidate), "--output-dir", str(tmp_path)])
    assert code == 2
    assert "responses.toml" in one_error_line(capsys.readouterr().err)


# --- hive validate ---------------------------------------------------------------------------


def test_validate_prints_ok_for_a_valid_candidate(capsys):
    assert main(["validate", str(EXACT_SEED)]) == 0
    assert capsys.readouterr().out.strip() == "OK"


def test_validate_lists_problems_and_returns_1(tmp_path, capsys):
    candidate = tmp_path / "broken"
    shutil.copytree(EXACT_SEED, candidate)
    (candidate / "candidate.toml").write_text('schema_version = 1\nagent = "scripted"\n[valves]\nx = 1\n')
    assert main(["validate", str(candidate)]) == 1
    out = capsys.readouterr().out
    assert "valves" in out
    assert "OK" not in out


def test_validate_of_a_directory_without_a_manifest_is_a_user_error(tmp_path, capsys):
    assert main(["validate", str(tmp_path)]) == 2
    assert "candidate.toml" in one_error_line(capsys.readouterr().err)


def test_validate_of_an_unknown_agent_is_a_user_error(tmp_path, capsys):
    (tmp_path / "candidate.toml").write_text('schema_version = 1\nagent = "nonexistent"\n')
    assert main(["validate", str(tmp_path)]) == 2
    assert "nonexistent" in one_error_line(capsys.readouterr().err)


# --- user errors -----------------------------------------------------------------------------


def test_missing_config_is_a_one_line_user_error(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.toml")]) == 2
    assert "not found" in one_error_line(capsys.readouterr().err)


def test_config_with_several_problems_is_still_one_line(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text('[experiment]\nname = "x"\n[loop]\ntrial = 1\n')
    assert main(["run", str(config)]) == 2
    line = one_error_line(capsys.readouterr().err)
    assert "seed_candidate" in line and "max_usd" in line


def test_bad_arguments_return_2(capsys):
    assert main(["eval"]) == 2
    assert main(["nonsense"]) == 2


def test_no_subcommand_returns_2(capsys):
    assert main([]) == 2


def test_python_dash_m_hive_runs_the_cli(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "hive", "validate", str(EXACT_SEED)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "OK"
