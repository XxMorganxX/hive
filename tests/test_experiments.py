"""Every shipped experiment config must load and resolve, without running anything external."""

import subprocess
from pathlib import Path

import pytest

import hive.agents.deepagents.adapter as deepagents_adapter
import hive.agents.pi.adapter as pi_adapter
import hive.agents.strands.adapter as strands_adapter
from hive.candidate import Candidate
from hive.config import load_experiment
from hive.registry import make_agent_adapter, make_overseer, make_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted((REPO_ROOT / "experiments").glob("*.toml"))


@pytest.fixture(autouse=True)
def no_external_processes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"an experiment config tried to run a process: {args}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


def test_the_four_experiments_exist():
    assert {path.stem for path in EXPERIMENTS} >= {
        "pi-exact-answer",
        "pi-python-functions",
        "dryrun-exact-answer",
        "dryrun-python-functions",
    }


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_experiment_loads_and_its_seed_is_valid(path):
    config = load_experiment(path)
    assert config.name == path.stem
    seed = Candidate.load(config.seed_candidate)
    adapter = make_agent_adapter(seed.agent)
    assert adapter.validate(seed) == []


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_experiment_suite_and_overseer_resolve(path):
    config = load_experiment(path)
    suite = make_suite(config.suite_name, config.suite_settings)
    assert suite.tasks("search")
    make_overseer(config.overseer_kind, config.overseer_settings)


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_seed_adapter_builds_with_the_experiment_agent_settings(path, monkeypatch):
    # build() checks each pinned framework version by running the framework; record the pin instead.
    pinned: list[str] = []
    record = lambda binary, version: pinned.append(version)  # noqa: E731
    monkeypatch.setattr(pi_adapter, "check_pi_version", record)
    monkeypatch.setattr(deepagents_adapter, "check_deepagents_version", record)
    monkeypatch.setattr(strands_adapter, "check_strands_version", record)
    # Python framework adapters refuse to build an OpenAI run without a key; a placeholder suffices offline.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-not-used")
    config = load_experiment(path)
    seed = Candidate.load(config.seed_candidate)
    make_agent_adapter(seed.agent).build(seed, config.model, config.agent_settings)
    expected_pins = {"pi": ["0.75.3"], "deepagents": ["0.7.28"], "strands": ["1.61.1"]}
    assert pinned == expected_pins.get(seed.agent, [])


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_experiment_outputs_go_to_the_gitignored_experience_dir(path):
    assert load_experiment(path).output_dir == REPO_ROOT / "experience"


def test_dry_runs_spend_nothing():
    for path in EXPERIMENTS:
        if path.stem.startswith("dryrun-"):
            config = load_experiment(path)
            assert config.overseer_kind == "null"
            assert Candidate.load(config.seed_candidate).agent == "scripted"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_framework_experiments_fix_low_thinking_and_dry_runs_leave_it_unset(path):
    # The reasoning-effort level is part of the frozen model (P3), so each config states it.
    expected = None if path.stem.startswith("dryrun-") else "low"
    assert load_experiment(path).model.thinking == expected
