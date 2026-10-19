import json
import math
from pathlib import Path

import pytest

from hive.config import ConfigError, ExperimentConfig, load_experiment, snapshot
from hive.contracts import ModelSpec

SECTIONS = {
    "experiment": {"name": '"demo"', "seed_candidate": '"candidates/seed"'},
    "model": {"provider": '"openai"', "id": '"gpt-5-nano"'},
    "suite": {"name": '"exact_answer"'},
    "loop": {"max_usd": "5.0"},
    "overseer": {"kind": '"null"'},
}


def render(sections: dict[str, dict[str, str]]) -> str:
    lines = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {value}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def write_config(tmp_path: Path, overrides: dict[str, dict[str, str | None]] | None = None, extra: str = "") -> Path:
    """Write the minimal valid config with per-key overrides; a value of None deletes that key."""
    sections = {name: dict(values) for name, values in SECTIONS.items()}
    for section, values in (overrides or {}).items():
        target = sections.setdefault(section, {})
        for key, value in values.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
    path = tmp_path / "experiments" / "demo.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(sections) + extra)
    return path


def problems_for(path: Path) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        load_experiment(path)
    return caught.value.problems


# --- loading and defaults -------------------------------------------------------


def test_minimal_config_loads(tmp_path):
    config = load_experiment(write_config(tmp_path))
    assert isinstance(config, ExperimentConfig)
    assert config.name == "demo"


def test_evaluation_defaults_to_one_trial_and_no_concurrency(tmp_path):
    config = load_experiment(write_config(tmp_path))
    assert (config.trials_per_task, config.concurrency) == (1, 1)


def test_loop_defaults_to_one_generation_and_zero_margin(tmp_path):
    config = load_experiment(write_config(tmp_path))
    assert (config.generations, config.promotion_margin) == (1, 0.0)


def test_output_dir_defaults_to_experience_next_to_config(tmp_path):
    config = load_experiment(write_config(tmp_path))
    assert config.output_dir == (tmp_path / "experiments" / "experience").resolve()


def test_missing_plugin_sections_give_empty_settings(tmp_path):
    config = load_experiment(write_config(tmp_path))
    assert (config.agent_settings, config.suite_settings, config.overseer_settings) == ({}, {}, {})


def test_model_section_becomes_model_spec(tmp_path):
    config = load_experiment(write_config(tmp_path))
    assert config.model == ModelSpec(provider="openai", id="gpt-5-nano")


def test_model_thinking_defaults_to_none(tmp_path):
    assert load_experiment(write_config(tmp_path)).model.thinking is None


def test_model_thinking_flows_into_the_model_spec(tmp_path):
    config = load_experiment(write_config(tmp_path, {"model": {"thinking": '"low"'}}))
    assert config.model == ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")


@pytest.mark.parametrize("value", ['""', '"  "', "1", "true"])
def test_model_thinking_must_be_a_non_empty_string(tmp_path, value):
    assert problems_for(write_config(tmp_path, {"model": {"thinking": value}})) == [
        "[model] thinking must be a non-empty string"
    ]


def test_config_path_is_absolute(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = load_experiment(Path("experiments/demo.toml"))
    assert config.config_path == path.resolve()


def test_load_accepts_string_path(tmp_path):
    config = load_experiment(str(write_config(tmp_path)))
    assert config.name == "demo"


# --- paths ----------------------------------------------------------------------


def test_relative_seed_candidate_resolves_against_config_directory_not_cwd(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = load_experiment(path)
    assert config.seed_candidate == (tmp_path / "experiments" / "candidates" / "seed").resolve()


def test_relative_output_dir_resolves_against_config_directory(tmp_path):
    config = load_experiment(write_config(tmp_path, {"experiment": {"output_dir": '"../runs"'}}))
    assert config.output_dir == (tmp_path / "runs").resolve()


def test_absolute_seed_candidate_is_kept(tmp_path):
    absolute = tmp_path / "elsewhere" / "seed"
    config = load_experiment(write_config(tmp_path, {"experiment": {"seed_candidate": json.dumps(str(absolute))}}))
    assert config.seed_candidate == absolute.resolve()


# --- plug-in settings -----------------------------------------------------------


def test_agent_section_becomes_agent_settings(tmp_path):
    config = load_experiment(write_config(tmp_path, {"agent": {"timeout_s": "120", "pi_version": '"0.75.3"'}}))
    assert config.agent_settings == {"timeout_s": 120, "pi_version": "0.75.3"}


def test_suite_name_is_separated_from_suite_settings(tmp_path):
    config = load_experiment(write_config(tmp_path, {"suite": {"limit": "3"}}))
    assert (config.suite_name, config.suite_settings) == ("exact_answer", {"limit": 3})


def test_overseer_kind_is_separated_from_overseer_settings(tmp_path):
    config = load_experiment(write_config(tmp_path, {"overseer": {"kind": '"claude_code"', "model": '"sonnet"'}}))
    assert (config.overseer_kind, config.overseer_settings) == ("claude_code", {"model": "sonnet"})


def test_plugin_sections_accept_arbitrary_keys(tmp_path):
    config = load_experiment(write_config(tmp_path, {"agent": {"anything_goes": "true"}}))
    assert config.agent_settings == {"anything_goes": True}


# --- explicit values ------------------------------------------------------------


def test_explicit_evaluation_and_loop_values_are_used(tmp_path):
    config = load_experiment(
        write_config(
            tmp_path,
            {
                "evaluation": {"trials_per_task": "3", "concurrency": "4"},
                "loop": {"generations": "5", "promotion_margin": "0.05", "max_usd": "12.5"},
            },
        )
    )
    assert (config.trials_per_task, config.concurrency, config.generations) == (3, 4, 5)
    assert (config.promotion_margin, config.max_usd) == (0.05, 12.5)


def test_zero_generations_is_allowed(tmp_path):
    config = load_experiment(write_config(tmp_path, {"loop": {"generations": "0"}}))
    assert config.generations == 0


def test_integer_is_accepted_where_float_expected(tmp_path):
    config = load_experiment(write_config(tmp_path, {"loop": {"max_usd": "5"}}))
    assert config.max_usd == 5.0 and isinstance(config.max_usd, float)


# --- file-level errors ----------------------------------------------------------


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_experiment(tmp_path / "nope.toml")


def test_invalid_toml_raises_config_error(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[experiment\nname = ")
    with pytest.raises(ConfigError, match="TOML"):
        load_experiment(path)


def test_config_error_message_names_the_file(tmp_path):
    path = write_config(tmp_path, {"loop": {"max_usd": None}})
    with pytest.raises(ConfigError, match="demo.toml"):
        load_experiment(path)


# --- required values ------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("experiment", "name"),
        ("experiment", "seed_candidate"),
        ("model", "provider"),
        ("model", "id"),
        ("suite", "name"),
        ("loop", "max_usd"),
        ("overseer", "kind"),
    ],
)
def test_missing_required_key_is_reported(tmp_path, section, key):
    problems = problems_for(write_config(tmp_path, {section: {key: None}}))
    assert problems == [f"[{section}] {key} is required"]


def test_missing_required_section_reports_each_required_key(tmp_path):
    path = tmp_path / "c.toml"
    sections = {name: values for name, values in SECTIONS.items() if name != "model"}
    path.write_text(render(sections))
    assert problems_for(path) == ["[model] provider is required", "[model] id is required"]


@pytest.mark.parametrize(
    ("section", "key"),
    [("experiment", "name"), ("experiment", "seed_candidate"), ("model", "id"), ("suite", "name"), ("overseer", "kind")],
)
def test_empty_required_string_is_an_error(tmp_path, section, key):
    problems = problems_for(write_config(tmp_path, {section: {key: '""'}}))
    assert problems == [f"[{section}] {key} must be a non-empty string"]


@pytest.mark.parametrize(
    ("section", "key"),
    [("experiment", "name"), ("experiment", "output_dir"), ("model", "provider"), ("suite", "name"), ("overseer", "kind")],
)
def test_non_string_where_string_expected_is_an_error(tmp_path, section, key):
    problems = problems_for(write_config(tmp_path, {section: {key: "7"}}))
    assert problems == [f"[{section}] {key} must be a non-empty string"]


@pytest.mark.parametrize("name", ['"../escape"', '"a/b"', '"."', '".."', '"has space"'])
def test_experiment_name_must_be_a_single_safe_path_segment(tmp_path, name):
    problems = problems_for(write_config(tmp_path, {"experiment": {"name": name}}))
    assert len(problems) == 1 and problems[0].startswith("[experiment] name must be")


# --- numeric validation ---------------------------------------------------------


@pytest.mark.parametrize(("section", "key"), [("evaluation", "trials_per_task"), ("evaluation", "concurrency"), ("loop", "generations")])
def test_boolean_is_rejected_where_integer_expected(tmp_path, section, key):
    problems = problems_for(write_config(tmp_path, {section: {key: "true"}}))
    assert len(problems) == 1 and problems[0].startswith(f"[{section}] {key} must be an integer")


@pytest.mark.parametrize(("section", "key"), [("loop", "promotion_margin"), ("loop", "max_usd")])
def test_boolean_is_rejected_where_number_expected(tmp_path, section, key):
    problems = problems_for(write_config(tmp_path, {section: {key: "true"}}))
    assert len(problems) == 1 and problems[0].startswith(f"[{section}] {key} must be a number")


def test_float_is_rejected_where_integer_expected(tmp_path):
    problems = problems_for(write_config(tmp_path, {"evaluation": {"trials_per_task": "2.0"}}))
    assert len(problems) == 1 and problems[0].startswith("[evaluation] trials_per_task must be an integer")


def test_string_is_rejected_where_number_expected(tmp_path):
    problems = problems_for(write_config(tmp_path, {"loop": {"max_usd": '"5"'}}))
    assert len(problems) == 1 and problems[0].startswith("[loop] max_usd must be a number")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("evaluation", "trials_per_task", "0"),
        ("evaluation", "concurrency", "0"),
        ("loop", "generations", "-1"),
    ],
)
def test_integer_below_minimum_is_an_error(tmp_path, section, key, value):
    problems = problems_for(write_config(tmp_path, {section: {key: value}}))
    assert len(problems) == 1 and problems[0].startswith(f"[{section}] {key} must be an integer >=")


def test_negative_promotion_margin_is_an_error(tmp_path):
    problems = problems_for(write_config(tmp_path, {"loop": {"promotion_margin": "-0.1"}}))
    assert problems == ["[loop] promotion_margin must be a number >= 0, got -0.1"]


@pytest.mark.parametrize("value", ["0", "0.0", "-1.0"])
def test_non_positive_max_usd_is_an_error(tmp_path, value):
    problems = problems_for(write_config(tmp_path, {"loop": {"max_usd": value}}))
    assert len(problems) == 1 and problems[0].startswith("[loop] max_usd must be a number > 0")


@pytest.mark.parametrize(("key", "value"), [("max_usd", "inf"), ("max_usd", "nan"), ("promotion_margin", "nan"), ("promotion_margin", "inf")])
def test_non_finite_number_is_an_error(tmp_path, key, value):
    problems = problems_for(write_config(tmp_path, {"loop": {key: value}}))
    assert len(problems) == 1 and problems[0].startswith(f"[loop] {key} must be a finite number")


# --- unknown keys and sections --------------------------------------------------


def test_unknown_top_level_section_is_an_error(tmp_path):
    problems = problems_for(write_config(tmp_path, {"evalution": {"trials_per_task": "3"}}))
    assert len(problems) == 1 and "unknown section [evalution]" in problems[0]


def test_unknown_top_level_key_is_an_error(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("stray = 1\n" + render(SECTIONS))
    problems = problems_for(path)
    assert len(problems) == 1 and "unknown top-level key 'stray'" in problems[0]


@pytest.mark.parametrize(
    ("section", "key"),
    [("experiment", "seed"), ("model", "temperature"), ("evaluation", "trials"), ("loop", "max_dollars")],
)
def test_unknown_key_in_strict_section_is_an_error(tmp_path, section, key):
    problems = problems_for(write_config(tmp_path, {section: {key: "1"}}))
    assert problems == [f"[{section}] unknown key {key!r}"]


def test_unknown_key_does_not_fall_back_to_default_silently(tmp_path):
    problems = problems_for(write_config(tmp_path, {"evaluation": {"trial_per_task": "5"}}))
    assert problems == ["[evaluation] unknown key 'trial_per_task'"]


def test_section_that_is_not_a_table_is_an_error(tmp_path):
    path = tmp_path / "c.toml"
    sections = {name: values for name, values in SECTIONS.items() if name != "suite"}
    path.write_text('suite = "exact_answer"\n' + render(sections))
    assert problems_for(path) == ["[suite] must be a table"]


# --- error aggregation ----------------------------------------------------------


def test_all_problems_are_reported_together(tmp_path):
    path = write_config(
        tmp_path,
        {
            "experiment": {"name": None},
            "evaluation": {"trials_per_task": "true", "concurency": "2"},
            "loop": {"max_usd": None, "promotion_margin": "-1"},
            "bogus": {"x": "1"},
        },
    )
    problems = problems_for(path)
    assert len(problems) == 6


def test_config_error_message_lists_every_problem(tmp_path):
    path = write_config(tmp_path, {"experiment": {"name": None}, "loop": {"max_usd": None}})
    with pytest.raises(ConfigError) as caught:
        load_experiment(path)
    message = str(caught.value)
    assert "[experiment] name is required" in message and "[loop] max_usd is required" in message


# --- snapshot -------------------------------------------------------------------


def test_snapshot_is_json_serializable(tmp_path):
    config = load_experiment(
        write_config(
            tmp_path,
            {
                "agent": {"timeout_s": "120", "nested": "{ depth = 2, tags = ['a', 'b'] }"},
                "suite": {"started": "2026-10-19T08:00:00Z", "day": "2026-10-19", "at": "08:00:00"},
            },
        )
    )
    json.dumps(snapshot(config), allow_nan=False)


def test_snapshot_records_resolved_paths_as_strings(tmp_path):
    config = load_experiment(write_config(tmp_path))
    data = snapshot(config)
    assert data["seed_candidate"] == str(config.seed_candidate)
    assert data["output_dir"] == str(config.output_dir)
    assert data["config_path"] == str(config.config_path)


def test_snapshot_records_model_and_plugin_choices(tmp_path):
    config = load_experiment(write_config(tmp_path, {"suite": {"limit": "3"}}))
    data = snapshot(config)
    assert data["model"] == {"provider": "openai", "id": "gpt-5-nano", "thinking": None}
    assert (data["suite_name"], data["suite_settings"], data["overseer_kind"]) == ("exact_answer", {"limit": 3}, "null")


def test_snapshot_records_model_thinking(tmp_path):
    config = load_experiment(write_config(tmp_path, {"model": {"thinking": '"low"'}}))
    assert snapshot(config)["model"]["thinking"] == "low"


def test_snapshot_records_every_tunable(tmp_path):
    data = snapshot(load_experiment(write_config(tmp_path)))
    for key in ("trials_per_task", "concurrency", "generations", "promotion_margin", "max_usd", "name"):
        assert key in data


def test_snapshot_converts_non_finite_setting_to_string(tmp_path):
    config = load_experiment(write_config(tmp_path, {"agent": {"limit": "inf"}}))
    assert snapshot(config)["agent_settings"]["limit"] == "inf"
    assert math.isinf(config.agent_settings["limit"])


# --- path-valued plug-in settings -----------------------------------------------------------


def test_relative_file_and_dir_settings_resolve_against_the_config_file(tmp_path):
    path = write_config(
        tmp_path,
        {"suite": {"tasks_file": '"data/tasks.jsonl"'}, "agent": {"cache_dir": '"cache"'}},
    )
    config = load_experiment(path)
    assert config.suite_settings["tasks_file"] == str((path.parent / "data" / "tasks.jsonl").resolve())
    assert config.agent_settings["cache_dir"] == str((path.parent / "cache").resolve())


def test_absolute_path_settings_are_kept(tmp_path):
    absolute = tmp_path / "elsewhere" / "tasks.jsonl"
    config = load_experiment(write_config(tmp_path, {"suite": {"tasks_file": f'"{absolute}"'}}))
    assert config.suite_settings["tasks_file"] == str(absolute.resolve())


def test_settings_without_a_path_suffix_are_left_alone(tmp_path):
    # Binaries like pi_bin are looked up on PATH, not relative to the config.
    config = load_experiment(write_config(tmp_path, {"overseer": {"claude_bin": '"claude"', "model": '"sonnet"'}}))
    assert config.overseer_settings["claude_bin"] == "claude"
    assert config.overseer_settings["model"] == "sonnet"


def test_a_non_string_path_setting_is_a_problem(tmp_path):
    problems = problems_for(write_config(tmp_path, {"suite": {"tasks_file": "3"}}))
    assert any("tasks_file" in problem for problem in problems)
