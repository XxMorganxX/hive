"""The experiment config: the single source of truth for every tunable (P3, P12, P13).

One TOML file fixes everything an experiment varies or holds constant: the model
under test, which agent adapter, suite and overseer to use and their settings,
how many trials, how many generations, and the hard dollar cap.

Validation is strict on purpose. A typo such as `trial_per_task = 5` would
otherwise fall back to the default and silently change the experiment, so unknown
keys are errors. Every problem is collected and reported at once, so fixing a
config takes one round trip rather than one per mistake.

Plug-in sections ([agent], [suite], [overseer]) are open: their extra keys belong
to the plug-in, which validates them itself. One convention applies to all of them:
a key ending in `_file`, `_dir` or `_path` is a path, and a relative one resolves against the
config file's directory, like `seed_candidate` and `output_dir` do.

[model] also holds what the model gateway (hive/gateway) needs: `prices` (USD per million
tokens, required for every provider but the offline `scripted` one, because the gateway
computes each trial's cost from it, P12) and `upstream_url` (defaulted by provider). The
environment variable holding the provider's key is fixed here by provider, not configured:
the key is read once by the controller and never written to any file.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hive.contracts import ModelSpec
from hive.pricing import Prices, parse_prices

KNOWN_SECTIONS = ("experiment", "model", "agent", "suite", "evaluation", "loop", "overseer")

# Sections whose keys are all HIVE's own; any other key in them is a typo.
STRICT_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "experiment": ("name", "seed_candidate", "output_dir"),
    "model": ("provider", "id", "thinking", "prices", "upstream_url"),
    "evaluation": ("trials_per_task", "concurrency", "max_model_calls_per_trial"),
    "loop": ("generations", "promotion_margin", "max_usd"),
}

DEFAULT_OUTPUT_DIR = "experience"
DEFAULT_TRIALS_PER_TASK = 1
DEFAULT_CONCURRENCY = 1
DEFAULT_GENERATIONS = 1
DEFAULT_PROMOTION_MARGIN = 0.0
DEFAULT_MAX_MODEL_CALLS_PER_TRIAL = 50

# The offline provider: the scripted agent calls no model, so it gets no gateway, prices or key.
OFFLINE_PROVIDER = "scripted"
# Per real provider: where the gateway forwards to by default, and the environment variable the
# controller reads the key from. Fixed in code so a config can't point the key lookup elsewhere.
DEFAULT_UPSTREAM_URLS = {"openai": "https://api.openai.com"}
PROVIDER_KEY_ENV = {"openai": "OPENAI_API_KEY"}
KNOWN_PROVIDERS = tuple(sorted({OFFLINE_PROVIDER, *PROVIDER_KEY_ENV}))

# The name becomes a directory under output_dir, so it must not be able to climb out of it.
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


class ConfigError(Exception):
    """The experiment config is unusable. `problems` lists every reason, not just the first."""

    def __init__(self, source: Path, problems: list[str]):
        self.source = source
        self.problems = list(problems)
        listing = "\n".join(f"  - {problem}" for problem in self.problems)
        count = len(self.problems)
        super().__init__(f"{source}: {count} problem{'s' if count != 1 else ''} in experiment config:\n{listing}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    config_path: Path
    seed_candidate: Path
    output_dir: Path
    model: ModelSpec
    agent_settings: Mapping[str, Any]
    suite_name: str
    suite_settings: Mapping[str, Any]
    trials_per_task: int
    concurrency: int
    generations: int
    promotion_margin: float
    max_usd: float
    overseer_kind: str
    overseer_settings: Mapping[str, Any]
    # The model gateway's settings. Defaults keep hand-built configs (tests) on the offline path.
    prices: Prices | None = None  # [model] prices; None only for the scripted provider
    upstream_url: str | None = None  # [model] upstream_url or the provider's default; None for scripted
    max_model_calls_per_trial: int = DEFAULT_MAX_MODEL_CALLS_PER_TRIAL

    @property
    def uses_gateway(self) -> bool:
        """Every real provider's traffic goes through the model gateway; the scripted one has none."""
        return self.model.provider != OFFLINE_PROVIDER

    @property
    def provider_key_env(self) -> str | None:
        """The environment variable holding the provider's key (never the key itself)."""
        return PROVIDER_KEY_ENV.get(self.model.provider)


class _Section:
    """Reads typed values out of one TOML table, appending problems instead of raising.

    `table` is None when the section exists but is not a table; that problem has
    already been reported, so reads return defaults without piling on more errors.
    """

    def __init__(self, name: str, table: dict[str, Any] | None, problems: list[str]):
        self.name = name
        self.table = table
        self.problems = problems

    def problem(self, text: str) -> None:
        self.problems.append(f"[{self.name}] {text}")

    def _get(self, key: str, required: bool) -> Any:
        if self.table is None:
            return None
        if key not in self.table:
            if required:
                self.problem(f"{key} is required")
            return None
        return self.table[key]

    def string(self, key: str, default: str | None = None) -> str | None:
        value = self._get(key, required=default is None)
        if value is None:  # absent (TOML has no null); a missing required key is already reported
            return default
        if not isinstance(value, str) or not value.strip():
            self.problem(f"{key} must be a non-empty string")
            return None
        return value

    def optional_string(self, key: str) -> str | None:
        """A non-empty string, or None when the key is absent (TOML has no null)."""
        value = self._get(key, required=False)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            self.problem(f"{key} must be a non-empty string")
            return None
        return value

    def integer(self, key: str, default: int, minimum: int) -> int | None:
        value = self._get(key, required=False)
        if value is None:
            return default
        # bool is a subclass of int; `trials_per_task = true` is a mistake, not 1.
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            self.problem(f"{key} must be an integer >= {minimum}, got {value!r}")
            return None
        return value

    def number(self, key: str, default: float | None, minimum: float, exclusive: bool = False) -> float | None:
        value = self._get(key, required=default is None)
        if value is None:
            return default
        bound = f"{'>' if exclusive else '>='} {minimum:g}"
        if isinstance(value, bool) or not isinstance(value, int | float):
            self.problem(f"{key} must be a number {bound}, got {value!r}")
            return None
        if not math.isfinite(value):
            self.problem(f"{key} must be a finite number {bound}, got {value!r}")
            return None
        if value < minimum or (exclusive and value == minimum):
            self.problem(f"{key} must be a number {bound}, got {value!r}")
            return None
        return float(value)

    def reject_unknown_keys(self) -> None:
        if self.table is None:
            return
        allowed = STRICT_SECTION_KEYS[self.name]
        for key in self.table:
            if key not in allowed:
                self.problem(f"unknown key {key!r}")

    def settings(self, excluding: str | None = None) -> dict[str, Any]:
        if self.table is None:
            return {}
        return {key: value for key, value in self.table.items() if key != excluding}


def _read_toml(config_path: Path) -> dict[str, Any]:
    try:
        text = config_path.read_text()
    except FileNotFoundError:
        raise ConfigError(config_path, ["config file not found"]) from None
    except OSError as error:
        raise ConfigError(config_path, [f"could not read config file: {error}"]) from error
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(config_path, [f"not valid TOML: {error}"]) from error


def _resolve_path(base: Path, text: str) -> Path:
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _absolute_path(base: Path, text: str) -> str:
    """Absolute and normalized, but symlinks kept: a venv's bin/python is a symlink to a base
    interpreter, and following it would run without the venv's packages."""
    path = Path(text).expanduser()
    return os.path.normpath(path if path.is_absolute() else base / path)


# Plug-in settings named like these are paths. They resolve against the config file, as
# seed_candidate and output_dir do, so a config behaves the same from any working directory.
PATH_SETTING_SUFFIXES = ("_file", "_dir", "_path")


def _resolve_path_settings(section: str, settings: dict[str, Any], base: Path, problems: list[str]) -> dict[str, Any]:
    resolved = dict(settings)
    for key, value in settings.items():
        if not key.endswith(PATH_SETTING_SUFFIXES):
            continue
        if not isinstance(value, str) or not value:
            problems.append(f"[{section}] {key} must be a non-empty path string, got {value!r}")
            continue
        resolved[key] = _absolute_path(base, value)
    return resolved


# Reasoning-effort levels every framework adapter and the gateway agree on.
SHARED_THINKING_LEVELS = ("minimal", "low", "medium", "high")


def load_experiment(path: Path | str) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    data = _read_toml(config_path)
    problems: list[str] = []

    known = ", ".join(f"[{name}]" for name in KNOWN_SECTIONS)
    for key, value in data.items():
        if key not in KNOWN_SECTIONS:
            what = f"unknown section [{key}]" if isinstance(value, dict) else f"unknown top-level key {key!r}"
            problems.append(f"{what}; known sections: {known}")

    sections: dict[str, _Section] = {}
    for name in KNOWN_SECTIONS:
        table = data.get(name, {})
        if not isinstance(table, dict):
            problems.append(f"[{name}] must be a table")
            table = None
        sections[name] = _Section(name, table, problems)
    for name in STRICT_SECTION_KEYS:
        sections[name].reject_unknown_keys()

    experiment = sections["experiment"]
    name = experiment.string("name")
    if name is not None and (not _SAFE_NAME.fullmatch(name) or name in (".", "..")):
        experiment.problem(
            f"name must be a single path segment of letters, digits, '.', '_' or '-', got {name!r}"
        )
    seed_candidate = experiment.string("seed_candidate")
    output_dir = experiment.string("output_dir", default=DEFAULT_OUTPUT_DIR)

    model = sections["model"]
    provider = model.string("provider")
    model_id = model.string("id")
    # Reasoning effort is part of the frozen model (P3). For a real provider it must be stated, and
    # be a level that Pi, Deep Agents, Strands and the gateway all send the same way on the wire;
    # "unset", "off" and "none" each meant something different to one of them.
    thinking = model.optional_string("thinking")
    if provider is not None and provider != "scripted":
        if thinking is None and "thinking" not in (model.table or {}):  # a bad value is already reported
            model.problem(f"thinking is required for provider {provider!r}; one of {list(SHARED_THINKING_LEVELS)}")
        elif thinking is not None and thinking not in SHARED_THINKING_LEVELS:
            model.problem(f"thinking {thinking!r} is not supported; use one of {list(SHARED_THINKING_LEVELS)}")
    prices, upstream_url = _gateway_settings(model, provider)

    suite = sections["suite"]
    suite_name = suite.string("name")

    evaluation = sections["evaluation"]
    trials_per_task = evaluation.integer("trials_per_task", DEFAULT_TRIALS_PER_TASK, minimum=1)
    concurrency = evaluation.integer("concurrency", DEFAULT_CONCURRENCY, minimum=1)
    max_model_calls_per_trial = evaluation.integer(
        "max_model_calls_per_trial", DEFAULT_MAX_MODEL_CALLS_PER_TRIAL, minimum=1
    )

    loop = sections["loop"]
    generations = loop.integer("generations", DEFAULT_GENERATIONS, minimum=0)
    promotion_margin = loop.number("promotion_margin", DEFAULT_PROMOTION_MARGIN, minimum=0.0)
    # No default: every run has a hard cap (P12), so the author must choose one.
    max_usd = loop.number("max_usd", None, minimum=0.0, exclusive=True)

    overseer = sections["overseer"]
    overseer_kind = overseer.string("kind")

    base = config_path.parent
    agent_settings = _resolve_path_settings("agent", sections["agent"].settings(), base, problems)
    if "prices" in agent_settings:
        problems.append("[agent] prices moved to [model] prices: cost is a property of the model, computed by the gateway")
    suite_settings = _resolve_path_settings("suite", suite.settings(excluding="name"), base, problems)
    overseer_settings = _resolve_path_settings("overseer", overseer.settings(excluding="kind"), base, problems)

    if problems:
        raise ConfigError(config_path, problems)

    return ExperimentConfig(
        name=name,
        config_path=config_path,
        seed_candidate=_resolve_path(base, seed_candidate),
        output_dir=_resolve_path(base, output_dir),
        model=ModelSpec(provider=provider, id=model_id, thinking=thinking),
        agent_settings=agent_settings,
        suite_name=suite_name,
        suite_settings=suite_settings,
        trials_per_task=trials_per_task,
        concurrency=concurrency,
        generations=generations,
        promotion_margin=promotion_margin,
        max_usd=max_usd,
        overseer_kind=overseer_kind,
        overseer_settings=overseer_settings,
        prices=prices,
        upstream_url=upstream_url,
        max_model_calls_per_trial=max_model_calls_per_trial,
    )


def _gateway_settings(model: _Section, provider: str | None) -> tuple[Prices | None, str | None]:
    """([model] prices, upstream URL), with problems reported on `model`."""
    if provider is not None and provider not in KNOWN_PROVIDERS:
        model.problem(f"provider {provider!r} is not supported; known providers: {', '.join(KNOWN_PROVIDERS)}")
        provider = None  # no further problems about a provider we don't know
    prices = None
    table = model.table.get("prices") if model.table is not None else None
    if table is not None:
        try:
            prices = parse_prices(table)
        except ValueError as error:
            model.problem(f"prices: {error}")
    elif provider is not None and provider != OFFLINE_PROVIDER:
        model.problem(f"prices is required for provider {provider!r}: the gateway computes every trial's cost from it (P12)")
    upstream_url = model.optional_string("upstream_url")
    if upstream_url is not None:
        problem = _upstream_url_problem(upstream_url)
        if problem is not None:
            model.problem(f"upstream_url {problem}, got {upstream_url!r}")
            upstream_url = None
    elif provider is not None:
        upstream_url = DEFAULT_UPSTREAM_URLS.get(provider)
    return prices, upstream_url


def _upstream_url_problem(url: str) -> str | None:
    # The gateway checks the same; checking here makes a bad URL a config error, not a crash mid-run.
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "must be an http(s) URL"
    if parts.query or parts.fragment:
        return "must not have a query or fragment"
    if parts.path.rstrip("/").endswith("/v1"):
        return "is the provider's origin; the gateway appends /v1/..., so drop the trailing /v1"
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ModelSpec):
        return {"provider": value.provider, "id": value.id, "thinking": value.thinking}
    if isinstance(value, Prices):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime.date | datetime.time):  # datetime is a date subclass
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # "inf" / "nan": strict JSON has no spelling for them
    if value is None or isinstance(value, bool | int | float | str):
        return value
    # Never fail here: run_meta.json must be written even when the run is going wrong (P13).
    return repr(value)


def snapshot(config: ExperimentConfig) -> dict[str, Any]:
    """The resolved config as a JSON-safe dict, recorded in run_meta.json."""
    return {field.name: _json_safe(getattr(config, field.name)) for field in dataclasses.fields(config)}
