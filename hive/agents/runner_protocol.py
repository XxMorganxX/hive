"""The controller side of a framework runner: turn what a trial subprocess left behind into
one AgentResult.

Python frameworks (Deep Agents, Strands) run each trial in a runner script inside their
own venv (see hive/agents/process.py for why). The runner writes two files into the
trial's trace dir, and this module is the one place that reads them, so every adapter
classifies runs the same way:

- `trace.jsonl`, appended as the run goes. Every model call is one line:
      {"type": "model_call", "model": "<model the provider reported>",
       "usage": {"input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"}}
  `input_tokens` excludes cache reads. Other line types are the runner's own trace and are
  ignored here. Usage and the model check come from this file, not from result.json,
  because a run killed on timeout never writes result.json but has still spent money (P12).
- `result.json`, written once at the end:
      {"status": "completed" | "errored", "final_text", "error", "error_class",
       "extensions": [{"path", "loaded", "error", "registered"}], "runtime_errors": [...]}

Neither framework computes dollars, so cost comes from the experiment's price table.
"""

from __future__ import annotations

import json
import math
import re
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hive.agents.process import ProcessOutcome
from hive.contracts import AGENT_ERROR_CLASSES, AgentResult, ExtensionLoad, LoadReport, ModelSpec, Usage

TRACE_FILE = "trace.jsonl"
RESULT_FILE = "result.json"

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
_REQUIRED_PRICES = ("input", "output")
# Providers report dated snapshots of the requested model ("gpt-5-nano-2025-08-07").
_SNAPSHOT_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})")
_STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class Prices:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * self.cache_write
        ) / 1_000_000


def parse_prices(table: Any) -> Prices:
    """The `prices` table from an adapter's settings. Required: without it, spend would read as $0."""
    if not isinstance(table, Mapping):
        raise ValueError("prices must be a table of USD per million tokens, e.g. { input = 0.05, output = 0.40 }")
    unknown = sorted(set(table) - set(_PRICE_FIELDS))
    if unknown:
        raise ValueError(f"unknown price keys {unknown}; allowed: {list(_PRICE_FIELDS)}")
    missing = [name for name in _REQUIRED_PRICES if name not in table]
    if missing:
        raise ValueError(f"prices must include {missing}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"price {name!r} must be a finite number >= 0, got {value!r}")
    return Prices(**{name: float(value) for name, value in table.items()})


def model_matches(reported: str, model: ModelSpec) -> bool:
    """True when a provider-reported model name is the experiment's model (P3)."""
    name = reported
    if "/" in name:
        provider, _, name = name.partition("/")
        if provider != model.provider:
            return False
    if name == model.id:
        return True
    return name.startswith(model.id) and _SNAPSHOT_SUFFIX.fullmatch(name[len(model.id) :]) is not None


def interpret_runner(
    trace_dir: Path, outcome: ProcessOutcome, *, model: ModelSpec, prices: Prices, wall_clock_s: float
) -> AgentResult:
    calls = _model_calls(trace_dir / TRACE_FILE)
    tokens = Usage(**{field: sum(_count(call, field) for call in calls) for field in _TOKEN_FIELDS})
    usage = Usage(**{field: getattr(tokens, field) for field in _TOKEN_FIELDS}, cost_usd=prices.cost(tokens))

    def errored(error_class: str, error: str, load_report: LoadReport | None = None) -> AgentResult:
        return AgentResult(status="errored", final_text="", usage=usage, wall_clock_s=wall_clock_s,
                           error=error, error_class=error_class, load_report=load_report)  # fmt: skip

    if outcome.launch_error is not None:
        return errored("agent_error", f"could not start the runner: {outcome.launch_error}")

    # A breach of P3 outranks everything else: the run is not part of this experiment at all.
    others = sorted({str(call.get("model", "")) for call in calls if not model_matches(str(call.get("model", "")), model)})
    result = _read_result(trace_dir / RESULT_FILE)
    load_report = _load_report(result) if result is not None else None
    if others:
        names = ", ".join(repr(name) for name in others)
        return errored("model_mismatch", f"model calls went to {names}, not {model.qualified}", load_report)
    if outcome.timed_out:
        return errored("timeout", "the run exceeded its time limit and was stopped", load_report)
    if result is None:
        return errored("agent_error", _crash_description(outcome))

    if result.get("status") == "completed":
        final_text = result.get("final_text")
        if not isinstance(final_text, str):
            return errored("agent_error", "the runner reported no final text", load_report)
        return AgentResult(status="completed", final_text=final_text, usage=usage, wall_clock_s=wall_clock_s,
                           load_report=load_report)  # fmt: skip
    error = str(result.get("error") or "the runner reported an error without a message")
    error_class = result.get("error_class")
    if error_class not in AGENT_ERROR_CLASSES:
        error = f"[runner error_class {error_class!r}] {error}"
        error_class = "agent_error"
    return errored(error_class, error, load_report)


def _model_calls(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    calls = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a kill, or runner noise
        if isinstance(entry, dict) and entry.get("type") == "model_call":
            calls.append(entry)
    return calls


def _count(call: Mapping[str, Any], field: str) -> int:
    usage = call.get("usage")
    value = usage.get(field, 0) if isinstance(usage, Mapping) else 0
    # Only trust non-negative ints: a bogus count must not shrink or inflate the spend.
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _read_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return result if isinstance(result, dict) else None


def _load_report(result: Mapping[str, Any]) -> LoadReport:
    loads = []
    for entry in result.get("extensions") or ():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            continue
        registered = entry.get("registered") or ()
        loads.append(
            ExtensionLoad(
                path=entry["path"],
                loaded=entry.get("loaded") is True,
                error=str(entry["error"]) if entry.get("error") is not None else None,
                registered=tuple(str(item) for item in registered if isinstance(item, str)),
            )
        )
    runtime_errors = tuple(str(error) for error in result.get("runtime_errors") or () if error)
    return LoadReport(extensions=tuple(loads), runtime_errors=runtime_errors)


def _crash_description(outcome: ProcessOutcome) -> str:
    stderr = outcome.stderr.decode(errors="replace").strip()[-_STDERR_TAIL_CHARS:]
    if outcome.returncode is not None and outcome.returncode < 0:
        try:
            name = signal.Signals(-outcome.returncode).name
        except ValueError:
            name = f"signal {-outcome.returncode}"
        head = f"the runner was killed by {name}"
    else:
        head = f"the runner exited with code {outcome.returncode} and wrote no result"
    return f"{head}: {stderr}" if stderr else head
