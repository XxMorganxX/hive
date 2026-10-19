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
  The runner writes "extensions" only once its load phase has run, and leaves the key out
  when the run failed before it (a broken import, a model that could not be built). Absent,
  the run has no load report, so the controller keeps the runner's own error instead of
  calling it a load failure (hive.extensions.load_problem). An entry with "loaded": false
  needs an "error": an errored run's load report counts against the candidate only where a
  file failed with a reason.

How a runner is started, and what it must do with what it is given:

- Model access never travels in the runner's environment, where every shell command the agent
  runs could read it (`env`, `ps eww -p $PPID`). The adapter sends one JSON line on stdin
  through run_process_group(stdin_text=...): for a real provider, the per-trial gateway
  endpoint, `{"endpoint": {"base_url": "...", "api_key": "<per-trial token>"}}`, so the
  provider's own key never enters the trial at all. The runner reads stdin to EOF and closes
  it first thing, before importing any candidate code, passes the values explicitly to its
  client's constructor (`base_url=...`, `api_key=...`), and never sets them in os.environ.
- Shell and tool children get an explicit minimal environment, without the token.
  It must still carry hive.agents.process.TRIAL_MARKER_ENV from the runner's environment:
  the marker is how the controller finds and kills a command that left every process group
  it knows of (see hive/agents/process.py).

Neither framework computes dollars. For a real provider every call goes through the model
gateway, and the controller replaces this usage with the gateway's ledger, which is priced from
`[model] prices` (hive/evaluate.py). The token counts here stay as the runner's own account;
`prices`, when given, prices them, and without it their cost is 0.
"""

from __future__ import annotations

import json
import re
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hive.agents.process import ProcessOutcome
from hive.contracts import AGENT_ERROR_CLASSES, AgentResult, ExtensionLoad, LoadReport, ModelSpec, Usage
from hive.pricing import Prices, parse_prices

__all__ = ["RESULT_FILE", "TRACE_FILE", "Prices", "interpret_runner", "model_matches", "parse_prices"]

TRACE_FILE = "trace.jsonl"
RESULT_FILE = "result.json"

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
# Providers report dated snapshots of the requested model ("gpt-5-nano-2025-08-07").
_SNAPSHOT_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})")
_STDERR_TAIL_CHARS = 2000


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
    trace_dir: Path, outcome: ProcessOutcome, *, model: ModelSpec, prices: Prices | None = None, wall_clock_s: float
) -> AgentResult:
    calls = _model_calls(trace_dir / TRACE_FILE)
    tokens = Usage(**{field: sum(_count(call, field) for call in calls) for field in _TOKEN_FIELDS})
    cost = prices.cost(tokens) if prices is not None else 0.0
    usage = Usage(**{field: getattr(tokens, field) for field in _TOKEN_FIELDS}, cost_usd=cost)

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


def _load_report(result: Mapping[str, Any]) -> LoadReport | None:
    if "extensions" not in result:  # the runner never reached its load phase
        return None
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
