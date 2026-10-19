#!/usr/bin/env python3
"""A stand-in for the Strands venv's interpreter, so adapter tests need no framework install.

The adapter talks to its venv interpreter in exactly two ways, and this fake answers both:

- `python -I -c <code>`: the version check. Prints the strands-agents version.
- `python -I <runner.py> <trace_dir>/run_spec.json`: one trial. Instead of running the real
  runner, it records how it was invoked (argv, cwd, environment) to
  `<trace_dir>/fake_record.json`, then plays a canned scenario: it writes trace.jsonl,
  result.json and child_pgids.txt the way the runner protocol says a runner does.

The adapter passes the subprocess an explicit, minimal environment, so the scenario can't
travel in an environment variable. A test makes a small wrapper script that calls this file
with `--fake-scenario NAME` and `--fake-version VERSION` before the adapter's own arguments;
those two are stripped before anything is recorded.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def take_option(argv: list[str], name: str, default: str) -> str:
    if name in argv:
        index = argv.index(name)
        value = argv[index + 1]
        del argv[index : index + 2]
        return value
    return default


ARGV = sys.argv[1:]
SCENARIO = take_option(ARGV, "--fake-scenario", "completed")
VERSION = take_option(ARGV, "--fake-version", "1.61.1")


def model_call(trace: Path, model: str, input_tokens: int = 100, output_tokens: int = 20, cache_read: int = 30) -> None:
    line = {
        "type": "model_call",
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": 0,
        },
    }
    with trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def loads(spec: dict, loaded: bool = True, error: str | None = None) -> list[dict]:
    return [
        {
            "path": extension["path"],
            "loaded": loaded,
            "error": error,
            "registered": ["before_tool_call", "tool:word_count"] if loaded else [],
        }
        for extension in spec.get("extensions", [])
    ]


def write_result(spec: dict, **fields) -> None:
    result = {
        "status": "errored",
        "final_text": "",
        "error": None,
        "error_class": None,
        "extensions": loads(spec),
        "runtime_errors": [],
    }
    result.update(fields)
    Path(spec["result_file"]).write_text(json.dumps(result), encoding="utf-8")


def run_trial(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    trace_dir = Path(spec["trace_dir"])
    record = {"argv": ARGV, "cwd": os.getcwd(), "env": dict(os.environ), "spec": spec}
    (trace_dir / "fake_record.json").write_text(json.dumps(record), encoding="utf-8")
    trace = Path(spec["trace_file"])
    model_id = spec["model"]["id"]

    if SCENARIO == "completed":
        model_call(trace, model_id)
        model_call(trace, model_id, input_tokens=200, output_tokens=40, cache_read=0)
        write_result(spec, status="completed", final_text="ANSWER: 42")
    elif SCENARIO == "completed_snapshot":
        model_call(trace, f"{model_id}-2025-08-07")
        write_result(spec, status="completed", final_text="ANSWER: 42")
    elif SCENARIO == "model_mismatch":
        model_call(trace, "gpt-4o")
        write_result(spec, status="completed", final_text="ANSWER: 42")
    elif SCENARIO == "harness_load_error":
        write_result(
            spec,
            error="extensions/guard.py: NameError: name 'oops' is not defined",
            error_class="harness_load_error",
            extensions=loads(spec, loaded=False, error="NameError: name 'oops' is not defined"),
        )
    elif SCENARIO == "extension_error":
        model_call(trace, model_id)
        write_result(
            spec,
            error="extension code threw during the run",
            error_class="extension_error",
            runtime_errors=["extensions/guard.py: BeforeToolCallEvent hook raised ValueError: boom"],
        )
    elif SCENARIO == "quota":
        write_result(spec, error="429 insufficient_quota: You exceeded your current quota", error_class="quota")
    elif SCENARIO == "provider_error":
        write_result(spec, error="InternalServerError: 500", error_class="provider_error")
    elif SCENARIO == "step_limit":
        model_call(trace, model_id)
        write_result(spec, error="the run hit its step limit (max_turns=2)", error_class="agent_error")
    elif SCENARIO == "unknown_error_class":
        write_result(spec, error="something odd", error_class="weird_class")
    elif SCENARIO == "missing_final_text":
        model_call(trace, model_id)
        write_result(spec, status="completed", final_text=None)
    elif SCENARIO == "crash":
        model_call(trace, model_id)
        print("Traceback (most recent call last):\nRuntimeError: runner bug", file=sys.stderr)
        return 1
    elif SCENARIO == "killed":
        model_call(trace, model_id)
        os.kill(os.getpid(), signal.SIGKILL)
    elif SCENARIO == "hang":
        model_call(trace, model_id)
        time.sleep(60)
    elif SCENARIO == "orphan":
        # A shell tool child in its own session, as the vended shell starts them; the runner
        # lists its pgid and exits, leaving the child for the controller to reap.
        child = subprocess.Popen(["sleep", "300"], start_new_session=True)
        (trace_dir / "orphan.pid").write_text(str(child.pid))
        with Path(spec["child_pgids_file"]).open("a") as handle:
            handle.write(f"{child.pid}\n")
        model_call(trace, model_id)
        write_result(spec, status="completed", final_text="done")
    else:
        print(f"unknown fake scenario {SCENARIO!r}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    args = [arg for arg in ARGV if arg != "-I"]
    if args[:1] == ["-c"]:
        if VERSION == "missing":
            print("importlib.metadata.PackageNotFoundError: No package metadata was found for strands-agents", file=sys.stderr)
            return 1
        print(VERSION)
        return 0
    return run_trial(Path(args[1]))


if __name__ == "__main__":
    sys.exit(main())
