#!/usr/bin/env python3
"""A stand-in for the Deep Agents venv's interpreter, so adapter tests need no framework and
never call a model.

The adapter runs exactly two kinds of command with its `python_path`:

- `<python> -I -c <code>`: the pinned-version check. The fake prints the version in
  `fake_version.txt` beside itself, or 0.7.28.
- `<python> -I <runner.py> <trace_dir>/run_spec.json`: one trial. The fake records how it
  was started (argv, cwd, environment, and the stdin it read to EOF, as the real runner does)
  in `<trace_dir>/fake_record.json`, then plays the
  scenario named by the task instruction ("SCENARIO <name>", default "completed"), writing
  trace.jsonl / result.json / child_pgids.txt in the runner protocol's formats.

Every invocation is also appended to `invocations.jsonl` beside the fake, so tests can count
version checks. The adapter passes the subprocess a scrubbed environment, so nothing can
reach the fake through environment variables: files beside it, the run spec and the
credentials line on stdin are the only channels, exactly as for the real runner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTED_MODEL = "gpt-5-nano-2025-08-07"


def log_invocation() -> None:
    with (HERE / "invocations.jsonl").open("a") as log:
        log.write(json.dumps({"argv": sys.argv[1:]}) + "\n")


def version_check() -> int:
    version_file = HERE / "fake_version.txt"
    print(version_file.read_text().strip() if version_file.is_file() else "0.7.28")
    return 0


def model_call(trace: Path, model: str = REPORTED_MODEL, input_tokens: int = 100, output_tokens: int = 20,
               cache_read: int = 50) -> None:  # fmt: skip
    line = {
        "type": "model_call",
        "model": model,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "cache_read_tokens": cache_read, "cache_write_tokens": 0},  # fmt: skip
    }
    with trace.open("a") as file:
        file.write(json.dumps(line) + "\n")


def loads(spec: dict, loaded: bool = True, error: str | None = None) -> list[dict]:
    return [
        {"path": entry["path"], "loaded": loaded, "error": error, "registered": ["before_run"] if loaded else []}
        for entry in spec.get("extensions", [])
    ]


def write_result(spec: dict, **fields) -> None:
    result = {"status": "completed", "final_text": "", "error": None, "error_class": None,
              "extensions": loads(spec), "runtime_errors": [], **fields}  # fmt: skip
    Path(spec["result_file"]).write_text(json.dumps(result))


def start_detached_sleeper(spec: dict) -> int:
    # Like a framework shell command: its own session, so a group kill of the runner misses it.
    sleeper = subprocess.Popen(["sleep", "300"], start_new_session=True)
    with open(spec["child_pgids_file"], "a") as pgids:
        pgids.write(f"{sleeper.pid}\n")
    (Path(spec["trace_dir"]) / "sleeper.pid").write_text(str(sleeper.pid))
    return sleeper.pid


def trial(spec_path: Path) -> int:
    stdin = sys.stdin.read()  # the credentials line; the tests' key is a placeholder
    spec = json.loads(spec_path.read_text())
    trace_dir = Path(spec["trace_dir"])
    (trace_dir / "fake_record.json").write_text(
        json.dumps({"argv": sys.argv, "cwd": os.getcwd(), "env": dict(os.environ), "stdin": stdin})
    )
    instruction = spec["instruction"]
    scenario = instruction.split()[1] if instruction.startswith("SCENARIO ") else "completed"
    trace = Path(spec["trace_file"])

    if scenario == "completed":
        model_call(trace)
        model_call(trace, input_tokens=200, output_tokens=30, cache_read=0)
        write_result(spec, final_text="ANSWER: 42")
    elif scenario == "load_error":
        write_result(spec, status="errored", error="extensions/a.py failed to load", error_class="harness_load_error",
                     extensions=loads(spec, loaded=False, error="SyntaxError: bad"))  # fmt: skip
    elif scenario == "extension_error":
        model_call(trace)
        write_result(spec, status="errored", error="extension code threw", error_class="extension_error",
                     runtime_errors=["extensions/a.py:3 in before_model: ValueError: boom"])  # fmt: skip
    elif scenario == "quota":
        write_result(spec, status="errored", error="insufficient_quota", error_class="quota")
    elif scenario == "wrong_model":
        model_call(trace, model="gpt-4o-2024-08-06")
        write_result(spec, final_text="ANSWER: 42")
    elif scenario == "bogus_class":
        write_result(spec, status="errored", error="odd", error_class="not_a_class")
    elif scenario == "crash":
        print("Traceback: the runner fell over", file=sys.stderr)
        return 3
    elif scenario == "hang":
        model_call(trace)
        start_detached_sleeper(spec)
        while True:
            time.sleep(1)
    elif scenario == "orphan":
        start_detached_sleeper(spec)
        model_call(trace)
        write_result(spec, final_text="done")
    else:
        print(f"unknown scenario {scenario!r}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    log_invocation()
    args = sys.argv[1:]
    if args[:2] == ["-I", "-c"]:
        return version_check()
    if len(args) == 3 and args[0] == "-I":
        return trial(Path(args[2]))
    print(f"unexpected invocation: {args}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
