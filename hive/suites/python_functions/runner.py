"""Hidden-case runner for the python_functions suite. Runs in its own process, never in the grader's.

Usage: python -I runner.py <solution path> <cases path> <function name>

Agent-written code can loop forever, crash the interpreter or print anything, so the
grader runs it here, in a separate process it can time out and kill. This script is
stdlib only and imports nothing from HIVE, because `-I` keeps HIVE off `sys.path`.

It prints exactly one JSON line to the original stdout:
    {"passed": n, "total": m, "failures": ["wrong answer" | "<ExceptionType>", ...]}
plus an "error" key when the solution could not be loaded at all. It never prints case
inputs or expected values, and it is not a sandbox: it only keeps honest mistakes
(prints, crashes, wrong types) from corrupting the result.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any


def _strictly_equal(actual: Any, expected: Any) -> bool:
    # bool is a subclass of int in Python, so 1 == True; a function specified to return
    # a bool must return one, and one returning a count must not return True.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(map(_strictly_equal, actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_strictly_equal(actual[k], expected[k]) for k in actual)
    return actual == expected


def _matches(actual: Any, expected: Any) -> bool:
    # Expected values come from JSON, so compare in JSON terms: tuples become lists. The
    # round trip also strips subclasses, so a returned object can't fake equality via __eq__.
    try:
        normalized = json.loads(json.dumps(actual))
    except (TypeError, ValueError, RecursionError):
        return False
    return _strictly_equal(normalized, expected)


def _silence_stdout() -> Any:
    """Point fd 1 at /dev/null and return a private stream on the original stdout.

    Redirecting the file descriptor, not just sys.stdout, also silences print(),
    sys.__stdout__ and os.write(1, ...) from the solution, so only the result line
    reaches the grader.
    """
    sys.stdout.flush()
    result_stream = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return result_stream


def run(solution_path: str, cases: list[dict[str, Any]], function_name: str) -> dict[str, Any]:
    total = len(cases)
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    if spec is None or spec.loader is None:
        return {"passed": 0, "total": total, "failures": [], "error": "solution could not be loaded"}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as error:  # SystemExit and friends too: any import-time exit is a failed load
        return {"passed": 0, "total": total, "failures": [], "error": type(error).__name__}

    function = getattr(module, function_name, None)
    if not callable(function):
        return {"passed": 0, "total": total, "failures": [], "error": f"function {function_name} not defined"}

    passed = 0
    failures: list[str] = []
    for case in cases:
        try:
            actual = function(*case.get("args", []), **case.get("kwargs", {}))
        except BaseException as error:
            failures.append(type(error).__name__)
            continue
        if _matches(actual, case["expected"]):
            passed += 1
        else:
            failures.append("wrong answer")
    return {"passed": passed, "total": total, "failures": failures}


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: runner.py <solution path> <cases path> <function name>", file=sys.stderr)
        return 2
    solution_path, cases_path, function_name = argv[1:]
    with open(cases_path, encoding="utf-8") as cases_file:
        cases = json.load(cases_file)
    # The solution runs in this process: don't hand it the cases path or leave bytecode behind.
    sys.argv = [argv[0]]
    sys.dont_write_bytecode = True
    result_stream = _silence_stdout()
    result = run(solution_path, cases, function_name)
    result_stream.write(json.dumps(result) + "\n")
    result_stream.flush()
    return 0


if __name__ == "__main__":
    exit_code = main(sys.argv)
    # Skip interpreter shutdown: threads or atexit hooks the solution left behind must not
    # hold the process open after the result is written.
    sys.stderr.flush()
    os._exit(exit_code)
