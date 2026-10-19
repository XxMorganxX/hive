#!/usr/bin/env python3
"""A stand-in for the `pi` executable, so adapter tests never call a real (paid) model.

It records how it was invoked (argv, cwd, stdin, selected env) as JSON to the path in
FAKE_PI_RECORD, then plays the scenario named in FAKE_PI_SCENARIO (default "success").
Event shapes mirror Pi 0.75.3's `--mode json` output: a session header line, then one
JSON line per session event.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

VERSION = os.environ.get("FAKE_PI_VERSION", "0.75.3")


def usage(input_tokens: int, output_tokens: int, cache_read: int, cache_write: int, cost: float) -> dict:
    return {
        "input": input_tokens,
        "output": output_tokens,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": input_tokens + output_tokens + cache_read + cache_write,
        "cost": {"input": cost / 2, "output": cost / 2, "cacheRead": 0, "cacheWrite": 0, "total": cost},
    }


def assistant(content: list, stop_reason: str, usage_block: dict | None, error_message: str | None = None) -> dict:
    message = {"role": "assistant", "content": content, "stopReason": stop_reason}
    if usage_block is not None:
        message["usage"] = usage_block
    if error_message is not None:
        message["errorMessage"] = error_message
    return {"type": "message_end", "message": message}


HEADER = {"type": "session", "version": 3, "id": "fake-session", "cwd": "."}
USER = {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "task"}]}}
TOOL_RESULT = {"type": "message_end", "message": {"role": "toolResult", "content": [{"type": "text", "text": "ok"}]}}

SUCCESS_EVENTS = [
    HEADER,
    {"type": "agent_start"},
    USER,
    assistant(
        [{"type": "text", "text": "Let me check."}, {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {}}],
        "toolUse",
        usage(100, 20, 5, 1, 0.01),
    ),
    TOOL_RESULT,
    assistant(
        [
            {"type": "thinking", "thinking": "the answer is clear"},
            {"type": "text", "text": "ANSWER: "},
            {"type": "text", "text": "42"},
        ],
        "stop",
        usage(200, 30, 7, 2, 0.02),
    ),
    {"type": "agent_end"},
]


def emit(events: list) -> None:
    for event in events:
        sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def record_invocation(stdin_text: str) -> None:
    path = os.environ.get("FAKE_PI_RECORD")
    if not path:
        return
    with open(path, "w") as handle:
        json.dump({"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": stdin_text}, handle)


def main() -> int:
    if sys.argv[1:] == ["--version"]:
        print(VERSION)
        return 0

    # Pi reads piped stdin as part of the prompt, so the adapter must never leave it open.
    stdin_text = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
    record_invocation(stdin_text)

    scenario = os.environ.get("FAKE_PI_SCENARIO", "success")
    if scenario == "success":
        emit(SUCCESS_EVENTS)
    elif scenario == "provider_error":
        emit([HEADER, USER, assistant([], "error", usage(10, 0, 0, 0, 0.001), "500 Internal Server Error")])
    elif scenario == "aborted":
        emit([HEADER, USER, assistant([], "aborted", None, "Request was aborted")])
    elif scenario == "quota":
        emit([HEADER, USER, assistant([], "error", None, "429 You exceeded your current quota (insufficient_quota)")])
    elif scenario == "no_assistant":
        emit([HEADER, USER])
        sys.stderr.write("Error: No API key found for openai\n")
        return 1
    elif scenario == "malformed":
        sys.stdout.write("this is not json\n{\"truncated\": \n42\n\n")
        emit(SUCCESS_EVENTS)
    elif scenario == "extension_load_error":
        # Pi 0.75.3 reports a failed extension load as a startup diagnostic and exits 1.
        sys.stderr.write('Error: Failed to load extension "/c/extensions/bad.ts": Failed to load extension: boom\n')
        return 1
    elif scenario == "extension_runtime_error":
        # An extension handler that throws mid-run is reported and the run carries on without it.
        sys.stderr.write("Extension error (/c/extensions/bad.ts): handler threw\n")
        emit(SUCCESS_EVENTS)
    elif scenario == "timeout":
        # A grandchild in the same process group: the adapter must kill the whole group.
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pid_path = os.environ.get("FAKE_PI_GRANDCHILD_PID")
        if pid_path:
            with open(pid_path, "w") as handle:
                handle.write(str(grandchild.pid))
        emit([HEADER])
        sys.stderr.write("partial stderr before hang\n")
        sys.stderr.flush()
        time.sleep(60)
    else:
        sys.stderr.write(f"fake_pi: unknown scenario {scenario!r}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
