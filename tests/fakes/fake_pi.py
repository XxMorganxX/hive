#!/usr/bin/env python3
"""A stand-in for the `pi` executable, so adapter tests never call a real (paid) model.

It records how it was invoked (argv, cwd, stdin, its agent dir) as JSON to the path in
FAKE_PI_RECORD, then plays the scenario named in FAKE_PI_SCENARIO (default "success").
Its behaviour mirrors Pi 0.75.3 where the adapter depends on it:

- `--version` goes to stderr when stdin is not a TTY (main.js takes over stdout first).
- `--mode json` prints a session header line, then one JSON line per session event, and a
  normal run ends with `agent_end`.
- Every assistant message carries the `provider` and `model` it was sent to.
- The session file in `--session-dir` starts with a `model_change` and a
  `thinking_level_change` entry recording the model and level the run started with.
- Bash tool commands run detached (their own session); on SIGTERM Pi kills those groups
  and exits 143 (print-mode.js), while SIGKILL leaves them running.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

VERSION = os.environ.get("FAKE_PI_VERSION", "0.75.3")


def flag_value(flag: str, default: str = "") -> str:
    argv = sys.argv[1:]
    for index, arg in enumerate(argv[:-1]):
        if arg == flag:
            return argv[index + 1]
    return default


def flag_values(flag: str) -> list[str]:
    argv = sys.argv[1:]
    return [argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == flag]


PROVIDER, _, MODEL_ID = flag_value("--model", "openai/gpt-5-nano").partition("/")
THINKING = flag_value("--thinking", "medium")


def usage(input_tokens: int, output_tokens: int, cache_read: int, cache_write: int, cost: float) -> dict:
    return {
        "input": input_tokens,
        "output": output_tokens,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": input_tokens + output_tokens + cache_read + cache_write,
        "cost": {"input": cost / 2, "output": cost / 2, "cacheRead": 0, "cacheWrite": 0, "total": cost},
    }


def assistant(
    content: list,
    stop_reason: str,
    usage_block: dict | None,
    error_message: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "api": "openai-responses",
        "provider": provider or PROVIDER,
        "model": model or MODEL_ID,
        "stopReason": stop_reason,
    }
    if usage_block is not None:
        message["usage"] = usage_block
    if error_message is not None:
        message["errorMessage"] = error_message
    return {"type": "message_end", "message": message}


HEADER = {"type": "session", "version": 3, "id": "fake-session", "cwd": "."}
AGENT_START = {"type": "agent_start"}
AGENT_END = {"type": "agent_end", "messages": []}
USER = {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "task"}]}}
TOOL_RESULT = {"type": "message_end", "message": {"role": "toolResult", "content": [{"type": "text", "text": "ok"}]}}
TOOL_USE = assistant(
    [{"type": "text", "text": "Let me check."}, {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {}}],
    "toolUse",
    usage(100, 20, 5, 1, 0.01),
)


def final_answer(stop_reason: str = "stop", **model_fields: str) -> dict:
    return assistant(
        [
            {"type": "thinking", "thinking": "the answer is clear"},
            {"type": "text", "text": "ANSWER: "},
            {"type": "text", "text": "42"},
        ],
        stop_reason,
        usage(200, 30, 7, 2, 0.02),
        **model_fields,
    )


SUCCESS_EVENTS = [HEADER, AGENT_START, USER, TOOL_USE, TOOL_RESULT, final_answer(), AGENT_END]


def emit(events: list) -> None:
    for event in events:
        sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def write_session(extra_entries: list[dict] | None = None) -> None:
    """The session file a fresh Pi session writes, plus any entries an extension caused."""
    session_dir = flag_value("--session-dir")
    if not session_dir:
        return
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    entries = [
        HEADER,
        {"type": "model_change", "id": "e1", "provider": PROVIDER, "modelId": MODEL_ID},
        {"type": "thinking_level_change", "id": "e2", "thinkingLevel": THINKING},
        *(extra_entries or []),
    ]
    with open(Path(session_dir) / "2026-01-01T00-00-00_fake-session.jsonl", "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def snapshot_agent_dir() -> dict:
    """What Pi would see in its agent dir: symlink targets, and settings.json's text."""
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if not agent_dir or not os.path.isdir(agent_dir):
        return {}
    snapshot = {}
    for name in sorted(os.listdir(agent_dir)):
        path = os.path.join(agent_dir, name)
        entry: dict = {"symlink": os.readlink(path) if os.path.islink(path) else None}
        if name == "settings.json":
            with open(path) as handle:
                entry["text"] = handle.read()
        snapshot[name] = entry
    return snapshot


def record_invocation(stdin_text: str) -> None:
    path = os.environ.get("FAKE_PI_RECORD")
    if not path:
        return
    with open(path, "w") as handle:
        json.dump(
            {
                "argv": sys.argv[1:],
                "cwd": os.getcwd(),
                "stdin": stdin_text,
                "agent_dir": os.environ.get("PI_CODING_AGENT_DIR"),
                "agent_dir_entries": snapshot_agent_dir(),
                "pid": os.getpid(),
            },
            handle,
        )


def spawn_sleepers() -> None:
    """One child in Pi's own group and one detached child, as Pi's bash tool spawns them."""
    attached = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    detached = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)

    def on_sigterm(signum, frame):
        # Pi's print-mode handler: killTrackedDetachedChildren(), then exit 143.
        try:
            os.killpg(detached.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_sigterm)
    for env_name, pid in (("FAKE_PI_GRANDCHILD_PID", attached.pid), ("FAKE_PI_DETACHED_PID", detached.pid)):
        pid_path = os.environ.get(env_name)
        if pid_path:
            with open(pid_path, "w") as handle:
                handle.write(str(pid))


def extension_stack_trace() -> str:
    """What Node prints when extension code throws outside any handler Pi catches."""
    extension = flag_values("-e")[0]
    return (
        f"{extension}:2\n"
        '    setTimeout(() => { throw new Error("boom from candidate extension") }, 0)\n'
        "                       ^\n\n"
        "Error: boom from candidate extension\n"
        f"    at Timeout._onTimeout ({extension}:2:30)\n"
        "    at listOnTimeout (node:internal/timers:588:17)\n\n"
        "Node.js v24.4.1\n"
    )


def main() -> int:
    if sys.argv[1:] == ["--version"]:
        stream = sys.stdout if sys.stdin is not None and sys.stdin.isatty() else sys.stderr
        stream.write(VERSION + "\n")
        return 0

    # Pi reads piped stdin as part of the prompt, so the adapter must never leave it open.
    stdin_text = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
    record_invocation(stdin_text)

    scenario = os.environ.get("FAKE_PI_SCENARIO", "success")
    if scenario == "success":
        write_session()
        emit(SUCCESS_EVENTS)
    elif scenario == "length":
        emit([HEADER, AGENT_START, USER, final_answer("length"), AGENT_END])
    elif scenario == "provider_error":
        emit([HEADER, USER, assistant([], "error", usage(10, 0, 0, 0, 0.001), "500 Internal Server Error"), AGENT_END])
    elif scenario == "aborted":
        emit([HEADER, USER, assistant([], "aborted", None, "Request was aborted"), AGENT_END])
    elif scenario == "quota":
        quota = "429 You exceeded your current quota (insufficient_quota)"
        emit([HEADER, USER, assistant([], "error", None, quota), AGENT_END])
    elif scenario == "no_assistant":
        emit([HEADER, USER])
        sys.stderr.write("Error: No API key found for openai\n")
        return 1
    elif scenario == "malformed":
        sys.stdout.write('this is not json\n{"truncated": \n42\n\n')
        emit(SUCCESS_EVENTS)
    elif scenario == "model_switch":
        # An extension called pi.setModel(): later turns go to a different model.
        write_session([{"type": "model_change", "id": "e3", "provider": "openai", "modelId": "gpt-5.5-pro"}])
        emit([HEADER, AGENT_START, USER, TOOL_USE, TOOL_RESULT, final_answer(model="gpt-5.5-pro"), AGENT_END])
    elif scenario == "provider_switch":
        emit([HEADER, AGENT_START, USER, final_answer(provider="anthropic"), AGENT_END])
    elif scenario == "thinking_switch":
        # An extension called pi.setThinkingLevel() mid-run.
        write_session([{"type": "thinking_level_change", "id": "e3", "thinkingLevel": "xhigh"}])
        emit([HEADER, AGENT_START, USER, TOOL_USE, {"type": "thinking_level_changed", "level": "xhigh"}])
        emit([TOOL_RESULT, final_answer(), AGENT_END])
    elif scenario == "session_model_change":
        # A model switch that shows only in the session file (switched back before the next turn).
        write_session([{"type": "model_change", "id": "e3", "provider": "openai", "modelId": "gpt-5.5-pro"}])
        emit(SUCCESS_EVENTS)
    elif scenario == "crash_after_tool_use":
        # Node heap OOM, the OOM killer or a stray kill: Pi dies with no agent_end.
        emit([HEADER, AGENT_START, USER, TOOL_USE, {"type": "tool_execution_start", "toolCallId": "t1"}])
        sys.stderr.write("FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory\n")
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGKILL)
    elif scenario == "exit_without_agent_end":
        # An uncaught exception in print mode exits 1 after the last event it got out.
        emit([HEADER, AGENT_START, USER, final_answer()])
        sys.stderr.write("TypeError: Cannot read properties of undefined (reading 'x')\n")
        return 1
    elif scenario == "tool_use_last":
        # The run ended (agent_end) without the model's own final reply.
        emit([HEADER, AGENT_START, USER, TOOL_USE, TOOL_RESULT, AGENT_END])
    elif scenario == "extension_load_error":
        # Pi 0.75.3 reports a failed extension load as a startup diagnostic and exits 1.
        sys.stderr.write('Error: Failed to load extension "/c/extensions/bad.ts": Failed to load extension: boom\n')
        return 1
    elif scenario == "extension_runtime_error":
        # An extension handler that throws mid-run is reported and the run carries on without it.
        sys.stderr.write("Extension error (/c/extensions/bad.ts): handler threw\n")
        emit(SUCCESS_EVENTS)
    elif scenario == "extension_crash_at_start":
        sys.stderr.write(extension_stack_trace())
        return 1
    elif scenario == "extension_crash_mid_run":
        emit([HEADER, AGENT_START, USER, TOOL_USE])
        sys.stderr.write(extension_stack_trace())
        return 1
    elif scenario == "unrelated_crash":
        emit([HEADER, AGENT_START, USER, TOOL_USE])
        sys.stderr.write("Error: boom\n    at main (/opt/pi/dist/main.js:12:3)\n")
        return 1
    elif scenario == "timeout":
        spawn_sleepers()
        emit([HEADER])
        sys.stderr.write("partial stderr before hang\n")
        sys.stderr.flush()
        time.sleep(60)
    elif scenario == "timeout_ignores_sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        emit([HEADER])
        time.sleep(60)
    else:
        sys.stderr.write(f"fake_pi: unknown scenario {scenario!r}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
