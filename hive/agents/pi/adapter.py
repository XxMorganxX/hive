"""The Pi adapter: the only code in HIVE that knows Pi's CLI, file layout and event format.

A Pi candidate is a directory of Pi-native files (see VALVES.md). Each task runs as one
`pi --mode json -p` process. Behaviour below was checked against Pi 0.75.3's installed
source (dist/cli/args.js, dist/main.js, dist/core/resource-loader.js, dist/modes/print-mode.js):

- `--system-prompt` / `--append-system-prompt` read the value as a file when it is an
  existing path, else use it as literal text. Without them Pi discovers SYSTEM.md and
  APPEND_SYSTEM.md in `<cwd>/.pi/` and the user's global agent dir; an empty value stops
  that discovery and still gives the stock preamble, so we always pass one.
- `--no-extensions` / `--no-skills` drop discovered resources but keep explicit `-e` / `--skill`.
- `--tools` is an allowlist over built-in and extension tools alike.
- A positional argument starting with "-" or "@" is parsed as a flag or a file reference,
  and piped stdin is read into the prompt. So such instructions go on stdin, and stdin is
  otherwise closed (an inherited open stdin would make Pi wait forever).
- JSON mode exits 0 on provider errors, so the exit code is ignored (P8).
- A failed extension load prints `Failed to load extension "<path>": ...` and exits 1; an
  extension that throws mid-run prints `Extension error (<path>): ...` and the run goes on.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hive.candidate import MANIFEST, Candidate, content_files
from hive.contracts import AgentResult, ModelSpec, Task, Usage

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_THINKING = "low"
DEFAULT_TOOLS = ("read", "bash", "edit", "write")
VALVE_NAMES = ("thinking", "tools")

SYSTEM_PROMPT_FILE = "SYSTEM.md"
APPEND_SYSTEM_PROMPT_FILE = "APPEND_SYSTEM.md"
EXTENSIONS_DIR = "extensions"
SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
EXTENSION_SUFFIXES = (".ts", ".js")
ALLOWED_TOP_LEVEL = {MANIFEST, "README.md", SYSTEM_PROMPT_FILE, APPEND_SYSTEM_PROMPT_FILE, EXTENSIONS_DIR, SKILLS_DIR}

DEFAULT_SETTINGS: Mapping[str, Any] = {"pi_bin": "pi", "pi_version": None, "timeout_s": 600, "offline": True}

# Isolation from the user's global Pi config. The agent dir itself is left alone so auth works.
ISOLATION_FLAGS = ("--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files")

EXTENSION_ERROR_PATTERN = re.compile(r"Failed to load extension|Extension error \(")
QUOTA_PATTERN = re.compile(r"\b429\b|rate[ _-]?limit|too many requests|quota|billing", re.IGNORECASE)

VERSION_CHECK_TIMEOUT_S = 30
# After killing the process group, how long to wait for the pipes to drain.
KILL_DRAIN_TIMEOUT_S = 10
STDERR_TAIL_CHARS = 2000

VALVES_DOC = Path(__file__).with_name("VALVES.md")


class PiVersionMismatch(RuntimeError):
    """The installed Pi is not the version the experiment pinned (P13)."""


@dataclass(frozen=True)
class PiLayout:
    """The Pi-native files of a candidate, as absolute paths, plus anything that doesn't fit."""

    system_prompt: Path | None
    append_system_prompt: Path | None
    extensions: tuple[Path, ...]
    skills: tuple[Path, ...]
    problems: tuple[str, ...]


def read_layout(root: Path) -> PiLayout:
    """Classify every content file. Uses the candidate's own file list, so hashed == checked."""
    problems: list[str] = []
    extensions: list[Path] = []
    skill_files: dict[str, set[str]] = {}

    for relative in content_files(root):
        parts = PurePosixPath(relative.as_posix()).parts
        top = parts[0]
        if top not in ALLOWED_TOP_LEVEL:
            problems.append(f"unknown top-level entry '{top}' (allowed: {', '.join(sorted(ALLOWED_TOP_LEVEL))})")
        elif top in (EXTENSIONS_DIR, SKILLS_DIR) and len(parts) == 1:
            problems.append(f"'{top}' must be a directory")
        elif top not in (EXTENSIONS_DIR, SKILLS_DIR) and len(parts) > 1:
            problems.append(f"'{top}' must be a file, not a directory")
        elif top == EXTENSIONS_DIR:
            if len(parts) > 2:
                problems.append(f"'{relative.as_posix()}': extensions must be single files directly in extensions/")
            elif relative.suffix not in EXTENSION_SUFFIXES:
                problems.append(f"'{relative.as_posix()}': extensions must end in .ts or .js")
            else:
                extensions.append(root / relative)
        elif top == SKILLS_DIR:
            if len(parts) == 2:
                problems.append(f"'{relative.as_posix()}': skills/ may only contain skill directories")
            else:
                skill_files.setdefault(parts[1], set()).add("/".join(parts[2:]))

    # Empty skill directories hold no content files, so look for them directly.
    skills_root = root / SKILLS_DIR
    if skills_root.is_dir():
        for entry in skills_root.iterdir():
            if entry.is_dir() and not entry.name.startswith(".") and entry.name not in skill_files:
                skill_files[entry.name] = set()

    skills: list[Path] = []
    for name in sorted(skill_files):
        if SKILL_FILE in skill_files[name]:
            skills.append(skills_root / name)
        else:
            problems.append(f"'skills/{name}' has no {SKILL_FILE}")

    system_prompt = root / SYSTEM_PROMPT_FILE
    append_prompt = root / APPEND_SYSTEM_PROMPT_FILE
    if system_prompt.is_file() and not system_prompt.read_text(errors="replace").strip():
        problems.append(f"{SYSTEM_PROMPT_FILE} is empty; delete it to use Pi's stock system prompt")

    return PiLayout(
        system_prompt=system_prompt if system_prompt.is_file() else None,
        append_system_prompt=append_prompt if append_prompt.is_file() else None,
        extensions=tuple(sorted(extensions)),
        skills=tuple(skills),
        problems=tuple(problems),
    )


def valve_problems(valves: Mapping[str, Any]) -> list[str]:
    problems = [
        f"unknown valve '{name}' (allowed: {', '.join(VALVE_NAMES)})" for name in valves if name not in VALVE_NAMES
    ]
    thinking = valves.get("thinking", DEFAULT_THINKING)
    if not isinstance(thinking, str) or thinking not in THINKING_LEVELS:
        problems.append(f"valve 'thinking' must be one of {', '.join(THINKING_LEVELS)}; got {thinking!r}")
    tools = valves.get("tools", list(DEFAULT_TOOLS))
    if not isinstance(tools, list):
        problems.append(f"valve 'tools' must be a list of tool names; got {tools!r}")
    else:
        for tool in tools:
            # Tool names travel comma-joined in one --tools argument.
            if not isinstance(tool, str) or not tool.strip() or "," in tool or tool != tool.strip():
                problems.append(f"valve 'tools' has an invalid tool name {tool!r}")
    return problems


def resolve_settings(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_SETTINGS, **defaults, **overrides}
    unknown = sorted(set(merged) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"unknown Pi adapter settings: {', '.join(unknown)} (allowed: {', '.join(DEFAULT_SETTINGS)})")
    if not isinstance(merged["pi_bin"], str) or not merged["pi_bin"]:
        raise ValueError("Pi setting 'pi_bin' must be a non-empty string")
    if merged["pi_version"] is not None and not isinstance(merged["pi_version"], str):
        raise ValueError("Pi setting 'pi_version' must be a string or unset")
    timeout = merged["timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"Pi setting 'timeout_s' must be a positive number; got {timeout!r}")
    if not isinstance(merged["offline"], bool):
        raise ValueError("Pi setting 'offline' must be true or false")
    return merged


def check_pi_version(pi_bin: str, pinned: str) -> None:
    try:
        completed = subprocess.run(
            [pi_bin, "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=VERSION_CHECK_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PiVersionMismatch(f"could not run '{pi_bin} --version' to check the pinned version {pinned}: {error}")
    installed = completed.stdout.strip()
    if installed != pinned:
        raise PiVersionMismatch(f"Pi version {installed or '(none reported)'} is installed but {pinned} is pinned")


class PiAdapter:
    name = "pi"

    def __init__(self, settings: Mapping[str, Any] | None = None):
        self._settings = dict(settings or {})

    def validate(self, candidate: Candidate) -> list[str]:
        try:
            problems = []
            if candidate.agent != self.name:
                problems.append(f"agent is '{candidate.agent}', expected '{self.name}'")
            problems.extend(valve_problems(candidate.valves))
            problems.extend(read_layout(candidate.root).problems)
            return problems
        except Exception as error:  # validate must never raise; an unreadable candidate is a problem
            return [f"could not inspect candidate at {candidate.root}: {error}"]

    def build(self, candidate: Candidate, model: ModelSpec, settings: Mapping[str, Any]) -> PiAgent:
        problems = self.validate(candidate)
        if problems:
            raise ValueError(f"candidate {candidate.id} is not a valid Pi candidate: " + "; ".join(problems))
        resolved = resolve_settings(self._settings, settings)
        if resolved["pi_version"] is not None:
            check_pi_version(resolved["pi_bin"], resolved["pi_version"])

        layout = read_layout(candidate.root)
        tools = candidate.valves.get("tools", list(DEFAULT_TOOLS))
        options = [
            "--mode", "json", "-p",
            *ISOLATION_FLAGS,
            *(["--offline"] if resolved["offline"] else []),
            "--model", model.qualified,
            "--thinking", candidate.valves.get("thinking", DEFAULT_THINKING),
            *(["--tools", ",".join(tools)] if tools else ["--no-tools"]),
            # An empty value still blocks Pi's SYSTEM.md / APPEND_SYSTEM.md discovery (see module doc).
            "--system-prompt", str(layout.system_prompt or ""),
            "--append-system-prompt", str(layout.append_system_prompt or ""),
        ]  # fmt: skip
        for extension in layout.extensions:
            options += ["-e", str(extension)]
        for skill in layout.skills:
            options += ["--skill", str(skill)]
        return PiAgent(pi_bin=resolved["pi_bin"], options=tuple(options), timeout_s=float(resolved["timeout_s"]))

    def describe_valves(self) -> str:
        return VALVES_DOC.read_text()


@dataclass(frozen=True)
class PiAgent:
    pi_bin: str
    options: tuple[str, ...]  # everything except the per-run session dir and the instruction
    timeout_s: float

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        trace_dir.mkdir(parents=True, exist_ok=True)
        argv = [self.pi_bin, *self.options, "--session-dir", str(trace_dir / "session")]
        stdin_text: str | None = None
        if task.instruction.startswith(("-", "@")):
            stdin_text = task.instruction
        else:
            argv.append(task.instruction)

        started = time.monotonic()
        stdout, stderr, timed_out, launch_error = self._execute(argv, workdir, stdin_text)
        wall_clock_s = time.monotonic() - started

        (trace_dir / "events.jsonl").write_bytes(stdout)
        (trace_dir / "stderr.txt").write_bytes(stderr)

        if launch_error is not None:
            return errored("agent_error", f"could not start Pi ({self.pi_bin}): {launch_error}", Usage(), wall_clock_s)
        return interpret_run(stdout.decode(errors="replace"), stderr.decode(errors="replace"), timed_out, wall_clock_s, self.timeout_s)

    def _execute(self, argv: list[str], workdir: Path, stdin_text: str | None) -> tuple[bytes, bytes, bool, OSError | None]:
        try:
            # A new session makes Pi the leader of its own process group, so a timeout can
            # kill everything it spawned (bash tool commands included), not just Pi itself.
            process = subprocess.Popen(
                argv,
                cwd=workdir,
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            return b"", b"", False, error

        stdin_bytes = stdin_text.encode() if stdin_text is not None else None
        try:
            stdout, stderr = process.communicate(input=stdin_bytes, timeout=self.timeout_s)
            return stdout, stderr, False, None
        except subprocess.TimeoutExpired as expired:
            kill_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=KILL_DRAIN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Something outside the group still holds the pipes; keep what we have.
                stdout, stderr = expired.stdout or b"", expired.stderr or b""
            return stdout, stderr, True, None
        finally:
            if process.poll() is None:  # e.g. KeyboardInterrupt mid-run: never leak a Pi process
                kill_process_group(process)


def kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()


def errored(error_class: str, error: str, usage: Usage, wall_clock_s: float) -> AgentResult:
    return AgentResult(
        status="errored", final_text="", usage=usage, wall_clock_s=wall_clock_s, error=error, error_class=error_class
    )


def interpret_run(stdout: str, stderr: str, timed_out: bool, wall_clock_s: float, timeout_s: float) -> AgentResult:
    """Turn Pi's event stream and stderr into one result. The exit code is deliberately unused."""
    assistant_messages = [
        event["message"]
        for event in parse_events(stdout)
        if event.get("type") == "message_end"
        and isinstance(event.get("message"), dict)
        and event["message"].get("role") == "assistant"
    ]
    usage = sum((usage_of(message) for message in assistant_messages), Usage())

    if timed_out:
        return errored("timeout", f"Pi exceeded the {timeout_s:g}s time limit", usage, wall_clock_s)

    extension_errors = [line for line in stderr.splitlines() if EXTENSION_ERROR_PATTERN.search(line)]
    if extension_errors:
        # The candidate's own code failed, so this run is not the harness as specified.
        return errored("harness_load_error", "\n".join(extension_errors), usage, wall_clock_s)

    if not assistant_messages:
        tail = stderr.strip()[-STDERR_TAIL_CHARS:] or "(no stderr)"
        return errored("agent_error", f"Pi produced no assistant message. stderr: {tail}", usage, wall_clock_s)

    last = assistant_messages[-1]
    stop_reason = last.get("stopReason")
    if stop_reason in ("error", "aborted"):
        message = str(last.get("errorMessage") or f"Pi stopped with stopReason={stop_reason}")
        error_class = "quota" if QUOTA_PATTERN.search(message) else "provider_error"
        return errored(error_class, message, usage, wall_clock_s)

    return AgentResult(status="completed", final_text=text_of(last), usage=usage, wall_clock_s=wall_clock_s)


def parse_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated or non-JSON line must not cost the whole trial
        if isinstance(event, dict):
            events.append(event)
    return events


def text_of(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def usage_of(message: Mapping[str, Any]) -> Usage:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    cost = usage.get("cost")
    return Usage(
        input_tokens=int(number(usage.get("input"))),
        output_tokens=int(number(usage.get("output"))),
        cache_read_tokens=int(number(usage.get("cacheRead"))),
        cache_write_tokens=int(number(usage.get("cacheWrite"))),
        cost_usd=number(cost.get("total")) if isinstance(cost, dict) else 0.0,
    )


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
