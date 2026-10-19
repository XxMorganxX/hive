"""The Pi adapter: the only code in HIVE that knows Pi's CLI, file layout and event format.

A Pi candidate is a directory of Pi-native files (see VALVES.md). Each task runs as one
`pi --mode json -p` process. Behaviour below was checked against Pi 0.75.3's installed
source (dist/config.js, dist/cli/args.js, dist/main.js, dist/core/resource-loader.js,
dist/core/sdk.js, dist/core/skills.js, dist/modes/print-mode.js, pi-agent-core agent-loop.js):

- `--system-prompt` / `--append-system-prompt` read the value as a file when it is an
  existing path, else use it as literal text. Without them Pi discovers SYSTEM.md and
  APPEND_SYSTEM.md in `<cwd>/.pi/` and the user's global agent dir; an empty value stops
  that discovery and still gives the stock preamble, so we always pass one.
- `--no-extensions` / `--no-skills` drop discovered resources but keep explicit `-e` / `--skill`.
- The flags don't stop Pi reading settings.json (retry, compaction, shell, thinking budgets)
  and models.json (base URLs, costs) from its agent dir. Pi takes that dir from
  $PI_CODING_AGENT_DIR (config.js getAgentDir), so every run gets a fresh one: links to the
  user's credentials and fd/rg binaries, a pinned settings.json and no models.json.
- `--tools` is an allowlist over built-in and extension tools alike.
- A positional argument starting with "-" or "@" is parsed as a flag or a file reference,
  and piped stdin is read into the prompt. So such instructions go on stdin, and stdin is
  otherwise closed (an inherited open stdin would make Pi wait forever).
- With stdin not a TTY, Pi takes over stdout before printing anything, so `--version`
  arrives on stderr.
- JSON mode exits 0 on provider errors, so the exit code is not a verdict (P8). What is:
  a normal run always ends its stream with `agent_end`, after the last assistant message.
  A run killed by a signal, or one whose stream stops early, crashed.
- Every assistant message carries the `provider` and `model` it went to. Extensions can
  call setModel / setThinkingLevel, and Pi's model resolver fuzzy-matches ids without a
  warning, so the adapter checks those fields, and the session file's model and thinking
  entries, against the experiment's ModelSpec (P3).
- A failed extension load prints `Failed to load extension "<path>": ...` and exits 1; an
  extension handler that throws mid-run prints `Extension error (<path>): ...` and the run
  goes on. Extension code that throws outside a handler crashes Node with a stack trace
  naming the file. A tool's execute() that throws becomes an error tool result instead.
- Bash tool commands run detached, in their own session. Pi kills them on SIGTERM, not on
  SIGKILL, so a timeout sends SIGTERM first.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hive.candidate import MANIFEST, Candidate, content_files
from hive.contracts import AgentResult, ModelSpec, Task, Usage

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")
# Used when the experiment's ModelSpec leaves `thinking` unset.
DEFAULT_THINKING = "low"
DEFAULT_TOOLS = ("read", "bash", "edit", "write")
VALVE_NAMES = ("tools",)

SYSTEM_PROMPT_FILE = "SYSTEM.md"
APPEND_SYSTEM_PROMPT_FILE = "APPEND_SYSTEM.md"
EXTENSIONS_DIR = "extensions"
SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
EXTENSION_SUFFIXES = (".ts", ".js")
ALLOWED_TOP_LEVEL = {MANIFEST, "README.md", SYSTEM_PROMPT_FILE, APPEND_SYSTEM_PROMPT_FILE, EXTENSIONS_DIR, SKILLS_DIR}

DEFAULT_SETTINGS: Mapping[str, Any] = {"pi_bin": "pi", "pi_version": None, "timeout_s": 600, "offline": True}

# Isolation from resources Pi would discover in the user's global config and the project.
ISOLATION_FLAGS = ("--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files")

ENV_AGENT_DIR = "PI_CODING_AGENT_DIR"
# Linked, never copied, from the user's real agent dir: credentials are secrets, and bin/
# holds the fd/rg binaries Pi's grep/find tools need when it can't download them offline.
LINKED_AGENT_ENTRIES = ("auth.json", "bin")
# The settings.json every run gets: Pi's own defaults for the pinned version, whatever the
# user's settings say, so retry, compaction, shell and thinking-budget settings can't drift.
PINNED_SETTINGS: Mapping[str, Any] = {}

EXTENSION_ERROR_PATTERN = re.compile(r"Failed to load extension|Extension error \(")
QUOTA_PATTERN = re.compile(r"\b429\b|rate[ _-]?limit|too many requests|quota|billing", re.IGNORECASE)
COMPLETED_STOP_REASONS = ("stop", "length")

VERSION_CHECK_TIMEOUT_S = 30
# How long Pi gets after SIGTERM to kill its detached bash commands before the SIGKILL.
TERM_GRACE_S = 5.0
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
    skill_names: dict[str, str] = {}  # the name Pi will use -> the directory that claimed it first
    for name in sorted(skill_files):
        if SKILL_FILE not in skill_files[name]:
            problems.append(f"'skills/{name}' has no {SKILL_FILE}")
            continue
        skills.append(skills_root / name)
        problems.extend(skill_problems(skills_root / name, skill_names))

    system_prompt = root / SYSTEM_PROMPT_FILE
    append_prompt = root / APPEND_SYSTEM_PROMPT_FILE
    if system_prompt.is_file() and not system_prompt.read_text(encoding="utf-8", errors="replace").strip():
        problems.append(f"{SYSTEM_PROMPT_FILE} is empty; delete it to use Pi's stock system prompt")

    return PiLayout(
        system_prompt=system_prompt if system_prompt.is_file() else None,
        append_system_prompt=append_prompt if append_prompt.is_file() else None,
        extensions=tuple(sorted(extensions)),
        skills=tuple(skills),
        problems=tuple(problems),
    )


def skill_problems(skill_dir: Path, names_seen: dict[str, str]) -> list[str]:
    """Pi silently drops a skill with no frontmatter description, or whose name is taken
    (core/skills.js), and print mode never reports it. Catch both before paying for a run."""
    where = f"skills/{skill_dir.name}"
    frontmatter = parse_frontmatter((skill_dir / SKILL_FILE).read_text(encoding="utf-8", errors="replace"))
    problems = []
    if not frontmatter.get("description", "").strip():
        problems.append(
            f"'{where}/{SKILL_FILE}' has no 'description' in its frontmatter; Pi skips a skill without one"
        )
    name = frontmatter.get("name", "").strip() or skill_dir.name
    if name in names_seen:
        problems.append(f"'{where}': skill name '{name}' is already used by '{names_seen[name]}'; Pi loads only the first")
    else:
        names_seen[name] = where
    return problems


FRONTMATTER_KEY = re.compile(r"^([^\s#:][^:]*):(?:\s+(.*))?$")
BLOCK_SCALAR = re.compile(r"^[|>][+-]?[0-9]?(\s+#.*)?$")
YAML_NULLS = ("", "~", "null", "Null", "NULL")


def parse_frontmatter(text: str) -> dict[str, str]:
    """Top-level string fields of a `---` YAML frontmatter block, located as Pi locates it
    (utils/frontmatter.js). Stdlib has no YAML parser, so this reads the forms a skill header
    uses: plain, quoted, multi-line plain and `|` / `>` block scalars."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}
    end = normalized.find("\n---", 3)
    if end == -1:
        return {}
    lines = normalized[4:end].split("\n")

    fields: dict[str, str] = {}
    index = 0
    while index < len(lines):
        match = FRONTMATTER_KEY.match(lines[index])
        index += 1
        if not match:
            continue
        key, value = match.group(1).strip(), (match.group(2) or "").strip()
        # Indented lines that follow belong to this key (a block scalar or a wrapped plain value).
        continuation = []
        while index < len(lines) and (not lines[index].strip() or lines[index][:1] in (" ", "\t")):
            continuation.append(lines[index].strip())
            index += 1
        if BLOCK_SCALAR.match(value):
            fields[key] = "\n".join(continuation).strip()
        elif value[:1] in ("'", '"'):
            quoted = " ".join([value, *continuation]).strip()
            fields[key] = quoted[1:-1] if len(quoted) >= 2 and quoted[-1] == quoted[0] else quoted[1:]
        else:
            plain = " ".join([re.sub(r"\s+#.*$", "", value), *continuation]).strip()
            fields[key] = "" if plain in YAML_NULLS else plain
    return fields


def valve_problems(valves: Mapping[str, Any]) -> list[str]:
    problems = [
        f"unknown valve '{name}' (allowed: {', '.join(VALVE_NAMES)})" for name in valves if name not in VALVE_NAMES
    ]
    tools = valves.get("tools", list(DEFAULT_TOOLS))
    if not isinstance(tools, list):
        problems.append(f"valve 'tools' must be a list of tool names; got {tools!r}")
    else:
        for tool in tools:
            # Tool names travel comma-joined in one --tools argument.
            if not isinstance(tool, str) or not tool.strip() or "," in tool or tool != tool.strip():
                problems.append(f"valve 'tools' has an invalid tool name {tool!r}")
    return problems


def thinking_level(model: ModelSpec) -> str:
    """The reasoning effort is part of the frozen model spec (P3), never a candidate valve."""
    level = DEFAULT_THINKING if model.thinking is None else model.thinking
    if level not in THINKING_LEVELS:
        raise ValueError(f"model thinking level must be one of {', '.join(THINKING_LEVELS)}; got {level!r}")
    return level


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
    installed = reported_version(completed.stdout, completed.stderr)
    if installed != pinned:
        raise PiVersionMismatch(f"Pi version {installed or '(none reported)'} is installed but {pinned} is pinned")


def reported_version(stdout: str, stderr: str) -> str:
    # Pi prints the version to stdout on a TTY and to stderr otherwise (see module doc).
    for stream in (stdout, stderr):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return ""


def user_agent_dir() -> Path:
    """The user's real Pi agent dir, resolved the way Pi resolves it (config.js getAgentDir)."""
    configured = os.environ.get(ENV_AGENT_DIR)
    if configured:
        return Path(configured).expanduser().absolute()
    return Path.home() / ".pi" / "agent"


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
        thinking = thinking_level(model)
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
            "--thinking", thinking,
            *(["--tools", ",".join(tools)] if tools else ["--no-tools"]),
            # An empty value still blocks Pi's SYSTEM.md / APPEND_SYSTEM.md discovery (see module doc).
            "--system-prompt", str(layout.system_prompt or ""),
            "--append-system-prompt", str(layout.append_system_prompt or ""),
        ]  # fmt: skip
        for extension in layout.extensions:
            options += ["-e", str(extension)]
        for skill in layout.skills:
            options += ["--skill", str(skill)]
        return PiAgent(
            pi_bin=resolved["pi_bin"],
            options=tuple(options),
            timeout_s=float(resolved["timeout_s"]),
            model=model,
            extensions=layout.extensions,
            # Read now, from the environment HIVE started in, not from whatever a run changes.
            user_agent_dir=user_agent_dir(),
        )

    def describe_valves(self) -> str:
        return VALVES_DOC.read_text(encoding="utf-8")


@dataclass(frozen=True)
class RawRun:
    """What one Pi process left behind, before any interpretation."""

    stdout: str
    stderr: str
    session_entries: tuple[dict[str, Any], ...]  # from the session file(s) in the trace dir
    returncode: int | None  # None when the process never started or was killed on timeout
    timed_out: bool
    wall_clock_s: float


@dataclass(frozen=True)
class ProcessOutcome:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool
    launch_error: OSError | None


@dataclass(frozen=True)
class PiAgent:
    pi_bin: str
    options: tuple[str, ...]  # everything except the per-run session dir and the instruction
    timeout_s: float
    model: ModelSpec  # what every assistant message must report (P3)
    extensions: tuple[Path, ...]  # the candidate's extension files, to recognise their crashes
    user_agent_dir: Path  # the user's real Pi agent dir, source of the credential links

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        trace_dir.mkdir(parents=True, exist_ok=True)
        session_dir = trace_dir / "session"
        argv = [self.pi_bin, *self.options, "--session-dir", str(session_dir)]
        stdin_text: str | None = None
        if task.instruction.startswith(("-", "@")):
            stdin_text = task.instruction
        else:
            argv.append(task.instruction)

        started = time.monotonic()
        try:
            with isolated_agent_dir(self.user_agent_dir) as agent_dir:
                env = {**os.environ, ENV_AGENT_DIR: str(agent_dir)}
                outcome = self._execute(argv, workdir, stdin_text, env)
        except OSError as error:  # the temp agent dir could not be made; Pi never ran
            outcome = ProcessOutcome(b"", b"", None, False, error)
        wall_clock_s = time.monotonic() - started

        (trace_dir / "events.jsonl").write_bytes(outcome.stdout)
        (trace_dir / "stderr.txt").write_bytes(outcome.stderr)

        if outcome.launch_error is not None:
            message = f"could not start Pi ({self.pi_bin}): {outcome.launch_error}"
            return errored("agent_error", message, Usage(), wall_clock_s)
        raw = RawRun(
            stdout=outcome.stdout.decode(errors="replace"),
            stderr=outcome.stderr.decode(errors="replace"),
            session_entries=read_session_entries(session_dir),
            returncode=outcome.returncode,
            timed_out=outcome.timed_out,
            wall_clock_s=wall_clock_s,
        )
        return interpret_run(raw, model=self.model, extensions=self.extensions, timeout_s=self.timeout_s)

    def _execute(self, argv: list[str], workdir: Path, stdin_text: str | None, env: Mapping[str, str]) -> ProcessOutcome:
        try:
            # A new session makes Pi the leader of its own process group, so a timeout reaches
            # Pi and its plain children. Bash tool commands run detached in sessions of their
            # own; only Pi's SIGTERM handler reaches them (see stop_process_group).
            process = subprocess.Popen(
                argv,
                cwd=workdir,
                env=dict(env),
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            return ProcessOutcome(b"", b"", None, False, error)

        stdin_bytes = stdin_text.encode() if stdin_text is not None else None
        try:
            stdout, stderr = process.communicate(input=stdin_bytes, timeout=self.timeout_s)
            return ProcessOutcome(stdout, stderr, process.returncode, False, None)
        except subprocess.TimeoutExpired as expired:
            stop_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=KILL_DRAIN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Something outside the group still holds the pipes; keep what we have.
                stdout, stderr = expired.stdout or b"", expired.stderr or b""
            return ProcessOutcome(stdout, stderr, None, True, None)
        finally:
            if process.poll() is None:  # e.g. KeyboardInterrupt mid-run: never leak a Pi process
                stop_process_group(process)


@contextmanager
def isolated_agent_dir(real_agent_dir: Path) -> Iterator[Path]:
    """A fresh Pi agent dir outside the repo, removed afterwards. One per run, because Pi
    writes into it (lock files, settings it changes) and runs happen concurrently."""
    agent_dir = Path(tempfile.mkdtemp(prefix="hive-pi-agent-"))
    try:
        for name in LINKED_AGENT_ENTRIES:
            source = real_agent_dir / name
            if source.exists():
                (agent_dir / name).symlink_to(source)
        (agent_dir / "settings.json").write_text(json.dumps(PINNED_SETTINGS), encoding="utf-8")
        yield agent_dir
    finally:
        # rmtree unlinks the symlinks themselves; it never follows them into the real dir.
        shutil.rmtree(agent_dir, ignore_errors=True)


def stop_process_group(process: subprocess.Popen) -> None:
    """SIGTERM first, so Pi's handler kills its detached bash commands; then SIGKILL whatever
    is left in the group once the grace period is over."""
    signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    signal_group(process, signal.SIGKILL)
    process.wait()


def signal_group(process: subprocess.Popen, signum: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def read_session_entries(session_dir: Path) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for path in sorted(session_dir.glob("*.jsonl")) if session_dir.is_dir() else ():
        entries.extend(parse_events(path.read_text(encoding="utf-8", errors="replace")))
    return tuple(entries)


def errored(error_class: str, error: str, usage: Usage, wall_clock_s: float) -> AgentResult:
    return AgentResult(
        status="errored", final_text="", usage=usage, wall_clock_s=wall_clock_s, error=error, error_class=error_class
    )


def interpret_run(raw: RawRun, *, model: ModelSpec, extensions: Sequence[Path], timeout_s: float) -> AgentResult:
    """Turn Pi's event stream, session file and stderr into one result.

    The checks run from "this run is not the experiment" to "the provider failed", so a run
    that broke P3 is never reported as an ordinary error.
    """
    events = parse_events(raw.stdout)
    assistant_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("type") == "message_end"
        and isinstance(event.get("message"), dict)
        and event["message"].get("role") == "assistant"
    ]
    assistant_messages = [events[index]["message"] for index in assistant_indexes]
    usage = sum((usage_of(message) for message in assistant_messages), Usage())
    wall_clock_s = raw.wall_clock_s
    tail = raw.stderr.strip()[-STDERR_TAIL_CHARS:] or "(no stderr)"

    mismatch = model_mismatch(events, assistant_messages, raw.session_entries, model)
    if mismatch is not None:
        return errored("model_mismatch", mismatch, usage, wall_clock_s)

    if raw.timed_out:
        return errored("timeout", f"Pi exceeded the {timeout_s:g}s time limit", usage, wall_clock_s)

    extension_failure = extension_failure_of(raw.stderr, raw.returncode, extensions)
    if extension_failure is not None:
        # The candidate's own code failed, so this run is not the harness as specified.
        return errored("harness_load_error", extension_failure, usage, wall_clock_s)

    if raw.returncode is not None and raw.returncode < 0:
        message = f"Pi was killed by {signal_name(-raw.returncode)} before the run ended. stderr: {tail}"
        return errored("agent_error", message, usage, wall_clock_s)

    if not assistant_messages:
        return errored("agent_error", f"Pi produced no assistant message. stderr: {tail}", usage, wall_clock_s)

    last = assistant_messages[-1]
    stop_reason = last.get("stopReason")
    if stop_reason in ("error", "aborted"):
        message = str(last.get("errorMessage") or f"Pi stopped with stopReason={stop_reason}")
        error_class = "quota" if QUOTA_PATTERN.search(message) else "provider_error"
        return errored(error_class, message, usage, wall_clock_s)

    ended = any(event.get("type") == "agent_end" for event in events[assistant_indexes[-1] + 1 :])
    if not ended:
        message = f"Pi exited (code {raw.returncode}) with no agent_end after its last message. stderr: {tail}"
        return errored("agent_error", message, usage, wall_clock_s)

    if stop_reason not in COMPLETED_STOP_REASONS:
        message = f"Pi's run ended on stopReason={stop_reason!r}, not on the model's final reply"
        return errored("agent_error", message, usage, wall_clock_s)

    return AgentResult(status="completed", final_text=text_of(last), usage=usage, wall_clock_s=wall_clock_s)


def model_mismatch(
    events: Sequence[Mapping[str, Any]],
    assistant_messages: Sequence[Mapping[str, Any]],
    session_entries: Sequence[Mapping[str, Any]],
    model: ModelSpec,
) -> str | None:
    """Why this run did not use the experiment's model and thinking level, or None (P3)."""
    expected = (model.provider, model.id)
    for message in assistant_messages:
        used = (message.get("provider"), message.get("model"))
        if used != expected:
            return f"Pi answered with {used[0]}/{used[1]} but the experiment fixed {model.qualified}"

    for entry in session_entries:
        if entry.get("type") == "model_change" and (entry.get("provider"), entry.get("modelId")) != expected:
            return (
                f"the session switched to {entry.get('provider')}/{entry.get('modelId')} "
                f"but the experiment fixed {model.qualified}"
            )

    # A fresh session records its starting level once (sdk.js), clamped to what the model
    # supports; any later entry, or a live thinking_level_changed event, is a mid-run change.
    thinking_entries = [entry for entry in session_entries if entry.get("type") == "thinking_level_change"]
    changed_events = [event for event in events if event.get("type") == "thinking_level_changed"]
    if len(thinking_entries) > 1 or changed_events:
        level = changed_events[-1].get("level") if changed_events else thinking_entries[-1].get("thinkingLevel")
        return f"the thinking level was changed mid-run to {level!r}; the experiment fixes it"
    return None


def extension_failure_of(stderr: str, returncode: int | None, extensions: Sequence[Path]) -> str | None:
    """Pi's own extension error lines, or a crash whose stack trace names a candidate extension."""
    reported = [line for line in stderr.splitlines() if EXTENSION_ERROR_PATTERN.search(line)]
    if reported:
        return "\n".join(reported)
    if returncode in (None, 0) or not extensions:
        return None
    # The loader may report either the path as given or its resolved form (/tmp vs /private/tmp).
    paths = {str(path) for path in extensions} | {str(path.resolve()) for path in extensions}
    if any(path in stderr for path in paths):
        return f"a candidate extension crashed Pi (exit code {returncode}). stderr: {stderr.strip()[-STDERR_TAIL_CHARS:]}"
    return None


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


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
