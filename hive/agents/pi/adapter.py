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
- Every assistant message carries a `provider` and `model`, filled in from the model Pi was
  configured with, never from the provider's response (pi-ai openai-responses.js). Extensions
  can call setModel / setThinkingLevel, and Pi's model resolver fuzzy-matches ids without a
  warning, so the adapter checks those fields, and the session file's model and thinking
  entries, against the experiment's ModelSpec (P3). A rewrite of the request payload itself
  (`before_provider_request`) would not show in those fields, so validation rejects any
  extension that hooks it.
- Usage comes from each assistant message's `usage`. Compaction's summary call is the
  exception: it emits no message and Pi keeps no usage for it, only a `compaction` entry in
  the session file, so the adapter adds an upper-bound estimate per entry (compaction_usage).
- Extensions (see extension_spec() and EXTENSIONS_API.md). Pi loads every `-e` file, and if
  any failed it prints `Error: Failed to load extension "<path>": ...` per failure and exits
  1 before the session header (main.js reportDiagnostics; core/extensions/loader.js). So
  once the header or `agent_start` is on stdout, every `-e` file loaded: that is the
  positive half of the load report, which Pi has nowhere else. An extension handler that
  throws mid-run prints `Extension error (<path>): ...` (print-mode.js) and the run goes on.
  Extension code that throws outside a handler crashes Node with a stack trace naming the
  file. A tool's execute() that throws, and a `tool_call` handler that throws, become error
  tool results the model sees, with nothing on stderr (runner.js emitToolCall has no catch).
- Bash tool commands run detached, in their own session. Pi kills them on SIGTERM, not on
  SIGKILL, so a timeout sends SIGTERM first. A command that daemonizes, or a process an
  extension starts, can outlive even that, so every run carries a trial marker and what it
  left behind is killed afterwards (hive.agents.process.TrialProcesses).
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

from hive.agents.process import TRIAL_MARKER_ENV, TrialProcesses, new_trial_marker
from hive.candidate import MANIFEST, Candidate, content_files
from hive.contracts import (
    EXTENSIONS_DIR,
    AgentResult,
    ExtensionLoad,
    ExtensionSpec,
    LifecyclePoint,
    LoadReport,
    ModelSpec,
    Task,
    Usage,
)
from hive.extensions import load_problem

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")
# Used when the experiment's ModelSpec leaves `thinking` unset.
DEFAULT_THINKING = "low"
DEFAULT_TOOLS = ("read", "bash", "edit", "write")
VALVE_NAMES = ("tools",)

SYSTEM_PROMPT_FILE = "SYSTEM.md"
APPEND_SYSTEM_PROMPT_FILE = "APPEND_SYSTEM.md"
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

# Compaction's summary call, under those pinned defaults (Pi 0.75.3 settings-manager.js:
# compaction.reserveTokens = 16384; compaction.js caps the history summary at 0.8 of it and a
# split turn's prefix summary at 0.5). Pi records no usage for the call, so these caps are
# what the adapter charges for its output: an upper bound, so the budget errs high (P12).
COMPACTION_RESERVE_TOKENS = 16384
COMPACTION_SUMMARY_MAX_OUTPUT_TOKENS = int(0.8 * COMPACTION_RESERVE_TOKENS)
COMPACTION_TURN_PREFIX_MAX_OUTPUT_TOKENS = int(0.5 * COMPACTION_RESERVE_TOKENS)
# compaction.js joins a split turn's two summaries with this heading.
SPLIT_TURN_SUMMARY_MARKER = "**Turn Context (split turn):**"

# Pi's own extension diagnostics on stderr (see module doc).
LOAD_ERROR_LINE = re.compile(r'Failed to load extension "([^"]+)": ?(.*)$')
DIAGNOSTIC_START = re.compile(r"^(?:Error|Warning): ")
RUNTIME_ERROR_PREFIX = "Extension error ("
RUNTIME_ERROR_LINE = re.compile(r"^Extension error \((.+?)\): ?(.*)$")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
QUOTA_PATTERN = re.compile(r"\b429\b|rate[ _-]?limit|too many requests|quota|billing", re.IGNORECASE)
COMPLETED_STOP_REASONS = ("stop", "length")

VERSION_CHECK_TIMEOUT_S = 30
# How long Pi gets after SIGTERM to kill its detached bash commands before the SIGKILL.
TERM_GRACE_S = 5.0
# After killing the process group, how long to wait for the pipes to drain.
KILL_DRAIN_TIMEOUT_S = 10
STDERR_TAIL_CHARS = 2000

VALVES_DOC = Path(__file__).with_name("VALVES.md")
EXTENSIONS_API_DOC = Path(__file__).with_name("EXTENSIONS_API.md")


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


# --- extensions ---------------------------------------------------------------------------
#
# The catalog below maps HIVE's neutral lifecycle points onto Pi 0.75.3's extension API. Every
# entry was checked against the installed dist/core/extensions/{types.d.ts,runner.js,loader.js},
# dist/core/sdk.js and pi-agent-core agent-loop.js, and the failure rows by running Pi offline
# with a fake provider. The caveats carry the semantics that differ from other frameworks,
# because a bare "supported" would mislead the overseer.

PI_VERSION = "0.75.3"

PI_LIFECYCLE = (
    LifecyclePoint(
        id="before_run",
        support="supported",
        native_api='pi.on("session_start"), pi.on("input"), pi.on("before_agent_start")',
        can_modify='input returns {action: "transform", text} to rewrite the task prompt; before_agent_start '
        "returns {message} to add a custom message the model sees; session_start only observes",
        caveats="before_agent_start fires once per prompt, and HIVE sends one prompt per task. "
        'input returning {action: "handled"} skips the model, so the run ends with no answer (agent_error).',
    ),
    LifecyclePoint(
        id="system_prompt",
        support="supported",
        native_api='pi.on("before_agent_start")',
        can_modify="return {systemPrompt} to replace the system prompt; event.systemPrompt is the current one "
        "and event.systemPromptOptions its structured parts",
        caveats="Handlers chain in load order, each seeing the previous result. For fixed text, editing "
        "SYSTEM.md or APPEND_SYSTEM.md is simpler and cheaper to get right.",
    ),
    LifecyclePoint(
        id="before_model_call",
        support="supported",
        native_api='pi.on("context")',
        can_modify="return {messages} to replace the message list sent on this model call",
        caveats="event.messages is a deep copy and the edit applies to this one call only: it is not saved, "
        "so a handler must re-apply it on every call. Handlers chain.",
    ),
    LifecyclePoint(
        id="after_model_call",
        support="partial",
        native_api='pi.on("message_end"), pi.on("after_provider_response")',
        can_modify="message_end returns {message} to replace the finished message (same role only); "
        "after_provider_response observes the HTTP status and headers",
        caveats="There is no retry hook: an extension cannot have a model call redone. message_end fires for "
        "user, assistant and tool-result messages alike; check event.message.role.",
    ),
    LifecyclePoint(
        id="wrap_model_call",
        support="unsupported",
        native_api='pi.on("before_provider_request")',
        can_modify="nothing: validation rejects an extension that hooks before_provider_request",
        caveats="There is no wrapper that receives the model call itself. before_provider_request can replace "
        "the serialized payload, model included, and Pi would still report the configured model (P3), so "
        "HIVE forbids it. Use context (before_model_call) to change the messages a call sends.",
    ),
    LifecyclePoint(
        id="before_tool_call",
        support="supported",
        native_api='pi.on("tool_call")',
        can_modify="return {block: true, reason} to cancel the call; rewrite its arguments by mutating "
        "event.input in place (Pi does not re-validate them)",
        caveats="Handlers run in load order and stop at the first block; later handlers see earlier "
        "mutations. A throw inside a tool_call handler is not reported as an extension error: Pi turns it "
        "into an error tool result the model sees, and the trial is not marked.",
    ),
    LifecyclePoint(
        id="after_tool_call",
        support="supported",
        native_api='pi.on("tool_result")',
        can_modify="return {content, details, isError} to replace the tool result",
        caveats="Handlers chain in load order, each seeing the previous result.",
    ),
    LifecyclePoint(
        id="register_tool",
        support="supported",
        native_api="pi.registerTool(ToolDefinition)",
        can_modify="add a tool: name, label, description, TypeBox parameters and execute()",
        caveats="The tool is active only if its name is also in [valves].tools; validation rejects a "
        'registerTool({ name: "..." }) string literal missing from it. A tool named like a built-in '
        "replaces the built-in. A throw in execute() becomes an error tool result the model sees.",
    ),
    LifecyclePoint(
        id="compaction",
        support="supported",
        native_api='pi.on("session_before_compact"), pi.on("session_compact"), ctx.compact()',
        can_modify="session_before_compact returns {cancel: true}, or {compaction: {summary, firstKeptEntryId, "
        "tokensBefore}} to supply your own summary; session_compact observes the result",
        caveats="Fires only when the context grows past Pi's threshold (Pi's defaults; the experiment pins them), "
        "or when an extension calls ctx.compact(). Pi's own summary is a model call with no usage record, so "
        "HIVE charges an upper-bound estimate for it; supplying {compaction} from session_before_compact "
        "makes no call and costs nothing.",
    ),
    LifecyclePoint(
        id="after_run",
        support="partial",
        native_api='pi.on("agent_end"), pi.on("turn_end") with pi.sendUserMessage',
        can_modify='agent_end only observes. To continue the run, call pi.sendUserMessage(text, {deliverAs: '
        '"followUp"}) from turn_end while the run is still going',
        caveats="No hook returns a continue signal. The follow-up route comes from Pi's loop code and is "
        "not verified in print mode; check the trace before relying on it.",
    ),
)

# Markdown: each entry carries its own code spans (hive.extensions renders them as bullets).
PI_FORBIDDEN = (
    "`pi.setModel`",
    "`pi.setThinkingLevel`",
    "`pi.registerProvider`",
    "`pi.unregisterProvider`",
    '`pi.on("before_provider_request")`: validation rejects any extension that hooks it, since a payload '
    "rewrite can change the model unseen (P3)",
)
FORBIDDEN_EVENTS = ("before_provider_request",)

PI_ENTRY_CONVENTION = (
    "default export the factory `(pi: ExtensionAPI) => void | Promise<void>`. Pi calls it once at "
    "startup, before the run, to register handlers and tools. The file is loaded with jiti: TypeScript "
    "is transpiled but never type-checked, and only Node built-ins, Pi's own packages and `typebox` "
    "can be imported."
)

# Pi events -> the neutral points they serve, for the static `registered` list. An event that
# serves two points (before_agent_start) is listed under both.
PI_EVENT_POINTS: Mapping[str, tuple[str, ...]] = {
    "session_start": ("before_run",),
    "input": ("before_run",),
    "before_agent_start": ("before_run", "system_prompt"),
    "context": ("before_model_call",),
    "message_end": ("after_model_call",),
    "after_provider_response": ("after_model_call",),
    "before_provider_request": ("wrap_model_call",),
    "tool_call": ("before_tool_call",),
    "tool_result": ("after_tool_call",),
    "session_before_compact": ("compaction",),
    "session_compact": ("compaction",),
    "agent_end": ("after_run",),
    "turn_end": ("after_run",),
}

# Matched against source whose comments and string contents are blanked (see code_views), so
# text inside a string never counts as a call. An optional type argument may follow `.on`.
ON_CALL = re.compile(r"""\.on\s*(?:<[^<>]*>\s*)?\(\s*(?=["'`])""")
EVENT_NAME = re.compile(r"[A-Za-z_]+")
REGISTER_TOOL_CALL = re.compile(r"\.registerTool\s*\(\s*")
NAME_KEY = re.compile(r"name\s*:\s*")
LITERAL_STRING = re.compile(r"""^(?:"([^"\\\n]*)"|'([^'\\\n]*)'|`([^`\\$]*)`)""")


def pi_extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        language="typescript",
        suffixes=EXTENSION_SUFFIXES,
        entry_convention=PI_ENTRY_CONVENTION,
        framework_version=PI_VERSION,
        lifecycle=PI_LIFECYCLE,
        forbidden=PI_FORBIDDEN,
        api_reference=EXTENSIONS_API_DOC.read_text(encoding="utf-8"),
    )


def registered_by(source: str) -> tuple[str, ...]:
    """What an extension's source hooks: neutral point ids, plus "tool:<name>" per literal tool name.

    Static and best effort: Pi records nothing about what an extension registered, and running
    the code to find out would mean running candidate code in the controller. Calls inside
    comments or string literals are ignored; calls built dynamically (a computed event name, a
    tool object passed by variable) are missed, though any registerTool call still counts as
    "register_tool".
    """
    _, skeleton = code_views(source)
    registered = {point for event in hooked_events(source) for point in PI_EVENT_POINTS.get(event, ())}
    if REGISTER_TOOL_CALL.search(skeleton):
        registered.add("register_tool")
    registered.update(f"tool:{name}" for name in literal_tool_names(source))
    return tuple(sorted(registered))


def hooked_events(source: str) -> tuple[str, ...]:
    """Event names passed as a string literal to `.on(...)`, in source order."""
    code, skeleton = code_views(source)
    events: list[str] = []
    for match in ON_CALL.finditer(skeleton):
        name = code[match.end() + 1 : string_end(code, match.end()) - 1]
        if EVENT_NAME.fullmatch(name) and name not in events:
            events.append(name)
    return tuple(events)


def literal_tool_names(source: str) -> tuple[str, ...]:
    """Names in `registerTool({ name: "<literal>", ... })` calls, in source order. A name that is a
    variable, an expression or an interpolated template can't be known statically and is skipped."""
    code, skeleton = code_views(source)
    names: list[str] = []
    for match in REGISTER_TOOL_CALL.finditer(skeleton):
        if code.startswith("{", match.end()):
            name = object_literal_name(code, match.end())
            if name is not None and name not in names:
                names.append(name)
    return tuple(names)


def object_literal_name(code: str, start: int) -> str | None:
    """The string-literal value of the top-level `name` key of the object literal at code[start]."""
    depth = 0
    index = start
    while index < len(code):
        char = code[index]
        if char in "'\"`":
            index = string_end(code, index)
            continue
        if char in "{([":
            depth += 1
        elif char in "})]":
            depth -= 1
            if depth == 0:
                return None
        elif depth == 1 and (index == 0 or not (code[index - 1].isalnum() or code[index - 1] in "_$.")):
            # Only a key of this object: `parameters: Type.Object({ name: ... })` is one level deeper.
            key = NAME_KEY.match(code, index)
            if key is not None:
                value = LITERAL_STRING.match(code[key.end() :])
                if value is None:
                    return None
                return next(group for group in value.groups() if group is not None)
        index += 1
    return None


def string_end(code: str, start: int) -> int:
    """Index just past the string literal opening at code[start] (or the end of the code)."""
    quote = code[start]
    index = start + 1
    while index < len(code):
        if code[index] == "\\":
            index += 2
            continue
        if code[index] == quote:
            return index + 1
        index += 1
    return len(code)


def code_views(source: str) -> tuple[str, str]:
    """Two views of the source, each the same length as it, so an offset in one is the same
    place in the other: `code` has comments blanked, and `skeleton` has string-literal
    contents blanked too (quotes kept). Search the skeleton for calls; read literals from code."""
    code: list[str] = []
    skeleton: list[str] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char in "'\"`":
            end = string_end(source, index)
            literal = source[index:end]
            code.append(literal)
            closed = end - index >= 2 and literal[-1] == char
            inner = literal[1:-1] if closed else literal[1:]
            skeleton.append(char + blank(inner) + (char if closed else ""))
            index = end
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index)
            end = len(source) if newline == -1 else newline
        elif source.startswith("/*", index):
            close = source.find("*/", index + 2)
            end = len(source) if close == -1 else close + 2
        else:
            code.append(char)
            skeleton.append(char)
            index += 1
            continue
        blanked = blank(source[index:end])
        code.append(blanked)
        skeleton.append(blanked)
        index = end
    return "".join(code), "".join(skeleton)


def blank(text: str) -> str:
    """Spaces in place of text, newlines kept."""
    return re.sub(r"[^\n]", " ", text)


def string_literals(source: str) -> Iterator[str]:
    """The contents of every string literal outside comments."""
    code, skeleton = code_views(source)
    index = 0
    while index < len(skeleton):
        if skeleton[index] in "'\"`":
            end = string_end(code, index)
            yield code[index + 1 : end - 1]
            index = end
        else:
            index += 1


def forbidden_hook_problems(extensions: Sequence[Path], root: Path) -> list[str]:
    """Extensions that hook an event HIVE forbids (P3). Static: an `.on("<event>")` call, or the
    event name as a whole string literal (so a constant passed to `.on` is caught too). A name
    assembled at runtime is not caught; the guide forbids it all the same."""
    problems = []
    for path in extensions:
        source = path.read_text(encoding="utf-8", errors="replace")
        named = set(hooked_events(source)) | set(string_literals(source))
        for event in FORBIDDEN_EVENTS:
            if event in named:
                problems.append(
                    f"'{path.relative_to(root).as_posix()}' hooks {event}, which is forbidden: it can rewrite the "
                    "request payload's model, and Pi would still report the configured model (P3)"
                )
    return problems


def extension_tool_problems(extensions: Sequence[Path], root: Path, valves: Mapping[str, Any]) -> list[str]:
    """Pi silently leaves out an extension tool whose name is not in --tools; catch the cases
    where the name is a string literal. A name built at runtime can't be checked (VALVES.md)."""
    tools = valves.get("tools", list(DEFAULT_TOOLS))
    if not isinstance(tools, list):
        return []  # valve_problems already reports it
    allowed = {tool for tool in tools if isinstance(tool, str)}
    problems = []
    for path in extensions:
        for name in literal_tool_names(path.read_text(encoding="utf-8", errors="replace")):
            if name not in allowed:
                problems.append(
                    f"'{path.relative_to(root).as_posix()}' registers tool '{name}', which is not in "
                    "[valves].tools, so Pi would leave it inactive; add it to tools or remove the registration"
                )
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
            layout = read_layout(candidate.root)
            problems.extend(layout.problems)
            problems.extend(extension_tool_problems(layout.extensions, candidate.root, candidate.valves))
            problems.extend(forbidden_hook_problems(layout.extensions, candidate.root))
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
        extensions = tuple(
            ExtensionFile(
                relative=path.relative_to(candidate.root).as_posix(),
                path=path,
                registered=registered_by(path.read_text(encoding="utf-8", errors="replace")),
            )
            for path in layout.extensions
        )
        for extension in extensions:
            options += ["-e", str(extension.path)]
        for skill in layout.skills:
            options += ["--skill", str(skill)]
        return PiAgent(
            pi_bin=resolved["pi_bin"],
            options=tuple(options),
            timeout_s=float(resolved["timeout_s"]),
            model=model,
            extensions=extensions,
            # Read now, from the environment HIVE started in, not from whatever a run changes.
            user_agent_dir=user_agent_dir(),
        )

    def describe_valves(self) -> str:
        return VALVES_DOC.read_text(encoding="utf-8")

    def extension_spec(self) -> ExtensionSpec:
        return pi_extension_spec()


@dataclass(frozen=True)
class ExtensionFile:
    """One candidate extension as a run needs it."""

    relative: str  # candidate-relative, as the load report names it: "extensions/guard.ts"
    path: Path  # absolute, as passed with -e and as Pi names it on stderr
    registered: tuple[str, ...]  # static, from registered_by(); read once at build time


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
    extensions: tuple[ExtensionFile, ...]  # the candidate's extension files, for the load report
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
            # Nothing loaded or failed: there is no load report (see load_report_of).
            message = f"could not start Pi ({self.pi_bin}): {outcome.launch_error}"
            return errored("agent_error", message, Usage(), wall_clock_s, None)
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
        marker = new_trial_marker()
        env = {**env, TRIAL_MARKER_ENV: marker}
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
        # What a daemonizing bash command or an extension's own child leaves behind, outside
        # every group Pi or we would signal (see module doc).
        tracked = TrialProcesses(process.pid, marker)
        tracked.start()

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
            tracked.kill_all()


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


def errored(
    error_class: str, error: str, usage: Usage, wall_clock_s: float, load_report: LoadReport | None = None
) -> AgentResult:
    return AgentResult(
        status="errored",
        final_text="",
        usage=usage,
        wall_clock_s=wall_clock_s,
        error=error,
        error_class=error_class,
        load_report=load_report,
    )


def interpret_run(
    raw: RawRun, *, model: ModelSpec, extensions: Sequence[ExtensionFile], timeout_s: float
) -> AgentResult:
    """Turn Pi's event stream, session file and stderr into one result.

    The checks run from "this run is not the experiment" to "the provider failed", so a run
    that broke P3 is never reported as an ordinary error. Every result carries a load report,
    except one from a Pi that stopped before loading anything (see load_report_of).
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
    input_rate, output_rate = token_rates(assistant_messages)
    usage += compaction_usage(raw.session_entries, input_rate=input_rate, output_rate=output_rate)
    wall_clock_s = raw.wall_clock_s
    tail = raw.stderr.strip()[-STDERR_TAIL_CHARS:] or "(no stderr)"
    report, report_is_evidence = load_report_of(raw, events, extensions)

    def failed(error_class: str, error: str) -> AgentResult:
        return errored(error_class, error, usage, wall_clock_s, report)

    mismatch = model_mismatch(events, assistant_messages, raw.session_entries, model)
    if mismatch is not None:
        return failed("model_mismatch", mismatch)

    if raw.timed_out:
        return failed("timeout", f"Pi exceeded the {timeout_s:g}s time limit")

    if report_is_evidence:
        # The controller's own rule (hive.extensions.load_problem), so both always agree: a file
        # that did not load is harness_load_error, a throw during the run is extension_error.
        problem = load_problem([extension.relative for extension in extensions], report)
        if problem is not None:
            return failed(*problem)

    if raw.returncode is not None and raw.returncode < 0:
        killed_by = signal_name(-raw.returncode)
        return failed("agent_error", f"Pi was killed by {killed_by} before the run ended. stderr: {tail}")

    if not assistant_messages:
        return failed("agent_error", f"Pi produced no assistant message. stderr: {tail}")

    last = assistant_messages[-1]
    stop_reason = last.get("stopReason")
    if stop_reason in ("error", "aborted"):
        message = str(last.get("errorMessage") or f"Pi stopped with stopReason={stop_reason}")
        return failed("quota" if QUOTA_PATTERN.search(message) else "provider_error", message)

    ended = any(event.get("type") == "agent_end" for event in events[assistant_indexes[-1] + 1 :])
    if not ended:
        message = f"Pi exited (code {raw.returncode}) with no agent_end after its last message. stderr: {tail}"
        return failed("agent_error", message)

    if stop_reason not in COMPLETED_STOP_REASONS:
        return failed("agent_error", f"Pi's run ended on stopReason={stop_reason!r}, not on the model's final reply")

    return AgentResult(
        status="completed", final_text=text_of(last), usage=usage, wall_clock_s=wall_clock_s, load_report=report
    )


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


def load_report_of(
    raw: RawRun, events: Sequence[Mapping[str, Any]], extensions: Sequence[ExtensionFile]
) -> tuple[LoadReport | None, bool]:
    """What this run loaded, and whether that is evidence about the extensions at all.

    Pi has no positive load report, but it exits at startup when any `-e` file fails to load,
    so a run that got as far as its session header loaded them all (see module doc). There is
    no evidence when Pi stopped before the run for a reason that names no extension (no model,
    a bad flag, a timeout before the header): then the report is None, as for a Pi that never
    started, and the run's failure is Pi's, not the candidate's. The controller keeps an
    errored run's own class when it has no report (hive.extensions.load_problem).
    """
    stderr = ANSI_ESCAPE.sub("", raw.stderr)
    by_path = extension_paths(extensions)
    load_errors = startup_load_errors(stderr)
    runtime_errors = list(runtime_errors_of(stderr, by_path))
    started = any(event.get("type") in ("session", "agent_start") for event in events)
    crashed_in = crash_named_files(stderr, raw.returncode, extensions)

    if load_errors:
        named = {by_path[path].relative: message for path, message in load_errors.items() if path in by_path}
        unattributed = "; ".join(message for path, message in load_errors.items() if path not in by_path)
        others = "Pi exits at startup when any extension fails to load, so this one never ran"
        if unattributed:
            others += f" (failures Pi reported: {unattributed})"
        loads = [
            ExtensionLoad(e.relative, loaded=False, error=f"failed to load: {named[e.relative]}")
            if e.relative in named
            else unknown_load(e, others)
            for e in extensions
        ]
        return LoadReport(tuple(loads), tuple(runtime_errors)), True
    if started:
        crash_detail = f"Pi crashed (exit code {raw.returncode}): {stderr.strip()[-STDERR_TAIL_CHARS:]}"
        runtime_errors += [f"{extension.relative}: {crash_detail}" for extension in crashed_in]
        loads = [ExtensionLoad(e.relative, loaded=True, registered=e.registered) for e in extensions]
        return LoadReport(tuple(loads), tuple(runtime_errors)), True
    if crashed_in:
        crash_detail = f"Pi crashed at startup (exit code {raw.returncode}): {stderr.strip()[-STDERR_TAIL_CHARS:]}"
        loads = [
            ExtensionLoad(e.relative, loaded=False, error=crash_detail)
            if e in crashed_in
            else unknown_load(e, "Pi crashed at startup in another extension")
            for e in extensions
        ]
        return LoadReport(tuple(loads), tuple(runtime_errors)), True
    return None, False


def unknown_load(extension: ExtensionFile, why: str) -> ExtensionLoad:
    return ExtensionLoad(extension.relative, loaded=False, error=f"unknown: {why}")


def extension_paths(extensions: Sequence[ExtensionFile]) -> dict[str, ExtensionFile]:
    """Every form in which Pi may name an extension: the path as given and its resolved form
    (/tmp vs /private/tmp), so attribution doesn't depend on how the path was spelled."""
    paths: dict[str, ExtensionFile] = {}
    for extension in extensions:
        paths[str(extension.path)] = extension
        paths.setdefault(str(extension.path.resolve()), extension)
    return paths


def startup_load_errors(stderr: str) -> dict[str, str]:
    """Pi's `Failed to load extension "<path>": <error>` diagnostics, by path. The loader's error
    (a syntax error with its code frame, say) can span lines; it runs to the next diagnostic."""
    errors: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stderr.splitlines():
        match = LOAD_ERROR_LINE.search(line)
        if match:
            current = errors.setdefault(match.group(1), [])
            current.append(match.group(2))
        elif DIAGNOSTIC_START.match(line):
            current = None
        elif current is not None:
            current.append(line)
    return {path: "\n".join(parts).strip()[:STDERR_TAIL_CHARS] for path, parts in errors.items()}


def runtime_errors_of(stderr: str, by_path: Mapping[str, ExtensionFile]) -> Iterator[str]:
    """`Extension error (<path>): <error>` lines, as "<candidate path>: <error>". A path that is
    no candidate file is kept as Pi printed it: the throw still happened."""
    for line in stderr.splitlines():
        if not line.startswith(RUNTIME_ERROR_PREFIX):
            continue
        # Match known paths first: a path may itself contain "): ".
        known = next((path for path in by_path if line.startswith(f"{RUNTIME_ERROR_PREFIX}{path}): ")), None)
        if known is not None:
            yield f"{by_path[known].relative}: {line[len(RUNTIME_ERROR_PREFIX) + len(known) + 3 :]}"
        elif (match := RUNTIME_ERROR_LINE.match(line)) is not None:
            yield f"{match.group(1)}: {match.group(2)}"


def crash_named_files(
    stderr: str, returncode: int | None, extensions: Sequence[ExtensionFile]
) -> list[ExtensionFile]:
    """Candidate extensions named in the stack trace of a Pi that exited abnormally: code that
    threw outside any handler Pi catches (a timer, an unawaited promise) and took Node down."""
    if returncode in (None, 0):
        return []
    # Pi's own extension-error lines name the file too, but they are handled throws, not crashes.
    trace = "\n".join(line for line in stderr.splitlines() if not line.startswith(RUNTIME_ERROR_PREFIX))
    return [e for e in extensions if str(e.path) in trace or str(e.path.resolve()) in trace]


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


def token_rates(assistant_messages: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """(USD per input token, USD per output token), from what Pi charged this run's own messages.
    Pi prices every call from the configured model, so its summary call costs the same rates."""
    totals = {"input": [0.0, 0.0], "output": [0.0, 0.0]}  # kind -> [tokens, dollars]
    for message in assistant_messages:
        usage = message.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if not isinstance(cost, dict):
            continue
        for kind, total in totals.items():
            tokens, dollars = number(usage.get(kind)), number(cost.get(kind))
            if tokens > 0:
                total[0] += tokens
                total[1] += dollars
    (input_tokens, input_dollars), (output_tokens, output_dollars) = totals["input"], totals["output"]
    input_rate = input_dollars / input_tokens if input_tokens else 0.0
    output_rate = output_dollars / output_tokens if output_tokens else 0.0
    return input_rate, output_rate


def compaction_usage(
    session_entries: Sequence[Mapping[str, Any]], *, input_rate: float, output_rate: float
) -> Usage:
    """An upper-bound estimate of compaction's summary calls, which Pi makes but never reports.

    Each `compaction` entry in the session file is one compaction. Unless an extension supplied
    the summary (`fromHook`), Pi made one summary call, or two for a split turn, over at most
    the context it recorded (`tokensBefore`), each capped at its maximum output. A compaction
    cut short (a timeout mid-summary) writes no entry, so its call is not counted.
    """
    total = Usage()
    for entry in session_entries:
        if entry.get("type") != "compaction" or entry.get("fromHook") is True:
            continue
        input_tokens = int(number(entry.get("tokensBefore")))
        output_tokens = COMPACTION_SUMMARY_MAX_OUTPUT_TOKENS
        if SPLIT_TURN_SUMMARY_MARKER in str(entry.get("summary", "")):
            output_tokens += COMPACTION_TURN_PREFIX_MAX_OUTPUT_TOKENS
        total += Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * input_rate + output_tokens * output_rate,
        )
    return total


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
