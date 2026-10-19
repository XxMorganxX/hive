"""The Deep Agents adapter, controller side: the only HIVE code that knows a Deep Agents
candidate's layout and how a trial of it is launched.

A Deep Agents candidate is a directory of files this adapter owns (see VALVES.md): a
system prompt, extensions (LangChain `AgentMiddleware` and tools, in Python) and skills.
The framework never runs in the controller. Each trial is one subprocess:

    <python_path> -I runner.py <trace_dir>/run_spec.json      (cwd = the trial workdir)

`python_path` is the interpreter of the pinned venv `.venvs/deepagents` (see
scripts/setup-framework-venvs.sh), and runner.py imports only the stdlib and the framework.
Why a subprocess, not an import: candidate code must not run in the controller; a timeout
must be able to stop the run; the framework needs openai 3.x, which conflicts with the other
framework venvs; and deepagents keeps process-global profile registries an extension could
mutate. This module therefore imports only the stdlib and hive.

Behaviour below was checked against deepagents 0.7.28 / langchain 1.4.6 / langchain-core
1.6.12 / langchain-openai 1.7.2 (installed source and offline runs):

- The subprocess environment is built from nothing: PATH, LANG and a fresh HOME under the trace
  dir. LangSmith tracing turns on from LANGSMITH_* / LANGCHAIN_* variables and the OpenAI SDK
  honours OPENAI_BASE_URL, so an inherited environment could leak traces or redirect the model
  (P3, P10). `-I` keeps PYTHONPATH, user site-packages and the script directory off sys.path.
- Model access goes on the runner's stdin as one JSON line, never in its environment, argv or
  the run spec: a process's exec-time environment stays readable by its children
  (`ps eww $PPID`), so the agent's own shell could print it into the trace.
  - With a `ModelEndpoint` (the controller's model gateway; every real provider): the line is
    `{"endpoint": {"base_url": ..., "api_key": <per-trial token>}}` and the runner builds
    `ChatOpenAI(base_url=..., api_key=token)`. The real provider key never enters the trial,
    and this adapter never reads it. The gateway decides P3 (model and effort) and P12 (usage
    and cost) for every request, including any an extension sends past the runner.
  - Legacy, without an endpoint (provider "openai" only): `{"credentials": {"OPENAI_API_KEY":
    ...}}`, read from the controller's environment when the trial starts. The real key is then
    in the trial process, next to candidate code, and nothing outside it checks or bills the
    calls; kept only for running the adapter without a gateway.
- The verdict comes from the runner's result.json and trace.jsonl, read by
  hive.agents.runner_protocol, never from the exit code (P8). The trace's model check and token
  counts are a second line behind the gateway: the reported usage carries tokens but no dollars,
  because cost is priced once, from the gateway's ledger (`[model] prices`, hive.pricing).
- The framework starts shell commands in sessions of their own, so the runner lists their
  process groups in child_pgids.txt and the process helper kills them after every trial.
- validate() never runs candidate code: each extension is parsed with `ast` and checked for
  the entry convention and for APIs that would switch the model (P3). Importing an
  extension happens only in the runner, and a file that fails there is reported in the
  load report, which the controller checks (hive.extensions.load_problem).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hive.agents.process import run_process_group
from hive.agents.runner_protocol import RESULT_FILE, TRACE_FILE, interpret_runner
from hive.candidate import MANIFEST, Candidate, content_files
from hive.contracts import (
    EXTENSIONS_DIR,
    AgentResult,
    ExtensionSpec,
    LifecyclePoint,
    ModelEndpoint,
    ModelSpec,
    Task,
    Usage,
)
from hive.pricing import Prices

# The version every fact in this module, the runner, VALVES.md and EXTENSIONS_API.md was
# checked against (P13). requirements.txt pins the whole lock around it.
DEEPAGENTS_VERSION = "0.7.28"

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"
VALVES_DOC = HERE / "VALVES.md"
EXTENSIONS_API_DOC = HERE / "EXTENSIONS_API.md"
WORKDIR_PROMPT = HERE / "WORKDIR_PROMPT.md"

RUN_SPEC_FILE = "run_spec.json"
CHILD_PGIDS_FILE = "child_pgids.txt"
STDOUT_FILE = "runner.stdout"
STDERR_FILE = "runner.stderr"
HOME_DIR = "home"

SYSTEM_PROMPT_FILE = "SYSTEM.md"
SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
EXTENSION_SUFFIX = ".py"
ENTRY_FUNCTION = "extension"
ALLOWED_TOP_LEVEL = {MANIFEST, "README.md", SYSTEM_PROMPT_FILE, EXTENSIONS_DIR, SKILLS_DIR}

# deepagents' FilesystemMiddleware tools (FsToolName) and the subagent tool.
FILESYSTEM_TOOLS = ("ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute")
SUBAGENT_TOOL = "task"
VALVE_NAMES = ("tools", "general_purpose_subagent")
DEFAULT_TOOLS = FILESYSTEM_TOOLS
DEFAULT_GENERAL_PURPOSE_SUBAGENT = False

# ModelSpec.thinking is passed to OpenAI as the reasoning effort, unchanged.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_THINKING = "low"
# Provider -> the environment variable its credential comes from on the legacy path (a run
# without a gateway endpoint). "scripted" replays canned turns for tests and needs none; there
# are deliberately no Bedrock, Anthropic or Google paths, so a typo can never fall through to a
# default cloud provider.
PROVIDER_CREDENTIALS: Mapping[str, str | None] = {"openai": "OPENAI_API_KEY", "scripted": None}
# What the runner's trace is priced at: nothing. Dollars come only from the gateway's ledger,
# so the token counts this adapter reports are never mistaken for the authoritative spend (P12).
UNPRICED = Prices(input=0.0, output=0.0)

DEFAULT_SETTINGS: Mapping[str, Any] = {
    "python_path": None,  # required: the pinned venv's interpreter
    "deepagents_version": DEEPAGENTS_VERSION,
    "timeout_s": 600,  # wall clock per trial, enforced by killing the process
    "term_grace_s": 5.0,  # between SIGTERM and SIGKILL on a timeout
    "max_model_calls": 100,  # per trial: every request through the agent's chat model, subagent and retries included
    "recursion_limit": 1000,  # LangGraph super-steps: model, tool and middleware nodes all count
    "shell_timeout_s": 120,  # default per `execute` command
    "provider_timeout_s": 120,  # per model request
    "max_retries": 2,  # per model request, inside the OpenAI SDK; each retry is a gateway call
    "scripted_turns_path": None,  # provider "scripted" only: the turns the test model replays
}
_POSITIVE_NUMBERS = ("timeout_s", "provider_timeout_s")
# Settings that used to exist, with where their job went, so an old config fails with a pointer.
REMOVED_SETTINGS: Mapping[str, str] = {
    "prices": "cost now comes from the model gateway's ledger; move the table to [model] prices",
}
_POSITIVE_INTEGERS = ("max_model_calls", "recursion_limit", "shell_timeout_s")

VERSION_CHECK_TIMEOUT_S = 60
VERSION_CHECK_CODE = "import importlib.metadata as m; print(m.version('deepagents'))"

# Names an extension must not use: each builds or swaps a chat model or a provider profile,
# which would change the model the experiment fixed (P3). validate() rejects them statically,
# and since a static scan can be dodged, the runner also checks every request made through the
# agent's chat model and the controller every reply the runner traced. All of this is in-process
# detection, a second line: extension code runs in the runner's process and can reach the model
# object's token and provider client. The boundary is the gateway, which checks, counts and
# bills every request made with the trial's token, including one sent through that client
# directly (which the runner neither checks, counts nor traces).
FORBIDDEN_NAMES = (
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatAnthropicVertex", "ChatGoogleGenerativeAI",
    "ChatBedrock", "ChatBedrockConverse", "init_chat_model", "register_harness_profile",
    "register_provider_profile", "HarnessProfile", "HarnessProfileConfig", "ProviderProfile",
    # Shipped middleware that builds or calls a model of its own.
    "ModelFallbackMiddleware", "LLMToolSelectorMiddleware", "LLMToolEmulator", "SummarizationMiddleware",
    "create_summarization_middleware",
)  # fmt: skip
FORBIDDEN_OVERRIDE_KEYWORDS = ("model", "model_settings")

SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

ENTRY_CONVENTION = (
    "define a top-level, synchronous function `def extension():` taking no arguments and returning a "
    "list whose items are `langchain.agents.middleware.AgentMiddleware` instances (subclasses, or the "
    "objects the `@before_model` / `@wrap_tool_call` / ... decorators return) and/or "
    "`langchain_core.tools.BaseTool` instances (e.g. made with `@tool`). Middleware is applied to the "
    "main agent in list order (first = outermost), after the deepagents core stack; tools are added to "
    "the agent, and each tool's name must also be in the `tools` valve. Hooks must be sync methods: the "
    "runner uses the sync invoke path, so an async-only hook fails to load. Each file is imported on its "
    "own, so it cannot import another extension file."
)

LIFECYCLE: tuple[LifecyclePoint, ...] = (
    LifecyclePoint(
        id="before_run",
        support="supported",
        native_api="AgentMiddleware.before_agent(self, state, runtime) -> dict | None (or @before_agent)",
        can_modify="Return a state update, e.g. {'messages': [HumanMessage(...)]} (appended by the add_messages "
        "reducer). With @hook_config(can_jump_to=['end']) return {'jump_to': 'end'} to end before any model call.",
        caveats="Once per run, main agent only.",
    ),
    LifecyclePoint(
        id="system_prompt",
        support="supported",
        native_api="AgentMiddleware.wrap_model_call + request.override(system_message=SystemMessage(...)); "
        "@dynamic_prompt",
        can_modify="The system message of that one model call (request.system_prompt is its current text). "
        "Not persisted: rewrite it on every call.",
        caveats="The base prompt is SYSTEM.md plus HIVE's working-directory note. deepagents' filesystem "
        "section is already in it; SkillsMiddleware (inner to yours) appends the skills list after you.",
    ),
    LifecyclePoint(
        id="before_model_call",
        support="supported",
        native_api="AgentMiddleware.before_model(self, state, runtime) -> dict | None; or wrap_model_call + "
        "request.override(messages=[...])",
        can_modify="before_model: a state update (messages are appended; remove one with RemoveMessage(id=...)). "
        "wrap_model_call: the messages sent on that one call.",
        caveats="before_model edits persist in state for every later call; wrap_model_call edits apply to one "
        "call only.",
    ),
    LifecyclePoint(
        id="after_model_call",
        support="supported",
        native_api="AgentMiddleware.after_model(self, state, runtime) -> dict | None; or wrap_model_call "
        "inspecting handler(request)",
        can_modify="after_model: a state update; with @hook_config(can_jump_to=['model']) return "
        "{'jump_to': 'model'} to call the model again. wrap_model_call: inspect the ModelResponse, call "
        "handler again to retry, or return a different AIMessage.",
        caveats="A jump_to retry keeps the first reply in state; a retry inside wrap_model_call discards it. "
        "Every retry is a paid model call and counts toward the trial's model-call limit.",
    ),
    LifecyclePoint(
        id="wrap_model_call",
        support="supported",
        native_api="AgentMiddleware.wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | "
        "AIMessage (or @wrap_model_call)",
        can_modify="request.override(messages=, system_message=, tools=, tool_choice=) before calling "
        "handler; call handler zero or more times; short-circuit by returning an AIMessage.",
        caveats="Never override model= or model_settings=: the model and its reasoning effort are fixed (P3), "
        "and HIVE checks every request sent through the agent's chat model (each handler call) and every reply "
        "(model_mismatch). Each handler call is one model call toward the trial's model-call limit, which the "
        "main agent and the subagent share. The first middleware in your list is the outermost.",
    ),
    LifecyclePoint(
        id="before_tool_call",
        support="supported",
        native_api="AgentMiddleware.wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | "
        "Command (or @wrap_tool_call)",
        can_modify="request.tool_call (name, args, id): rewrite it with request.override(tool_call={...}); "
        "block the call by returning ToolMessage(content=..., tool_call_id=request.tool_call['id'], "
        "status='error') without calling handler.",
        caveats="An exception raised by your hook ends the run as extension_error. An exception raised by the "
        "tool itself becomes an error ToolMessage (HIVE adds ToolErrorMiddleware inside your middleware). "
        "A middleware named like a deepagents stack entry or a HIVE guard fails to load "
        "(harness_load_error), except the 'SummarizationMiddleware' slot.",
    ),
    LifecyclePoint(
        id="after_tool_call",
        support="supported",
        native_api="AgentMiddleware.wrap_tool_call: result = handler(request), then return a new ToolMessage",
        can_modify="The ToolMessage the model sees (content, status), or a langgraph Command.",
        caveats="Main agent only: the general-purpose subagent (off by default) runs its tools without your "
        "middleware.",
    ),
    LifecyclePoint(
        id="register_tool",
        support="supported",
        native_api="Return a BaseTool (langchain_core.tools.tool) from extension(), or set the `tools` class "
        "attribute of an AgentMiddleware",
        can_modify="Adds a tool the model can call, with the schema inferred from its signature and docstring.",
        caveats="Its name must be listed in the `tools` valve, or the extension fails to load "
        "(harness_load_error). A tool must not reuse a built-in tool's name.",
    ),
    LifecyclePoint(
        id="compaction",
        support="partial",
        native_api="A middleware whose `name` is 'SummarizationMiddleware' replaces deepagents' built-in "
        "summarization in place (e.g. a ContextEditingMiddleware subclass with `name` overridden; the only "
        "stack name an extension may take, and only one extension may take it); or trim messages in "
        "before_model / wrap_model_call",
        can_modify="How the conversation is shortened before model calls.",
        caveats="extension() gets no model and building one is forbidden (P3), so a model-written summary "
        "can't be produced by an extension; the built-in summarizer uses the experiment's model.",
    ),
    LifecyclePoint(
        id="after_run",
        support="supported",
        native_api="AgentMiddleware.after_agent(self, state, runtime) -> dict | None (or @after_agent)",
        can_modify="A state update; with @hook_config(can_jump_to=['model']) return {'jump_to': 'model'} "
        "(plus e.g. a new HumanMessage) to continue the run.",
        caveats="A run is completed only when it ends on a reply with no tool calls. Continuations count toward "
        "the trial's model-call limit.",
    ),
)

FORBIDDEN_APIS = (
    "ModelRequest.override(model=...) or setting request.model",
    "ModelRequest.override(model_settings=...) (it changes the reasoning effort and other model parameters)",
    "constructing any chat model: ChatOpenAI, init_chat_model, ChatAnthropic, ... (the model is fixed, P3)",
    "middleware that builds or calls a model of its own: ModelFallbackMiddleware, LLMToolSelectorMiddleware, "
    "LLMToolEmulator, SummarizationMiddleware (langchain's or deepagents')",
    "deepagents.register_harness_profile / register_provider_profile / HarnessProfile",
    "reading provider credentials (OPENAI_API_KEY, or the token the model object holds) or setting OPENAI_BASE_URL",
    "calling the provider client behind the model directly (request.model.client / root_client, openai.OpenAI): "
    "such calls skip the runner's checks and trace; the model gateway still checks and bills them, and refuses "
    "any other model or effort (model_mismatch), so they are forbidden and never free",
)


class DeepAgentsVersionMismatch(RuntimeError):
    """The venv's deepagents is not the version the experiment pinned (P13)."""


# --- validation ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepAgentsLayout:
    """The adapter-owned files of a candidate, as absolute paths, plus anything that doesn't fit."""

    system_prompt: Path | None
    extensions: tuple[Path, ...]
    skills_dir: Path | None
    problems: tuple[str, ...]


def read_layout(root: Path) -> DeepAgentsLayout:
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
            elif relative.suffix != EXTENSION_SUFFIX:
                problems.append(f"'{relative.as_posix()}': extensions must be Python files ending in .py")
            else:
                extensions.append(root / relative)
                problems.extend(extension_problems(root / relative, relative.as_posix()))
        elif top == SKILLS_DIR:
            if len(parts) == 2:
                problems.append(f"'{relative.as_posix()}': skills/ may only contain skill directories")
            else:
                skill_files.setdefault(parts[1], set()).add("/".join(parts[2:]))

    skills_root = root / SKILLS_DIR
    for name in sorted(skill_files):
        if SKILL_FILE not in skill_files[name]:
            problems.append(f"'skills/{name}' has no {SKILL_FILE}")
        else:
            problems.extend(skill_problems(skills_root / name))

    system_prompt = root / SYSTEM_PROMPT_FILE
    if system_prompt.is_file() and not system_prompt.read_text(encoding="utf-8", errors="replace").strip():
        problems.append(f"{SYSTEM_PROMPT_FILE} is empty; delete it to run without a system prompt of your own")
    return DeepAgentsLayout(
        system_prompt=system_prompt if system_prompt.is_file() else None,
        extensions=tuple(sorted(extensions)),
        skills_dir=skills_root if skill_files else None,
        problems=tuple(problems),
    )


def extension_problems(path: Path, where: str) -> list[str]:
    """Static checks of one extension file. Parses it; never imports or runs it."""
    try:
        tree = ast.parse(path.read_bytes(), filename=where)
    except (SyntaxError, ValueError) as error:  # ValueError: null bytes; SyntaxError covers bad encodings
        return [f"'{where}' is not valid Python (syntax error): {error}"]
    problems = []
    entries = [node for node in tree.body if getattr(node, "name", None) == ENTRY_FUNCTION]
    entry = entries[-1] if entries else None  # the last definition is the one the module ends up with
    if entry is None or not isinstance(entry, ast.FunctionDef | ast.AsyncFunctionDef):
        problems.append(f"'{where}' must define a top-level function `def extension():` (the entry convention)")
    elif isinstance(entry, ast.AsyncFunctionDef):
        problems.append(f"'{where}': `extension` must be a plain function, not async (the runner calls it synchronously)")
    else:
        arguments = entry.args
        required = len(arguments.posonlyargs) + len(arguments.args) - len(arguments.defaults)
        required += sum(1 for default in arguments.kw_defaults if default is None)
        if required > 0:
            problems.append(f"'{where}': `extension()` must take no arguments (it is called with none)")
    problems.extend(_forbidden_uses(tree, where))
    return problems


def _forbidden_uses(tree: ast.AST, where: str) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.add(node.attr)
        elif isinstance(node, ast.alias) and node.name.rpartition(".")[2] in FORBIDDEN_NAMES:
            found.add(node.name.rpartition(".")[2])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "override":
            for keyword in node.keywords:
                if keyword.arg in FORBIDDEN_OVERRIDE_KEYWORDS:
                    found.add(f"override({keyword.arg}=...)")
    return [
        f"'{where}' uses {name}, which would change the model the experiment fixed (P3)" for name in sorted(found)
    ]


def skill_problems(skill_dir: Path) -> list[str]:
    """deepagents' SkillsMiddleware skips a skill silently when its frontmatter lacks a name or
    a description, and warns when the name breaks the Agent Skills rules. Catch both up front."""
    where = f"skills/{skill_dir.name}"
    fields = _frontmatter_fields((skill_dir / SKILL_FILE).read_text(encoding="utf-8", errors="replace"))
    problems = []
    if not fields.get("description"):
        problems.append(f"'{where}/{SKILL_FILE}' needs a non-empty 'description' in its frontmatter")
    name = fields.get("name", "")
    if name != skill_dir.name or not SKILL_NAME.match(name) or len(name) > 64:
        problems.append(
            f"'{where}/{SKILL_FILE}': frontmatter 'name' must equal the directory name '{skill_dir.name}', "
            "in lowercase letters, digits and single hyphens (at most 64 characters)"
        )
    return problems


def _frontmatter_fields(text: str) -> dict[str, str]:
    """Top-level `key: value` lines of a leading `---` block. The stdlib has no YAML parser; a
    skill header needs only plain or quoted one-line name and description values."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, separator, value = line.partition(":")
        if separator and key and not key[0].isspace():
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            fields[key.strip()] = value.strip()
    return {}  # no closing marker: not frontmatter


def valve_problems(valves: Mapping[str, Any]) -> list[str]:
    problems = [
        f"unknown valve '{name}' (allowed: {', '.join(VALVE_NAMES)})" for name in valves if name not in VALVE_NAMES
    ]
    subagent = valves.get("general_purpose_subagent", DEFAULT_GENERAL_PURPOSE_SUBAGENT)
    if not isinstance(subagent, bool):
        problems.append(f"valve 'general_purpose_subagent' must be true or false; got {subagent!r}")
        subagent = False
    tools = valves.get("tools", list(DEFAULT_TOOLS))
    if not isinstance(tools, list):
        return [*problems, f"valve 'tools' must be a list of tool names; got {tools!r}"]
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, str) or not tool.strip() or tool != tool.strip():
            problems.append(f"valve 'tools' has an invalid tool name {tool!r}")
            continue
        if tool in seen:
            problems.append(f"valve 'tools' lists '{tool}' more than once")
        seen.add(tool)
    if SUBAGENT_TOOL in seen and not subagent:
        problems.append(f"valve 'tools' lists '{SUBAGENT_TOOL}', which exists only with general_purpose_subagent = true")
    if subagent and SUBAGENT_TOOL not in seen:
        problems.append(f"general_purpose_subagent = true needs '{SUBAGENT_TOOL}' in the 'tools' valve to be usable")
    return problems


# --- settings and build -----------------------------------------------------------------------


def resolve_settings(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_SETTINGS, **defaults, **overrides}
    removed = sorted(set(merged) & set(REMOVED_SETTINGS))
    if removed:
        raise ValueError("; ".join(f"'{name}' is no longer a Deep Agents setting: {REMOVED_SETTINGS[name]}"
                                   for name in removed))  # fmt: skip
    unknown = sorted(set(merged) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown Deep Agents adapter settings: {', '.join(unknown)} (allowed: {', '.join(DEFAULT_SETTINGS)})"
        )
    if not isinstance(merged["python_path"], str) or not merged["python_path"]:
        raise ValueError("Deep Agents setting 'python_path' (the pinned venv's interpreter) is required")
    if merged["deepagents_version"] != DEEPAGENTS_VERSION:
        raise ValueError(
            f"Deep Agents setting 'deepagents_version' is {merged['deepagents_version']!r}, but this adapter, its "
            f"runner and its extension reference were checked against {DEEPAGENTS_VERSION} only"
        )
    for name in _POSITIVE_NUMBERS:
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"Deep Agents setting '{name}' must be a positive number; got {value!r}")
    grace = merged["term_grace_s"]
    if isinstance(grace, bool) or not isinstance(grace, int | float) or grace < 0:
        raise ValueError(f"Deep Agents setting 'term_grace_s' must be a number >= 0; got {grace!r}")
    for name in _POSITIVE_INTEGERS:
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Deep Agents setting '{name}' must be a positive integer; got {value!r}")
    retries = merged["max_retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError(f"Deep Agents setting 'max_retries' must be an integer >= 0; got {retries!r}")
    turns = merged["scripted_turns_path"]
    if turns is not None and (not isinstance(turns, str) or not turns):
        raise ValueError("Deep Agents setting 'scripted_turns_path' must be a path string or unset")
    return merged


def reasoning_effort(model: ModelSpec) -> str:
    """The reasoning effort is part of the frozen model spec (P3), never a candidate valve."""
    level = DEFAULT_THINKING if model.thinking is None else model.thinking
    if level not in REASONING_EFFORTS:
        raise ValueError(f"model thinking level must be one of {', '.join(REASONING_EFFORTS)}; got {level!r}")
    return level


def check_deepagents_version(python_path: str, pinned: str) -> None:
    """Ask the venv which deepagents it has. Run once per build, with the same scrubbed env as a trial."""
    try:
        completed = subprocess.run(
            [python_path, "-I", "-c", VERSION_CHECK_CODE],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=VERSION_CHECK_TIMEOUT_S,
            env=_base_environment(home=None),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DeepAgentsVersionMismatch(f"could not run '{python_path}' to check the pinned deepagents {pinned}: {error}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    installed = lines[-1] if completed.returncode == 0 and lines else ""
    if installed != pinned:
        detail = installed or f"(none reported; {completed.stderr.strip()[-300:] or 'no output'})"
        # A venv's bin/python is a symlink; the interpreter it points to runs without the venv.
        hint = "" if (Path(python_path).parent.parent / "pyvenv.cfg").is_file() else (
            f" ({python_path} is not inside a venv: python_path must name the venv's own bin/python, "
            "not the interpreter that symlink points to)"
        )  # fmt: skip
        raise DeepAgentsVersionMismatch(
            f"deepagents {detail} is installed at {python_path} but {pinned} is pinned; "
            f"rebuild the venv with scripts/setup-framework-venvs.sh deepagents{hint}"
        )


def _base_environment(home: Path | None) -> dict[str, str]:
    # Built from nothing (see module doc): no tracing, cloud, proxy or Python variables leak in.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": os.environ.get("LANG", "C.UTF-8")}
    env["HOME"] = str(home) if home is not None else os.environ.get("HOME", "/")
    return env


class DeepAgentsAdapter:
    name = "deepagents"

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

    def build(self, candidate: Candidate, model: ModelSpec, settings: Mapping[str, Any]) -> DeepAgentsAgent:
        problems = self.validate(candidate)
        if problems:
            raise ValueError(f"candidate {candidate.id} is not a valid Deep Agents candidate: " + "; ".join(problems))
        if model.provider not in PROVIDER_CREDENTIALS:
            raise ValueError(
                f"Deep Agents adapter supports model providers {', '.join(PROVIDER_CREDENTIALS)}; got {model.provider!r}"
            )
        effort = reasoning_effort(model)
        resolved = resolve_settings(self._settings, settings)
        turns = resolved["scripted_turns_path"]
        if (model.provider == "scripted") != (turns is not None):
            raise ValueError("Deep Agents setting 'scripted_turns_path' is required with provider 'scripted' and "
                             "allowed only with it")  # fmt: skip
        # No credential is read here: with the gateway, the key never leaves it, and the legacy
        # path (no endpoint) reads it when a trial starts, so the agent object never holds it.
        check_deepagents_version(resolved["python_path"], resolved["deepagents_version"])

        layout = read_layout(candidate.root)
        valves = candidate.valves
        return DeepAgentsAgent(
            python_path=resolved["python_path"],
            candidate_root=candidate.root,
            system_prompt=layout.system_prompt.read_text(encoding="utf-8") if layout.system_prompt else None,
            extensions=tuple(
                (path.relative_to(candidate.root).as_posix(), str(path)) for path in layout.extensions
            ),
            skills_dir=layout.skills_dir,
            tools=tuple(valves.get("tools", list(DEFAULT_TOOLS))),
            general_purpose_subagent=valves.get("general_purpose_subagent", DEFAULT_GENERAL_PURPOSE_SUBAGENT),
            model=model,
            reasoning_effort=effort,
            limits={name: resolved[name] for name in
                    ("max_model_calls", "recursion_limit", "shell_timeout_s", "provider_timeout_s", "max_retries")},
            timeout_s=float(resolved["timeout_s"]),
            term_grace_s=float(resolved["term_grace_s"]),
            scripted_turns_file=str(Path(turns).expanduser().resolve()) if turns is not None else None,
            workdir_prompt=WORKDIR_PROMPT.read_text(encoding="utf-8"),
        )  # fmt: skip

    def describe_valves(self) -> str:
        return VALVES_DOC.read_text(encoding="utf-8")

    def extension_spec(self) -> ExtensionSpec:
        return ExtensionSpec(
            language="python",
            suffixes=(EXTENSION_SUFFIX,),
            entry_convention=ENTRY_CONVENTION,
            framework_version=DEEPAGENTS_VERSION,
            lifecycle=LIFECYCLE,
            forbidden=FORBIDDEN_APIS,
            api_reference=EXTENSIONS_API_DOC.read_text(encoding="utf-8"),
        )


# --- one trial ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepAgentsAgent:
    python_path: str
    candidate_root: Path
    system_prompt: str | None
    extensions: tuple[tuple[str, str], ...]  # (candidate-relative path, absolute file)
    skills_dir: Path | None
    tools: tuple[str, ...]
    general_purpose_subagent: bool
    model: ModelSpec
    reasoning_effort: str
    limits: Mapping[str, int | float]
    timeout_s: float
    term_grace_s: float
    scripted_turns_file: str | None
    workdir_prompt: str

    def __repr__(self) -> str:
        return f"DeepAgentsAgent(model={self.model.qualified!r}, candidate_root={str(self.candidate_root)!r})"

    def run(self, task: Task, workdir: Path, trace_dir: Path, endpoint: ModelEndpoint | None = None) -> AgentResult:
        """One trial. With `endpoint`, every model request goes to it with its per-trial token.

        Usage in the result is the runner's traced token count with no cost: the controller takes
        usage and cost from the gateway's ledger, which is the authority (P12).
        """
        started = time.monotonic()
        try:
            return self._run(task, workdir, trace_dir, endpoint, started)
        except Exception as error:  # run never raises (Agent contract); an adapter fault is this trial's error
            return AgentResult(status="errored", final_text="", usage=Usage(), wall_clock_s=time.monotonic() - started,
                               error=f"the Deep Agents adapter failed: {type(error).__name__}: {error}",
                               error_class="agent_error")  # fmt: skip

    def _run(self, task: Task, workdir: Path, trace_dir: Path, endpoint: ModelEndpoint | None,
             started: float) -> AgentResult:  # fmt: skip
        access = self.model_access(endpoint)
        trace_dir.mkdir(parents=True, exist_ok=True)
        home = trace_dir / HOME_DIR
        home.mkdir(exist_ok=True)
        spec_path = trace_dir / RUN_SPEC_FILE
        spec_path.write_text(json.dumps(self.run_spec(task, workdir.resolve(), trace_dir), indent=2) + "\n")
        outcome = run_process_group(
            [self.python_path, "-I", str(RUNNER), str(spec_path)],
            cwd=workdir,
            env=_base_environment(home=home),
            stdin_text=json.dumps(access) + "\n",
            timeout_s=self.timeout_s,
            stdout_path=trace_dir / STDOUT_FILE,
            stderr_path=trace_dir / STDERR_FILE,
            term_grace_s=self.term_grace_s,
            extra_pgids_file=trace_dir / CHILD_PGIDS_FILE,
        )
        return interpret_runner(trace_dir, outcome, model=self.model, prices=UNPRICED,
                                wall_clock_s=time.monotonic() - started)  # fmt: skip

    def model_access(self, endpoint: ModelEndpoint | None) -> dict[str, Any]:
        """The stdin line's payload: the gateway endpoint, or (legacy) the provider credential.

        The legacy key is read from the controller's environment here, when the trial starts, so
        it is never kept on this object; a missing key fails the trial before anything runs.
        """
        if endpoint is not None:
            return {"endpoint": {"base_url": endpoint.base_url, "api_key": endpoint.api_key}}
        name = PROVIDER_CREDENTIALS[self.model.provider]
        if name is None:
            return {"credentials": {}}
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"no model endpoint was given and {name} is not set for model provider "
                             f"{self.model.provider!r} (the legacy no-gateway path)")  # fmt: skip
        return {"credentials": {name: value}}

    def run_spec(self, task: Task, workdir: Path, trace_dir: Path) -> dict[str, Any]:
        """Everything the runner needs, as plain JSON. Never a key or token: those travel on stdin."""
        return {
            "instruction": task.instruction,
            "workdir": str(workdir),
            "trace_dir": str(trace_dir),
            "candidate_root": str(self.candidate_root),
            "system_prompt": self.system_prompt,
            # replace, not format: a workdir path may contain braces.
            "workdir_note": self.workdir_prompt.replace("{workdir}", str(workdir)),
            "extensions": [{"path": relative, "file": file} for relative, file in self.extensions],
            "skills_dir": str(self.skills_dir) if self.skills_dir is not None else None,
            "tools": list(self.tools),
            "builtin_tools": [*FILESYSTEM_TOOLS, SUBAGENT_TOOL],  # names an extension tool must not reuse
            "general_purpose_subagent": self.general_purpose_subagent,
            "model": {"provider": self.model.provider, "id": self.model.id, "reasoning_effort": self.reasoning_effort},
            "limits": dict(self.limits),
            "scripted_turns_file": self.scripted_turns_file,
            "trace_file": str(trace_dir / TRACE_FILE),
            "result_file": str(trace_dir / RESULT_FILE),
            "child_pgids_file": str(trace_dir / CHILD_PGIDS_FILE),
        }
