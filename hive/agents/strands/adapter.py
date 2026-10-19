"""The Strands adapter, controller side: candidate checks, settings, and one subprocess per trial.

Strands (strands-agents) is a Python framework with 50+ dependencies, so it lives in its own
pinned venv (.venvs/strands) and this module imports only the standard library and hive.
Each trial runs `runner.py` under that venv's interpreter, as its own process group, through
hive.agents.process.run_process_group, and hive.agents.runner_protocol turns what the runner
left in the trace dir into one AgentResult. Why a subprocess, from the checked behaviour of
strands-agents 1.61.1:

- the vended shell and file_editor resolve against the process-global os.getcwd(), so the
  trial's workdir must be the process's cwd;
- candidate extensions are arbitrary Python that must never run in the controller, and their
  import side effects (sys.modules, monkeypatches) are process-global;
- Strands has no wall-clock limit and its cancel signal does not interrupt a running tool, so
  only killing the process enforces a timeout. Shell children run in sessions of their own;
  the runner lists their pgids in child_pgids.txt and run_process_group kills those too.

What the runner gets is passed explicitly: a run_spec.json naming every candidate file, a
minimal environment, and the provider credential as one JSON line on stdin. The key is never
in the runner's environment, so no shell, tool or child process inherits it, and `env` in the
agent's shell can't copy it into the trace. Strands silently defaults to Bedrock and boto3
probes AWS metadata and credential files, so AWS discovery is switched off here and Bedrock is
disabled in the runner.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hive.agents.process import ProcessOutcome, run_process_group
from hive.agents.runner_protocol import RESULT_FILE, TRACE_FILE, Prices, interpret_runner, parse_prices
from hive.candidate import MANIFEST, Candidate, content_files
from hive.contracts import EXTENSIONS_DIR, AgentResult, ExtensionSpec, LifecyclePoint, ModelSpec, Task
from hive.extensions import expected_extension_files

PACKAGE_DIR = Path(__file__).parent
RUNNER = PACKAGE_DIR / "runner.py"
VALVES_DOC = PACKAGE_DIR / "VALVES.md"
EXTENSIONS_API_DOC = PACKAGE_DIR / "EXTENSIONS_API.md"
EXTENSION_SPEC_FILE = PACKAGE_DIR / "extension_spec.toml"
WORKDIR_NOTE_FILE = PACKAGE_DIR / "WORKDIR_NOTE.md"

# --- candidate layout ---------------------------------------------------------------------

SYSTEM_PROMPT_FILE = "SYSTEM.md"
SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
EXTENSION_SUFFIX = ".py"
ALLOWED_TOP_LEVEL = {MANIFEST, "README.md", SYSTEM_PROMPT_FILE, EXTENSIONS_DIR, SKILLS_DIR}

VALVE_NAMES = ("tools", "conversation_window")
# The SDK's own vended tools (strands.vended_tools) the runner can give the agent. Any other
# name in the tools valve must be a tool registered by one of the candidate's extensions.
VENDED_TOOLS = ("shell", "file_editor", "python_repl")
DEFAULT_TOOLS = ("shell", "file_editor")
# Strands' own default (SlidingWindowConversationManager.window_size), stated so it is visible.
DEFAULT_CONVERSATION_WINDOW = 40

# Strands' AgentSkills rules (vended_plugins/skills/skill.py), applied with strict=True in the runner.
SKILL_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
MAX_SKILL_NAME_LENGTH = 64

# Static P3 checks on extension source. The runner's model guard and the recording client
# still check at run time; this only rejects the obvious cases before any money is spent.
FORBIDDEN_MODULES = ("strands.models", "strands._middleware", "openai", "boto3", "botocore", "anthropic")
FORBIDDEN_STRANDS_NAMES = ("BedrockModel", "ModelRouter")
# Agent attributes an extension must not assign. Only an assignment through a receiver that
# names an agent counts (agent.model, event.agent.model, self._agent.model, sub_agent.model):
# a plugin's own `self.model` switches nothing, and the runtime guard checks agent.model anyway.
FORBIDDEN_ASSIGNED_ATTRIBUTES = ("model", "_model_router", "_aux_model")
FORBIDDEN_CALLS = ("update_config",)

# --- model ---------------------------------------------------------------------------------

# "scripted" replays model turns from a file (settings `script_file`). It exists for HIVE's own
# tests, so a full framework run can be driven offline; experiments use "openai".
PROVIDERS = ("openai", "scripted")
# OpenAI's reasoning_effort values for Chat Completions.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_THINKING = "low"

# --- settings ------------------------------------------------------------------------------

REQUIRED_SETTINGS = ("python_path", "strands_version", "prices")
DEFAULT_SETTINGS: Mapping[str, Any] = {
    "timeout_s": 600,
    "term_grace_s": 5.0,
    # Model turns per run (Strands' limits={"turns": N}); a run that hits it never answered.
    "max_turns": 50,
    # Total attempts per model call when it is throttled or the server fails. Strands' default is
    # 6 attempts with up to 240 s backoff, which stalls a trial for minutes on a billing failure.
    "max_model_attempts": 3,
    "script_file": None,
}
SETTING_NAMES = (*REQUIRED_SETTINGS, *DEFAULT_SETTINGS)
RETRY_INITIAL_DELAY_S = 1
RETRY_MAX_DELAY_S = 8

# --- subprocess ----------------------------------------------------------------------------

PASSED_ENV = ("PATH", "HOME", "LANG")
PROVIDER_CREDENTIAL = {"openai": "OPENAI_API_KEY", "scripted": None}
# Set for every run: no AWS metadata probe, no AWS config or credential files, no telemetry export.
PINNED_ENV: Mapping[str, str] = {
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_CONFIG_FILE": os.devnull,
    "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    "OTEL_SDK_DISABLED": "true",
}
RUN_SPEC_FILE = "run_spec.json"
CHILD_PGIDS_FILE = "child_pgids.txt"
MESSAGES_FILE = "messages.json"
RUNNER_STDOUT_FILE = "runner.stdout.txt"
RUNNER_STDERR_FILE = "runner.stderr.txt"
VERSION_CHECK_TIMEOUT_S = 60
VERSION_CHECK_CODE = "import importlib.metadata as m; print(m.version('strands-agents'))"


class StrandsVersionMismatch(RuntimeError):
    """The venv's strands-agents is not the version the experiment pinned (P13)."""


# --- extension spec ----------------------------------------------------------------------


def load_extension_spec() -> ExtensionSpec:
    data = tomllib.loads(EXTENSION_SPEC_FILE.read_text(encoding="utf-8"))
    return ExtensionSpec(
        language=data["language"],
        suffixes=tuple(data["suffixes"]),
        entry_convention=data["entry_convention"],
        framework_version=f"{data['framework']} {data['version']}",
        lifecycle=tuple(LifecyclePoint(**point) for point in data["lifecycle"]),
        forbidden=tuple(data["forbidden"]),
        api_reference=EXTENSIONS_API_DOC.read_text(encoding="utf-8"),
    )


def checked_strands_version() -> str:
    """The strands-agents version the extension spec and API reference were checked against."""
    return tomllib.loads(EXTENSION_SPEC_FILE.read_text(encoding="utf-8"))["version"]


# --- candidate checks ----------------------------------------------------------------------


@dataclass(frozen=True)
class StrandsLayout:
    """The Strands-native files of a candidate, plus anything that doesn't fit."""

    system_prompt: Path | None
    skills: tuple[Path, ...]
    problems: tuple[str, ...]


def read_layout(root: Path) -> StrandsLayout:
    """Classify every content file. Uses the candidate's own file list, so hashed == checked."""
    problems: list[str] = []
    skill_files: dict[str, set[str]] = {}
    for relative in content_files(root):
        parts = PurePosixPath(relative.as_posix()).parts
        top = parts[0]
        where = relative.as_posix()
        if top not in ALLOWED_TOP_LEVEL:
            problems.append(f"unknown top-level entry '{top}' (allowed: {', '.join(sorted(ALLOWED_TOP_LEVEL))})")
        elif top in (EXTENSIONS_DIR, SKILLS_DIR) and len(parts) == 1:
            problems.append(f"'{top}' must be a directory")
        elif top not in (EXTENSIONS_DIR, SKILLS_DIR) and len(parts) > 1:
            problems.append(f"'{top}' must be a file, not a directory")
        elif top == EXTENSIONS_DIR:
            if len(parts) > 2:
                problems.append(f"'{where}': extensions must be single files directly in extensions/")
            elif relative.suffix != EXTENSION_SUFFIX:
                problems.append(f"'{where}': extensions must end in .py")
            else:
                problems.extend(extension_problems(where, root / relative))
        elif top == SKILLS_DIR:
            if len(parts) == 2:
                problems.append(f"'{where}': skills/ may only contain skill directories")
            else:
                skill_files.setdefault(parts[1], set()).add("/".join(parts[2:]))

    # Empty skill directories hold no content files, so look for them directly.
    skills_root = root / SKILLS_DIR
    if skills_root.is_dir():
        for entry in skills_root.iterdir():
            if entry.is_dir() and not entry.name.startswith(".") and entry.name not in skill_files:
                skill_files[entry.name] = set()
    skills = []
    for name in sorted(skill_files):
        if SKILL_FILE not in skill_files[name]:
            problems.append(f"'skills/{name}' has no {SKILL_FILE}")
            continue
        skills.append(skills_root / name)
        problems.extend(skill_problems(skills_root / name))

    system_prompt = root / SYSTEM_PROMPT_FILE
    if system_prompt.is_file() and not system_prompt.read_text(encoding="utf-8", errors="replace").strip():
        problems.append(f"{SYSTEM_PROMPT_FILE} is empty; delete it to run without a candidate system prompt")
    return StrandsLayout(
        system_prompt=system_prompt if system_prompt.is_file() else None,
        skills=tuple(skills),
        problems=tuple(problems),
    )


def extension_problems(where: str, path: Path) -> list[str]:
    """Static checks of one extension file. Parses it with ast; never imports or runs it."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"'{where}' is not valid UTF-8"]
    try:
        tree = ast.parse(source, filename=where)
    except SyntaxError as error:
        return [f"'{where}': syntax error at line {error.lineno}: {error.msg}"]

    problems = []
    entries = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "extension"
    ]
    if not entries:
        problems.append(f"'{where}' must define a top-level function extension() returning a strands.plugins.Plugin")
    elif isinstance(entries[-1], ast.AsyncFunctionDef):
        problems.append(f"'{where}': extension() must not be async")
    elif _required_arguments(entries[-1].args):
        problems.append(f"'{where}': extension() must take no arguments")
    problems.extend(f"'{where}' {finding}: an extension must not switch the model or provider (P3)"
                    for finding in _forbidden_uses(tree))  # fmt: skip
    return problems


def _required_arguments(arguments: ast.arguments) -> int:
    positional = len(arguments.posonlyargs) + len(arguments.args) - len(arguments.defaults)
    keyword_only = sum(1 for default in arguments.kw_defaults if default is None)
    return positional + keyword_only


def _forbidden_uses(tree: ast.AST) -> list[str]:
    def forbidden_module(name: str) -> bool:
        return any(name == module or name.startswith(module + ".") for module in FORBIDDEN_MODULES)

    findings = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            findings += [f"line {line}: importing {alias.name}" for alias in node.names if forbidden_module(alias.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if forbidden_module(node.module):
                findings.append(f"line {line}: importing from {node.module}")
            elif node.module == "strands":
                findings += [f"line {line}: importing strands.{alias.name}" for alias in node.names
                             if alias.name in FORBIDDEN_STRANDS_NAMES or alias.name == "models"]  # fmt: skip
        elif isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for element in ast.walk(target):
                    if (isinstance(element, ast.Attribute) and element.attr in FORBIDDEN_ASSIGNED_ATTRIBUTES
                            and _names_an_agent(element.value)):  # fmt: skip
                        findings.append(f"line {line}: assigning .{element.attr} of an agent")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
            findings.append(f"line {line}: calling .{node.func.attr}()")
    return findings


def _names_an_agent(receiver: ast.expr) -> bool:
    """Whether `receiver` reads as an agent: a name or attribute like agent, sub_agent, event.agent."""
    if isinstance(receiver, ast.Name):
        name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        name = receiver.attr
    else:
        return False
    return name.lstrip("_").lower().endswith("agent")


def skill_problems(skill_dir: Path) -> list[str]:
    """AgentSkills in strict mode refuses these at load time; catch them before paying for a run."""
    where = f"skills/{skill_dir.name}"
    problems = []
    name = skill_dir.name
    if len(name) > MAX_SKILL_NAME_LENGTH or not SKILL_NAME.match(name) or "--" in name:
        problems.append(f"'{where}': skill names must be 1-64 lowercase letters, digits and single hyphens")
    frontmatter = parse_frontmatter((skill_dir / SKILL_FILE).read_text(encoding="utf-8", errors="replace"))
    if frontmatter is None:
        return [*problems, f"'{where}/{SKILL_FILE}' must start with a '---' frontmatter block"]
    declared = frontmatter.get("name", "")
    if not declared:
        problems.append(f"'{where}/{SKILL_FILE}' has no 'name' in its frontmatter")
    elif declared != name:
        problems.append(f"'{where}/{SKILL_FILE}': name '{declared}' must equal its directory name '{name}'")
    if not frontmatter.get("description", ""):
        problems.append(f"'{where}/{SKILL_FILE}' has no 'description' in its frontmatter")
    return problems


def parse_frontmatter(text: str) -> dict[str, str] | None:
    """Top-level `key: value` lines of a `---` frontmatter block; None when there is none.

    The stdlib has no YAML parser. Skill headers only need `name` and `description`, so this
    reads plain and quoted scalars; the runner's strict Skill.from_file is the full check.
    """
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return None
    end = normalized.find("\n---", 3)
    if end == -1:
        return None
    fields = {}
    for line in normalized[4:end].split("\n"):
        if not line or line[0] in " \t#" or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        fields[key.strip()] = value.strip()
    return fields


def valve_problems(valves: Mapping[str, Any]) -> list[str]:
    problems = [f"unknown valve '{name}' (allowed: {', '.join(VALVE_NAMES)})" for name in valves if name not in VALVE_NAMES]
    tools = valves.get("tools", list(DEFAULT_TOOLS))
    if not isinstance(tools, list):
        problems.append(f"valve 'tools' must be a list of tool names; got {tools!r}")
    else:
        for tool in tools:
            if not isinstance(tool, str) or not tool.strip() or tool != tool.strip():
                problems.append(f"valve 'tools' has an invalid tool name {tool!r}")
    window = valves.get("conversation_window", DEFAULT_CONVERSATION_WINDOW)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        problems.append(f"valve 'conversation_window' must be an integer >= 1; got {window!r}")
    return problems


# --- settings and model --------------------------------------------------------------------


@dataclass(frozen=True)
class StrandsSettings:
    python_path: str
    strands_version: str
    prices: Prices
    timeout_s: float
    term_grace_s: float
    max_turns: int
    max_model_attempts: int
    script_file: str | None


def resolve_settings(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> StrandsSettings:
    merged = {**DEFAULT_SETTINGS, **defaults, **overrides}
    unknown = sorted(set(merged) - set(SETTING_NAMES))
    if unknown:
        raise ValueError(f"unknown Strands adapter settings: {', '.join(unknown)} (allowed: {', '.join(SETTING_NAMES)})")
    missing = [name for name in REQUIRED_SETTINGS if name not in merged]
    if missing:
        raise ValueError(f"Strands adapter settings must include {', '.join(missing)}")
    for name in ("python_path", "strands_version"):
        if not isinstance(merged[name], str) or not merged[name].strip():
            raise ValueError(f"Strands setting '{name}' must be a non-empty string")
    prices = parse_prices(merged["prices"])

    def number(name: str, minimum: float, exclusive: bool) -> float:
        value = merged[name]
        too_small = value <= minimum if exclusive else value < minimum
        if isinstance(value, bool) or not isinstance(value, int | float) or too_small:
            bound = f"{'>' if exclusive else '>='} {minimum:g}"
            raise ValueError(f"Strands setting '{name}' must be a number {bound}; got {value!r}")
        return float(value)

    def count(name: str) -> int:
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Strands setting '{name}' must be an integer >= 1; got {value!r}")
        return value

    script_file = merged["script_file"]
    if script_file is not None and (not isinstance(script_file, str) or not script_file):
        raise ValueError("Strands setting 'script_file' must be a path")
    return StrandsSettings(
        python_path=merged["python_path"],
        strands_version=merged["strands_version"],
        prices=prices,
        timeout_s=number("timeout_s", 0, exclusive=True),
        term_grace_s=number("term_grace_s", 0, exclusive=False),
        max_turns=count("max_turns"),
        max_model_attempts=count("max_model_attempts"),
        script_file=script_file,
    )


def model_config(model: ModelSpec, script_file: str | None) -> dict[str, Any]:
    """What the runner needs to build the model. The model is fixed by the experiment (P3)."""
    if model.provider not in PROVIDERS:
        raise ValueError(f"the Strands adapter supports model providers {', '.join(PROVIDERS)}; got {model.provider!r}")
    if model.provider == "scripted":
        if script_file is None:
            raise ValueError("the scripted provider needs a script_file setting with the model turns to replay")
        if not Path(script_file).is_file():
            raise ValueError(f"Strands setting 'script_file' names no file: {script_file}")
        return {"provider": "scripted", "id": model.id, "reasoning_effort": None}
    if script_file is not None:
        raise ValueError("Strands setting 'script_file' is only for the test-only scripted provider")
    effort = DEFAULT_THINKING if model.thinking is None else model.thinking
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"model thinking level must be one of {', '.join(REASONING_EFFORTS)}; got {effort!r}")
    return {"provider": model.provider, "id": model.id, "reasoning_effort": effort}


def subprocess_env(source: Mapping[str, str]) -> dict[str, str]:
    """The trial's whole environment, built explicitly: nothing else from the controller leaks in.

    It never holds the provider credential; that goes on stdin (credentials_input)."""
    env = {name: source[name] for name in PASSED_ENV if name in source}
    env.update(PINNED_ENV)
    return env


def credentials_input(provider: str, source: Mapping[str, str]) -> str:
    """The runner's stdin: one JSON line, {"credentials": {NAME: value}}. Empty when the provider
    needs none or the controller has none; the runner then reports the missing key itself."""
    credential = PROVIDER_CREDENTIAL.get(provider)
    credentials = {credential: source[credential]} if credential and source.get(credential) else {}
    return json.dumps({"credentials": credentials}) + "\n"


def check_strands_version(python_path: str, pinned: str) -> None:
    try:
        completed = subprocess.run(
            [python_path, "-I", "-c", VERSION_CHECK_CODE],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=subprocess_env(os.environ),
            timeout=VERSION_CHECK_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        message = f"could not run {python_path} to check the pinned strands-agents {pinned}: {error}"
        raise StrandsVersionMismatch(message) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or ["no output"]
        raise StrandsVersionMismatch(f"strands-agents is not importable with {python_path}: {detail[0]}")
    installed = completed.stdout.strip()
    if installed != pinned:
        raise StrandsVersionMismatch(f"strands-agents {installed or '(none reported)'} is installed but {pinned} is pinned")


# --- the adapter ---------------------------------------------------------------------------


class StrandsAdapter:
    name = "strands"

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

    def build(self, candidate: Candidate, model: ModelSpec, settings: Mapping[str, Any]) -> StrandsAgent:
        problems = self.validate(candidate)
        if problems:
            raise ValueError(f"candidate {candidate.id} is not a valid Strands candidate: " + "; ".join(problems))
        resolved = resolve_settings(self._settings, settings)
        config = model_config(model, resolved.script_file)
        checked = checked_strands_version()
        if resolved.strands_version != checked:
            # The overseer's extension guide describes the checked version; it would mislead for another.
            raise StrandsVersionMismatch(
                f"strands_version {resolved.strands_version} is pinned, but this adapter's extension spec and API "
                f"reference were checked against {checked}; re-check them before changing the pin"
            )
        check_strands_version(resolved.python_path, resolved.strands_version)

        layout = read_layout(candidate.root)
        system_prompt = layout.system_prompt.read_text(encoding="utf-8") if layout.system_prompt else None
        extensions = tuple(
            (relative, str(candidate.root / relative))
            for relative in expected_extension_files(candidate, self.extension_spec())
        )
        return StrandsAgent(
            python_path=resolved.python_path,
            timeout_s=resolved.timeout_s,
            term_grace_s=resolved.term_grace_s,
            prices=resolved.prices,
            model=model,
            model_config=config,
            script_file=str(Path(resolved.script_file).resolve()) if resolved.script_file else None,
            candidate_root=str(candidate.root),
            system_prompt=system_prompt,
            extensions=extensions,
            skills=tuple(str(skill) for skill in layout.skills),
            tools=tuple(candidate.valves.get("tools", list(DEFAULT_TOOLS))),
            conversation_window=candidate.valves.get("conversation_window", DEFAULT_CONVERSATION_WINDOW),
            max_turns=resolved.max_turns,
            max_model_attempts=resolved.max_model_attempts,
        )

    def describe_valves(self) -> str:
        return VALVES_DOC.read_text(encoding="utf-8")

    def extension_spec(self) -> ExtensionSpec:
        return load_extension_spec()


@dataclass(frozen=True)
class StrandsAgent:
    python_path: str
    timeout_s: float
    term_grace_s: float
    prices: Prices
    model: ModelSpec  # what every provider response must report (P3)
    model_config: Mapping[str, Any]
    script_file: str | None
    candidate_root: str
    system_prompt: str | None  # SYSTEM.md, or None
    extensions: tuple[tuple[str, str], ...]  # (candidate-relative path, absolute file)
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    conversation_window: int
    max_turns: int
    max_model_attempts: int

    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        workdir = Path(workdir).resolve()
        trace_dir = Path(trace_dir).resolve()
        started = time.monotonic()
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            # A file left by an earlier run in this dir would be read as this run's result.
            for name in (RESULT_FILE, TRACE_FILE, CHILD_PGIDS_FILE, MESSAGES_FILE):
                (trace_dir / name).unlink(missing_ok=True)
            spec_path = trace_dir / RUN_SPEC_FILE
            spec_path.write_text(json.dumps(self.run_spec(task, workdir, trace_dir), indent=2), encoding="utf-8")
            outcome = run_process_group(
                [self.python_path, "-I", str(RUNNER), str(spec_path)],
                cwd=workdir,
                env=subprocess_env(os.environ),
                stdin_text=credentials_input(self.model.provider, os.environ),
                timeout_s=self.timeout_s,
                stdout_path=trace_dir / RUNNER_STDOUT_FILE,
                stderr_path=trace_dir / RUNNER_STDERR_FILE,
                term_grace_s=self.term_grace_s,
                extra_pgids_file=trace_dir / CHILD_PGIDS_FILE,
            )
        except OSError as error:  # the trace dir could not be written; the runner never ran
            outcome = ProcessOutcome(b"", b"", None, False, error)
        return interpret_runner(
            trace_dir, outcome, model=self.model, prices=self.prices, wall_clock_s=time.monotonic() - started
        )

    def run_spec(self, task: Task, workdir: Path, trace_dir: Path) -> dict[str, Any]:
        """Everything the runner needs, explicitly. Never holds the API key (that goes on stdin)."""
        note = WORKDIR_NOTE_FILE.read_text(encoding="utf-8").strip().replace("{workdir}", str(workdir))
        system_prompt = f"{self.system_prompt.rstrip()}\n\n{note}" if self.system_prompt else note
        return {
            "instruction": task.instruction,
            "workdir": str(workdir),
            "trace_dir": str(trace_dir),
            "candidate_root": self.candidate_root,
            "system_prompt": system_prompt,
            "extensions": [{"path": relative, "file": file} for relative, file in self.extensions],
            "skills": list(self.skills),
            "tools": list(self.tools),
            "vended_tools": list(VENDED_TOOLS),
            "conversation_window": self.conversation_window,
            "model": dict(self.model_config),
            "script_file": self.script_file,
            "max_turns": self.max_turns,
            "retry": {
                "max_attempts": self.max_model_attempts,
                "initial_delay_s": RETRY_INITIAL_DELAY_S,
                "max_delay_s": RETRY_MAX_DELAY_S,
            },
            "trace_file": str(trace_dir / TRACE_FILE),
            "result_file": str(trace_dir / RESULT_FILE),
            "child_pgids_file": str(trace_dir / CHILD_PGIDS_FILE),
            "messages_file": str(trace_dir / MESSAGES_FILE),
        }
