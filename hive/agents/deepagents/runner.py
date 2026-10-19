"""One Deep Agents trial, run inside the framework's pinned venv (.venvs/deepagents).

Started by hive/agents/deepagents/adapter.py as

    <venv python> -I runner.py <trace_dir>/run_spec.json        (cwd = the trial workdir)

with one JSON line on stdin, `{"credentials": {"OPENAI_API_KEY": "..."}}` (empty for the
scripted provider). This file imports only the stdlib and the framework, never hive: it runs in another venv,
and nothing in the controller may depend on the framework. It speaks the runner protocol of
hive/agents/runner_protocol.py through files:

- `trace.jsonl`, appended as the run goes: one `model_call` line per provider response (the
  model the provider reported, and its usage with input_tokens excluding cache reads), plus
  free-form lines (`model_request`, `tool_start`, `tool_end`, `tool_error`, ...) for the
  overseer's evidence.
- `result.json`, written once at the end, whatever happened, unless the process is killed.
- `child_pgids.txt`: the process group of every shell command, so the controller can kill
  commands the agent left running.

Checked against deepagents 0.7.28, langchain 1.4.6, langchain-core 1.6.12, langgraph 1.2.16
and langchain-openai 1.7.2. Why the less obvious choices:

- Usage is summed per model call from `on_llm_end` callbacks, not from the final state: the
  final state holds only the main agent's messages, so subagent calls would go unpaid (P12).
- Every request that goes through the agent's chat model is checked in `on_chat_model_start`
  (model id and reasoning effort) and counted there against `max_model_calls`, and the callback
  raises before the request is sent: an extension that overrides the model or its settings must
  not get a paid call through (P3), and every handler call inside wrap_model_call, every
  summarization call and every subagent call is a paid call. One counter per trial, held by the
  callback object (not graph state), so subagents share it. Replies are checked by the controller.
- `LocalShellBackend.execute` starts each command in a new session and waits on pipes, so a
  group kill misses it and `cmd &` blocks until its timeout. HIVE's subclass records each
  command's process group and waits on the shell itself, with output going to files.
- HIVE's guards sit after the candidate's middleware: a tool allowlist (the `tools` valve) and
  ToolErrorMiddleware, so a tool's own exception becomes an error result while an extension
  hook's exception still ends the run. deepagents replaces any stack entry whose name a custom
  middleware reuses, in place, even entries inner to the guards; so an extension middleware
  named like a stack entry (except the summarization slot) or a guard fails to load.
- The general-purpose subagent is off unless the valve asks for it. User middleware never
  reaches it (graph.py), so when it is on, HIVE builds it explicitly with its guards, and binds
  the experiment's recursion_limit on it: create_agent binds 9_999, and a subagent's bound
  config wins over the parent's run config.
- The provider credential never enters the environment. A process's exec-time environment
  stays readable by its children (`ps eww $PPID`, /proc/<pid>/environ) after os.environ.pop,
  so the key arrives on stdin, read to EOF before anything else runs, and goes straight to the
  client constructor. Shell commands get an explicit PATH/HOME/LANG environment.
- Extension code runs in this process, so it can reach the built model object, which holds the
  key and the provider client. The request checks, the call count and the trace see only calls
  made through the langchain model; they detect mistakes and plain violations, but they are not
  a sandbox, and the static checks in validate() are not either.
"""

from __future__ import annotations

import sys

# First, before any import: extensions are imported in place from the candidate, which must not
# change, and `-I` ignores PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# What each AgentMiddleware hook is in the controller's neutral vocabulary (hive.contracts
# LIFECYCLE_POINTS). wrap_tool_call can both gate a call and rewrite its result.
HOOK_POINTS = {
    "before_agent": ("before_run",),
    "before_model": ("before_model_call",),
    "wrap_model_call": ("wrap_model_call",),
    "after_model": ("after_model_call",),
    "wrap_tool_call": ("before_tool_call", "after_tool_call"),
    "after_agent": ("after_run",),
}
SUMMARIZATION_SLOT = "SummarizationMiddleware"
# Every middleware name deepagents 0.7.28 puts in a stack it builds (graph.py create_deep_agent,
# main agent and subagents; some only with options HIVE never sets). A custom middleware with
# one of these names replaces that entry in place, wherever it sits, including inner to HIVE's
# guards. Only SUMMARIZATION_SLOT is open to extensions (the documented compaction slot).
DEEPAGENTS_STACK_NAMES = frozenset({
    "FilesystemMiddleware", "SubAgentMiddleware", "_ParentNotifierMiddleware", SUMMARIZATION_SLOT,
    "PatchToolCallsMiddleware", "AsyncSubAgentMiddleware", "SkillsMiddleware", "AnthropicPromptCachingMiddleware",
    "MemoryMiddleware", "HumanInTheLoopMiddleware", "UnsupportedContentMiddleware", "_ToolExclusionMiddleware",
})  # fmt: skip
OPENAI_CREDENTIAL = "OPENAI_API_KEY"
# HIVE_TRIAL_ID is the process helper's trial marker (hive/agents/process.py): a shell command
# that daemonizes is still found and killed after the trial by it.
SHELL_ENV_VARIABLES = ("PATH", "HOME", "LANG", "HIVE_TRIAL_ID")
MAIN_AGENT_NAME = "main"
TRACE_TEXT_CHARS = 4000
STDERR_EXCERPT_LINES = 12


class ExtensionContractError(Exception):
    """An extension file loaded but does not follow the entry convention."""


class ModelSpecViolation(Exception):
    """A request was about to go to another model or reasoning effort than the experiment fixed."""


class ModelCallLimitReached(Exception):
    """The trial's model calls (main agent and subagents together) reached max_model_calls."""


# --- trace and result files -------------------------------------------------------------------


class TraceFile:
    def __init__(self, path: Path):
        self._file = path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> None:
        line = {"type": kind, "t": round(time.time(), 3), **fields}
        self._file.write(json.dumps(line, default=str) + "\n")
        self._file.flush()  # a run killed on timeout still leaves every line written so far


def write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)  # the controller never sees half a result


def clip(text: Any, limit: int = TRACE_TEXT_CHARS) -> str:
    text = text if isinstance(text, str) else str(text)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


# --- the run ------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    # First thing, before the spec or any framework or candidate code: take the key off stdin.
    try:
        credentials, credentials_error = read_credentials(), None
    except Exception as error:  # noqa: BLE001 (reported in result.json like any other runner failure)
        credentials, credentials_error = {}, error
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    # No "extensions" key until the load phase has run: a run that failed before it has no load
    # report, and an empty one would tell the controller every extension file went missing.
    result: dict[str, Any] = {"status": "errored", "final_text": "", "error": None, "error_class": "agent_error",
                              "runtime_errors": []}  # fmt: skip
    trial = None
    try:
        if credentials_error is not None:
            raise credentials_error
        trial = Trial(spec, credentials)
        try:
            result.update(trial.run())
        finally:
            trial.cleanup()
    except BaseException as error:  # noqa: BLE001 (whatever happens, the controller gets a result)
        result.update(error=f"the runner failed: {type(error).__name__}: {error}", error_class="agent_error")
        if trial is not None and trial.loads is not None:
            result.setdefault("extensions", trial.loads)
        traceback.print_exc()
    write_result(Path(spec["result_file"]), result)
    return 0


def read_credentials() -> dict[str, str]:
    """The provider credentials the adapter sends as one JSON line on stdin, read to EOF.

    stdin is then pointed at /dev/null, so nothing started later can read the pipe. The key is
    never put in os.environ (see the module doc). An empty stdin means no credentials.
    """
    text = sys.stdin.read() if sys.stdin is not None else ""
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, 0)
    finally:
        os.close(devnull)
    if not text.strip():
        return {}
    payload = json.loads(text)
    credentials = payload.get("credentials") if isinstance(payload, dict) else None
    if not isinstance(credentials, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in credentials.items()
    ):
        raise ValueError('stdin must hold one JSON object {"credentials": {"NAME": "value", ...}}')
    return credentials


class Trial:
    def __init__(self, spec: dict[str, Any], credentials: dict[str, str] | None = None):
        self.spec = spec
        self.credentials = credentials if credentials is not None else {}
        self.loads: list[dict[str, Any]] | None = None  # set once the load phase has run
        self.workdir = Path(spec["workdir"])
        self.trace = TraceFile(Path(spec["trace_file"]))
        self.extension_files = {
            os.path.realpath(entry["file"]): entry["path"] for entry in spec["extensions"]
        }  # absolute file -> candidate-relative path, to attribute tracebacks
        self.temporary_dirs: list[str] = []

    def cleanup(self) -> None:
        for directory in self.temporary_dirs:
            shutil.rmtree(directory, ignore_errors=True)

    def run(self) -> dict[str, Any]:
        # Framework imports happen here, inside main()'s catch-all: a broken venv is a result too.
        fw = Framework()
        model = build_model(fw, self.spec, self.credentials)
        self.credentials.clear()  # the model holds the key now; main() and this object keep no copy

        loads, middleware, tools = load_extensions(
            fw, self.spec["extensions"], allowed=set(self.spec["tools"]), builtin=set(self.spec["builtin_tools"])
        )
        self.loads = loads
        self.trace.write("extensions_loaded", extensions=loads)
        if not all(load["loaded"] for load in loads):
            failed = [f"{load['path']}: {load['error']}" for load in loads if not load["loaded"]]
            return {"error_class": "harness_load_error", "extensions": loads,
                    "error": "extension files failed to load: " + "; ".join(failed)}  # fmt: skip

        callbacks = make_trace_callbacks(fw, self.trace, self.spec["model"], self.spec["limits"]["max_model_calls"])
        try:
            agent = self.build_agent(fw, model, middleware, tools)
        except Exception as error:  # noqa: BLE001 (e.g. a tool deepagents rejects): the candidate can't run as written
            self.trace.write("build_error", error=f"{type(error).__name__}: {error}")
            return {"error_class": "harness_load_error", "extensions": loads,
                    "error": f"the agent could not be built from the candidate: {type(error).__name__}: {error}"}  # fmt: skip

        limits = self.spec["limits"]
        try:
            output = agent.invoke(
                {"messages": [{"role": "user", "content": self.spec["instruction"]}]},
                config={"callbacks": [callbacks], "recursion_limit": limits["recursion_limit"]},
            )
        except BaseException as error:  # noqa: BLE001 (extension code may raise SystemExit or KeyboardInterrupt)
            traceback.print_exc()
            return {"extensions": loads, **self.classify_failure(fw, error, callbacks)}
        if callbacks.violations:
            # An extension caught the violation and carried on; the run still left the experiment (P3).
            return {"extensions": loads, "error_class": "model_mismatch", "error": "; ".join(callbacks.violations)}
        if callbacks.limit_reached:
            # An extension caught the limit error and carried on; the run still used up its calls.
            return {"extensions": loads, "error_class": "agent_error", "error": self.call_limit_message()}
        return {"extensions": loads, **final_answer(fw, output)}

    def call_limit_message(self) -> str:
        return f"the run hit the model call limit ({self.spec['limits']['max_model_calls']}) without finishing"

    def build_agent(self, fw: Framework, model: Any, extension_middleware: list, extension_tools: list) -> Any:
        spec, limits = self.spec, self.spec["limits"]
        allowed = list(spec["tools"])
        backend = make_shell_backend(fw, self.workdir, Path(spec["child_pgids_file"]), limits["shell_timeout_s"])
        skills = None
        if spec.get("skills_dir"):
            # A copy: the agent can write under /skills/, and must not change the candidate.
            copy = tempfile.mkdtemp(prefix="hive-skills-")
            self.temporary_dirs.append(copy)
            shutil.copytree(spec["skills_dir"], copy, dirs_exist_ok=True)
            backend = fw.CompositeBackend(default=backend,
                                          routes={"/skills/": fw.FilesystemBackend(root_dir=copy, virtual_mode=True)})  # fmt: skip
            skills = ["/skills/"]

        subagents: list[dict[str, Any]] = []
        if spec["general_purpose_subagent"]:
            # Built explicitly, not left to deepagents: its auto-added subagent would run
            # without HIVE's guards (tool allowlist, tool errors).
            subagents.append({**fw.GENERAL_PURPOSE_SUBAGENT,
                              "middleware": hive_guards(fw, [t for t in allowed if t != "task"])})  # fmt: skip
            bind_subagent_recursion_limit(fw, limits["recursion_limit"])
        else:
            # The only switch for the auto-added subagent is a harness profile for this model.
            key = f"{fw.get_model_provider(model)}:{fw.get_model_identifier(model)}"
            fw.register_harness_profile(
                key, fw.HarnessProfile(general_purpose_subagent=fw.GeneralPurposeSubagentProfile(enabled=False))
            )

        prompt_parts = [part for part in (spec.get("system_prompt"), spec["workdir_note"]) if part and part.strip()]
        return fw.create_deep_agent(
            model,
            tools=extension_tools,
            system_prompt="\n\n".join(part.strip() for part in prompt_parts),
            middleware=[*extension_middleware, *hive_guards(fw, allowed)],
            subagents=subagents,
            skills=skills,
            backend=backend,
            name=MAIN_AGENT_NAME,
        )

    def classify_failure(self, fw: Framework, error: BaseException, callbacks: Any) -> dict[str, Any]:
        described = f"{type(error).__name__}: {error}"
        self.trace.write("run_error", error=clip(described))
        if isinstance(error, ModelSpecViolation) or callbacks.violations:
            return {"error_class": "model_mismatch", "error": "; ".join(callbacks.violations) or described}
        if isinstance(error, ModelCallLimitReached) or callbacks.limit_reached:
            return {"error_class": "agent_error", "error": self.call_limit_message()}
        if fw.is_quota_error(error):
            return {"error_class": "quota", "error": f"the provider refused for rate limit or quota: {described}"}
        if fw.is_provider_error(error):
            return {"error_class": "provider_error", "error": f"the model provider failed: {described}"}
        if isinstance(error, fw.GraphRecursionError):
            limit = self.spec["limits"]["recursion_limit"]
            return {"error_class": "agent_error",
                    "error": f"the run hit the step limit (recursion limit: {limit} graph steps) without finishing"}  # fmt: skip
        if isinstance(error, fw.ModelCallLimitExceededError):
            # Only an extension's own ModelCallLimitMiddleware raises this; HIVE's limit is ModelCallLimitReached.
            return {"error_class": "agent_error", "error": f"an extension's model call limit ended the run: {described}"}
        origin = self.extension_frame(error)
        if origin is not None:
            return {"error_class": "extension_error", "runtime_errors": [f"{origin}: {described}"],
                    "error": f"extension code threw during the run: {origin}: {described}"}  # fmt: skip
        return {"error_class": "agent_error", "error": f"the run crashed: {clip(described, 1000)}"}

    def extension_frame(self, error: BaseException) -> str | None:
        """'extensions/x.py:LINE in FUNC' for the innermost frame of extension code, if any."""
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            for frame in reversed(traceback.extract_tb(current.__traceback__)):
                path = self.extension_files.get(os.path.realpath(frame.filename))
                if path is not None:
                    return f"{path}:{frame.lineno} in {frame.name}"
            current = current.__cause__ or current.__context__
        return None


def final_answer(fw: Framework, output: Any) -> dict[str, Any]:
    """Completed only on the framework's own end-of-run signal: a model reply with no tool calls."""
    messages = output.get("messages") if isinstance(output, dict) else None
    last = messages[-1] if messages else None
    if isinstance(last, fw.AIMessage) and not last.tool_calls:
        return {"status": "completed", "final_text": last.text, "error": None, "error_class": None}
    kind = type(last).__name__ if last is not None else "nothing"
    detail = " with tool calls still pending" if isinstance(last, fw.AIMessage) else ""
    return {"error_class": "agent_error", "error": f"the run ended on {kind}{detail}, not on a final model reply"}


# --- the framework, imported lazily ---------------------------------------------------------------


class Framework:
    """The framework names this runner uses, imported once, inside main()'s error handling."""

    def __init__(self) -> None:
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from deepagents._models import get_model_identifier, get_model_provider
        from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
        from deepagents.backends.protocol import ExecuteResponse
        from deepagents.middleware import subagents as subagents_module
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
        from langchain.agents.middleware import (
            AgentMiddleware,
            ContextEditingMiddleware,
            ToolErrorMiddleware,
        )
        from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.exceptions import ModelAPIError, ModelError, ModelRateLimitError
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import BaseTool
        from langgraph.errors import GraphRecursionError

        self.GeneralPurposeSubagentProfile = GeneralPurposeSubagentProfile
        self.HarnessProfile = HarnessProfile
        self.create_deep_agent = create_deep_agent
        self.register_harness_profile = register_harness_profile
        self.get_model_identifier = get_model_identifier
        self.get_model_provider = get_model_provider
        self.CompositeBackend = CompositeBackend
        self.FilesystemBackend = FilesystemBackend
        self.LocalShellBackend = LocalShellBackend
        self.ExecuteResponse = ExecuteResponse
        self.GENERAL_PURPOSE_SUBAGENT = GENERAL_PURPOSE_SUBAGENT
        self.subagents_module = subagents_module
        self.AgentMiddleware = AgentMiddleware
        self.ContextEditingMiddleware = ContextEditingMiddleware
        self.ToolErrorMiddleware = ToolErrorMiddleware
        self.ModelCallLimitExceededError = ModelCallLimitExceededError
        self.BaseCallbackHandler = BaseCallbackHandler
        self.ModelAPIError = ModelAPIError
        self.ModelError = ModelError
        self.ModelRateLimitError = ModelRateLimitError
        self.BaseChatModel = BaseChatModel
        self.AIMessage = AIMessage
        self.ToolMessage = ToolMessage
        self.ChatGeneration = ChatGeneration
        self.ChatResult = ChatResult
        self.BaseTool = BaseTool
        self.GraphRecursionError = GraphRecursionError

    def is_quota_error(self, error: BaseException) -> bool:
        if isinstance(error, self.ModelRateLimitError) or getattr(error, "status_code", None) == 429:
            return True
        text = f"{getattr(error, 'code', '')} {error}".lower()
        return isinstance(error, self.ModelError) and ("insufficient_quota" in text or "billing" in text)

    def is_provider_error(self, error: BaseException) -> bool:
        if isinstance(error, self.ModelError):
            return True
        try:
            import openai
        except ImportError:
            return False
        return isinstance(error, openai.OpenAIError)


def build_model(fw: Framework, spec: dict[str, Any], credentials: dict[str, str]) -> Any:
    model = spec["model"]
    if model["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        key = credentials.get(OPENAI_CREDENTIAL)
        if not key:
            raise ValueError(f"no {OPENAI_CREDENTIAL} arrived on stdin for model provider 'openai'")
        limits = spec["limits"]
        return ChatOpenAI(
            model=model["id"],
            api_key=key,  # explicit: the key is never in os.environ for the client to find
            use_responses_api=True,
            reasoning={"effort": model["reasoning_effort"]},
            timeout=limits["provider_timeout_s"],
            max_retries=limits["max_retries"],
        )
    if model["provider"] == "scripted":
        turns = json.loads(Path(spec["scripted_turns_file"]).read_text(encoding="utf-8"))
        return scripted_model_class(fw)(model_name=model["id"], reasoning={"effort": model["reasoning_effort"]},
                                        turns=turns)  # fmt: skip
    raise ValueError(f"unsupported model provider {model['provider']!r}")


def scripted_model_class(fw: Framework) -> type:
    """A test-only chat model that replays scripted turns (provider "scripted").

    The stock langchain fakes can't bind tools, and deep agents always bind them. Each turn is
    {"text", "tool_calls": [{"name", "args"}], "usage": {"input", "output", "cache_read"},
    "model"} or {"raise": "rate_limit" | "insufficient_quota" | "provider" | "runtime",
    "message"}; the last turn repeats once the script runs out. Like ChatOpenAI, `input`
    counts cache reads too, it reports `model_name` in response_metadata, and bound tools
    and settings show up in the callbacks' invocation_params.
    """

    class ScriptedChatModel(fw.BaseChatModel):
        model_name: str
        reasoning: dict
        turns: list
        position: int = 0

        @property
        def _llm_type(self) -> str:
            return "hive-scripted"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"model_name": self.model_name, "reasoning": self.reasoning}

        def _get_ls_params(self, stop: Any = None, **kwargs: Any) -> Any:
            params = super()._get_ls_params(stop=stop, **kwargs)
            params["ls_provider"] = "scripted"
            return params

        def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
            return self.bind(tools=[{"type": "function", "name": tool_name(tool)} for tool in tools], **kwargs)

        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
            turn = self.turns[min(self.position, len(self.turns) - 1)]
            self.position += 1
            if "raise" in turn:
                raise scripted_error(fw, turn["raise"], turn.get("message", "scripted failure"))
            calls = [
                {"name": call["name"], "args": call.get("args", {}), "id": call.get("id") or f"call_{self.position}_{index}",
                 "type": "tool_call"}
                for index, call in enumerate(turn.get("tool_calls", []))
            ]  # fmt: skip
            usage = turn.get("usage", {})
            cache_read = usage.get("cache_read", 0)
            input_tokens, output_tokens = usage.get("input", 0), usage.get("output", 0)
            message = fw.AIMessage(
                content=turn.get("text", ""),
                tool_calls=calls,
                usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                                "input_token_details": {"cache_read": cache_read}},
                response_metadata={"model_name": turn.get("model", self.model_name), "model_provider": "scripted",
                                   "finish_reason": "tool_calls" if calls else "stop"},
            )  # fmt: skip
            return fw.ChatResult(generations=[fw.ChatGeneration(message=message)])

    return ScriptedChatModel


def scripted_error(fw: Framework, kind: str, message: str) -> Exception:
    if kind == "rate_limit":
        return fw.ModelRateLimitError(message)
    if kind == "insufficient_quota":
        error = fw.ModelRateLimitError(f"Error code: 429 - insufficient_quota: {message}")
        error.code = "insufficient_quota"
        return error
    if kind == "provider":
        return fw.ModelAPIError(message)
    return RuntimeError(message)


def tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("function", {}).get("name") or "")
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", ""))


# --- extensions -------------------------------------------------------------------------------------


def load_extensions(fw: Framework, entries: list[dict[str, str]], *, allowed: set[str], builtin: set[str]):
    """Import every extension file and call its extension(); report each one, loaded or not.

    `allowed` is the tools valve; `builtin` the framework's own tool names, which an extension
    tool must not reuse (two tools with one name would leave which one runs to the framework).
    """
    loads, middleware, tools = [], [], []
    middleware_owners: dict[str, str] = {}  # middleware name -> the extension file that returned it
    for index, entry in enumerate(entries):
        load: dict[str, Any] = {"path": entry["path"], "loaded": False, "error": None, "registered": []}
        try:
            items = call_entry_function(import_isolated(entry["file"], index))
            registered: list[str] = []
            for item in items:
                registered += registration_of(fw, item, allowed, builtin)
            file_middleware = [item for item in items if isinstance(item, fw.AgentMiddleware)]
            names = check_middleware_names(file_middleware, entry["path"], middleware_owners)
            middleware += file_middleware
            tools += [item for item in items if isinstance(item, fw.BaseTool)]
            middleware_owners.update(names)
            load.update(loaded=True, registered=list(dict.fromkeys(registered)))
        except BaseException as error:  # noqa: BLE001 (an extension may raise anything, KeyboardInterrupt included)
            load["error"] = describe_load_error(error, entry["file"])
        loads.append(load)
    return loads, middleware, tools


def check_middleware_names(items: list, path: str, owners: dict[str, str]) -> dict[str, str]:
    """Reject a middleware name deepagents would swap into its stack in place, or one already taken.

    deepagents replaces a same-named stack entry instead of adding the middleware, even entries
    inner to HIVE's guards, and keeps only the last of two custom middleware sharing a stack
    name. Returns the names this file adds.
    """
    reserved = (DEEPAGENTS_STACK_NAMES - {SUMMARIZATION_SLOT}) | set(_GUARD_NAMES)
    names: dict[str, str] = {}
    for item in items:
        name = item.name
        if name in reserved:
            raise ExtensionContractError(
                f"it returns middleware named '{name}', a name deepagents or HIVE uses in the agent's middleware "
                f"stack; rename it (the only stack name an extension may take is '{SUMMARIZATION_SLOT}')"
            )
        owner = names.get(name) or owners.get(name)
        if owner is not None:
            raise ExtensionContractError(f"it returns middleware named '{name}', a duplicate of one from {owner}")
        names[name] = path
    return names


def import_isolated(file: str, index: int) -> Any:
    # A unique module name per file: two extensions named alike never share a module, and
    # registering it in sys.modules lets pydantic and dataclasses resolve its own types.
    name = f"hive_extension_{index}_{re.sub(r'\W', '_', Path(file).stem)}"
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def call_entry_function(module: Any) -> list:
    entry = getattr(module, "extension", None)
    if not callable(entry):
        raise ExtensionContractError("the file defines no `extension()` function")
    items = entry()
    if not isinstance(items, list | tuple):
        raise ExtensionContractError(f"extension() returned {type(items).__name__}, not a list")
    return list(items)


def registration_of(fw: Framework, item: Any, allowed: set[str], builtin: set[str]) -> list[str]:
    """What one returned object hooks, as neutral lifecycle ids and 'tool:<name>' entries."""
    if isinstance(item, fw.BaseTool):
        return tool_registration(item, allowed, builtin)
    if not isinstance(item, fw.AgentMiddleware):
        raise ExtensionContractError(
            f"extension() returned a {type(item).__name__}; each item must be an AgentMiddleware or a BaseTool"
        )
    registered: list[str] = []
    cls = type(item)
    for hook, points in HOOK_POINTS.items():
        # The same override test create_agent uses (langchain/agents/factory.py).
        sync_hook = getattr(cls, hook) is not getattr(fw.AgentMiddleware, hook)
        async_hook = getattr(cls, "a" + hook) is not getattr(fw.AgentMiddleware, "a" + hook)
        if async_hook and not sync_hook:
            raise ExtensionContractError(
                f"{item.name} defines only `a{hook}`; the runner invokes the agent synchronously, so define `{hook}`"
            )
        if sync_hook:
            registered += points
    if item.name == SUMMARIZATION_SLOT or isinstance(item, fw.ContextEditingMiddleware):
        registered.append("compaction")
    for tool in getattr(item, "tools", None) or ():
        registered += tool_registration(tool, allowed, builtin)
    return registered


def tool_registration(tool: Any, allowed: set[str], builtin: set[str]) -> list[str]:
    name = tool_name(tool)
    if name in builtin:
        raise ExtensionContractError(f"it registers a tool named '{name}', which is a built-in tool's name")
    if name not in allowed:
        raise ExtensionContractError(
            f"it registers the tool '{name}', which the [valves] tools list does not include; "
            "list it there or the model can never call it"
        )
    return ["register_tool", f"tool:{name}"]


def describe_load_error(error: BaseException, file: str) -> str:
    described = f"{type(error).__name__}: {error}"
    if isinstance(error, ExtensionContractError):
        return described
    lines = [frame.lineno for frame in traceback.extract_tb(error.__traceback__)
             if os.path.realpath(frame.filename) == os.path.realpath(file)]  # fmt: skip
    return f"{described} (line {lines[-1]})" if lines else described


# --- HIVE's guards ----------------------------------------------------------------------------------


def hive_guards(fw: Framework, allowed_tools: list[str]) -> list:
    """Inner to every extension middleware (load_extensions rejects names deepagents would swap
    in place), so an extension's hooks see the allowlisted tools and its exceptions are not
    turned into tool results. The model-call limit is not here: it is counted per trial in the
    trace callback, which sees every call, retries and subagents included."""
    classes = hive_middleware_classes(fw)
    return [
        classes["HiveToolAllowlist"](allowed_tools),
        classes["HiveToolErrors"](on_error=lambda error, request: tool_error_content(fw, error)),
    ]


def bind_subagent_recursion_limit(fw: Framework, recursion_limit: int) -> None:
    """Bind the experiment's step limit on every subagent deepagents compiles.

    create_agent binds recursion_limit=9_999 on each compiled agent, and a subagent's bound
    config wins over the parent's run config (deepagents subagents.py, `task`), so the limit
    passed to invoke never reaches it. deepagents compiles declarative subagents through the
    module-level create_sub_agent, so HIVE wraps that once for this process (one trial).
    """
    module = fw.subagents_module
    original = module.create_sub_agent
    if getattr(original, "hive_recursion_limit", None) == recursion_limit:
        return

    def create_sub_agent(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs).with_config(recursion_limit=recursion_limit)

    create_sub_agent.hive_recursion_limit = recursion_limit
    module.create_sub_agent = create_sub_agent


def tool_error_content(fw: Framework, error: Exception) -> str | None:
    # None lets the exception through: provider failures, limits and model checks from inside
    # a subagent's `task` call must end the run with their own class, not become tool output.
    passthrough = (ModelSpecViolation, ModelCallLimitReached, fw.ModelError, fw.ModelCallLimitExceededError,
                   fw.GraphRecursionError)  # fmt: skip
    if isinstance(error, passthrough) or fw.is_provider_error(error):
        return None
    return f"Error: {type(error).__name__}: {error}"


_GUARD_CLASSES: dict[str, type] = {}
_GUARD_NAMES = ("HiveToolAllowlist", "HiveToolErrors")  # the guards' middleware names (their class names)


def hive_middleware_classes(fw: Framework) -> dict[str, type]:
    if _GUARD_CLASSES:
        return _GUARD_CLASSES

    class HiveToolErrors(fw.ToolErrorMiddleware):
        """A tool's own exception becomes an error result the model sees, as in Pi."""

    class HiveToolAllowlist(fw.AgentMiddleware):
        """The `tools` valve: unlisted tools are hidden from the model and refused if called anyway."""

        def __init__(self, allowed: list[str]):
            self.allowed = frozenset(allowed)

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            visible = [tool for tool in request.tools if tool_name(tool) in self.allowed]
            return handler(request.override(tools=visible))

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            name = request.tool_call["name"]
            if name in self.allowed:
                return handler(request)
            return fw.ToolMessage(content=f"Error: the tool '{name}' is not available.",
                                  tool_call_id=request.tool_call["id"], name=name, status="error")  # fmt: skip

    _GUARD_CLASSES.update(HiveToolErrors=HiveToolErrors, HiveToolAllowlist=HiveToolAllowlist)
    return _GUARD_CLASSES


# --- shell backend ----------------------------------------------------------------------------------


def make_shell_backend(fw: Framework, workdir: Path, pgids_file: Path, timeout_s: int) -> Any:
    # An explicit environment: PATH, HOME, LANG and the trial marker only (the credential is never in
    # os.environ either).
    env = {name: os.environ[name] for name in SHELL_ENV_VARIABLES if name in os.environ}
    return _shell_backend_class(fw)(workdir, pgids_file=pgids_file, virtual_mode=True, timeout=timeout_s, env=env)


def _shell_backend_class(fw: Framework) -> type:
    class _HiveShellBackend(fw.LocalShellBackend):
        def __init__(self, root_dir: Path, *, pgids_file: Path, **kwargs: Any):
            super().__init__(root_dir, **kwargs)
            self._pgids_file = pgids_file
            # The model may pass the workdir's real path; map it onto the virtual root.
            self._root_prefixes = sorted({str(self.cwd), str(Path(root_dir).absolute())}, key=len, reverse=True)

        def _resolve_path(self, key: str) -> Path:
            for prefix in self._root_prefixes:
                if key == prefix or key.startswith(prefix + "/"):
                    key = key[len(prefix):] or "/"
                    break
            return super()._resolve_path(key)

        def execute(self, command: str, *, timeout: int | None = None) -> Any:
            if not command or not isinstance(command, str):
                return fw.ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1,
                                          truncated=False)  # fmt: skip
            effective_timeout = timeout if timeout is not None else self._default_timeout
            if effective_timeout <= 0:
                raise ValueError(f"timeout must be positive, got {effective_timeout}")
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                try:
                    process = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=stdout,
                                               stderr=stderr, env=self._env, cwd=str(self.cwd),
                                               start_new_session=True)  # fmt: skip
                except Exception as error:  # noqa: BLE001 (as the stock backend: an error result, not a crash)
                    return fw.ExecuteResponse(output=f"Error executing command ({type(error).__name__}): {error}",
                                              exit_code=1, truncated=False)  # fmt: skip
                with self._pgids_file.open("a") as pgids:
                    pgids.write(f"{process.pid}\n")  # its own session: pgid == pid
                try:
                    returncode = process.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    _kill_group(process.pid)
                    process.wait()
                    return fw.ExecuteResponse(
                        output=f"Error: Command timed out after {effective_timeout} seconds.", exit_code=124,
                        truncated=False,
                    )  # fmt: skip
                stdout.seek(0)
                stderr.seek(0)
                out = stdout.read().decode(errors="replace")
                err = stderr.read().decode(errors="replace")
            return self._response(out, err, returncode)

        def _response(self, out: str, err: str, returncode: int) -> Any:
            # The stock backend's output format (deepagents/backends/local_shell.py).
            parts = [out] if out else []
            if err:
                parts.extend(f"[stderr] {line}" for line in err.strip().split("\n"))
            output = "\n".join(parts) if parts else "<no output>"
            truncated = len(output) > self._max_output_bytes
            if truncated:
                output = output[: self._max_output_bytes] + f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            if returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {returncode}"
            return fw.ExecuteResponse(output=output, exit_code=returncode, truncated=truncated)

    return _HiveShellBackend


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# --- callbacks: the trace, usage and the request-side model check ----------------------------------


def trace_callbacks_class(fw: Framework) -> type:
    class _TraceCallbacks(fw.BaseCallbackHandler):
        raise_error = True  # so a ModelSpecViolation or the call limit stops the request before it is sent

        def __init__(self, trace: TraceFile, model: dict[str, Any], max_model_calls: int):
            self.trace = trace
            self.expected_model = model["id"]
            self.expected_effort = model["reasoning_effort"]
            self.violations: list[str] = []
            # One count per trial: this object is handed to invoke, and callbacks reach subagents,
            # summarization and every handler call an extension makes. Parallel `task` calls run
            # on threads, hence the lock.
            self.max_model_calls = max_model_calls
            self.model_calls = 0
            self.limit_reached = False
            self._lock = threading.Lock()
            self.agents_seen: set[str] = set()
            self.tool_names: dict[Any, str] = {}

        def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any, metadata: Any = None,
                                **kwargs: Any) -> None:  # fmt: skip
            params = kwargs.get("invocation_params") or {}
            model = params.get("model") or params.get("model_name")
            reasoning = params.get("reasoning")
            effort = reasoning.get("effort") if isinstance(reasoning, dict) else params.get("reasoning_effort")
            agent = (metadata or {}).get("lc_agent_name") or MAIN_AGENT_NAME
            line: dict[str, Any] = {"agent": agent, "model": model, "reasoning_effort": effort,
                                    "tools": [tool_name(tool) for tool in params.get("tools") or ()],
                                    "n_messages": len(messages[0]) if messages else 0}  # fmt: skip
            if agent not in self.agents_seen and messages and messages[0] and messages[0][0].type == "system":
                line["system_prompt"] = messages[0][0].text  # the effective prompt, framework additions included
            self.agents_seen.add(agent)
            self.trace.write("model_request", **line)
            if model != self.expected_model or effort != self.expected_effort:
                violation = (
                    f"a request was about to go to model {model!r} with reasoning effort {effort!r}; the experiment "
                    f"fixed {self.expected_model!r} with {self.expected_effort!r}"
                )
                self.violations.append(violation)
                raise ModelSpecViolation(violation)
            with self._lock:
                if self.model_calls >= self.max_model_calls:
                    self.limit_reached = True
                    refused = True
                else:
                    self.model_calls += 1
                    refused = False
            if refused:
                self.trace.write("model_call_limit", agent=agent, max_model_calls=self.max_model_calls)
                raise ModelCallLimitReached(f"the trial's {self.max_model_calls} model calls are used up")

        def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
            try:
                message = response.generations[0][0].message
            except (AttributeError, IndexError):
                message = None
            metadata = getattr(message, "response_metadata", None) or {}
            reported = metadata.get("model_name") or (response.llm_output or {}).get("model_name") or ""
            usage = getattr(message, "usage_metadata", None) or {}
            details = usage.get("input_token_details") or {}
            cache_read = int(details.get("cache_read") or 0)
            cache_write = int(details.get("cache_creation") or 0)
            self.trace.write(
                "model_call",
                model=reported,
                usage={"input_tokens": max(int(usage.get("input_tokens") or 0) - cache_read - cache_write, 0),
                       "output_tokens": int(usage.get("output_tokens") or 0),
                       "cache_read_tokens": cache_read, "cache_write_tokens": cache_write},
                text=clip(getattr(message, "text", "") or ""),
                tool_calls=[{"name": call["name"], "args": call["args"]} for call in getattr(message, "tool_calls", ())],
            )  # fmt: skip

        def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
            self.trace.write("model_error", error=clip(f"{type(error).__name__}: {error}"))

        def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any, **kwargs: Any) -> None:
            name = (serialized or {}).get("name") or kwargs.get("name") or ""
            self.tool_names[run_id] = name
            self.trace.write("tool_start", name=name, input=clip(input_str))

        def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
            content = getattr(output, "content", output)
            status = getattr(output, "status", None)
            self.trace.write("tool_end", name=self.tool_names.get(run_id, ""), status=status, output=clip(content))

        def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
            self.trace.write("tool_error", name=self.tool_names.get(run_id, ""),
                             error=clip(f"{type(error).__name__}: {error}"))  # fmt: skip

    return _TraceCallbacks


def make_trace_callbacks(fw: Framework, trace: TraceFile, model: dict[str, Any], max_model_calls: int) -> Any:
    return trace_callbacks_class(fw)(trace, model, max_model_calls)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
