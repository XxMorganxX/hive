"""HIVE's Strands runner: one trial, run inside the Strands venv (.venvs/strands).

hive/agents/strands/adapter.py starts it as `python -I runner.py <trace_dir>/run_spec.json`,
with the trial workdir as the working directory. It imports only the standard library,
strands and openai, never hive: it runs under the framework's interpreter, where hive is not
installed, and candidate code must never share a process with the controller. What it owes
the controller is the runner protocol (hive/agents/runner_protocol.py):

- trace.jsonl, appended as the run goes: one "model_call" line per provider response, with
  the model the provider reported and the usage (input_tokens excluding cache reads), plus
  HIVE's own event lines (hooks, tools, messages) as evidence for the overseer;
- result.json, written once at the end: status, final text, error class, a load record for
  every extension file, and extension code that threw during the run;
- child_pgids.txt: the process group of every shell the tools start, which Strands puts in
  sessions of its own; the controller kills them when the trial ends.

It catches everything and always writes result.json, unless it is killed.

Model integrity (P3). Strands builds a BedrockModel whenever no model is given, and merely
constructing one probes AWS, so BedrockModel is disabled outright. The OpenAI provider never
reports which model answered (it ignores chunk.model), so the OpenAI client is wrapped to
record it. Extensions run in this process and can replace agent.model, so a guard hook checks
the model object and its config around every model call. This is detection, not a sandbox.

Extensions. Each extensions/<name>.py defines `extension()` returning a strands Plugin named
<name>. Every hook callback an extension registers (via @hook, agent.add_hook or a
HookProvider in init_agent) is wrapped so a throw is recorded with the file's path before it
aborts the run, and every hook and tool is attributed to the file that registered it, which
is what the load report's `registered` lists.

The test-only "scripted" provider replays model turns from run_spec["script_file"]:
    {"turns": [turn, ...]}, where a turn is
    - a list of blocks, {"text": str} or {"tool": name, "input": {...}};
    - or {"blocks": [...], "model": "<reported model>", "usage": {"input_tokens",
      "output_tokens", "cache_read_tokens"}}, input_tokens counted the way OpenAI counts
      prompt_tokens (cache reads included);
    - or {"error": {"kind": "rate_limit" | "server_error" | "connection" | "context_overflow"
      | "runtime", "code": str, "message": str}}, raised as the OpenAI provider would.
"""

from __future__ import annotations

import sys

# Importing candidate extensions must not write __pycache__ into the candidate directory, and
# -I ignores PYTHONDONTWRITEBYTECODE, so say it here before anything is imported.
sys.dont_write_bytecode = True

import functools  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from collections.abc import AsyncGenerator  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

TEXT_LIMIT = 4000  # characters of a tool input, tool result or message kept in a trace line
ERROR_TEXT_LIMIT = 2000
COMPLETED_STOP_REASONS = ("end_turn", "stop_sequence")
CREDENTIAL = {"openai": "OPENAI_API_KEY"}

# The neutral lifecycle ids (hive.contracts.LIFECYCLE_POINTS, copied: this file can't import
# hive), in the contract's order, and the Strands events that hook each one.
LIFECYCLE_ORDER = ("before_run", "system_prompt", "before_model_call", "after_model_call", "wrap_model_call",
                   "before_tool_call", "after_tool_call", "register_tool", "compaction", "after_run")  # fmt: skip
EVENT_POINTS = {
    "BeforeInvocationEvent": "before_run",
    "BeforeModelCallEvent": "before_model_call",
    "AfterModelCallEvent": "after_model_call",
    "BeforeToolCallEvent": "before_tool_call",
    "BeforeToolsEvent": "before_tool_call",
    "AfterToolCallEvent": "after_tool_call",
    "AfterToolsEvent": "after_tool_call",
    "AfterInvocationEvent": "after_run",
}


class RunnerError(Exception):
    """A failure this runner classifies itself, with one of the contract's error classes."""

    def __init__(self, error_class: str, message: str):
        super().__init__(message)
        self.error_class = error_class


class BedrockDisabled(RuntimeError):
    """Raised by BedrockModel.__init__: HIVE never lets Strands fall back to Bedrock (P3)."""


class ModelSwitched(RuntimeError):
    """Raised by HIVE's model guard when the agent's model is not the experiment's (P3)."""


class ScriptExhausted(RuntimeError):
    """The scripted model was asked for more turns than the script holds."""


# --- trace -----------------------------------------------------------------------------------


class Trace:
    """trace.jsonl, appended line by line so a killed run still leaves its usage behind."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"type": kind, **fields}, default=str)
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def model_call(self, model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int,
                   requested: str | None = None) -> None:  # fmt: skip
        usage = {
            # Providers count cache reads inside the prompt tokens; the protocol keeps them apart.
            "input_tokens": max(0, input_tokens - cache_read_tokens),
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": 0,
        }
        self.write("model_call", model=model, requested=requested, usage=usage)


def clip(value: Any, limit: int = TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"


def describe(error: BaseException) -> str:
    return clip(f"{type(error).__name__}: {error}", ERROR_TEXT_LIMIT)


# --- extension records and attribution --------------------------------------------------------


@dataclass
class ExtensionRecord:
    path: str  # candidate-relative, e.g. "extensions/guard.py"
    file: str  # absolute
    loaded: bool = False
    error: str | None = None
    plugin: Any = None
    points: set[str] = field(default_factory=set)
    tools: list[str] = field(default_factory=list)

    def registered(self, active_tools: set[str]) -> list[str]:
        tools = [name for name in self.tools if name in active_tools]
        points = set(self.points) | ({"register_tool"} if tools else set())
        return [point for point in LIFECYCLE_ORDER if point in points] + [f"tool:{name}" for name in tools]

    def as_json(self, active_tools: set[str]) -> dict[str, Any]:
        registered = self.registered(active_tools) if self.loaded else []
        return {"path": self.path, "loaded": self.loaded, "error": self.error, "registered": registered}


class Attribution:
    """Which extension file registered which hook or tool, and what extension code threw.

    While an extension plugin is being attached (its init_agent, then its @hook and @tool
    registration), everything registered belongs to it. Outside that window, a callback is
    attributed by the file its code lives in.
    """

    def __init__(self, records: list[ExtensionRecord], trace: Trace):
        self.records = records
        self.trace = trace
        self.current: ExtensionRecord | None = None
        self.runtime_errors: list[str] = []
        self.by_file = {os.path.realpath(record.file): record for record in records}

    def owner_of(self, callback: Any) -> ExtensionRecord | None:
        if self.current is not None:
            return self.current
        return self.by_file.get(os.path.realpath(code_file(callback) or ""))

    def record_throw(self, record: ExtensionRecord, where: str, error: BaseException) -> None:
        message = f"{record.path}: {where} raised {describe(error)}"
        if message not in self.runtime_errors:
            self.runtime_errors.append(message)
            self.trace.write("extension_error", path=record.path, where=where, error=describe(error),
                             traceback=clip("".join(traceback.format_exception(error)), ERROR_TEXT_LIMIT))  # fmt: skip

    def file_on_stack(self, errors: list[BaseException]) -> ExtensionRecord | None:
        """The extension whose code was running when one of these errors was raised, if any."""
        for error in errors:
            frames = traceback.extract_tb(error.__traceback__)
            for frame in reversed(frames):
                record = self.by_file.get(os.path.realpath(frame.filename))
                if record is not None:
                    return record
        return None


def code_file(callback: Any) -> str | None:
    target = getattr(callback, "__func__", callback)
    target = inspect.unwrap(target) if callable(target) else target
    while isinstance(target, functools.partial):
        target = target.func
    code = getattr(target, "__code__", None)
    if code is None:
        code = getattr(getattr(type(target), "__call__", None), "__code__", None)
    return code.co_filename if code is not None else None


def install_guards(attribution: Attribution) -> None:
    """Patch Strands before any candidate code is imported. Each patch says why in place."""
    from strands.hooks import HookRegistry
    from strands.hooks._type_inference import infer_event_types
    from strands.interrupt import InterruptException
    from strands.models.bedrock import BedrockModel
    from strands.tools.registry import ToolRegistry

    # Strands builds a BedrockModel for Agent() with no model, a string model or aux_model, and
    # merely constructing one probes AWS. Nothing in a HIVE run may use any model but the
    # experiment's, so refuse it outright, before it can touch the network.
    def refuse_bedrock(self: Any, *args: Any, **kwargs: Any) -> None:
        attribution.trace.write("bedrock_blocked")
        raise BedrockDisabled(
            "BedrockModel is disabled by HIVE: Strands builds it when no model is given, and the experiment "
            "fixes the model (pass model=agent.model to any sub-agent)"
        )

    BedrockModel.__init__ = refuse_bedrock

    original_add_callback = HookRegistry.add_callback

    def add_callback(self: Any, event_type: Any, callback: Any, *, order: float = 0) -> None:
        record = attribution.owner_of(callback)
        if record is None:
            return original_add_callback(self, event_type, callback, order=order)
        types = event_type if isinstance(event_type, list) else [event_type] if event_type else None
        if types is None:
            types = infer_event_types(callback)  # raises as add_callback itself would
        names = [getattr(kind, "__name__", str(kind)) for kind in types]
        record.points.update(EVENT_POINTS[name] for name in names if name in EVENT_POINTS)
        where = f"{'/'.join(names)} hook {getattr(callback, '__qualname__', repr(callback))}"

        # Wrapped with functools.wraps, so Strands' own type-hint inference still sees the original.
        if inspect.iscoroutinefunction(callback):

            @functools.wraps(callback)
            async def guarded(event: Any) -> Any:
                try:
                    return await callback(event)
                except InterruptException:
                    raise  # Strands' human-in-the-loop control flow, not a failure
                except Exception as error:
                    attribution.record_throw(record, where, error)
                    raise
        else:

            @functools.wraps(callback)
            def guarded(event: Any) -> Any:
                try:
                    return callback(event)
                except InterruptException:
                    raise
                except Exception as error:
                    attribution.record_throw(record, where, error)
                    raise

        return original_add_callback(self, types, guarded, order=order)

    HookRegistry.add_callback = add_callback

    original_register_tool = ToolRegistry.register_tool

    def register_tool(self: Any, tool: Any) -> None:
        original_register_tool(self, tool)
        if attribution.current is not None:
            attribution.current.tools.append(tool.tool_name)

    ToolRegistry.register_tool = register_tool


# --- loading ---------------------------------------------------------------------------------


def load_extension(record: ExtensionRecord, index: int) -> None:
    """Import one extension file on its own and call its entry point. Never raises."""
    from strands.plugins import Plugin

    stem = Path(record.file).stem
    module_name = f"hive_candidate_extension_{index}_{re.sub(r'[^0-9A-Za-z_]', '_', stem)}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, record.file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {record.file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        entry = getattr(module, "extension", None)
        if not callable(entry):
            raise TypeError("the file defines no callable extension()")
        plugin = entry()
        if not isinstance(plugin, Plugin):
            raise TypeError(f"extension() returned {type(plugin).__name__}, not an instance of a strands.plugins.Plugin subclass")
        if plugin.name != stem:
            raise ValueError(f"the plugin is named {plugin.name!r}; it must be named {stem!r}, after its file")
    except BaseException as error:  # noqa: BLE001 - a candidate file failing in any way is a load failure
        if isinstance(error, KeyboardInterrupt):
            raise
        record.error = describe(error)
        return
    record.plugin = plugin
    record.loaded = True


def attach_window(plugin: Any, record: ExtensionRecord | None, attribution: Attribution) -> None:
    """Make the plugin's init_agent open its attribution window (None closes it)."""
    original = plugin.init_agent

    if inspect.iscoroutinefunction(original):

        async def init_agent(agent: Any) -> None:
            attribution.current = record
            return await original(agent)
    else:

        def init_agent(agent: Any) -> Any:
            attribution.current = record
            return original(agent)

    plugin.init_agent = init_agent


def end_of_extensions_plugin() -> Any:
    from strands.plugins import Plugin

    class EndOfExtensions(Plugin):
        """Attached after the extensions, so the SDK's own plugins are never attributed to one."""

        name = "hive:end-of-extensions"

    return EndOfExtensions()


# --- models ----------------------------------------------------------------------------------


def make_model(spec: dict[str, Any], trace: Trace) -> Any:
    config = spec["model"]
    if config["provider"] == "scripted":
        return make_scripted_model(json.loads(Path(spec["script_file"]).read_text(encoding="utf-8")), config["id"], trace)
    if config["provider"] == "openai":
        import openai
        from strands.models.openai import OpenAIModel

        if not os.environ.get(CREDENTIAL["openai"]):
            raise RunnerError("provider_error", f"{CREDENTIAL['openai']} is not set; the openai provider needs it")
        # max_retries=0: HIVE's retry strategy is the only retry layer, so attempts don't multiply.
        client = RecordingOpenAIClient(openai.AsyncOpenAI(max_retries=0), trace)
        params = {"reasoning_effort": config["reasoning_effort"]} if config.get("reasoning_effort") else {}
        return OpenAIModel(client=client, model_id=config["id"], params=params)
    raise RunnerError("agent_error", f"unknown model provider {config['provider']!r}")


class RecordingOpenAIClient:
    """An AsyncOpenAI wrapper that records the model the provider reports on every response.

    Strands' OpenAIModel reads only `choices` and `usage` from the stream and drops
    `chunk.model`, so without this the run could not show which model actually answered (P3).
    """

    def __init__(self, client: Any, trace: Trace):
        self._client = client
        self.chat = _Namespace(completions=RecordingCompletions(client.chat.completions, trace))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _Namespace:
    def __init__(self, **attributes: Any):
        self.__dict__.update(attributes)


class RecordingCompletions:
    def __init__(self, completions: Any, trace: Trace):
        self._completions = completions
        self._trace = trace

    async def create(self, **request: Any) -> Any:
        response = await self._completions.create(**request)
        if not request.get("stream"):
            recorder = UsageRecorder(self._trace, request.get("model"))
            recorder.see(response)
            recorder.finish()
            return response
        return RecordingStream(response, UsageRecorder(self._trace, request.get("model")))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class UsageRecorder:
    """Collects the reported model and usage of one response and writes its model_call line(s)."""

    def __init__(self, trace: Trace, requested: str | None):
        self.trace = trace
        self.requested = requested
        self.models: list[str] = []
        self.usage: Any = None
        self.finished = False

    def see(self, chunk: Any) -> None:
        model = getattr(chunk, "model", None)
        if model and model not in self.models:
            self.models.append(str(model))
        if getattr(chunk, "usage", None):
            self.usage = chunk.usage

    def finish(self) -> None:
        if self.finished or not self.models:  # nothing came back: no response, no spend
            return
        self.finished = True
        usage = self.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        self.trace.model_call(self.models[0], int(getattr(usage, "prompt_tokens", 0) or 0),
                              int(getattr(usage, "completion_tokens", 0) or 0), cached, self.requested)  # fmt: skip
        for other in self.models[1:]:  # a stream that changed model midway: every name is checked
            self.trace.model_call(other, 0, 0, 0, self.requested)


class RecordingStream:
    """Passes a streamed response through unchanged, recording each chunk. Strands iterates it
    twice (up to finish_reason, then the rest for usage), so it keeps one inner iterator."""

    def __init__(self, response: Any, recorder: UsageRecorder):
        self._response = response
        self._recorder = recorder
        self._iterator: Any = None

    def __aiter__(self) -> RecordingStream:
        if self._iterator is None:
            self._iterator = self._response.__aiter__()
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._iterator.__anext__()
        except BaseException:  # StopAsyncIteration at the end, or the stream failing midway
            self._recorder.finish()
            raise
        self._recorder.see(chunk)
        return chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def make_scripted_model(script: dict[str, Any], model_id: str, trace: Trace) -> Any:
    from strands.models.model import Model

    class ScriptedModel(Model):
        """A minimal strands Model that replays scripted turns, for HIVE's offline tests."""

        def __init__(self) -> None:
            self.config = {"model_id": model_id}
            self.turns = list(script.get("turns", []))

        def update_config(self, **model_config: Any) -> None:
            self.config.update(model_config)

        def get_config(self) -> dict[str, Any]:
            return self.config

        async def structured_output(
            self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise NotImplementedError("the scripted model has no structured output")
            yield  # pragma: no cover - makes this an async generator, as the interface requires

        async def stream(
            self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any
        ) -> AsyncGenerator[dict[str, Any], None]:
            if not self.turns:
                raise ScriptExhausted("the model script has no turns left")
            turn = self.turns.pop(0)
            if isinstance(turn, dict) and "error" in turn:
                raise scripted_error(turn["error"])
            blocks = turn["blocks"] if isinstance(turn, dict) else turn
            usage = turn.get("usage", {}) if isinstance(turn, dict) else {}
            reported = turn.get("model", self.config["model_id"]) if isinstance(turn, dict) else self.config["model_id"]
            input_tokens = int(usage.get("input_tokens", 10))
            output_tokens = int(usage.get("output_tokens", 5))
            cache_read = int(usage.get("cache_read_tokens", 0))

            yield {"messageStart": {"role": "assistant"}}
            stop_reason = "end_turn"
            for index, block in enumerate(blocks):
                if "text" in block:
                    yield {"contentBlockStart": {"start": {}}}
                    yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
                else:
                    stop_reason = "tool_use"
                    tool_use = {"name": block["tool"], "toolUseId": f"scripted-{len(self.turns)}-{index}"}
                    yield {"contentBlockStart": {"start": {"toolUse": tool_use}}}
                    yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(block.get("input", {}))}}}}
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": stop_reason}}
            trace.model_call(reported, input_tokens, output_tokens, cache_read, self.config["model_id"])
            metadata_usage = {"inputTokens": input_tokens, "outputTokens": output_tokens,
                              "totalTokens": input_tokens + output_tokens, "cacheReadInputTokens": cache_read}  # fmt: skip
            yield {"metadata": {"usage": metadata_usage, "metrics": {"latencyMs": 1}}}

    return ScriptedModel()


def scripted_error(error: dict[str, Any]) -> Exception:
    """The exception the real OpenAI provider would raise for this failure."""
    import httpx
    import openai
    from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

    kind = error.get("kind", "runtime")
    message = error.get("message", f"scripted {kind}")
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": message, "code": error.get("code")}

    def status(cls: type, code: int) -> Exception:
        return cls(message=message, response=httpx.Response(code, request=request, json={"error": body}), body=body)

    if kind == "rate_limit":
        # OpenAIModel maps every 429, insufficient_quota included, to ModelThrottledException.
        throttled = ModelThrottledException(message)
        throttled.__cause__ = status(openai.RateLimitError, 429)
        return throttled
    if kind == "server_error":
        return status(openai.InternalServerError, 500)
    if kind == "connection":
        return openai.APIConnectionError(request=request)
    if kind == "context_overflow":
        return ContextWindowOverflowException(message)
    return RuntimeError(message)


def make_retry_strategy(retry: dict[str, Any]) -> Any:
    import openai
    from strands import ModelRetryStrategy
    from strands.types.exceptions import ModelThrottledException

    class HiveRetry(ModelRetryStrategy):
        """Few, short retries, and none for a billing failure: Strands' default of 6 attempts with
        up to 240 s backoff stalls a trial for minutes, and retrying insufficient_quota is futile."""

        def is_retryable(self, exception: Exception) -> bool:
            if quota_exhausted(exception):
                return False
            if isinstance(exception, ModelThrottledException):
                return True
            return isinstance(exception, openai.APIConnectionError | openai.InternalServerError)

    return HiveRetry(max_attempts=retry["max_attempts"], initial_delay=retry["initial_delay_s"],
                     max_delay=retry["max_delay_s"])  # fmt: skip


# --- HIVE's own hooks ------------------------------------------------------------------------


def make_hive_hooks(trace: Trace, guard_state: dict[str, Any]) -> Any:
    from strands.hooks import (
        AfterInvocationEvent,
        AfterModelCallEvent,
        AfterToolCallEvent,
        BeforeInvocationEvent,
        BeforeModelCallEvent,
        BeforeToolCallEvent,
        HookProvider,
        MessageAddedEvent,
    )

    def check_model(event: Any) -> None:
        expected = guard_state.get("model")
        if expected is None:
            return
        problem = None
        if event.agent.model is not expected:
            problem = f"agent.model was replaced by {type(event.agent.model).__name__}"
        elif expected.get_config() != guard_state["config"]:
            problem = f"the model config changed to {clip(expected.get_config(), 500)}"
        if problem is not None:
            guard_state["problems"].append(problem)
            trace.write("model_guard", problem=problem)
            raise ModelSwitched(f"{problem}; the experiment fixes the model (P3)")

    class HiveHooks(HookProvider):
        """The trace and the model guard. Registered before any extension, so never attributed to one."""

        def register_hooks(self, registry: Any, **kwargs: Any) -> None:
            registry.add_callback(BeforeInvocationEvent, lambda event: trace.write("before_invocation"))
            # Last among BeforeModelCall hooks, so it sees what every extension hook did.
            registry.add_callback(BeforeModelCallEvent, self.before_model, order=math.inf)
            registry.add_callback(AfterModelCallEvent, self.after_model)
            registry.add_callback(BeforeToolCallEvent, lambda event: trace.write(
                "tool_call", name=event.tool_use.get("name"), input=clip(event.tool_use.get("input"))))
            registry.add_callback(AfterToolCallEvent, lambda event: trace.write(
                "tool_result", name=event.tool_use.get("name"), status=event.result.get("status"),
                content=clip(event.result.get("content"))))  # fmt: skip
            registry.add_callback(MessageAddedEvent, lambda event: trace.write(
                "message", role=event.message.get("role"), content=clip(event.message.get("content"))))
            registry.add_callback(AfterInvocationEvent, lambda event: trace.write(
                "after_invocation", stop_reason=getattr(event.result, "stop_reason", None)))

        def before_model(self, event: Any) -> None:
            check_model(event)
            trace.write("before_model_call", model_class=type(event.agent.model).__name__,
                        model_id=event.agent.model.get_config().get("model_id"))  # fmt: skip

        def after_model(self, event: Any) -> None:
            stop = event.stop_response.stop_reason if event.stop_response else None
            trace.write("after_model_call", stop_reason=stop, error=describe(event.exception) if event.exception else None)
            check_model(event)

    return HiveHooks()


def make_sandbox(pgids_file: Path) -> Any:
    from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment

    class PgidRecordingSandbox(NotASandboxLocalEnvironment):
        """The host sandbox Strands uses by default, plus a record of every shell it starts.

        Each command runs as `sh -c` in a new session, so the shell's pid ($$) is its process
        group; the shell appends it to child_pgids.txt before running the command. Killing the
        runner's own group never reaches these, so the controller kills each listed group.
        """

        async def execute_streaming(self, command: str, **kwargs: Any) -> AsyncGenerator[Any, None]:
            recorded = f"printf '%s\\n' \"$$\" >> {shlex.quote(str(pgids_file))}; {command}"
            async for chunk in super().execute_streaming(recorded, **kwargs):
                yield chunk

    return PgidRecordingSandbox()


def vended_tools(names: list[str]) -> dict[str, Any]:
    from strands import vended_tools as vended

    return {name: getattr(vended, name) for name in names}


def load_skills(skill_dirs: list[str]) -> Any:
    if not skill_dirs:
        return None
    from strands import AgentSkills, Skill

    # Path-loaded skills that fail validation are only logged, even with strict=True, so load
    # each one here, where a bad SKILL.md raises.
    return AgentSkills(skills=[Skill.from_file(path, strict=True) for path in skill_dirs], strict=True)


# --- classification ---------------------------------------------------------------------------


def exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    pending = [error]
    while pending:
        current = pending.pop(0)
        if current is None or any(current is seen for seen in chain):
            continue
        chain.append(current)
        pending += [getattr(current, "original_exception", None), current.__cause__, current.__context__]
    return chain


def error_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str):
        return code
    body = getattr(error, "body", None)
    return body.get("code") if isinstance(body, dict) and isinstance(body.get("code"), str) else None


def quota_exhausted(error: BaseException) -> bool:
    return any(error_code(item) == "insufficient_quota" for item in exception_chain(error))


def classify(error: BaseException, attribution: Attribution, guard_state: dict[str, Any]) -> tuple[str, str]:
    """(error_class, message) for an exception that ended the run, worst breach first."""
    import openai
    from strands.types.exceptions import EventLoopException, ModelThrottledException

    chain = exception_chain(error)
    root = next((item for item in reversed(chain) if not isinstance(item, EventLoopException)), error)
    if guard_state["problems"] or any(isinstance(item, ModelSwitched | BedrockDisabled) for item in chain):
        cause = next((item for item in chain if isinstance(item, ModelSwitched | BedrockDisabled)), root)
        return "model_mismatch", describe(cause)
    runner_error = next((item for item in chain if isinstance(item, RunnerError)), None)
    if runner_error is not None:
        return runner_error.error_class, str(runner_error)
    if attribution.runtime_errors:
        return "extension_error", "extension code threw during the run: " + "; ".join(attribution.runtime_errors)
    on_stack = attribution.file_on_stack(chain)
    if on_stack is not None:
        attribution.record_throw(on_stack, "extension code", root)
        return "extension_error", "extension code threw during the run: " + "; ".join(attribution.runtime_errors)
    if any(isinstance(item, ModelThrottledException | openai.RateLimitError) or getattr(item, "status_code", None) == 429
           for item in chain):  # fmt: skip
        detail = " (insufficient_quota: the account is out of credit)" if quota_exhausted(error) else ""
        return "quota", describe(root) + detail
    provider = next((item for item in chain if isinstance(item, openai.APIError)), None)
    if provider is not None:
        return "provider_error", describe(provider)
    return "agent_error", describe(root)


# --- the run ---------------------------------------------------------------------------------


@dataclass
class Outcome:
    status: str = "errored"
    final_text: str = ""
    error: str | None = None
    error_class: str | None = None


def run(spec: dict[str, Any], trace: Trace, records: list[ExtensionRecord], outcome: Outcome,
        state: dict[str, Any]) -> None:  # fmt: skip
    attribution = Attribution(records, trace)
    state["attribution"] = attribution
    install_guards(attribution)

    from strands import Agent
    from strands.agent.conversation_manager import SlidingWindowConversationManager

    # Load phase: anything failing here means the candidate did not run as written.
    for index, record in enumerate(records):
        load_extension(record, index)
        trace.write("extension_import", path=record.path, loaded=record.loaded, error=record.error)
    plugins = []
    for record in records:
        if record.loaded:
            attach_window(record.plugin, record, attribution)
            plugins.append(record.plugin)
    closing = end_of_extensions_plugin()
    attach_window(closing, None, attribution)
    plugins.append(closing)

    model = make_model(spec, trace)
    guard_state = state["guard"]
    try:
        skills = load_skills(spec["skills"])
        tools = vended_tools([name for name in spec["tools"] if name in spec["vended_tools"]])
        agent = Agent(
            model=model,
            system_prompt=spec["system_prompt"],
            tools=list(tools.values()),
            plugins=[*plugins, *([skills] if skills else [])],
            hooks=[make_hive_hooks(trace, guard_state)],
            callback_handler=None,
            conversation_manager=SlidingWindowConversationManager(window_size=spec["conversation_window"]),
            retry_strategy=make_retry_strategy(spec["retry"]),
            load_tools_from_directory=False,
            sandbox=make_sandbox(Path(spec["child_pgids_file"])),
        )
    except Exception as error:
        # Building the agent runs every extension's init_agent and registers its hooks and tools,
        # and loads the skills: a failure here is the candidate's files not loading.
        failed = attribution.current
        attribution.current = None
        if failed is not None:
            failed.loaded, failed.error, failed.plugin = False, describe(error), None
        raise RunnerError("harness_load_error", f"the agent could not be built from the candidate: {describe(error)}") from error
    attribution.current = None

    # Pi-like allowlist: a tool is offered only if the tools valve lists it.
    allowed = set(spec["tools"])
    extension_tools = {name for record in records for name in record.tools}
    for name in sorted(extension_tools - allowed):
        agent.tool_registry.registry.pop(name, None)
        agent.tool_registry.dynamic_tools.pop(name, None)
    active = set(agent.tool_registry.registry)
    state["active_tools"] = active
    for record in records:
        if record.loaded and type(agent.conversation_manager).__module__ == getattr(record.plugin, "__module__", None):
            record.points.add("compaction")
    unknown = sorted(allowed - active)
    trace.write("harness", tools=sorted(active), extensions=[record.as_json(active) for record in records])

    problems = [f"{record.path}: {record.error}" for record in records if not record.loaded]
    if unknown:
        problems.append(f"the tools valve lists {unknown}, which no vended tool or extension provides")
    if agent.model is not model:
        guard_state["problems"].append(f"agent.model was replaced by {type(agent.model).__name__} while loading")
        raise RunnerError("model_mismatch", guard_state["problems"][-1])
    if problems:
        raise RunnerError("harness_load_error", "; ".join(problems))

    # Run phase.
    guard_state["model"] = model
    guard_state["config"] = json.loads(json.dumps(model.get_config(), default=str))
    try:
        result = agent(spec["instruction"], limits={"turns": spec["max_turns"]})
    finally:
        dump_messages(agent, Path(spec["messages_file"]))
    if guard_state["problems"]:
        raise RunnerError("model_mismatch", "; ".join(guard_state["problems"]))
    if attribution.runtime_errors:  # a throw Strands swallowed is still the candidate's code failing
        raise RunnerError("extension_error", "extension code threw during the run: " + "; ".join(attribution.runtime_errors))
    stop_reason = result.stop_reason
    trace.write("run_end", stop_reason=stop_reason)
    if stop_reason in COMPLETED_STOP_REASONS:
        outcome.status, outcome.final_text = "completed", str(result).strip()
    elif stop_reason == "limit_turns":
        outcome.error_class = "agent_error"
        outcome.error = f"the run hit its step limit (max_turns={spec['max_turns']}) before giving a final answer"
    else:
        outcome.error_class = "agent_error"
        outcome.error = f"the run ended on stop_reason {stop_reason!r}, not on a final answer"


def dump_messages(agent: Any, path: Path) -> None:
    try:
        path.write_text(json.dumps(agent.messages, indent=2, default=str), encoding="utf-8")
    except Exception:  # evidence only; never let it mask the run's own outcome
        pass


def write_result(path: Path, outcome: Outcome, records: list[ExtensionRecord], state: dict[str, Any]) -> None:
    attribution: Attribution | None = state.get("attribution")
    active = state.get("active_tools", set())
    result = {
        "status": outcome.status,
        "final_text": outcome.final_text,
        "error": outcome.error,
        "error_class": outcome.error_class,
        "extensions": [record.as_json(active) for record in records],
        "runtime_errors": list(attribution.runtime_errors) if attribution else [],
    }
    # Written to a temporary name and renamed, so a kill mid-write never leaves half a result.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(result, indent=2), encoding="utf-8")
    partial.replace(path)


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    trace = Trace(Path(spec["trace_file"]))
    records = [ExtensionRecord(path=entry["path"], file=entry["file"]) for entry in spec["extensions"]]
    outcome = Outcome()
    state: dict[str, Any] = {"guard": {"model": None, "config": None, "problems": []}}
    try:
        run(spec, trace, records, outcome, state)
    except Exception as error:
        attribution = state.get("attribution") or Attribution(records, trace)
        state["attribution"] = attribution
        outcome.error_class, outcome.error = classify(error, attribution, state["guard"])
        trace.write("run_error", error_class=outcome.error_class, error=outcome.error,
                    traceback=clip("".join(traceback.format_exception(error)), ERROR_TEXT_LIMIT * 4))  # fmt: skip
    write_result(Path(spec["result_file"]), outcome, records, state)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
