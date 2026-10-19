Excerpted from the installed packages in HIVE's pinned Deep Agents venv: deepagents 0.7.28,
langchain 1.4.6 (`langchain/agents/middleware/types.py`, `langchain/agents/factory.py`),
langchain-core 1.6.12 (`langchain_core/messages`, `langchain_core/tools`), langgraph 1.2.16
(`langgraph/prebuilt/tool_node.py`, `langgraph/types.py`) and langchain-openai 1.7.2.
Signatures are verbatim; doc comments are trimmed to what an extension author needs, and
`# HIVE:` comments are ours.

### The entry convention and a minimal extension

Each `extensions/<name>.py` defines `def extension():` (sync, no arguments) returning a list
of `AgentMiddleware` and/or `BaseTool` objects. The runner imports each file on its own with
importlib, calls `extension()`, and passes the middleware (in list order) and the tools to
`create_deep_agent`. A file cannot import another extension file.

```python
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool


class CarefulShell(AgentMiddleware):
    """Blocks destructive shell commands and reminds the model to run its code."""

    def wrap_model_call(self, request, handler):
        reminder = "\n\nBefore you finish, run your code once with the execute tool."
        system = SystemMessage(content=(request.system_prompt or "") + reminder)
        return handler(request.override(system_message=system))

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] == "execute" and "rm -rf" in str(call["args"].get("command", "")):
            return ToolMessage(content="Blocked: destructive command.", tool_call_id=call["id"], status="error")
        return handler(request)


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


def extension():
    return [CarefulShell(), word_count]  # HIVE: "word_count" must also be in [valves] tools
```

### How HIVE assembles the main agent

`create_deep_agent(model, tools=[...your tools], system_prompt=SYSTEM.md + working-directory
note, middleware=[...your middleware, HIVE guards], backend=..., skills=...)`. The middleware
stack of the main agent, outermost first, as deepagents 0.7.28 builds it:

1. `FilesystemMiddleware` (the file and `execute` tools), `SubAgentMiddleware` (only with
   `general_purpose_subagent = true`), `_ParentNotifierMiddleware`, `SummarizationMiddleware`,
   `PatchToolCallsMiddleware`: the deepagents core.
2. **Your middleware**, from every extension file in file-name order, each file's list in order.
   deepagents swaps a middleware whose `name` equals a stack entry's into that entry's place,
   wherever it sits, so HIVE allows that for one name only: a middleware named
   `"SummarizationMiddleware"` replaces the built-in summarization in place (one extension at
   most). Any other name used in these stacks (`FilesystemMiddleware`, `SubAgentMiddleware`,
   `_ParentNotifierMiddleware`, `PatchToolCallsMiddleware`, `SkillsMiddleware`,
   `AnthropicPromptCachingMiddleware`, `UnsupportedContentMiddleware`, ...) or by HIVE's guards,
   and two middleware with the same `name`, fail to load (`harness_load_error`).
3. HIVE's guards: `HiveToolAllowlist` (the `tools` valve: hides unlisted tools from the model and
   refuses calls to them) and `HiveToolErrors` (a `ToolErrorMiddleware`: an exception raised by a
   tool becomes an error `ToolMessage` the model sees).
4. `SkillsMiddleware` (only with `skills/`), `AnthropicPromptCachingMiddleware` (a no-op for
   OpenAI), `UnsupportedContentMiddleware`.

`wrap_*` hooks nest in this order (the first is outermost: it sees the request first and the
response last). `before_*` hooks run in this order, `after_*` hooks in reverse.

The model-call limit is not a middleware: HIVE counts every request sent through the agent's
chat model, in a callback, against one limit per trial. Each `handler` call inside
`wrap_model_call` is one call, and the general-purpose subagent's calls count toward the same
limit. A request past the limit is refused before it is sent and the run ends as `agent_error`,
even if your code catches the error.

The run: `agent.invoke({"messages": [HumanMessage(task instruction)]})`, sync. Per run:
`before_agent` once, then per turn `before_model`, `wrap_model_call` around the provider call,
`after_model`, then `wrap_tool_call` around each tool call the reply asked for; `after_agent` at
the end. A hook that raises ends the run (`extension_error`).

Built-in tools (schemas the model sees): `ls(path)`, `read_file(file_path, offset, limit)`,
`write_file(file_path, content)`, `edit_file(file_path, old_string, new_string, replace_all)`,
`delete(file_path)`, `glob(pattern, path)`, `grep(pattern, path, glob, output_mode, max_count)`,
`execute(command, timeout)`, and `task(description, subagent_type)` with the subagent.

### `langchain.agents.middleware.AgentMiddleware`

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    state_schema: type[StateT] = _DefaultAgentState  # a TypedDict extending AgentState, to add state keys
    tools: Sequence[BaseTool]                          # extra tools this middleware registers
    @property
    def name(self) -> str: ...                         # defaults to the class name; must be unique

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]: ...
    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]: ...
    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...
    # Each has an async twin (abefore_agent, abefore_model, awrap_model_call, aafter_model,
    # awrap_tool_call, aafter_agent). HIVE: the runner invokes the sync path, so define the sync
    # method; a class that overrides only the async twin fails to load.
```

A hook counts as active only when the class overrides it (`m.__class__.before_model is not
AgentMiddleware.before_model`, factory.py). `before_*`/`after_*` return a state update (a dict
merged into state; `messages` go through the add-messages reducer, so returned messages are
appended, and `RemoveMessage(id=...)` deletes one) or `None` for no change.

```python
JumpTo = Literal["tools", "model", "end"]

def hook_config(*, can_jump_to: list[JumpTo] | None = None) -> Callable[[CallableT], CallableT]: ...
# Put on before_model / after_model / before_agent / after_agent; the hook may then return
# {"jump_to": "model" | "tools" | "end", ...}. "model" from after_agent continues the run.
```

Decorator forms (each returns an `AgentMiddleware` instance; `name=` sets its name):

```python
@before_agent / @before_model / @after_model / @after_agent
    (func=None, *, state_schema=None, tools: list[BaseTool] | None = None,
     can_jump_to: list[JumpTo] | None = None, name: str | None = None)
@wrap_model_call / @wrap_tool_call
    (func=None, *, state_schema=None, tools: list[BaseTool] | None = None, name: str | None = None)
@dynamic_prompt(func)  # func(request: ModelRequest) -> str | SystemMessage: the call's system prompt
```

### `ModelRequest`, `ModelResponse`, `ExtendedModelResponse`

```python
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel                  # HIVE: read-only. Replacing it is forbidden (P3). It also holds the
                                          # trial's gateway token and the provider client: reading them or
                                          # calling the client directly is forbidden (such calls skip the
                                          # runner's checks; the gateway still refuses and bills them).
    messages: list[AnyMessage]            # excluding the system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any]        # HIVE: read-only. Changing it is forbidden (P3).
    @property
    def system_prompt(self) -> str | None: ...  # the text of system_message
    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]: ...
    # keys: system_message, messages, tool_choice, tools, response_format, state
    # (and model, model_settings: HIVE forbids both; validate() rejects them)

class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]             # usually one AIMessage
    structured_response: ResponseT | None = None

class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None   # an extra state update applied after the model node
```

Assigning attributes on a `ModelRequest` is deprecated; use `override`, which returns a copy.

### `langgraph.prebuilt.tool_node.ToolCallRequest`

```python
@dataclass
class ToolCallRequest:
    tool_call: ToolCall           # {"name": str, "args": dict, "id": str}
    tool: BaseTool | None
    state: Any
    runtime: ToolRuntime
    def override(self, **overrides: Unpack[_ToolCallRequestOverrides]) -> ToolCallRequest: ...
    # keys: tool_call, tool, state
```

### Messages, tools and commands

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage
ToolMessage(content: str | list, tool_call_id: str, status: Literal["success", "error"] = "success", ...)
AIMessage(content: str | list, tool_calls: list[ToolCall] = [], ...)  # .text: the reply as a str
RemoveMessage(id: str)                 # in a state update: delete the message with that id

from langchain_core.tools import tool, BaseTool
@tool                                   # schema from the signature and docstring (docstring required)
def name(arg: type, ...) -> type: ...

from langgraph.types import Command
Command(*, update: Any | None = None, goto=(), ...)  # a state update returned from wrap_tool_call
```

### Shipped middleware you may use or subclass

From `langchain.agents.middleware`: `ToolCallLimitMiddleware(*, tool_name=None, thread_limit=None,
run_limit=None, exit_behavior='continue')`, `ToolRetryMiddleware`, `TodoListMiddleware` (adds a
`write_todos` tool: list it in `tools`), `ContextEditingMiddleware(*, edits=None,
token_count_method='approximate')` with `ClearToolUsesEdit(trigger=100000, clear_at_least=0,
keep=3, clear_tool_inputs=False, exclude_tools=(), placeholder='[cleared]')`.

Never use the ones that build or call another model: `ModelFallbackMiddleware`,
`LLMToolSelectorMiddleware`, `LLMToolEmulator`, `SummarizationMiddleware` (langchain's or
deepagents'), and never `ChatOpenAI` / `init_chat_model` or any other chat model class.
