Pinned to **strands-agents 1.61.1** (with openai 2.54.0), Python 3.12. Every signature below was excerpted
from the installed package (`strands/plugins/plugin.py`, `strands/plugins/decorator.py`,
`strands/tools/decorator.py`, `strands/hooks/registry.py`, `strands/hooks/events.py`,
`strands/types/tools.py`, `strands/types/content.py`, `strands/agent/agent.py`,
`strands/agent/conversation_manager/conversation_manager.py`). Only what an extension needs is shown.

### A complete extension (`extensions/guard.py`)

```python
from strands import tool
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook


class Guard(Plugin):
    name = "guard"  # must equal the file's stem

    def init_agent(self, agent) -> None:  # optional; runs first, once, when the agent is built
        agent.state.set("blocked", 0)

    @hook  # registered for the event type in the annotation
    def block_rm(self, event: BeforeToolCallEvent) -> None:
        command = str(event.tool_use["input"].get("command", ""))
        if event.tool_use["name"] == "shell" and "rm -rf" in command:
            event.cancel_tool = "rm -rf is blocked; delete files one by one"
            event.agent.state.set("blocked", event.agent.state.get("blocked") + 1)

    @hook
    def shorten(self, event: AfterToolCallEvent) -> None:
        content = event.result.get("content", [])
        if content and "text" in content[0] and len(content[0]["text"]) > 20000:
            event.result = {**event.result, "content": [{"text": content[0]["text"][:20000] + "\n[truncated]"}]}

    @tool  # offered to the model only if "word_count" is listed in the tools valve
    def word_count(self, text: str) -> int:
        """Count the words in text. (The docstring is the tool description the model sees.)"""
        return len(text.split())


def extension():
    return Guard()
```

### `strands.plugins`

```python
class Plugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...            # a class attribute `name = "..."` satisfies it
    def __init__(self) -> None: ...       # discovers @hook and @tool methods; call super().__init__() if you override
    @property
    def hooks(self) -> list[HookCallback]: ...
    @property
    def tools(self) -> list[DecoratedFunctionTool]: ...
    def init_agent(self, agent: "Agent") -> None | Awaitable[None]: ...   # sync or async

def hook(func: HookCallback | None = None)   # @hook or @hook(); event type(s) inferred from the annotation;
                                             # a union (A | B) registers for each member
```

Attachment order per plugin (`plugins/registry.py` `add_and_init`): duplicate-name check, `init_agent(agent)`,
then the `@hook` methods, then the `@tool` methods. A duplicate plugin name raises `ValueError`. Strands itself
lets a `@tool` silently replace a same-named tool; HIVE does not: a tool name already taken by a vended tool or
another extension is `harness_load_error`. `agent.tool_registry.register_tool(...)` or `.replace(...)` once the
run has started is refused (`ToolRefused`) and ends the run as `extension_error`.

### `strands.tool`

```python
def tool(func=None, description: str | None = None, inputSchema: dict | None = None,
         name: str | None = None, context: bool | str = False)
```

The schema comes from the type hints, the description from the docstring (Args: sections describe
parameters). `context=True` injects `tool_context: ToolContext` (fields: `tool_use`, `agent`,
`invocation_state`, `cancel_signal`). A tool may be sync or async. A raised exception becomes a
`{"status": "error"}` result for the model; the run continues.

### Hooks without a Plugin decorator

```python
# strands.agent.agent.Agent
def add_hook(self, callback: HookCallback[TEvent],
             event_type: type[TEvent] | list[type[TEvent]] | None = None, *, order: float = 0) -> None

# strands.hooks.HookRegistry (agent.hooks)
def add_callback(self, event_type: type[TEvent] | list[type[TEvent]] | None,
                 callback: HookCallback[TEvent], *, order: float = 0) -> None
def add_hook(self, hook: HookProvider) -> None

class HookProvider(Protocol):
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None: ...

class HookOrder:  # lower runs first; equal orders keep registration order
    SDK_FIRST = -100; INTERVENTION_OUTPUT = -90; DEFAULT = 0; MODEL_ROUTING = 50; INTERVENTION_INPUT = 90; SDK_LAST = 100
```

Callbacks take one event argument and may be `async def`, except for `AgentInitializedEvent`. "After"
events (AfterModelCall, AfterToolCall, AfterTools, AfterInvocation) run callbacks in **reverse** order.
Any exception a callback raises propagates and **ends the run** (`EventLoopException`; HIVE: `extension_error`).

### Events (`from strands.hooks import ...`)

Every event has `agent` (the running `Agent`). Assigning any field not listed as writable raises
`AttributeError("Property <name> is not writable")`, which ends the run.

| Event | Read | Writable |
|---|---|---|
| `BeforeInvocationEvent` | `invocation_state: dict`, `messages: list[Message] \| None` | `messages`, `cancel: bool \| str` |
| `AfterInvocationEvent` | `invocation_state`, `result: AgentResult \| None` | `resume: str \| list[ContentBlock] \| list[Message] \| None` |
| `BeforeModelCallEvent` | `invocation_state`, `projected_input_tokens: int \| None` | `cancel: bool \| str` |
| `AfterModelCallEvent` | `invocation_state`, `stop_response: ModelStopResponse \| None` (`.message`, `.stop_reason`), `exception: Exception \| None` | `retry: bool` |
| `BeforeToolsEvent` | `message: Message` (the assistant turn holding the tool calls), `invocation_state` | `cancel: bool \| str` |
| `AfterToolsEvent` | `message`, `invocation_state` | `end_turn: bool \| str \| list[ContentBlock]` |
| `BeforeToolCallEvent` | `tool_use: ToolUse`, `selected_tool: AgentTool \| None`, `invocation_state` | `tool_use`, `selected_tool`, `cancel_tool: bool \| str` |
| `AfterToolCallEvent` | `tool_use`, `selected_tool`, `result: ToolResult`, `exception: Exception \| None`, `cancel_message: str \| None`, `duration: float \| None` | `result`, `retry: bool` |
| `MessageAddedEvent` | `message: Message` | none |
| `AgentInitializedEvent` | (agent only) | none (sync callbacks only) |

### Data shapes (`strands.types`)

```python
ToolUse    = TypedDict: {"toolUseId": str, "name": str, "input": Any}
ToolResult = TypedDict: {"toolUseId": str, "status": Literal["success", "error"],
                         "content": list[ToolResultContent]}   # ToolResultContent keys: text | json | image | document
Message    = TypedDict: {"role": Literal["user", "assistant"], "content": list[ContentBlock]}
ContentBlock keys: text, toolUse, toolResult, reasoningContent, image, document, cachePoint, ...
AgentResult (dataclass): stop_reason, message, metrics, state, interrupts, structured_output, checkpoint
StopReason: "end_turn" | "tool_use" | "max_tokens" | "limit_turns" | "limit_total_tokens" | "cancelled" | ...
```

### The agent, as hooks see it (`event.agent`)

- `agent.messages: list[Message]`: the conversation, mutable in place (edits persist).
- `agent.system_prompt: str | None`: a property with a setter; HIVE's prompt ends with the absolute working directory.
- `agent.state`: per-run key/value store, `get(key=None)`, `set(key, value)` (JSON-serializable values), `delete(key)`.
- `agent.tool_names: list[str]`, `agent.hooks: HookRegistry`, `agent.cancel_signal: threading.Event`.
- `agent.model`: **read only for extensions**. Assigning it, or calling `agent.model.update_config(...)`, is `model_mismatch`.
- `agent.conversation_manager`: may be replaced in `init_agent` by a subclass of
  `strands.agent.conversation_manager.ConversationManager`:

```python
class ConversationManager(ABC):
    def __init__(self) -> None: ...      # call super().__init__()
    @abstractmethod
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None: ...                       # after each invocation
    @abstractmethod
    def reduce_context(self, agent: "Agent", e: Exception | None = None, **kwargs: Any) -> None: ...  # on overflow (e set)

class SlidingWindowConversationManager(ConversationManager):   # HIVE's default
    def __init__(self, window_size: int = 40, should_truncate_results: bool = True, *,
                 per_turn: bool | int = False, pin_first: int | None = None, proactive_compression=None): ...
```
