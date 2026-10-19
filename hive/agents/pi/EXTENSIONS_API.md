Excerpted from the installed Pi 0.75.3 package,
`@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts` (plus `AgentToolResult` from
`@earendil-works/pi-agent-core/dist/types.d.ts` and the compaction types from
`dist/core/compaction/compaction.d.ts`). Only what a HIVE extension author needs is kept:
interactive UI, commands, shortcuts, flags, renderers and session-tree types are cut, because
HIVE runs Pi headless (`--mode json -p`, `ctx.hasUI === false`). Declarations are verbatim
except for trimmed doc comments; `// HIVE:` comments are ours.

### Imports available to an extension

Pi loads each file with jiti (TypeScript is transpiled, never type-checked). Besides Node's
built-ins, only these packages resolve: `@earendil-works/pi-coding-agent` (types such as
`ExtensionAPI`), `@earendil-works/pi-agent-core`, `@earendil-works/pi-ai`, `@earendil-works/pi-tui`
and `typebox` (for tool parameter schemas: `import { Type } from "typebox"`).

### Minimal extension

```ts
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
    pi.on("tool_call", async (event, ctx) => {
        if (event.toolName === "bash" && String(event.input.command).includes("rm -rf")) {
            return { block: true, reason: "destructive command blocked" };
        }
    });
    pi.registerTool({
        name: "word_count", // HIVE: must also be listed in [valves].tools
        label: "Word count",
        description: "Count the words in a string.",
        parameters: Type.Object({ text: Type.String() }),
        async execute(toolCallId, params) {
            const count = params.text.split(/\s+/).filter(Boolean).length;
            return { content: [{ type: "text", text: String(count) }], details: { count } };
        },
    });
}
```

### Handlers and context

```ts
/** Handler function type for events */
export type ExtensionHandler<E, R = undefined> = (event: E, ctx: ExtensionContext) => Promise<R | void> | R | void;

/** Context passed to extension event handlers. */
export interface ExtensionContext {
    ui: ExtensionUIContext;          // HIVE: no UI in print mode; hasUI is false
    hasUI: boolean;
    /** Current working directory */  // HIVE: the task's working directory
    cwd: string;
    /** Session manager (read-only) */
    sessionManager: ReadonlySessionManager;
    /** Model registry for API key resolution */
    modelRegistry: ModelRegistry;
    /** Current model (may be undefined) */
    model: Model<any> | undefined;
    /** Whether the agent is idle (not streaming) */
    isIdle(): boolean;
    /** The current abort signal, or undefined when the agent is not streaming. */
    signal: AbortSignal | undefined;
    /** Abort the current agent operation */
    abort(): void;
    /** Whether there are queued messages waiting */
    hasPendingMessages(): boolean;
    /** Gracefully shutdown pi and exit. Available in all contexts. */
    shutdown(): void;
    /** Get current context usage for the active model. */
    getContextUsage(): ContextUsage | undefined;
    /** Trigger compaction without awaiting completion. */
    compact(options?: CompactOptions): void;
    /** Get the current effective system prompt. */
    getSystemPrompt(): string;
}
export interface ContextUsage {
    tokens: number | null;
    contextWindow: number;
    percent: number | null;
}
export interface CompactOptions {
    customInstructions?: string;
    onComplete?: (result: CompactionResult) => void;
    onError?: (error: Error) => void;
}
```

### Tools

```ts
/** Tool definition for registerTool(). */
export interface ToolDefinition<TParams extends TSchema = TSchema, TDetails = unknown, TState = any> {
    /** Tool name (used in LLM tool calls) */
    name: string;
    /** Human-readable label for UI */
    label: string;
    /** Description for LLM */
    description: string;
    /** Optional one-line snippet for the Available tools section in the default system prompt. Custom tools are omitted from that section when this is not provided. */
    promptSnippet?: string;
    /** Optional guideline bullets appended to the default system prompt Guidelines section when this tool is active. */
    promptGuidelines?: string[];
    /** Parameter schema (TypeBox) */
    parameters: TParams;
    /** Optional compatibility shim to prepare raw tool call arguments before schema validation. Must return an object conforming to TParams. */
    prepareArguments?: (args: unknown) => Static<TParams>;
    /** "sequential": one at a time with other tool calls; "parallel": may run concurrently. */
    executionMode?: ToolExecutionMode;
    /** Execute the tool. */
    execute(toolCallId: string, params: Static<TParams>, signal: AbortSignal | undefined, onUpdate: AgentToolUpdateCallback<TDetails> | undefined, ctx: ExtensionContext): Promise<AgentToolResult<TDetails>>;
    // HIVE: renderCall / renderResult (TUI rendering) cut.
}

// From @earendil-works/pi-agent-core
export interface AgentToolResult<T> {
    /** Text or image content returned to the model. */
    content: (TextContent | ImageContent)[];
    /** Arbitrary structured details for logs or UI rendering. */
    details: T;
    /** Hint that the agent should stop after the current tool batch.
     * Early termination only happens when every finalized tool result in the batch sets this to true. */
    terminate?: boolean;
}
```

### Events an extension can handle, and what a handler may return

```ts
/** Fired when a session is started, loaded, or reloaded */
export interface SessionStartEvent {
    type: "session_start";
    reason: "startup" | "reload" | "new" | "resume" | "fork";
    previousSessionFile?: string;
}
/** Source of user input */
export type InputSource = "interactive" | "rpc" | "extension";
/** Fired when user input is received, before agent processing */
export interface InputEvent {
    type: "input";
    text: string;
    images?: ImageContent[];
    source: InputSource;
}
export type InputEventResult = {
    action: "continue";
} | {
    action: "transform";
    text: string;
    images?: ImageContent[];
} | {
    action: "handled";
};
/** Fired after user submits prompt but before agent loop. */
export interface BeforeAgentStartEvent {
    type: "before_agent_start";
    /** The raw user prompt text (after expansion). */
    prompt: string;
    images?: ImageContent[];
    /** The fully assembled system prompt string. */
    systemPrompt: string;
    /** Structured options used to build the system prompt. */
    systemPromptOptions: BuildSystemPromptOptions;
}
export interface BeforeAgentStartEventResult {
    message?: Pick<CustomMessage, "customType" | "content" | "display" | "details">;
    /** Replace the system prompt for this turn. If multiple extensions return this, they are chained. */
    systemPrompt?: string;
}
/** Fired when an agent loop starts */
export interface AgentStartEvent {
    type: "agent_start";
}
/** Fired when an agent loop ends */
export interface AgentEndEvent {
    type: "agent_end";
    messages: AgentMessage[];
}
/** Fired at the start of each turn */
export interface TurnStartEvent {
    type: "turn_start";
    turnIndex: number;
    timestamp: number;
}
/** Fired at the end of each turn */
export interface TurnEndEvent {
    type: "turn_end";
    turnIndex: number;
    message: AgentMessage;
    toolResults: ToolResultMessage[];
}
/** Fired before each LLM call. Can modify messages. */
export interface ContextEvent {
    type: "context";
    messages: AgentMessage[];          // HIVE: a deep copy; the edit applies to this call only
}
export interface ContextEventResult {
    messages?: AgentMessage[];
}
/** Fired before a provider request is sent. Can replace the payload. */
export interface BeforeProviderRequestEvent {
    type: "before_provider_request";
    payload: unknown;                  // HIVE: never change the model in it (P3)
}
export type BeforeProviderRequestEventResult = unknown;
/** Fired after a provider response is received and before the response stream is consumed. */
export interface AfterProviderResponseEvent {
    type: "after_provider_response";
    status: number;
    headers: Record<string, string>;
}
/** Fired when a message ends */
export interface MessageEndEvent {
    type: "message_end";
    message: AgentMessage;             // HIVE: user, assistant and toolResult messages alike
}
export interface MessageEndEventResult {
    /** Replace the finalized message. The replacement must keep the original message role. */
    message?: AgentMessage;
}
interface ToolCallEventBase {
    type: "tool_call";
    toolCallId: string;
}
export interface BashToolCallEvent extends ToolCallEventBase {
    toolName: "bash";
    input: BashToolInput;
}
// HIVE: ReadToolCallEvent, EditToolCallEvent, WriteToolCallEvent, GrepToolCallEvent,
// FindToolCallEvent and LsToolCallEvent follow the same shape for "read", "edit", "write",
// "grep", "find" and "ls".
export interface CustomToolCallEvent extends ToolCallEventBase {
    toolName: string;
    input: Record<string, unknown>;
}
/**
 * Fired before a tool executes. Can block.
 *
 * `event.input` is mutable. Mutate it in place to patch tool arguments before execution.
 * Later `tool_call` handlers see earlier mutations. No re-validation is performed after mutation.
 */
export type ToolCallEvent = BashToolCallEvent | ReadToolCallEvent | EditToolCallEvent | WriteToolCallEvent | GrepToolCallEvent | FindToolCallEvent | LsToolCallEvent | CustomToolCallEvent;
export interface ToolCallEventResult {
    /** Block tool execution. To modify arguments, mutate `event.input` in place instead. */
    block?: boolean;
    reason?: string;
}
/** Narrow a ToolCallEvent by tool name. Built-in names narrow automatically; custom tools need type parameters.
 * Direct narrowing via `event.toolName === "bash"` doesn't work, because CustomToolCallEvent.toolName is `string`. */
export declare function isToolCallEventType(toolName: "bash", event: ToolCallEvent): event is BashToolCallEvent;
export declare function isToolCallEventType<TName extends string, TInput extends Record<string, unknown>>(toolName: TName, event: ToolCallEvent): event is ToolCallEvent & {
    toolName: TName;
    input: TInput;
};
interface ToolResultEventBase {
    type: "tool_result";
    toolCallId: string;
    input: Record<string, unknown>;
    content: (TextContent | ImageContent)[];
    isError: boolean;
}
export interface CustomToolResultEvent extends ToolResultEventBase {
    toolName: string;
    details: unknown;
}
// HIVE: BashToolResultEvent etc. narrow `toolName` and `details` per built-in tool.
/** Fired after a tool executes. Can modify result. */
export type ToolResultEvent = BashToolResultEvent | ReadToolResultEvent | EditToolResultEvent | WriteToolResultEvent | GrepToolResultEvent | FindToolResultEvent | LsToolResultEvent | CustomToolResultEvent;
export declare function isBashToolResult(e: ToolResultEvent): e is BashToolResultEvent;
export interface ToolResultEventResult {
    content?: (TextContent | ImageContent)[];
    details?: unknown;
    isError?: boolean;
}
/** Fired when a tool starts / streams / finishes executing (observe only) */
export interface ToolExecutionStartEvent {
    type: "tool_execution_start";
    toolCallId: string;
    toolName: string;
    args: any;
}
export interface ToolExecutionEndEvent {
    type: "tool_execution_end";
    toolCallId: string;
    toolName: string;
    result: any;
    isError: boolean;
}
/** Fired before context compaction (can be cancelled or customized) */
export interface SessionBeforeCompactEvent {
    type: "session_before_compact";
    preparation: CompactionPreparation;
    branchEntries: SessionEntry[];
    customInstructions?: string;
    signal: AbortSignal;
}
export interface SessionBeforeCompactResult {
    cancel?: boolean;
    compaction?: CompactionResult;
}
/** Fired after context compaction */
export interface SessionCompactEvent {
    type: "session_compact";
    compactionEntry: CompactionEntry;
    fromExtension: boolean;
}

// From dist/core/compaction/compaction.d.ts
export interface CompactionPreparation {
    /** UUID of first entry to keep */
    firstKeptEntryId: string;
    /** Messages that will be summarized and discarded */
    messagesToSummarize: AgentMessage[];
    /** Messages that will be turned into turn prefix summary (if splitting) */
    turnPrefixMessages: AgentMessage[];
    /** Whether this is a split turn (cut point in middle of turn) */
    isSplitTurn: boolean;
    tokensBefore: number;
    /** Summary from previous compaction, for iterative update */
    previousSummary?: string;
    fileOps: FileOperations;
    settings: CompactionSettings;
}
export interface CompactionResult<T = unknown> {
    summary: string;
    firstKeptEntryId: string;
    tokensBefore: number;
    details?: T;
}
```

### ExtensionAPI (the factory's argument)

```ts
export interface ExtensionAPI {
    on(event: "session_start", handler: ExtensionHandler<SessionStartEvent>): void;
    on(event: "session_before_compact", handler: ExtensionHandler<SessionBeforeCompactEvent, SessionBeforeCompactResult>): void;
    on(event: "session_compact", handler: ExtensionHandler<SessionCompactEvent>): void;
    on(event: "session_shutdown", handler: ExtensionHandler<SessionShutdownEvent>): void;
    on(event: "context", handler: ExtensionHandler<ContextEvent, ContextEventResult>): void;
    on(event: "before_provider_request", handler: ExtensionHandler<BeforeProviderRequestEvent, BeforeProviderRequestEventResult>): void;
    on(event: "after_provider_response", handler: ExtensionHandler<AfterProviderResponseEvent>): void;
    on(event: "before_agent_start", handler: ExtensionHandler<BeforeAgentStartEvent, BeforeAgentStartEventResult>): void;
    on(event: "agent_start", handler: ExtensionHandler<AgentStartEvent>): void;
    on(event: "agent_end", handler: ExtensionHandler<AgentEndEvent>): void;
    on(event: "turn_start", handler: ExtensionHandler<TurnStartEvent>): void;
    on(event: "turn_end", handler: ExtensionHandler<TurnEndEvent>): void;
    on(event: "message_start", handler: ExtensionHandler<MessageStartEvent>): void;
    on(event: "message_update", handler: ExtensionHandler<MessageUpdateEvent>): void;
    on(event: "message_end", handler: ExtensionHandler<MessageEndEvent, MessageEndEventResult>): void;
    on(event: "tool_execution_start", handler: ExtensionHandler<ToolExecutionStartEvent>): void;
    on(event: "tool_execution_update", handler: ExtensionHandler<ToolExecutionUpdateEvent>): void;
    on(event: "tool_execution_end", handler: ExtensionHandler<ToolExecutionEndEvent>): void;
    on(event: "tool_call", handler: ExtensionHandler<ToolCallEvent, ToolCallEventResult>): void;
    on(event: "tool_result", handler: ExtensionHandler<ToolResultEvent, ToolResultEventResult>): void;
    on(event: "input", handler: ExtensionHandler<InputEvent, InputEventResult>): void;
    // HIVE: also declared but irrelevant headless: resources_discover, session_before_switch,
    // session_before_fork, session_before_tree, session_tree, model_select,
    // thinking_level_select, user_bash. Pi 0.75.3 has no other event names; a handler
    // registered under any other name is silently never called.

    /** Register a tool that the LLM can call. */
    registerTool<TParams extends TSchema = TSchema, TDetails = unknown, TState = any>(tool: ToolDefinition<TParams, TDetails, TState>): void;
    /** Send a custom message to the session. */
    sendMessage<T = unknown>(message: Pick<CustomMessage<T>, "customType" | "content" | "display" | "details">, options?: {
        triggerTurn?: boolean;
        deliverAs?: "steer" | "followUp" | "nextTurn";
    }): void;
    /**
     * Send a user message to the agent. Always triggers a turn.
     * When the agent is streaming, use deliverAs to specify how to queue the message.
     */
    sendUserMessage(content: string | (TextContent | ImageContent)[], options?: {
        deliverAs?: "steer" | "followUp";
    }): void;
    /** Append a custom entry to the session for state persistence (not sent to LLM). */
    appendEntry<T = unknown>(customType: string, data?: T): void;
    /** Execute a shell command. */
    exec(command: string, args: string[], options?: ExecOptions): Promise<ExecResult>;
    /** Get the list of currently active tool names. */
    getActiveTools(): string[];
    /** Get all configured tools with parameter schema and source metadata. */
    getAllTools(): ToolInfo[];
    /** Set the active tools by name. */
    setActiveTools(toolNames: string[]): void;
    /** Get current thinking level. */
    getThinkingLevel(): ThinkingLevel;
    /** Shared event bus for extension communication. */
    events: EventBus;

    // HIVE: FORBIDDEN. The model and its thinking level are fixed by the experiment (P3);
    // a run that uses these is marked model_mismatch where HIVE can see it, and breaks the
    // experiment even where it can't.
    setModel(model: Model<any>): Promise<boolean>;
    setThinkingLevel(level: ThinkingLevel): void;
    registerProvider(name: string, config: ProviderConfig): void;
    unregisterProvider(name: string): void;

    // HIVE: cut (interactive only): registerCommand, registerShortcut, registerFlag, getFlag,
    // registerMessageRenderer, setSessionName, getSessionName, setLabel, getCommands.
}
```
