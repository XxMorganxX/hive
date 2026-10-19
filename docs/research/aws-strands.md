# AWS Strands Agents: evaluation as a HIVE base framework

Researched 2026-10-19. Sources: the live docs at strandsagents.com, PyPI/npm registries, the GitHub API, and shallow clones of the repos (I read the source; I did not run it). Every claim is tagged:

- **[V]** verified: read in source, docs, or registry metadata.
- **[I]** inferred: my reading of what the verified facts mean for HIVE.

---

## TL;DR

Strands is a strong candidate. In two respects it fits HIVE's actual pipeline better than anything else we've looked at:

1. **Harbor already ships a Strands adapter [V].** Harbor's `strands` installed agent (`src/harbor/agents/installed/strands.py`, added 2026-09-11, updated 2026-10-15) runs a **user-supplied Python factory** `module:symbol` with signature `(sandbox, model=None, **factory_kwargs) -> strands.Agent` inside the task container. It records every message to an events file, converts the run to an **ATIF trajectory**, and fills Harbor's token counts. Our optimizer would edit one Python package (the factory), and Harbor would run and score it unchanged.
2. **AWS is itself working on "harness optimization" on top of Strands [V].** Strands Labs publishes `strands-harness-optimizer` (Formula / RewardFunction / FormulaOptimizer / Trainer; an LLM optimizer that reads traces and rewrites the system prompt, skills, and tool descriptions). It also publishes `benchmark-harnesses` (Simple Strands Agent, a lean coding harness reporting strong SWE-Bench and Terminal-Bench 2 results). Its own blog says it benchmarks Strands harness against Claude Code, Codex, and oh-my-pi **on Harbor**.

The main weaknesses:

- Python is the first-class language: the evals SDK, the Harbor adapter, and the optimizer are Python-only.
- Bedrock is the default everywhere, though it is easy to override.
- Multi-agent topology is expressed in **code** (GraphBuilder, Swarm constructor), not as a declarative spec.
- Harbor's adapter must return a single `Agent`, so a Graph or Swarm has to be wrapped as an agent's tool.
- Churn is very high: weekly minor releases, a repo reorg in 2026, and the harness layer is 0.x.

---

## 0. What "Strands" is today: the package map

The ecosystem was reorganized in 2026. The old `sdk-python`, `sdk-typescript`, and `docs` repos now live in one monorepo, **`strands-agents/harness-sdk`** [V]. The `sdk-python` URL redirects there. `sdk-typescript` and `docs` are archived with "moved" banners. `agent-builder` is archived on GitHub [V].

| Package | Version (2026-10-19) | What it is | Status |
|---|---|---|---|
| `strands-agents` (PyPI) | **1.61.1** (2026-10-15); first release 2025-05-14 | Python SDK: agent loop, providers, tools, hooks, multi-agent | SemVer 1.x [V] |
| `@strands-agents/sdk` (npm) | **1.21.0** (2026-10-14); first release 2025-11-28 | TypeScript SDK, Node >= 22 | SemVer 1.x [V] |
| `strands-harness` (PyPI) / `@strands-agents/harness` (npm) | **0.3.1** / **0.3.0**; first release 2026-09-21 | "Fully assembled" batteries-included agent (`create_harness()`): shell, file, and web tools, sub-agent, memory, sessions, context management, tuned prompt | **0.x, minor = breaking** [V] |
| `@strands-agents/cli` | 0.2.1 | `strands "prompt"` terminal chat on top of the TS harness | 0.x [V] |
| `strands-agents-tools` | 0.8.9 (2026-09-15) | Community tools: `shell`, `editor`, `file_read/write`, `python_repl`, `workflow`, `swarm`, `agent_graph`, `use_agent`, `think`, `http_request`, `mcp_client`, memory backends, etc. | 0.x [V] |
| `strands-agents-evals` | **1.6.0** (2026-10-14); first release 2025-12-03 | Evals SDK (Python only; no npm package exists) | [V] |
| `strands-agents-builder` | 0.1.10 (2025-09-17) | Old interactive agent-builder CLI | **Archived; ignore** [V] |
| `strands-harness-optimizer` (Strands Labs) | 0.0.1 | Automatic harness optimization from rollouts | Experimental [V] |
| Strands Box (`strands-agents/box`) | 0.1.x, created 2026-10-02 | macOS Seatbelt sandbox plus Dogwood (Cedar-like) policy engine that can wrap any harness (has "Run pi in a box" and "Run Claude Code" guides) | Pre-release [V] |

- License: **Apache-2.0** across all of them [V].
- GitHub: harness-sdk has ~8.9k stars and ~1,026 open issues, and was pushed 2026-10-19 [V].

Doc URL pattern: `https://strandsagents.com/docs/user-guide/<path>/`. Source is at `harness-sdk/site/src/content/docs/user-guide/<path>.mdx` [V].

---

## 1. Architecture

### The model-driven loop [V]
Docs: `/docs/user-guide/sdk/agents/agent-loop/`. Source: `strands-py/src/strands/event_loop/event_loop.py`, `strands-py/src/strands/agent/agent.py`, `strands-ts/src/agent/agent.ts`.

The loop is classic ReAct with no planner graph:

1. Call the model.
2. If `stop_reason == tool_use`, validate and run the tools (errors go back to the model as error results, not exceptions).
3. Append the results and call the model again.
4. Stop on `end_turn`.

Built-in loop controls:

- **Per-invocation limits:** `agent(prompt, limits={"turns": N, "output_tokens": N, "total_tokens": N})`. Limits are checked at the top of each cycle. Stop reasons are `limit_turns`, `limit_total_tokens`, and `limit_output_tokens`. History stays valid, so the agent can be re-invoked.
- **Cancellation:** `agent.cancel()`, or an external `cancel_signal` (a `threading.Event` in Python, an `AbortSignal` in TS). It propagates into `as_tool()` children.
- **Retry strategies:** `retry_strategy=`.
- **Other reported stop reasons:** `cancelled`, `max_tokens`, `content_filtered`, `guardrail_intervened`, `stop_sequence`.

### The `Agent` class [V]
Python constructor parameters (`strands-py/src/strands/agent/agent.py:217-249`): `model, messages, tools, system_prompt, structured_output_model, callback_handler, conversation_manager, trace_attributes, aux_model, agent_id, name, description, state, context_manager, plugins, hooks, interventions, session_manager, memory_manager, tool_executor, retry_strategy, concurrent_invocation_mode, checkpointing, sandbox, storage, background_tasks`.

- A **string model means a Bedrock model ID** (`agent.py:382`).
- The SDK default is `global.anthropic.claude-sonnet-4-6` in `us-west-2` (`models/bedrock.py:45`).
- Invocation styles:
  - `agent("prompt")` (sync)
  - `await agent.invoke_async(...)`
  - `agent.stream_async(...)`
  - TS: `await agent.invoke(...)` and `agent.stream(...)`

### Tools [V]
Docs: `/docs/user-guide/sdk/tools/`.

- Python `@tool`-decorated functions, with schemas from type hints and docstrings. TS tools use Zod.
- Module or file paths as strings.
- `ToolProvider`s; MCP clients (`MCPClient.load_servers(...)`); other Agents (auto-converted to tools).
- Hot reload from a `./tools/` directory (`load_tools_from_directory=True`).
- **Tool executors:** concurrent (default) or sequential (`/sdk/tools/executors/`).
- `@tool(context=True)` gives access to `ToolContext` (agent, `invocation_state`, `cancel_signal`).
- **Vended (in-SDK) tools** in `strands.vended_tools`: `shell`, `file_editor`, `python_repl`, `http_request`, `web_fetch`, `subagent`, `swarm`, `a2a_client`, `mcp_router`, `notebook`, `handoff_to_user`, `sleep`, `stop`.

### Hooks system [V]
Docs: `/docs/user-guide/sdk/agents/hooks/` and `/hooks-events/`. Source: `strands-py/src/strands/hooks/events.py`.

- **Typed events:**
  - Agent lifecycle: `AgentInitialized`, `Before/AfterInvocation`, `MessageAdded`, `Before/AfterModelCall`, `BeforeTools/AfterTools` (batch), `Before/AfterToolCall`.
  - Multi-agent: `MultiAgentInitialized`, `Before/AfterMultiAgentInvocation`, `Before/AfterNodeCall`.
  - TS adds streaming events (`ModelStreamUpdate`, `ContentBlock`, `ToolResult`, `AgentResult`, `MultiAgentHandoff`, ...).
- **Writable properties** (this is how hooks change behavior, not just observe it):
  - `BeforeToolCallEvent`: `cancel_tool`, swap `selected_tool`, rewrite `tool_use` args.
  - `AfterToolCallEvent`: rewrite `result`, `retry`.
  - `AfterModelCallEvent`: `retry`.
  - `BeforeToolsEvent`: `cancel`.
  - `AfterToolsEvent`: `end_turn`.
  - `AfterInvocationEvent`: `resume` (re-enter the loop with new input).
- Registration: `agent.add_hook(cb, EventType)`, the `hooks=[HookProvider]` constructor argument, or bundled into **Plugins**.

### Middleware [V] (newer, `/sdk/agents/middleware/`)
- Wraps two stages: `InvokeModelStage` and `ExecuteToolStage`.
- Each stage has **Input** (transform the context, e.g. `context.replace(system_prompt=...)` for one model call), **Wrap** (call `next` 0, 1, or N times to short-circuit, pass through, or retry), and **Output** phases.
- This is the clean place for per-call prompt or context surgery.

### Conversation managers, context management, sessions [V]
- **Conversation managers** (`/sdk/agents/conversation-management/`): `NullConversationManager`, `SlidingWindowConversationManager(window_size)`, `SummarizingConversationManager` (custom summarizer prompt and model). They support message pinning (`pin_first`), proactive compression, and custom subclasses.
- **Context manager** (newer, `/sdk/context-management/`): `context_manager="auto" | "agentic"` or a dict of strategies.
  - Presets: `proactive_summarization`, `large_tool_offloading`, `overflow_protection`, `stale_tool_cleanup`.
  - Custom `Offload` strategies (Truncate / Summarize / Drop with conditions and targets), plus a stash and a retrieval tool.
  - The "auto" defaults were tuned on ContextBench (design doc `team/designs/0015-context-manager.md`).
  - Presets are **experimental** (added 2026-09-14).
- **Session managers** (`/sdk/agents/session-management/`): `FileSessionManager`, `S3SessionManager`, `RepositorySessionManager` (custom backends), and `SnapshotSessionManager`. They persist messages, state, and multi-agent state, and support immutable snapshots with restore (`/sdk/agents/snapshots/`).
- **Memory** (`/sdk/memory/`) is separate from sessions: a long-term `memory_manager` with stores.

### Plugins, interventions, steering [V]
- **Plugins** (`/sdk/plugins/`) bundle hooks and tools. Vended: `skills` (AgentSkills spec), `context_offloader`, `context_injector`, `goal` (**GoalLoop**: validate the response against a goal or validator function, inject feedback, retry), `steering`.
- **Interventions** (`/sdk/agents/interventions/`): `cedar` authorization, `hitl` (human approval), `steering` handlers that let the agent self-correct.

### Strands harness layer [V] (`/docs/user-guide/harness/`)
`create_harness(**opts)` returns a **plain `strands.Agent`** with:

- **Defaults:**
  - `model="bedrock/global.anthropic.claude-opus-5"`
  - built-in tools `shell, read, write, edit, web_fetch, web_search, programmatic_tool_caller, subagent` (a "generalist" delegate)
  - plugins `todos, environment`
  - `context_manager="auto"`, prompt caching, file memory in `./.agent/memory`, skills from `./.agent/skills`
  - a tuned `HARNESS_CONTRACT` system prompt (`harness-py/src/strands_harness/prompt.py`)
- **Overriding:** every default can be overridden. Unknown kwargs pass through to `Agent(...)`, and `system_prompt=` replaces the contract (`/harness/reference/configuration/`).
- **Model strings:** the harness, unlike the SDK, accepts `provider/name` strings: `bedrock`, `bedrock-mantle`, `anthropic`, `openai`, `google`, `ollama`, `litellm`.

---

## 2. Extension surfaces an optimizer could edit

| Surface | How it's expressed | Declarative? | Notes |
|---|---|---|---|
| System prompt | `system_prompt=` string or `SystemContentBlock` list; harness `instructions=` appends to the contract | **Data** (a string) | Easiest valve; can live in a `.md` or JSON file [V] |
| Per-call prompt or context rewrite | Middleware `InvokeModelStage.Input` | Code | [V] |
| Tool set selection | `tools=[...]` (module-path strings allowed); harness `builtin_tools=[names]` | **Mostly data**: names and paths | [V] |
| Tool descriptions and schemas | Docstrings / `@tool(name, description)`; `make_shell(description=...)`, `make_file_editor(...)` | Data inside code | Harbor's runner overrides descriptions this way [V] |
| New tools | Python functions | Code | Hot reload from `./tools/` [V] |
| Hooks (guards, retries, arg pinning, loop resume) | `HookProvider` classes / callbacks | Code | Powerful, typed [V] |
| Context management | `context_manager={"strategies": ["preset", Offload...]}` | **Semi-declarative** (preset names plus thresholds) | Presets experimental [V] |
| Conversation manager | Class plus kwargs (`window_size`, summary ratio, summarizer prompt) | Kwargs | [V] |
| Loop budgets | `limits={turns, total_tokens, output_tokens}`; Swarm/Graph limits | **Data** | [V] |
| Skills | AgentSkills directories (`SKILL.md`) via the `skills` plugin / harness `skills=` | **Data** (Markdown) | [V] |
| Model and reasoning | `Model` instance, or harness `model="provider/name"`, `effort=` | Data | [V] |
| Multi-agent topology | `GraphBuilder` / `Swarm(...)` / `as_tool()` / `make_swarm` / `make_subagent` | **Code** | No YAML/JSON topology spec [V] |
| Whole-agent config file | `strands.experimental.config_to_agent(json)`: keys `model` (Bedrock only), `prompt`, `tools`, `name` | **Declarative but thin and experimental** | `/sdk/experimental/agent-config/` [V] |

Overall [I]: the "harness" is a Python module, not a config document. The cleanest design for HIVE is to have the optimizer edit **one harness package**: a `factory.py` that builds the Agent, plus `prompt.md`, `tools/`, `hooks/`, `skills/`, and `config.yaml`. That package becomes Harbor's `factory_path`. This is very close to how Simple Strands Agent is organized (Hydra YAML for every knob).

---

## 3. Multi-agent primitives and topology fit

Docs: `/docs/user-guide/sdk/multi-agent/multi-agent-patterns/`. Source: `strands-py/src/strands/multiagent/{graph.py,swarm.py,spec.py,base.py}` and the matching TS files `strands-ts/src/multiagent/*` [V].

| Primitive | Semantics [V] | Python | TS |
|---|---|---|---|
| **Agents-as-tools** | Pass an `Agent` in `tools=[...]` or `agent.as_tool(name, description, preserve_context=False, delegate=False)`. A fresh context per call by default. `delegate=True` returns the child's answer verbatim, with no extra orchestrator turn. Cancellation propagates. | yes | yes |
| **Vended `subagent` tool** (`make_subagent`) | The model spawns a child at runtime. **Authority modes** per axis (`agent_type`, `instructions`, `tools`, `model`): `Fixed` / `Inherit` / `Open` / `Choice`. The child's tools can't exceed the parent's. Can fork the last N parent messages. | yes | yes |
| **Swarm** | A pool of agents. Each gets a `handoff_to_agent(agent_name, message, context)` tool. **One agent is active at a time.** Shared context shows the task, the handoff history, and contributed knowledge. Limits: `max_handoffs`, `max_iterations`, `execution_timeout`, `node_timeout`, ping-pong detection. | yes | yes |
| **Vended `swarm` tool** (`make_swarm`) | The model assembles a Swarm team at runtime under authority modes | yes | (not listed in TS vended tools) |
| **Graph** | `GraphBuilder().add_node(agent_or_multiagent).add_edge(a, b, condition=fn)`. Conditional edges (Python also has `invocation_state`-aware conditions), **parallel** execution of ready nodes, cycles with `set_max_node_executions`, `reset_on_revisit`, timeouts. **Nodes can be Swarms or Graphs (nesting)** and remote `A2AAgent`s. `serialize_state` / `deserialize_state` for resume. | yes | yes (`maxConcurrency`, `preserveContext`) |
| **Workflow** | In the SDK docs this is now **a pattern you implement in code** (sequential chaining). The older `workflow` tool still exists in `strands-agents-tools`. | pattern | pattern |
| **A2A** | `A2AAgent(url)` client (usable as a tool, Graph node, or Swarm member, mostly Python) plus an A2A server with a per-context agent factory. Pin a minor version (A2A is listed as an evolving standard). | yes | yes |
| `invocation_state` | A shared non-LLM dict propagated to every agent, tool, and hook in a Graph or Swarm | yes | n/a |

How each HIVE topology maps:

- **Hierarchical / manager-worker:** **excellent** [V+I]. Agents-as-tools is the native idiom. `subagent` with authority modes lets the manager create workers at runtime under constraints. Graphs can nest Swarms for sub-teams. Worker contexts are isolated by default, which matters for token budgets.
- **Peer-to-peer:** **partial** [I]. Swarm is P2P in *control* (any agent can hand off to any other with no central router), but it is **sequential handoff over shared context**, not concurrent agents messaging each other. True concurrent P2P needs one of:
  - a Graph with parallel nodes and cycles (structured, not free messaging);
  - A2A servers talking to each other (heavy);
  - a custom shared-blackboard tool.
  
  Strands Labs `robots` mentions a "peer-to-peer mesh", but for robots, not general agents [V].
- **Pipeline / DAG / fan-out-fan-in:** **good**, via Graph (parallel batches and join semantics) [V].
- **Expressing topology as an optimizable artifact:** **weak** [I]. Topology lives in Python builder calls. There is no first-party JSON or YAML graph spec, and `config_to_agent` is single-agent only. HIVE would need its own small topology schema compiled into GraphBuilder/Swarm calls, so the optimizer can make discrete, diffable topology edits.
- **Harbor constraint** [V]: the Harbor runner type-checks that the factory returns `strands.Agent` (`strands_runner.py` `_create_agent`). A Graph or Swarm must be wrapped: put it behind a tool on a thin top-level Agent, use `make_swarm`, or use agents-as-tools. [I] Wrapping is simple, but it puts an extra model turn at the top unless the tool uses `delegate=True`-style semantics.

---

## 4. Headless / programmatic use

Every Strands entry point is a library call [V]:

```python
from strands import Agent
from strands.models.anthropic import AnthropicModel

agent = Agent(model=AnthropicModel(model_id="claude-sonnet-5", max_tokens=8192),
              system_prompt=open("prompt.md").read(), tools=[...], callback_handler=None)
result = agent("Fix the failing test in /repo", limits={"turns": 60, "total_tokens": 2_000_000})
result.stop_reason, str(result), result.metrics.get_summary(), agent.messages
```

- TS: `const r = await new Agent({...}).invoke("...")`. `r.metrics`, `r.traces`, and `r.stopReason` are available [V].
- `callback_handler=None` silences stdout streaming [V].
- For Harbor [V]:
  - Write `agent.py` with `def create_agent(sandbox, model=None, **kw) -> Agent`, and point Harbor's `strands` agent at it via `factory` and `factory_path`.
  - Harbor installs `strands-agents[all]==<version>` with uv in the container (default pin 1.54.0; configurable), stages your directory, and runs `agent.invoke_async(instruction)`.
  - It writes `result.json` and `events.jsonl` and converts them to ATIF `trajectory.json` with token totals.
  - The `sandbox` passed in is a `TaskLocalSandbox` (a `PosixShellSandbox` subclass) that vends `sandbox_shell` and `sandbox_file_editor` tools bound to the task workdir.
  - The model is built from Harbor's provider config (Bedrock, Anthropic, Gemini, LlamaAPI, llama.cpp, OpenAI-compatible, ...).
- A harness via Harbor [I]: `create_harness(model=model, ...)` returns an `Agent`, so a factory can wrap the Strands harness. You'd need to route its tools through Harbor's sandbox (`sandbox=` passthrough to `Agent`). This is not verified end to end.

---

## 5. Tracing and observability

Docs: `/docs/user-guide/sdk/observability-evaluation/{traces,metrics,logs}/`.

- **OpenTelemetry built in** [V]:
  - Spans: agent invocation → event-loop cycle → model invoke → tool call.
  - Attributes follow GenAI semantic conventions: `gen_ai.request.model`, `gen_ai.usage.input_tokens/output_tokens/total_tokens/cache_read/cache_creation`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, user/assistant messages, and the choice.
  - Opt into the newest conventions with `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`.
  - Agent-level and multi-agent spans exist. `trace_attributes=` and baggage propagation are supported.
- **Local export without AWS** [V]:

  ```python
  from strands.telemetry import StrandsTelemetry
  t = StrandsTelemetry()
  t.setup_console_exporter(out=open("spans.jsonl","w"), formatter=lambda s: s.to_json()+"\n")
  t.setup_otlp_exporter(endpoint="http://localhost:4318")   # Jaeger/Langfuse/Phoenix/etc.
  t.setup_meter(enable_console_exporter=True)
  ```

  The docs include a Jaeger all-in-one local setup. If a global tracer provider is already configured, Strands uses it.
- **In-process metrics without any exporter** [V]: `result.metrics` (`EventLoopMetrics`) has:
  - `accumulated_usage` (input, output, total, and cache tokens)
  - `cycle_count` and `cycle_durations`
  - `tool_metrics` (calls, successes, errors, time per tool)
  - per-invocation history
  - **local execution traces** (an always-on timing tree of cycles, model calls, and tools), visible via `get_summary()`
  
  TS exposes `result.metrics` and `result.traces`.
- **Raw transcript**: `agent.messages` (the full conversation including tool calls and results) and `MessageAddedEvent` hooks. Harbor's runner uses exactly this [V].
- [I] For an optimizer that reads traces, the best inputs are: Harbor's ATIF `trajectory.json` (framework-neutral), plus a local OTel JSONL file, plus `metrics.get_summary()`. None of these need AWS.

---

## 6. Evals SDK (`strands-agents-evals` 1.6.0)

Repo `strands-agents/evals`; docs `/docs/user-guide/evals-sdk/` [V].

- **Core model:** `Case[In, Out]` (input, expected output, expected trajectory, metadata) → `Experiment(cases, evaluators)` → `experiment.run_evaluations(task_fn)` → `EvaluationReport`. Experiments save and load as JSON. Supports result caching.
- **Evaluators:**
  - **LLM-as-judge with rubrics:** Output, Correctness, Helpfulness, Faithfulness, Coherence, Conciseness, GoalSuccessRate, InstructionFollowing, ResponseRelevance, Refusal, Harmfulness, Stereotyping, PartialCompletion, RecoveryStrategy, FailureCommunication, multimodal variants.
  - **Trajectory and tool use:** TrajectoryEvaluator, ToolSelectionAccuracy, ToolParameterAccuracy, Interactions (multi-agent), SkillSelection, SkillInstructionFollowing, StructuredOutput.
  - **Deterministic evaluators** and helpers for exact, in-order, and any-order trajectory matching.
  - **Custom evaluators.**
- **Trace-based evaluation:** in-memory Strands session mapper plus mappers for generic GenAI OTel, OpenInference, LangChain, OpenAI Agents, and ADK. **Framework-agnostic**: the 2026-09-04 blog lists Claude Agents SDK (OpenInference), LangChain, OpenAI Agents, ADK, and others. Remote trace providers: CloudWatch, Langfuse, OpenSearch.
- **Detectors** (most relevant to HIVE):
  - `detect_failures`: ~20 failure categories with span locations and evidence.
  - `analyze_root_cause`: causal chain plus fix type.
  - `diagnose_session`: both of the above, producing deduplicated recommendations such as "system prompt change" or "tool description update".
  - Wire into `Experiment` via `DiagnosisConfig`.
  
  [I] This is essentially a ready-made "read traces and propose harness edits" component the outer optimizer could call or imitate.
- **Other features:** simulators (user simulation, LLM tool simulation), experiment generator, chaos testing (fault injection via plugin hooks), red-teaming (experimental).
- **CLI:** `strands-evals run | validate | report | diagnose | generate`, with `--fail-on` exit codes for CI.
- **Caveats** [V]:
  - Python only.
  - The judge and detector default model is Bedrock `global.anthropic.claude-sonnet-4-6` (`evaluators/evaluator.py:24`, `detectors/constants.py:4`).
  - `boto3` is a hard dependency.
  
  [I] Evaluators accept a model or Model instance, so pointing judges at the Anthropic API should work. Not tested.
- **Fit [I]:** Harbor's verifier stays the primary score for coding and terminal tasks (pass/fail tests). Strands Evals is useful as a **secondary** layer: trajectory-quality metrics, and failure detection plus RCA over traces to feed the optimizer.

---

## 7. Model / provider flexibility; working without AWS

- **Python providers** (`strands-py/src/strands/models/`) [V]: `bedrock` (default), `anthropic`, `openai`, `openai_responses`, `gemini`, `litellm`, `ollama`, `llamacpp`, `llamaapi`, `mistral`, `sagemaker`, `writer`, plus `routing` (ModelRouter) and a custom-provider interface. Installed via extras (`strands-agents[anthropic]`, `[openai]`, `[litellm]`, `[ollama]`, ...).
- **TS providers** (`strands-ts/src/models/`) [V]: `bedrock`, `anthropic`, `openai`, `google`, `vercel` (AI SDK adapter, which reaches many more providers). There is no native LiteLLM or Ollama in TS; use Vercel AI SDK or OpenAI-compatible endpoints. [I]
- **Without AWS** [V+I]: Works fine. Pass a `Model` instance (SDK) or `model="anthropic/..."` (harness). The gotchas:
  - a bare string model means Bedrock in the SDK;
  - `config_to_agent` only takes Bedrock strings;
  - evals and detectors default to Bedrock judges;
  - `boto3` is installed anyway.
  
  Harbor's model builder handles Anthropic and OpenAI-compatible keys from env [V].

---

## 8. Coding-agent capability out of the box; Harbor / Terminal-Bench

- **SDK vended tools** [V]: `shell`, `file_editor`, `python_repl`, routed through a **Sandbox** abstraction (`strands.sandbox`):
  - `PosixShellSandbox`
  - `DockerSandbox(container=...)`
  - `SshSandbox(host=...)`
  - "not-a-sandbox" local environment
  - custom sandboxes
  
  Docs: `/sdk/sandbox/`.
- **Strands harness** [V]: a general-purpose agent (the blog says "not a coding agent") with `shell` (stateless per call), `read/write/edit` (absolute paths, no `..`), web fetch and search, a programmatic tool caller, a `generalist` sub-agent, todos, environment awareness, context offloading, and caching.
  - Blog claims (vendor-reported, **not independently verified**):
    - 28% cheaper than other harnesses at similar accuracy across six benchmarks.
    - On Terminal-Bench 2.1 with Fable 5, 77% cheaper than Claude Code with a higher score.
  - The 2026-10-13 "94,000 trials" post compares Strands harness, OpenCode, Claude Code, **oh-my-pi**, Codex, and DeepSeek Harness on Harbor, across Terminal-Bench 2.1, SWE-bench Pro, and others.
- **Simple Strands Agent** (`strands-labs/benchmark-harnesses/simple-strands-agent`) [V]:
  - Minimal coding harness: `bash`, `str_replace_editor`, `think`, `submit`, Docker sandboxes, Hydra YAML configs.
  - Scripts for SWE-Bench Verified, SWE-Bench Pro, and **Terminal-Bench 2**.
  - Technical report arXiv 2606.17454 (claims SOTA-level results; vendor-reported).
- **Harbor adapter** [V]: `harbor-framework/harbor` `src/harbor/agents/installed/strands.py`, `strands_runner.py`, `strands_trajectory.py`. Harbor also has `pi.py`, `langgraph.py`, `claude_code.py`, etc. This means the HIVE "standard interface" could be **Harbor's agent interface itself**, with Pi and Strands as two adapters.

---

## 9. Maturity, license, API stability

- **License:** Apache-2.0 everywhere [V].
- **SDK maturity** [V]:
  - 1.0 since 2025. Python is at 1.61 after ~17 months, so roughly weekly minors. TS reached 1.x on 2025-11-28 and is now 1.21.
  - Large community and active AWS staffing. A `team/` folder holds design docs and tenets.
- **Versioning policy** (`/sdk/versioning-and-support/`) [V]:
  - SemVer, but with stated exceptions. **OTel GenAI conventions, MCP, and A2A** may break in minors.
  - "Pay-for-play" opt-in breaks are allowed in minors.
  - Anything in `strands.experimental` (e.g. `config_to_agent`, steering, checkpoint, some context-manager presets) is unguaranteed.
- **Harness** (`/harness/versioning/`) [V]: **0.x. Minor releases are breaking.** System prompt text, tool descriptions, and tuning can change in *any* release. Pin `strands-harness~=0.3.0`.
- **Churn signals** [V]:
  - The 2026 monorepo migration archived repos and moved docs.
  - Many features are weeks old: harness 2026-09-21, Box 2026-10-02, middleware, context strategies.
  - ~1k open issues.
  - The Harbor adapter pins SDK 1.54.0 by default while PyPI is at 1.61.1.
  - [I] Expect breakage in imports and paths over a semester. Pin exact versions in the harness package's lockfile.

---

## 10. Strengths and weaknesses for a self-editing harness project

### Strengths
1. **Already plugged into our eval stack** [V]. The Harbor adapter, the ATIF trajectory export, and token accounting exist. The contract (`factory(sandbox, model, **kw) -> Agent`) is small and stable enough to treat as HIVE's harness interface on the Python side.
2. **Many fine-grained, typed valves** [V]: prompt, tools and descriptions, hooks with writable events, middleware, context strategies, conversation managers, loop budgets, skills, sub-agent authority modes. All of them are plain Python an optimizer can edit and diff.
3. **First-class multi-agent primitives** [V], with nesting (Graph → Swarm → Agent), parallel Graph execution, runtime sub-agent spawning with constrained authority, and A2A for cross-process work. Hierarchical and manager-worker topologies map directly.
4. **Observability is free and local** [V]: OTel spans with GenAI conventions, in-process metrics and timing trees, and a raw message log. None need AWS.
5. **Prior art for exactly our project** [V]: Harness Optimizer (Formulas over prompt, skills, and tool descriptions; contrastive-reflection and multi-agent optimizers; reported AppWorld 72.6→95.8, vendor-reported). Evals detectors do root-cause analysis and suggest prompt or tool fixes. [I] HIVE can borrow its abstractions (Formula = editable surface, RewardFunction = Harbor score) or benchmark against it. The class novelty should then lean on **topology and hook and context edits**, which the optimizer doesn't yet cover. Its roadmap lists skills, memory, and tools Formulas.
6. **Lean coding baselines exist** [V]: Simple Strands Agent and Strands harness, both reported strong on Terminal-Bench, so we wouldn't start from zero.
7. **Same concepts in Python and TypeScript** [V]. This keeps a door open to sharing concepts with the Pi (TS) track.

### Weaknesses / risks
1. **Python-centric tooling** [V]. Evals, the Harbor adapter, Harness Optimizer, and `config_to_agent` are Python-only. If the team standardizes on TS (Pi), the Strands-TS route loses most of the payoff. [I] Choosing Strands effectively means choosing Python for the harness.
2. **Topology isn't data** [V/I]. There is no declarative multi-agent spec, so we'd build our own schema-to-GraphBuilder compiler for the optimizer to edit safely. True concurrent peer-to-peer messaging isn't native (Swarm is sequential handoff).
3. **Harbor wants an `Agent`, not a Graph or Swarm** [V]. Multi-agent runs need a wrapper agent. Harbor's ATIF conversion only sees the top agent's `MessageAddedEvent`s [I: inferred from the runner attaching a single hook]. Inner-agent turns may be missing from the trajectory unless we add our own OTel/JSONL export, so per-worker traces need extra plumbing.
4. **Churn** [V]: weekly minors, experimental flags on newer features (context presets, agent config, steering), and a 0.x harness whose prompts and tool descriptions change without notice. The Harbor default pin lags (1.54.0). Pin everything, and expect API drift.
5. **AWS-default gravity** [V]: string model = Bedrock, evals judges = Bedrock, `boto3` dependency. It's manageable, but every config path must pass explicit models. Otherwise you get silent credential errors.
6. **The vendor benchmarks are self-reported** [V]: cost and accuracy claims come from AWS blog posts. Treat them as hypotheses.
7. **Surface area** [I]. Hooks, middleware, plugins, interventions, conversation managers, and context managers overlap. An optimizer given all of them may make tangled edits. We should expose a *curated* subset as the editable surface.

### Verdict [I]
**Strong fit, conditional on the team accepting Python for the harness layer.** Strands plus Harbor's `strands` adapter gives HIVE:

- a working "run task, get score and trajectory" path on day one;
- a plain-Python harness package the optimizer can edit;
- native hierarchical and manager-worker multi-agent support.

It beats Pi on multi-agent primitives and eval tooling. It likely ties or loses on "a coding agent tuned for terminals" unless we start from Simple Strands Agent or Strands harness.

Recommended shape if chosen:
- A HIVE harness = a directory with `agent.py:create_agent(sandbox, model, **kw)`, `prompt.md`, `tools/`, `hooks/`, `skills/`, and a small `topology.yaml` compiled into Agent / as_tool / Swarm / Graph.
- Harbor runs it.
- Traces come from ATIF plus a local OTel JSONL exporter (so inner agents are captured).
- Optionally, Strands Evals detectors pre-digest failures for the Claude Code optimizer.

---

## Key URLs and paths

- Docs home: https://strandsagents.com/docs/user-guide/
- Agent loop: https://strandsagents.com/docs/user-guide/sdk/agents/agent-loop/
- Hooks / events: https://strandsagents.com/docs/user-guide/sdk/agents/hooks/ and https://strandsagents.com/docs/user-guide/sdk/agents/hooks-events/
- Middleware: https://strandsagents.com/docs/user-guide/sdk/agents/middleware/
- Conversation management: https://strandsagents.com/docs/user-guide/sdk/agents/conversation-management/
- Context management: https://strandsagents.com/docs/user-guide/sdk/context-management/
- Sessions: https://strandsagents.com/docs/user-guide/sdk/agents/session-management/
- Multi-agent overview: https://strandsagents.com/docs/user-guide/sdk/multi-agent/multi-agent-patterns/ (plus `/swarm/`, `/graph/`, `/graph-components/`, `/agents-as-tools/`, `/workflow/`, `/agent-to-agent/`)
- Traces: https://strandsagents.com/docs/user-guide/sdk/observability-evaluation/traces/ and metrics: `.../metrics/`
- Sandbox: https://strandsagents.com/docs/user-guide/sdk/sandbox/
- Agent config (experimental): https://strandsagents.com/docs/user-guide/sdk/experimental/agent-config/
- Versioning: https://strandsagents.com/docs/user-guide/sdk/versioning-and-support/ and https://strandsagents.com/docs/user-guide/harness/versioning/
- Harness: https://strandsagents.com/docs/user-guide/harness/ and the config reference https://strandsagents.com/docs/user-guide/harness/reference/configuration/
- Evals: https://strandsagents.com/docs/user-guide/evals-sdk/ (detectors: `/evals-sdk/detectors/`)
- Monorepo: https://github.com/strands-agents/harness-sdk (`strands-py/`, `strands-ts/`, `harness-py/`, `harness-ts/`, `site/`, `team/designs/`)
- Tools: https://github.com/strands-agents/tools
- Evals: https://github.com/strands-agents/evals
- Harbor adapter: https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/installed/strands.py (plus `strands_runner.py`, `strands_trajectory.py`)
- Harness Optimizer: https://github.com/strands-labs/harness-optimizer (blog: `site/src/content/blog/introducing-harness-optimizer.mdx`, 2026-07-10)
- Benchmark harnesses / SSA: https://github.com/strands-labs/benchmark-harnesses
- Harness benchmark posts: `site/src/content/blog/introducing-strands-harness.mdx` (2026-09-21) and `we-ran-94000-trials-to-measure-the-cost-of-harness-choice.mdx` (2026-10-13)
