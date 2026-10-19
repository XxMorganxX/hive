# LangChain Deep Agents: fit as a HIVE base harness

Research date: 2026-10-19. Sources: a shallow clone of `langchain-ai/deepagents` at commit `9a23fbb` (2026-10-19), `langchain-ai/deepagentsjs` (2026-10-19), `harbor-framework/harbor` (v0.24.0, 2026-10-18), plus the official docs and LangChain blog posts linked below. No Deep Agents code was run.

Labels: **[V]** = verified in source or docs. **[I]** = inferred (reasoning or general LangChain knowledge, not checked line by line).

---

## TL;DR

- **What it is.** Deep Agents is LangChain's "batteries-included agent harness". It is a Python library (`deepagents` 0.7.28, MIT) with a TypeScript twin (`deepagents` npm 1.15.1). It sits on LangChain's `create_agent()` loop, which runs on the LangGraph runtime. You call one function, `create_deep_agent(...)`, and get a compiled LangGraph graph back. **[V]**
- **Fit for HIVE: good, and closer to our project than expected.** LangChain ships `examples/better-harness`, an eval-driven harness optimizer. An outer Deep Agent edits declared "surfaces" of an inner agent (prompt, tools, skills, middleware, wiring), runs the train and holdout evals (through pytest *or Harbor*), and keeps only the edits that improve the score. That is HIVE's loop in miniature. LangChain also reports going from 52.8% to 66.5% on Terminal-Bench 2.0 by editing only the harness (prompt, tools, middleware). **[V]**
- **Main costs.** It is a deep, fast-moving stack (deepagents → langchain → langgraph → langsmith). It is pre-1.0, with breaking changes on minor versions. The 0.7.0 release removed the planning tool and the default prompt. LangSmith is the native trace path. Local or free tracing works, but we have to wire it ourselves. **[V]**
- **Harbor.** There is no `deepagents` agent in Harbor. Harbor's built-in **`langgraph`** agent runs any `langgraph.json` graph, Python or JS, and the Deep Agents repo uses exactly that to run Terminal-Bench 2.0. It reports token counts but **not ATIF trajectories**. Pi, by contrast, has native ATIF support in Harbor. **[V]**

---

## 1. Architecture

### Three layers  [V]
From `libs/ARCHITECTURE.md`:

```
Deep Agents   opinionated harness: defaults, middleware, backends, profiles
LangChain     create_agent(): model + tools + middleware -> agent loop
LangGraph     runtime: state, checkpoints, streaming, interrupts
```

Deep Agents adds no new runtime. `create_deep_agent()` (in `libs/deepagents/deepagents/graph.py`, about 1,000 lines) does six things:
1. Resolves the model and any *harness/provider profile* for it.
2. Resolves the *backend*, which decides where files, memory and shell live.
3. Assembles the middleware stack.
4. Builds the default `general-purpose` subagent plus any subagents you pass in.
5. Composes the system prompt as `USER → profile BASE → profile SUFFIX`.
6. Calls `langchain.agents.create_agent(...)`, then sets `.with_config(recursion_limit=9_999)`.

The result is a `CompiledStateGraph`. You run it with `.invoke` / `.ainvoke` / `.stream` / `.astream_events(version="v3")`.

### Signature (v0.7.28)  [V]
```python
create_deep_agent(model, tools=None, *, system_prompt=None, middleware=(), subagents=None,
    skills=None, memory=None, permissions=None, backend=None, interrupt_on=None,
    response_format=None, state_schema=None, context_schema=None, checkpointer=None,
    store=None, debug=False, name=None, cache=None)
```

### Default tools  [V]
- **Files:** `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `delete`. `delete` was added in 0.7.0 and only appears when the backend supports it.
- **Shell:** `execute`, only shown when the backend implements `SandboxBackendProtocol`.
- **Subagents:** `task`.

### Planning tool: no longer on by default  [V]
Many tutorials describe `write_todos` as a core Deep Agents feature. **0.7.0 (2026-07-29) removed it from the default stack.** Pass `middleware=[TodoListMiddleware()]` to get it back, and add it to each subagent separately. The same release:
- Made the default authored prompt **empty**. `BASE_AGENT_PROMPT` is deprecated and slated for removal in 0.9.0.
- Removed the built-in tool-usage prompt constants.

So Deep Agents today is much leaner by default than its reputation suggests (`libs/deepagents/CHANGELOG.md`, 0.7.0 section).

### Virtual filesystem and backends  [V]
(`deepagents/backends/`, docs: https://docs.langchain.com/oss/python/deepagents/backends)

| Backend | What it does |
|---|---|
| `StateBackend` (default) | Files live in LangGraph state, scoped to one thread. This is the "virtual filesystem". |
| `FilesystemBackend` | Real disk under `root_dir`. Since 0.7.0, `virtual_mode=True` anchors paths and blocks `..`. |
| `LocalShellBackend` | Real disk plus `execute` on the host. The Harbor entrypoint uses this inside the task container. |
| `StoreBackend` | LangGraph `BaseStore`; persists across threads, namespaced. |
| `CompositeBackend` | Routes path prefixes to different backends (e.g. `/memories/` → Store, everything else → sandbox). |
| `ContextHubBackend`, LangSmith sandbox, partners (Daytona, Modal, Runloop, Vercel in `libs/partners/`) | Remote or sandboxed storage and execution. |

Backends sit behind `BackendProtocol` / `SandboxBackendProtocol` (`backends/protocol.py`). That makes them a clean, swappable boundary.

### Middleware: the main extension mechanism  [V]
Middleware is LangChain's `AgentMiddleware`. Its hooks are `before_agent`, `before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`, and `after_agent`. Middleware can:
- rewrite the tool list or prompt before each model call;
- compact history;
- add typed state;
- veto or approve tool calls;
- re-enter the loop (for example, a "not done yet" check before the agent stops).

Default main-agent order (graph.py docstring and the customization docs):

1. `FilesystemMiddleware` (with permissions)
2. `SubAgentMiddleware` (only if sync subagents exist)
3. `SummarizationMiddleware`
4. `PatchToolCallsMiddleware`
5. `AsyncSubAgentMiddleware` (only if async subagents exist)
6. **your `middleware=`**
7. profile `extra_middleware`
8. `SkillsMiddleware`
9. Anthropic / Bedrock / Fireworks prompt caching (no-ops for other providers)
10. `MemoryMiddleware`
11. `HumanInTheLoopMiddleware`
12. `UnsupportedContentMiddleware`
13. `_ToolExclusionMiddleware`

Since 0.7, a user middleware whose `.name` matches a default replaces it in place. Profiles can also drop middleware by name with `excluded_middleware`, though the core scaffolding is protected.

Other notable shipped middleware:
- `RubricMiddleware`: when the agent tries to finish, a grader sub-agent scores the transcript against a rubric and pushes it to revise.
- `CodeInterpreterMiddleware`: a QuickJS sandbox for programmatic tool calls (experimental).
- LangChain built-ins such as `HumanInTheLoopMiddleware` and `SummarizationMiddleware`.

LangChain also has call-limit middleware. **[I]** Use it: the default `recursion_limit` is 9,999, so there is effectively no step cap unless we add one or rely on Harbor's timeout.

### State and persistence  [V]
- `DeepAgentState` extends `AgentState` (messages use a `DeltaChannel` reducer).
- A `checkpointer` gives resumability and HITL interrupts.
- A `store` gives cross-thread memory.
- `memory=["/AGENTS.md"]` loads memory files into the system prompt at startup.
- `skills=[...]` loads Agent-Skills-format `SKILL.md` folders on demand.

---

## 2. Extension surfaces an optimizer could edit

"Declarative" means plain data (strings, YAML, Markdown, dicts). "Code" means Python objects or functions.

| Surface | How it is set | Declarative or code | Optimizer notes |
|---|---|---|---|
| System prompt | `system_prompt=` string or `SystemMessage` | Declarative | Easiest lever; can live in a `.md` file. [V] |
| Per-model prompt / tool overrides | `HarnessProfileConfig`: `base_system_prompt`, `system_prompt_suffix`, `tool_description_overrides`, `excluded_tools`, `excluded_middleware`, `general_purpose_subagent`. Registered with `register_harness_profile("provider:model", ...)`. | **Declarative. Ships with `from_dict` / `to_dict` and a YAML example** (`profiles/harness/harness_profiles.py`). | Very good optimizer target: one YAML file covers prompt, tool descriptions, and which tools and middleware are on. [V] |
| Tool descriptions | `tool_description_overrides` in the profile | Declarative | Lets the optimizer rewrite built-in tool descriptions without touching code. [V] |
| Built-in tool set | `FilesystemMiddleware(tools=[...])` allowlist (`FsToolName`), or profile `excluded_tools` | Declarative list | [V] |
| Custom tools | `tools=[fn, BaseTool, MCP tool]` | Code (Python functions with docstrings and type hints) | The optimizer edits a `tools.py`. [V] |
| Middleware | `middleware=[...]` instances, or `@wrap_tool_call` / `@before_model` decorators | **Code** | The most powerful lever (LangChain's +13.7 pts mostly came from here). Needs two edits: the implementation *and* the registration (better-harness README calls this out). [V] |
| Subagents | `subagents=[{name, description, system_prompt, tools, model, middleware, interrupt_on, skills, permissions, response_format, mode}]` | Declarative dict, apart from tools and middleware. `deepagents-code` also loads subagents from `.deepagents/agents/<name>/AGENTS.md` (YAML frontmatter plus a prompt body). | Topology-as-data for the manager-worker pattern. [V] |
| Compiled subagents | `CompiledSubAgent(name, description, runnable)` | Code (any LangGraph graph) | Arbitrary custom sub-topologies. [V] |
| Skills | `skills=["/skills/"]` folders of `SKILL.md` | Declarative Markdown | Good for procedural knowledge the optimizer discovers. [V] |
| Memory | `memory=["/AGENTS.md"]` | Declarative Markdown | [V] |
| Context management | `SummarizationMiddleware` (replace by name), tool-output offload to the backend, prompt caching | Code (constructor args) | [V] |
| Backend / sandbox | `backend=` instance | Code | Usually fixed by the eval harness, not optimized. [V] |
| HITL / permissions | `interrupt_on={tool: bool/config}`, `permissions=[FilesystemPermission(...)]` | Declarative | Mostly irrelevant for headless evals. [V] |
| Model and reasoning | `model="provider:model"` or an instance; provider profiles | Declarative string | LangChain's "reasoning sandwich" (xhigh, then high, then xhigh) is itself a harness lever. [V] |
| Structured output | `response_format=` | Code (schema) | [V] |

**Takeaway.** A HIVE harness on Deep Agents can be one YAML/TOML spec (prompts, profile, subagent dicts, skill and memory files) plus two Python files (`tools.py`, `middleware.py`). The optimizer only ever edits those files. better-harness formalises this as "surfaces" of kind `module_attr` or `workspace_file` (`examples/better-harness/README.md`). [V for the pattern, I for our layout]

---

## 3. Multi-agent support

### What Deep Agents gives natively  [V]
- **Sync subagents through `task`.**
  - The parent blocks until the child finishes.
  - The child runs in an isolated context: it sees only the task description, and the parent sees only the final answer.
  - Several `task` calls in one turn run in parallel.
  - Each child has its own prompt, tools, model, middleware, skills, permissions, and `response_format`.
  - Runs are tagged `lc_agent_name` in traces.
  - A `general-purpose` subagent is added automatically; profiles can disable it.
- **Fork mode** (`mode="fork"`, beta, >=0.7.13). The child continues the parent's conversation instead of starting fresh. It cannot call `task` (there is a recursion guard).
- **Async subagents** (`AsyncSubAgent{name, description, graph_id, url?, headers?}`, beta).
  - Background runs on a remote Agent Protocol / LangGraph server.
  - Tools: `start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task`, `list_async_tasks`.
  - The parent can steer and cancel mid-flight (`middleware/async_subagents.py`; example in `examples/async-subagent-server/`).
- **Nesting.** Declarative subagents get a stack *without* `SubAgentMiddleware`, so by default only the root has `task`. Deeper hierarchies work by passing a `CompiledSubAgent` whose `runnable` is itself a `create_deep_agent(...)` with its own subagents. [V that the stack omits it; I that the compiled nesting works cleanly, though the docs say "any LangGraph `CompiledStateGraph` can be passed in as a sub-agent"]

### Mapping to the topologies HIVE cares about

| Topology | Native in Deep Agents? | How |
|---|---|---|
| Manager-worker | **Yes, first-class** | Root agent plus `subagents=[...]` through `task`, with parallel fan-out. [V] |
| Hierarchical (multi-level) | Yes, with a little code | Nested `CompiledSubAgent`s, each a deep agent. [I] |
| Peer-to-peer / handoffs / swarm | **No, not in Deep Agents itself** | Drop to LangGraph: a `StateGraph` whose nodes are deep agents, with tool calls that change the active agent (the "Handoffs" pattern in https://docs.langchain.com/oss/python/langchain/multi-agent). Wrap the result as a `CompiledSubAgent`, or use it as the top-level graph. [V the pattern is documented; I the exact wiring] |
| Router | Via LangGraph | Classifier node, conditional edges, then specialist agents. [V documented pattern] |
| Arbitrary graphs (debate, blackboard, pipelines) | Via LangGraph | LangGraph is a general state-machine runtime: nodes, conditional edges, `Command(goto=...)`, `Send` for map-reduce, subgraphs, shared state, checkpoints. [I, standard LangGraph] |

**What LangGraph adds.** Any directed graph, cyclic or not, of agents and functions with shared typed state, plus checkpointing, streaming and interrupts. So "topology" becomes code the optimizer edits, such as a `topology.py` that builds a `StateGraph`. The trade-off: once you leave manager-worker, the topology is *imperative Python graph code*, not a dict. [I]

---

## 4. Headless and programmatic use

### SDK (the path HIVE should use)  [V]
```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", system_prompt=PROMPT,
                          tools=[...], backend=LocalShellBackend(root_dir="/app"))
result = agent.invoke({"messages": [{"role": "user", "content": task}]})
final = result["messages"][-1].content   # full message list = main-agent trajectory
```
For live events, including subagent events, use `agent.stream(..., subgraphs=True, stream_mode=[...])` or `astream_events(version="v3")`. [V v3 exists; I flags]

### CLI: `deepagents-cli` is now **Deep Agents Code (`dcode`)**  [V]
- **Rename.** `libs/code` (`deepagents-code` 0.1.86) was forked from `deepagents-cli` at v0.1.0 on 2026-05-12. `libs/cli` is gone from main. Install with `curl -LsSf https://langch.in/dcode | bash`.
- **Headless flags** (`libs/code/deepagents_code/main.py`):
  - `-n/--non-interactive "TEXT"` runs one task and exits with no approvals. Shell stays off unless `--shell-allow-list` is set.
  - `-q/--quiet`, `--no-stream`, `--max-turns`, a time limit, `-y/--auto-approve` (`--yolo`), `--json` for some commands, `--trust-project-hooks`.
- **Extras on top of the SDK:**
  - hooks (`HOOKS.md`, Claude-Code-style `hooks.json`);
  - Python extensions (`EXTENSIONS.md`: `register_middleware`, `register_tool`, `register_backend_route`);
  - file-defined subagents;
  - cost tracking (`cost_tracking.py`, using `genai-prices`);
  - ACP server (`libs/acp`).
- `dcode` is the closest analogue to Pi or Claude Code. For HIVE, though, the **SDK entrypoint is the cleaner thing to optimize**. [I]

---

## 5. Tracing, tokens and cost

| Option | Paid? | Captures subagents? | Notes |
|---|---|---|---|
| **LangSmith** (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`) | Free Developer tier: 5k base traces/month, 14-day retention, 1 seat. Overage reportedly $2.50 per 1k base traces (https://www.langchain.com/pricing). | Yes, nested, tagged `lc_agent_name`. | Native path. Harbor's LangSmith plugin nests the agent trace under the Harbor trial and writes verifier reward as feedback (`libs/evals/CONTRIBUTING.md`). The Deep Agents team's own loop is Harbor → LangSmith → Trace Analyzer skill → harness edits. [V] |
| **OpenTelemetry through the LangSmith SDK** | Free if the backend is self-hosted | Yes | `pip install "langsmith[otel]"`, then `LANGSMITH_OTEL_ENABLED=true`, `LANGSMITH_TRACING=true`, `LANGSMITH_OTEL_ONLY=true`, `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`. Sends to any OTLP collector: Jaeger, Phoenix, Langfuse, or a file exporter (https://docs.langchain.com/langsmith/trace-with-opentelemetry). Whether a LangSmith key is needed in OTEL-only mode is not stated outright; the examples imply it is not. [V env vars, I key] |
| **Local JSONL through a LangChain callback handler** | Free | Yes; callbacks propagate into subgraphs. [I] | About 50 lines: a `BaseCallbackHandler` that logs `on_chat_model_start`, `on_llm_end` (with `usage_metadata`) and `on_tool_start` / `on_tool_end` per `run_id` and `parent_run_id`. Most control, and lets us emit ATIF ourselves. [I] |
| **Final state dump** | Free | **No.** Isolated subagents return only their final answer to the parent. | Harbor's `langgraph` agent writes `result.json` (the full serialized final state) and `summary.json` (answer plus aggregated `input_tokens`, `output_tokens`, `cache_read_tokens`), and fills `AgentContext.n_input_tokens` and the other token fields (`harbor/src/harbor/agents/installed/langgraph_runner.py`). [V] |
| Third-party (Langfuse, Arize Phoenix / OpenInference LangChain instrumentors) | Free self-host | Yes | Standard LangChain integrations. [I] |

- **Tokens.** Every LangChain `AIMessage` carries `usage_metadata`: input, output and total tokens, plus cache read/creation details. [V, used by Harbor's runner]
- **Cost.** The SDK has no cost field. LangSmith computes cost server-side. `dcode` estimates cost locally with `genai-prices` (`libs/code/deepagents_code/cost_tracking.py`, with `bundled_prices.json`). For HIVE we would price `usage_metadata` ourselves, or reuse `genai-prices`. [V]

---

## 6. Model and provider flexibility  [V]
- Any LangChain chat model with tool calling works, given as a `"provider:model"` string through `init_chat_model` or as a `BaseChatModel` instance.
- Anthropic and Google GenAI are core dependencies. The Harbor entrypoint also wires OpenAI, Fireworks, Baseten, Groq, NVIDIA, Ollama, OpenRouter and xAI (`libs/evals/deepagents_harbor/langgraph_project/langgraph.json`). Local models work through Ollama, vLLM or llama.cpp.
- **Harness profiles** (since 0.5.4) let one agent definition carry per-model prompt, tool and middleware tweaks. Built-ins exist for Claude Haiku 4.5, Sonnet 4.6 and Opus 4.7, OpenAI Codex, and NVIDIA Nemotron 3 Ultra (`profiles/harness/`).
- Note: with no explicit model, the default is `claude-sonnet-4-6`, but `model=None` is deprecated.
- OpenAI models default to the Responses API.

---

## 7. Harbor integration  [V]

- **No dedicated `deepagents` agent in Harbor.** Harbor v0.24.0 lists about 40 pre-integrated agents, including `pi`, `claude-code`, `codex`, `terminus-2` and **`langgraph`**, but none named deepagents or dcode (https://docs.harborframework.com/agents/pre-integrated-agents; `harbor/src/harbor/agents/installed/`).
- **The supported path is Harbor's generic `langgraph` installed agent** (`harbor/src/harbor/agents/installed/langgraph.py`, `langgraph_runner.py`, plus `langgraphjs_runner.mjs` for JS graphs). It:
  - copies a project with `langgraph.json` into the container and builds a venv;
  - loads the named graph;
  - invokes it with `{"messages":[{"role":"user","content":instruction}]}`;
  - writes `result.json`, `langgraph.txt` and `summary.json` (with token usage);
  - forwards provider keys and LangSmith env vars, and nests the trace under the Harbor parent trace.
  - Options: `project_path`, `graph`, `model_kwargs`, `configurable`, `dependency_overrides`, `python_version`, `node_version`.
  - Capabilities: MCP servers only. **No ATIF, no resume, no skills flag.**
- **The Deep Agents repo uses this for Terminal-Bench 2.0** (`libs/evals/CONTRIBUTING.md`; harbor pinned at `>=0.20,<0.21` in `libs/evals/pyproject.toml`):
  ```bash
  uv run harbor run --agent langgraph \
    --agent-kwarg project_path=deepagents_harbor/langgraph_project \
    --agent-kwarg config=langgraph.json --agent-kwarg graph=dcode \
    --dataset terminal-bench/terminal-bench-2 --model "$MODEL" -n 1 --env docker
  ```
  The project exposes the graphs `dcode`, `bare` and `tau3` (`langgraph_agent.py:make_graph`). They run inside the task container on `LocalShellBackend` with a short "you are in a Harbor sandbox, no human available" prompt.
- **History.** In Dec 2025 the post "Evaluating DeepAgents CLI on Terminal Bench 2.0" (https://www.langchain.com/blog/evaluating-deepagents-cli-on-terminal-bench-2-0) used a custom `BaseAgent` wrapper (`deepagents_harbor:DeepAgentsWrapper`, through `--agent-import-path`) with a `HarborSandbox` backend. That scored about 42.65% mean with Sonnet 4.5. The old `libs/harbor` has since been replaced by the generic `langgraph` route.
- **Harness-engineering result.** In Feb 2026 the post "Improving Deep Agents with harness engineering" (https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering) reported **52.8% → 66.5%** on TB2 with gpt-5.2-codex fixed, from top-30 to top-5. The changes:
  - prompt: a plan, build, verify, fix loop;
  - middleware: `PreCompletionChecklistMiddleware`, `LocalContextMiddleware`, `LoopDetectionMiddleware`, time-budget warnings;
  - a reasoning-effort "sandwich";
  - a LangSmith "Trace Analyzer" skill driving the iterations.

  The scores are self-reported. An earlier harness version scored 59.6% with Opus 4.6.
- **better-harness has a Harbor runner** (`examples/better-harness/better_harness/runners.py`, `HarborRunner`). It shells out to `harbor run -p <tasks_root> --task-name <t> [--agent-import-path ...]`, parses the Harbor result files into a pass/fail score, and captures trace links.

**ATIF gap.** If HIVE standardises on Harbor ATIF trajectories as the optimizer's trace format, Deep Agents through `langgraph` does not emit them. We would need one of:
- our own ATIF converter over `result.json` plus callback logs; or
- a custom `BaseAgent` / `BaseInstalledAgent` wrapper that writes ATIF; or
- a contribution to Harbor's `langgraph` agent.

Pi already has native ATIF. [V for the capability table; I for the effort]

---

## 8. Maturity, license, stability, docs  [V]

- **License.** MIT (SDK, `deepagents-code`, JS).
- **Activity.** About 30.4k GitHub stars, 4.3k forks and 206 open issues; the repo was created 2025-07-27 and last pushed 2026-10-19. The JS repo has about 1.6k stars. Development is very active: 0.7.0 through 0.7.28 shipped in under 12 weeks, often several releases a day.
- **Stability.**
  - "Development Status :: 4 - Beta" and pre-1.0 (0.7.x). Breaking changes land on minor versions; the 0.7.0 list is long (planning tool and prompts removed, `write_file` semantics, backend APIs, `read_file` output format).
  - Several features are explicitly beta: fork mode, async subagents, dynamic subagents, QuickJS interpreter.
  - Deprecations come with removal targets (e.g. `BASE_AGENT_PROMPT` in 0.9.0; `model=None` in 1.0).
  - The changelog is clean and detailed.
  - **Pin exact versions** (`deepagents==0.7.28`, plus a lockfile) for the semester.
- **Docs.** Strong. There is an official docs section (https://docs.langchain.com/oss/python/deepagents/overview, plus `customization`, `subagents`, `backends`, `profiles`, `memory`, `skills`, `permissions`, `event-streaming`, `going-to-production`), an API reference (https://reference.langchain.com/python/deepagents/), an in-repo `ARCHITECTURE.md`, and a large `libs/evals` suite with an `EVAL_CATALOG.md`. The docs pages are very long (100k+ chars), but accurate against source in the places checked.
- **Dependency weight.** `langchain`, `langchain-core`, `langchain-anthropic`, `langchain-google-genai`, `langsmith` and `wcmatch` pull in `langgraph` (1.2.15 in the lock). That is a lot of transitive surface compared with Pi.
- **Size.** The core SDK is about 26k lines of Python (graph.py about 1k; filesystem middleware about 3.9k; summarization about 2.4k; sandbox backend about 2.2k). Most of the runtime behaviour lives in LangChain and LangGraph, not in this repo.
- **JS (`deepagentsjs`).** Close feature parity (`createDeepAgent`, backends, middleware, skills, async subagents). It is a separate release line (1.15.1), not a port of each Python release. Relevant only if the team wants TypeScript next to Pi. Harbor's `langgraph` agent can run JS graphs too.

---

## 9. Deep Agents vs a minimal harness like Pi, for a *self-editing* harness

### Where Deep Agents is stronger
1. **The levers already exist as named, documented knobs:** prompt slots, tool-description overrides, tool allowlists, middleware hooks around every model and tool call, subagent specs, skills, memory, summarization. The optimizer gets a well-typed action space instead of having to invent one. [V]
2. **There is prior art for our exact loop.** `better-harness` (outer Deep Agent, inner harness, train/holdout, keep-if-better, Harbor runner) and LangChain's TB2 hill-climb (+13.7 pts). We can borrow the surface model and the overfitting guard (holdout split). [V]
3. **Manager-worker multi-agent is built in**, and LangGraph is available for any other topology. Pi would need its own subagent machinery for this (Pi deliberately ships without subagents; extensions add them). [V for Deep Agents; I for Pi details]
4. **Context management is built in:** summarization, large-output offload, prompt caching. On long Terminal-Bench tasks this matters. [V]
5. **Provider-agnostic, with per-model profiles.** [V]
6. **Rich tracing** if we accept LangSmith's free tier or wire OTel. [V]

### Where Deep Agents is weaker (for an optimizer editing code)
1. **Abstraction depth.** Behaviour is spread over four packages. A middleware edit means understanding hook order, state reducers, `ModelRequest` / `ModelResponse` types, private state attrs, and how subagent stacks differ from the main stack (subagents do *not* inherit main-agent middleware). An optimizer that edits outside the declared surfaces can easily make changes that type-check but do nothing, or break subtly. **Mitigation:** expose only declared surfaces (prompt and profile YAML, `tools.py`, `middleware.py`, subagent specs) and treat the framework as read-only. [I]
2. **Hidden defaults move under you.** 0.7.0 silently changed the prompt and planning defaults. A harness "baseline" is only reproducible with exact pins. The harness the optimizer sees is also not the whole harness: there are framework-injected prompt fragments, tool descriptions and middleware. Use `debug=True` or dump the final `ModelRequest` to show the optimizer the *effective* prompt and tools. [V for the changes, I for the mitigation]
3. **Python, while the team's primary candidate (Pi) is TypeScript.** A shared "standard harness interface" has to be language-neutral anyway (a CLI or Harbor agent contract). deepagentsjs exists if TypeScript is required. [V]
4. **Harbor ATIF gap** (see §7), against native ATIF for Pi. [V]
5. **Tracing defaults to a vendor SaaS.** The free tier (5k traces/month, 14-day retention) could be tight for big sweeps, since each Harbor trial is at least one trace. The OTel or local-callback route avoids this but is our work. [V tier; I fit]
6. **No step cap by default** (`recursion_limit` 9,999). Add call-limit middleware or rely on Harbor timeouts. [V]
7. **Heavier dependencies and install time** inside each Harbor container (Harbor builds a venv per trial). [I]

### Pi in one paragraph, for contrast  [I, from general knowledge]
Pi (`pi-mono` coding agent, TypeScript) is deliberately minimal: a tiny system prompt, about four tools (read, write, edit, bash), and a TypeScript extension API, with no built-in subagents or planning. Almost every behaviour is visible in a small codebase. That makes it *easier to edit wholesale* but gives the optimizer *fewer ready-made levers*: it has to author context management and multi-agent support from scratch. It has native Harbor ATIF and resume support [V, Harbor docs].

### Standard interface recommendation  [I]
Whichever base we pick, define the HIVE harness contract outside both frameworks:

```
run_task(instruction, workdir, model, harness_spec) -> {final_text, trajectory(ATIF), usage{in,out,cache}, cost}
```

Expose it as a Harbor agent: `--agent-import-path` for a custom wrapper, or `langgraph` / `pi`. A Deep Agents implementation is about 150 lines:
- load `harness_spec` (YAML: prompt, profile, subagents, skills, enabled middleware names);
- import `tools.py` and `middleware.py`;
- call `create_deep_agent`;
- run with an ATIF-emitting callback handler.

This keeps Deep Agents swappable with Pi, per the component-independence principle.

---

## Top risks

1. **API churn**, pre-1.0, with breaking changes on minors. Pin exact versions.
2. **The optimizer gets lost in framework depth.** Restrict it to declared surfaces and show it the effective prompt and tools.
3. **No ATIF from Harbor's `langgraph` agent.** We must build trajectory export.
4. **The free LangSmith tier may not cover large sweeps.** Plan for OTel or local JSONL.
5. **Python vs the team's TypeScript Pi.** It complicates a shared codebase; mitigated by a language-neutral Harbor contract or deepagentsjs.

## Key links
- Repo: https://github.com/langchain-ai/deepagents (SDK `libs/deepagents`, CLI `libs/code`, evals and Harbor `libs/evals`, optimizer `examples/better-harness`)
- JS: https://github.com/langchain-ai/deepagentsjs
- Docs: https://docs.langchain.com/oss/python/deepagents/overview, `/customization`, `/subagents`, `/backends`, `/profiles`
- Multi-agent patterns: https://docs.langchain.com/oss/python/langchain/multi-agent
- OTel: https://docs.langchain.com/langsmith/trace-with-opentelemetry
- Harbor agents: https://docs.harborframework.com/agents/pre-integrated-agents
- Blog, TB2 baseline: https://www.langchain.com/blog/evaluating-deepagents-cli-on-terminal-bench-2-0
- Blog, harness engineering: https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering
