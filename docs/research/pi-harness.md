# Pi coding agent harness: evaluation for HIVE

Research date: 2026-10-19. Source state: `github.com/earendil-works/pi` (formerly `badlogic/pi-mono`; the old URL redirects) at commit `d2e7d8b7`, 2026-10-19, and `github.com/laude-institute/harbor` (now `harbor-framework/harbor`) at commit `357c5e74`, 2026-10-18. I read the docs and source only. I ran nothing from either repo.

Legend: **[V]** means verified in the repo, docs or GitHub API. **[I]** means inferred, i.e. my reading of how pieces fit together, not tested.

---

## 0. Verdict

**Pi is a strong fit for the base harness.** Four reasons:

1. **Almost every behavior is a file the optimizer can edit.** The system prompt is built from named sections. Tools, the context transform, compaction, tool gating and loop continuation are all TypeScript extension hooks, loaded by `jiti` with no build step. Skills and prompt templates are Markdown.
2. **Headless control comes in three tiers.** These are `--print --mode json` (one shot), `--mode rpc` (a long-lived JSONL protocol) and an in-process TypeScript SDK.
3. **Traces are first class.** Each session is a JSONL tree that records per-message token usage and dollar cost, the system prompt and tool loadout, compactions and model changes.
4. **Harbor already ships a built-in `pi` agent.** It runs Pi in JSON mode and converts the session JSONL to Harbor's ATIF trajectory format. We would extend it rather than write one from scratch.

**The main gaps:**

- Multi-agent is not built in. Pi deliberately leaves out subagents; they come from an example extension or third-party packages.
- The API changes quickly. There were 50 "Breaking Changes" sections in about 10 months.
- The language is TypeScript/Node, while Harbor and most of our tooling are Python.
- Harbor's Pi adapter does not expose a way to inject a custom harness. We would need a small subclass.

---

## 1. Architecture

### Packages (all v1.3.0, released 2026-10-15) [V]

The npm scope was renamed from `@mariozechner/*` to `@earendil-works/*` at v0.74.0 (May 2026). Harbor's adapter hard-codes the rename version as `0.74.0`. Pi is now maintained by Earendil (Mario Zechner's company), and the license is still MIT.

| Package | Role |
|---|---|
| `@earendil-works/pi-ai` (`packages/ai`) | Unified multi-provider LLM API: streaming, tool calls, usage and cost, model catalog, and a `fauxProvider()` for scripted tests |
| `@earendil-works/pi-agent-core` (`packages/agent`) | Generic stateful agent loop (`Agent` class, `agent-loop.ts`): tool execution, event stream, `prepareRequest`/`finishTurn`/`beforeToolCall`/`afterToolCall` hooks |
| `@earendil-works/pi-coding-agent` (`packages/coding-agent`) | The `pi` CLI and SDK: sessions, extensions, skills, prompt templates, compaction, built-in tools, and the TUI/print/JSON/RPC modes |
| `@earendil-works/pi-tui` | Terminal UI library |
| `@earendil-works/pi-mcp` | Standalone MCP client (stdio and streamable HTTP, OAuth) |
| `@earendil-works/pi-codemode` | QuickJS-WASM sandbox where model-written JS calls injected tools |
| `@earendil-works/pi-durable` | **Experimental** durable harness: crash-safe conversations, a task graph, and owned child conversations (subagents) |
| `@earendil-works/pi-env` | Remote execution env over SSH (Rust daemon) for Durable |
| `@earendil-works/chord` | Hot-reloadable plugin/object runtime that Durable is built on |
| `@earendil-works/pi-telemetry` | Vendor-neutral span and metric contracts with an in-memory reference adapter |
| `@earendil-works/pi-evals` (`packages/evals`) | Pi's own behavioral evals (vitest-evals, Docker A/B "documentation-lift" runs) |

**Runtime** [V]: TypeScript on Node.js ≥ 22.19 (Bun is also supported for the SDK). Install with `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` or `curl -fsSL https://pi.dev/install.sh | sh`. Nix is also supported.

### Agent loop [V] (`packages/coding-agent/docs/how-pi-works.md`, `packages/agent/README.md`)

1. The user message is appended to the active branch of the session tree.
2. **Request build:** the system prompt (sections), the active-branch messages and the tool declarations go to the provider.
3. The provider streams an assistant message (text, thinking, tool calls). Tool calls run in parallel by default. Any tool marked `executionMode: "sequential"` forces the whole batch to run sequentially.
4. The tool results are appended. That is one **turn**. If there were tool calls, or messages are queued, another turn starts. Otherwise the run ends (`agent_end`).
5. Retries, overflow recovery, compaction-retry and steering or follow-up queues can start another run. `agent_settled` marks true idle.
6. Between turns, auto-compaction is checked: it fires when `contextTokens > contextWindow - reserveTokens`.

Steering messages are injected after the current turn. Follow-ups are injected after the run finishes. A tool can return `terminate: true` to skip the automatic follow-up model call.

There is **no built-in max-turns limit and no permission system**. Pi runs with the OS permissions of the process (`README.md` "Permissions & Containerization"). Harbor's adapter adds max-turns through a small extension (see §7).

---

## 2. Extension surfaces ("valves")

The table runs from most declarative to most code-heavy. Paths: `~/.pi/agent` (override with `PI_CODING_AGENT_DIR`, or SDK `agentDir`), and the project `.pi/`. Docs live at `packages/coding-agent/docs/*.md`, also published at https://pi.dev/docs/latest.

| Valve | Form | Where / how | Notes for the optimizer |
|---|---|---|---|
| **System prompt: full replace** | Markdown | `SYSTEM.md` (agent dir or `.pi/`), `--system-prompt <text\|file>`, SDK `systemPromptOverride` | Replaces only the `preamble`. Tools, rules, skills, context and cwd sections are still appended [V] |
| **System prompt: append** | Markdown | `APPEND_SYSTEM.md`, `--append-system-prompt` (repeatable) | Rendered as the `addendum` section [V] |
| **Context files** | Markdown | `AGENTS.md` / `CLAUDE.md` / `AGENTS.override.md`, discovered from the agent dir, cwd and its ancestors; `-nc` disables them | Rendered as the `project_context` section. Needs **no** project trust [V] |
| **System prompt: structured sections** | Code (extension) | `before_agent_start` → mutate `event.systemPromptOptions.sections.<name>`, `promptGuidelines`, `toolGuidelines`, `toolSnippets`; or `forceSystemPrompt` for a total replace | Sections: `preamble`, `tools`, `rules`, `docs`, `addendum`, `project_context`, `skills`, `cwd`, plus custom XML-tagged sections (`src/core/system-prompt.ts`). Section diffs are persisted, so traces show exactly which prompt was used [V] |
| **Tool selection** | Declarative | `defaultTools` setting (`["+grep","-write"]`), `--tools`, `--exclude-tools`, `--no-tools`; built-ins: `read bash edit write grep find ls powershell`, plus `codemode` and `tool_search` | [V] |
| **Custom tools** | Code | `pi.registerTool({name, description, parameters: TypeBox, execute, outputSchema?, executionMode?, exposure?, annotations?, prepareLoadout?, reachesTools?})`; SDK `customTools` | Exposure modes are `direct`, `model-only`, `codemode`, `deferred` and `hidden`. Tools can call other tools via `ctx.executeTool()`, and nested calls are recorded [V] |
| **Dynamic tool loadout** | Code | `pi.setActiveTools()`, `pi.getAllTools()` | Mid-session changes are persisted as system-message patches [V] |
| **MCP servers** | JSON | `mcp.json` (agent dir or `.pi/`), `pi mcp add`, `pi.registerMcpServer()` | Built in now (`builtin:mcp`) [V]. That it arrived around 0.86 is a guess from Harbor's version pin [I] |
| **Codemode** | Setting + code | `defaultTools: ["+codemode"]`, `codemode.mode: "on"\|"only"` | The model writes JS that orchestrates tools in a QuickJS sandbox. This is an interesting valve for tool-call efficiency [V] |
| **Skills** | Markdown dir | `skills/` or `.agents/skills/`, `--skill`, the Agent Skills spec (`SKILL.md` frontmatter) | Only name and description go into the prompt; the body is read on demand. `/skill:name` forces one [V] |
| **Prompt templates** | Markdown | `prompts/*.md` → `/name`, with `$1`, `$@` and `${1:-default}` substitution | Mostly an interactive feature. In a headless run, the task prompt can invoke one (`pi -p "/solve ..."`) [I] |
| **Extensions (lifecycle hooks)** | TypeScript | `extensions/*.ts` or `extensions/<dir>/index.ts`, `-e <path>`, settings `extensions: [...]`, SDK inline `extensionFactories` | See the event list below. Loaded by `jiti` with no compile step [V] |
| **Context management** | Code | `context` (filter or transform messages per request), `context_with_system` (whole transcript), `turn_end`/`agent_before_settle` can append `context_edit` entries (omit or replace earlier messages) | `context_edit` is append-only and branch-relative, so pruning is traceable [V] |
| **Compaction** | Settings + code | `compaction.enabled`, `reserveTokens` (16384), `keepRecentTokens` (20000), `modelOverrides`; `session_before_compact` hook supplies a custom summary; `ctx.compact()` / RPC `compact` | The default summary is a structured template (Goal, Constraints, Progress, Decisions, Next Steps, Critical Context). Example: `examples/extensions/custom-compaction.ts` [V] |
| **Loop control** | Code | `turn_end` / `agent_before_settle` return `continue: true` (one more model call), `ctx.abort()`, `tool_call` → `{block, reason}`, tool `terminate: true` | This is how to add verify-before-finish, budgets and reflection steps [V] |
| **Provider / model** | JSON + code | `models.json` (OpenAI/Anthropic-compatible endpoints, Ollama, vLLM, LM Studio), `pi.registerProvider()`, `pi.registerVirtualModel()` (per-request router) | Virtual models allow cascades, e.g. a cheap model first with escalation [V] |
| **Provider payload** | Code | `before_provider_headers`, `before_provider_request` (mutate the serialized payload), `after_provider_response`, `provider_stream_event` | [V] |
| **Retry / network** | Settings | `retry.enabled`, `retry.maxRetries` (3), `retry.baseDelayMs`, `httpIdleTimeoutMs` | [V] |
| **Thinking** | Setting / CLI | `defaultThinkingLevel`, `--thinking off..max`, `thinkingBudgets` | [V] |
| **Packages** | npm or git | `pi install npm:...`, `packages: [...]` in settings | Bundles extensions, skills, prompts and themes. Useful for versioning harness variants [V] |

**Full extension event list** [V] (`docs/extensions.md`; exact types in `src/core/extensions/types.ts`):

- Input: `input`, `before_agent_start`, `user_bash`
- Run: `agent_start`, `agent_end`, `agent_before_settle`, `agent_settled`
- Turn: `turn_start`, `context`, `context_with_system`, `turn_end`
- Provider: `before_provider_headers`, `before_provider_request`, `after_provider_response`, `provider_stream_event`
- Message: `message_start`, `message_update`, `message_end` (can replace the finalized message)
- Tool: `tool_execution_start`, `tool_call` (gate or mutate), `tool_execution_update`, `tool_result` (transform), `tool_execution_end`
- Session: `project_trust`, `session_start`, `resources_discover`, `session_before_switch`, `session_before_fork`, `session_shutdown`, `session_before_compact`, `session_compact_failed`, `session_before_tree`
- Other: `cache_warming_decision`, `mcp_servers_change`

The extension API also includes `pi.sendUserMessage`, `pi.sendMessage` (custom context message, optionally `triggerTurn`), `pi.appendEntry` (non-context state), `pi.events` (an inter-extension bus), `ctx.modelRegistry.streamSimple()` for nested model calls, and `ctx.sessionManager`.

**How declarative is it overall?** The prompt, AGENTS.md, skills, templates, settings, tool allowlists, models.json and mcp.json are plain files. Anything behavioral (loop control, context policy, custom tools, topology) is a TypeScript extension. For a Claude Code optimizer that edits files, that works well: it is one small `.ts` file per behavior and needs no build.

**Gotcha** [V] (`docs/security.md`): in print, JSON and RPC modes, **project-local `.pi/` resources (extensions, skills, settings, SYSTEM.md) are skipped** unless you pass `--approve`/`-a` or set `defaultProjectTrust: "always"` in agent-dir settings. AGENTS.md is not trust-gated. The simplest approach for HIVE is to put the harness in a dedicated `PI_CODING_AGENT_DIR` (the agent dir is always trusted), or pass `-e`, `--skill` and `--system-prompt` explicitly. [I]

---

## 3. Headless and programmatic use

| Mode | Command | Output | Use for HIVE |
|---|---|---|---|
| Print | `pi -p "task"` | Final assistant text on stdout. Exit code is nonzero on `error`/`aborted`/`length` stop reasons | Quick smoke tests |
| JSON | `pi --mode json -p "task"` (Harbor uses `--print --mode json`) | JSONL: session header, then every event (`agent_start`, `turn_*`, `message_*`, `tool_execution_*` with `durationMs`, `compaction_*`, `auto_retry_*`, `agent_settled`). `message_update` events are delta-only | **The default for eval runs.** One process per task. Failures show up in the stream, not always in the exit code |
| RPC | `pi --mode rpc --no-session` | JSONL commands on stdin (`prompt`, `steer`, `follow_up`, `abort`, `compact`, `set_model`, `get_session_stats`, `get_messages`, `get_entries`, `get_tree`, `fork`, `clone`, `export_html`, `bash`, ...). Responses are correlated by `id`; events stream on stdout | Long-lived workers (peer and manager-worker topologies), plus mid-run steering. A Python client is about 30 lines (`docs/rpc.md`). The TypeScript `RpcClient` has `promptAndWait()` |
| SDK | `createAgentSession({ cwd, model, resourceLoader, sessionManager, settingsManager, customTools, tools })` → `session.prompt()`, `session.subscribe()`, `session.getLastAssistantText()`, `session.dispose()` | In-process events | A custom orchestrator in TS. 14 examples in `packages/coding-agent/examples/sdk/` (01-minimal through 14-codemode-mcp). SDK sessions do **not** auto-load the codemode, tool_search and MCP built-ins |

[V] Useful flags: `--model provider/id[:thinking]`, `--provider`, `--thinking`, `--tools`, `--no-session`, `--session-dir`, `--session-id`, `--continue`, `-e`, `--skill`, `--system-prompt`, `--append-system-prompt`, `-nc`, `-ne`, `--offline`, `--approve`, and `--export <session.jsonl> [out.html]`.

**Running one task and getting a result** [V for flags, I for the recipe]:

```bash
PI_CODING_AGENT_DIR=$HIVE/harness/variant-07 \
pi --mode json -p --session-dir runs/$TASK_ID --model anthropic/claude-sonnet-4-6:medium \
   "$(cat task.md)" > runs/$TASK_ID/events.jsonl 2> runs/$TASK_ID/stderr.log
# result = last message_end with role=assistant; success = stopReason == "stop"; then run the task's verifier
```

---

## 4. Traces

**Session file** [V] (`docs/session-format.md`, `src/core/session-manager.ts`):

- Location: `~/.pi/agent/sessions/--<cwd>--/<timestamp>_<uuid>.jsonl`, or `--session-dir`.
- Format: JSONL, **version 3**, a **tree** of entries (`id`, `parentId`, ISO `timestamp`). The first line is the header `{type:"session",version:3,id,timestamp,cwd,parentSession?}`.
- Entry types: `message`, `model_change`, `thinking_level_change`, `usage`, `compaction`, `context_edit`, `branch_summary`, `custom`, `custom_message`, `label`, `session_info`.

What each message entry records:

| Message | Recorded fields |
|---|---|
| `system` | Prompt `sections` and `toolsAdded`/`toolsRemoved`. The first one holds the full loadout; later ones are patches. Replaying them gives the exact prompt and tools at every step |
| `user` | Content and an ms timestamp |
| `assistant` | Content blocks (text, thinking, toolCall with args), `provider`, `model`, `thinkingLevel`, `stopReason` (`stop`/`length`/`toolUse`/`error`/`aborted`), `errorMessage`, and **`usage`**: `{input, output, cacheRead, cacheWrite, reasoning?, totalTokens, cost: {input, output, cacheRead, cacheWrite, total}}` in USD |
| `toolResult` | `toolCallId`, `toolName`, content, `isError`, `details`, optional nested `usage`, and `nestedCalls` (each with name, args, status, `durationMs`) |
| Coding-agent extras | `bashExecution`, `custom`, `branchSummary`, `compactionSummary` (see `docs/message-types.md`) |

Other trace sources:

- **Timing:** every message carries an ms timestamp, so model latency per turn can be derived [I]. Tool `durationMs` is in the **JSON/RPC event stream** (`tool_execution_end`), not in the session file. Only nested calls keep `durationMs` in the file [V].
- **Compaction entries** record `summary`, `firstKeptEntryId`, `tokensBefore`, the summary's `usage`, and `details` with read/modified file lists [V].
- **Session totals:** RPC `get_session_stats` returns message and tool-call counts, tokens and cost. `src/core/usage-totals.ts` has `getUsageCostBreakdown(entries)` [V].
- **Export:** `pi --export s.jsonl out.html`, `/export` (HTML or JSONL), `/share` (gist or Radius). Harbor converts the session to **ATIF-v1.7** `trajectory.json`, with steps, tool calls, observations and per-step metrics (§7) [V].
- **Telemetry:** `pi-telemetry` defines span and metric contracts but no exporter. The coding-agent's `telemetry.ts` only handles install telemetry. For OpenTelemetry-style spans we would write an extension that listens to the events [V for the facts, I for the plan].
- **A trace-rich corpus exists:** Zechner publishes real Pi sessions on Hugging Face (`badlogicgames/pi-mono`, via `badlogic/pi-share-hf`). This could seed the optimizer's understanding of the trace format [V].

---

## 5. Multi-agent

**There is no built-in subagent support.** This is a design choice. The README says "Pi ships with powerful defaults but skips features like sub-agents and plan mode." [V]

Available building blocks:

1. **Reference subagent extension** [V] (`packages/coding-agent/examples/extensions/subagent/`, about 1,000 lines of TS). It registers a `subagent` tool with three modes: single `{agent, task}`, parallel `{tasks:[...]}` (max 8, 4 concurrent), and chain `{chain:[...]}` with a `{previous}` placeholder. Each child is a **separate `pi` process** spawned as `pi --mode json -p --no-session --model X --tools a,b --append-system-prompt <tmp> "Task: ..."`. Agents are Markdown files with frontmatter (`name`, `description`, `tools`, `model`) under `~/.pi/agent/agents/*.md`, for example `scout.md`, `planner.md`, `reviewer.md` and `worker.md`. It tracks per-child usage and cost and caps output at 50 KB per task. **This is effectively a manager-worker / hierarchical topology already.** The topology lives in Markdown agent defs plus prompt-template "workflows" (`implement.md` = scout → planner → worker), which an optimizer could edit declaratively.
2. **Durable package** [V] (`packages/durable`, **experimental, "API changes without notice"**). It has first-class owned child conversations (`ownership: {kind:"task"}`), foreground and background subagents (`test/examples/22-subagent-foreground.ts`, `23-subagent-background.ts`), child tasks with `allSettled`/`failFast` wait policies, a live `taskGraph()`, and per-conversation `usage()`. This is the most principled multi-agent substrate. It is **not** the `pi` CLI that Harbor runs, though.
3. **Third-party packages** [V that they exist; quality unverified]: `@tintinweb/pi-subagents`, `@narumitw/pi-subagents`, `@mjakl/pi-subagent` (has a depth guard), `pi-agent-teams` variants (shared task board and mailboxes), and `pi-agents-team` (a coordinator plus background **RPC** workers).
4. **Primitives** [V]: `pi.events` (an in-process bus between extensions), RPC mode (long-lived peers with `steer`/`follow_up`), `ctx.modelRegistry.streamSimple()` (cheap nested model calls), `AgentSessionRuntime.fork()`, and `handoff.ts` (context transfer to a new session).

**How to build each HIVE topology** [I]:

| Topology | Build | Trace story |
|---|---|---|
| Single-agent baseline | Stock `pi` with a pinned version and the default prompt and tools | One session JSONL |
| Hierarchical / manager-worker | Fork the reference `subagent` extension into our repo. Workers are `pi --mode json` children with `--session-dir runs/<task>/workers/<n>` (**drop `--no-session`** so child traces persist). Roles are `agents/*.md` | Parent JSONL plus N child JSONLs. Link them through the tool-call `details` |
| Peer-to-peer | N long-lived `pi --mode rpc` processes run by a small Python or TS orchestrator, which relays messages as `follow_up`/`steer` commands. Alternatively, a shared-mailbox extension (as in `pi-agent-teams`) | Per-peer JSONL plus an orchestrator log |
| Fixed multi-agent baselines | Freeze specific `agents/*.md` and workflow templates (e.g. scout→planner→worker, worker→reviewer→worker) | Same as above |

To run inside Harbor, the whole topology must run **inside the task container** from one `pi` invocation. A subagent extension does that, and so does an orchestrator script the adapter launches instead of `pi` [I].

---

## 6. Model and provider flexibility, cost tracking

[V] `packages/ai/src/providers/` has about 50 built-in providers, among them:

- Anthropic, OpenAI (plus `openai-codex` subscription)
- Google and Vertex, Bedrock, Azure, Mistral, Groq, Cerebras, xAI, DeepSeek
- OpenRouter, Together, Fireworks, Hugging Face
- GitHub Copilot, Vercel AI Gateway, Cloudflare
- Moonshot/Kimi, Qwen, Z.ai, MiniMax, NVIDIA, Baseten

Auth comes from env vars, `/login` (OAuth or subscriptions), `auth.json`, or `--api-key`.

- **Local models** [V]: llama.cpp router integration (`builtin:llama.cpp`, `docs/llama-cpp.md`). Ollama, LM Studio, vLLM and SGLang work through `models.json` compatible endpoints (`docs/models.md`).
- **Cross-provider handoff** [V]: the model can be switched mid-session (`model_change` entries), and thinking blocks from the previous model are dropped (1.3.0).
- **Cost** [V]: every assistant message has `usage.cost` in USD, computed from a bundled model catalog. pi.dev overlays a newer catalog at runtime; use `--offline` to freeze it. Compaction, branch summaries and cache warming record their own usage, and session totals include them. Cache-read and cache-write tokens are split out. **Reproducibility caveat:** the catalog refresh means prices and model metadata can change between runs unless you use `--offline` or the Nix pin [I].

---

## 7. Harbor integration

**Harbor has a built-in Pi agent** [V]: `src/harbor/agents/installed/pi.py` (1,108 lines), `AgentName.PI = "pi"`. Usage: `harbor run --dataset terminal-bench@2.0 --agent pi --model anthropic/<model>`. Harbor also has a built-in `openclaw.py`.

What it does:

- **Installs** Pi into the container via nvm and `npm install -g --ignore-scripts @earendil-works/pi-coding-agent@<version>`. Before 0.74.0 it uses the legacy `@mariozechner` name. The `version` kwarg pins it, and the default is `latest`.
- **Runs** `pi --print --mode json --session-dir <logs>/pi/sessions [--continue] --provider P --model M [--extension ...] [--thinking ...] "<instruction>"`, filtering out `message_update` lines and tee-ing the output to `pi.txt`.
- **Options** (`PiOptions`): `thinking`, `model_api` (custom endpoint style), `max_turns`, plus the generic `version`, `prompt_template_path` (a Jinja wrapper around the instruction) and `config`.
- **max_turns** is implemented as a generated extension. It appends a budget note to the system prompt on `before_agent_start` and calls `ctx.abort()` on `turn_end` once the count is reached. This is a nice concrete example of a "valve".
- **Custom endpoints:** writes `models.json` into an isolated `PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent` (provider `harbor-endpoint`).
- **Skills:** copies the Harbor task's skills into `~/.agents/skills/`.
- **MCP:** uses the third-party `pi-mcp-adapter@2.34.0` and **forces Pi into `>=0.84.1,<0.86.0` (default 0.85.1)** whenever MCP servers are configured. That is stale, because Pi 1.x has built-in MCP. Avoid MCP tasks, or patch this.
- **Traces:** after the run, it converts the active-branch session JSONL into **ATIF-v1.7** `trajectory.json` (steps, tool calls, observations, images, per-step and final token and cost metrics). Capabilities: `atif=True, resume=True, skills=True, mcp_servers=True`.

What it lacks for HIVE [V]: there is no kwarg to inject an arbitrary harness (extensions dir, SYSTEM.md, settings, agents). The plan [I] is to subclass `Pi` in our repo and load it with `--agent our_pkg.hive_pi:HivePi` (the old `--agent-import-path` is deprecated in favor of `--agent`). The subclass would:

1. Upload `harness/<variant>/` into the container as `PI_CODING_AGENT_DIR`.
2. Set `defaultProjectTrust`, or pass explicit `-e`, `--skill` and `--system-prompt` flags.
3. Keep the parent `populate_context_post_run` so ATIF conversion still works.
4. Also collect child-session JSONLs for multi-agent variants.

There is also a third-party PyPI package, `harbor-pi-sdk-agent` (by Kingwl, v0.1.x), that wraps the Pi SDK as a Harbor custom agent. I did not inspect it, and it appears to target the old `@mariozechner` package [V for existence, unverified for quality].

---

## 8. Maturity, license, docs, community, churn

- **License:** MIT [V].
- **Age and versions:** the repo was created 2025-08-09. The current version is 1.3.0; 1.0.0 shipped 2026-10-01 [V].
- **Community:** about 116k GitHub stars and about 14.9k forks; 350 open issues; pushed today [V via the GitHub API]. There is a Discord and a pi.dev package registry with many third-party packages. It is the foundation of OpenClaw [V, per the README].
- **Contribution policy:** "New issues and PRs from new contributors are auto-closed by default" (maintainers triage daily). Expect upstream fixes to be slow to get in; we should plan to work around issues rather than patch upstream [V].
- **Docs quality:** high. There are about 40 focused docs in `packages/coding-agent/docs/` (7.6k lines), and they are precise about lifecycle semantics. There are 60+ checked extension examples and 14 typechecked SDK examples. The exact types live in `src/core/extensions/types.ts`. The docs tell the agent to read the docs, and the system prompt even includes doc paths [V].
- **Release cadence and breaking changes:** 290 changelog version headings in about 10 months. That is roughly one release every 1–2 days in mid-2026, and still several per week in October (1.0.0 to 1.3.0 in two weeks). There are **50 "Breaking Changes" sections**. Recent breaks hit exactly the surfaces we would use:
  - `before_agent_start` became a union (1.2.0).
  - `shouldStopAfterTurn` was removed in favor of `finishTurn`, `SessionManager` became canonical, and the `TurnEndEvent` shape changed (0.87–0.99).
  - JSON-mode `message_update` became delta-only (0.84).
  - The custom-provider context type changed.
  - The Azure provider was renamed (1.0.x).
  - The npm scope was renamed (0.74).

  [V] The post-1.0 rate is lower but nonzero.
- **Engineering hygiene:** pinned deps, `--ignore-scripts`, an install lock, audit workflows, and its own eval package [V].

---

## 9. Risks and weak points for HIVE

1. **API churn** (high). Pin an exact Pi version (`version` kwarg in Harbor, `package-lock` locally) for the whole semester. Re-validate extensions before any bump. The optimizer must be told the pinned version, because the docs on `main` may describe newer APIs [I].
2. **Multi-agent is DIY.** The reference subagent extension uses process spawning and `--no-session` children by default, so child traces are lost unless we change that. Durable has the best model but is experimental and is not the CLI Harbor runs [V].
3. **Language split.** Harnesses are TS; Harbor and our analysis are Python. The JSON and RPC modes keep the boundary clean, but the optimizer edits TS extensions, so typos surface only at runtime. `jiti` gives no typecheck. Add a `tsc --noEmit` gate against the pinned `@earendil-works/pi-coding-agent` types before each eval [I].
4. **Harbor adapter gaps.** There is no harness injection, the MCP path is pinned to an old Pi, and max-turns is a blunt abort. All of these are fixable in a subclass [V].
5. **Headless trust trap.** Project `.pi/` resources are silently skipped in JSON and RPC modes unless trusted, so an "edited" harness may simply not load. Assert in every run that the expected extensions loaded. The session's first `system` message lists sections and tools, so this check is easy [V for the behavior, I for the check].
6. **No sandbox or permissions.** It relies on the Harbor container. Running it locally outside Docker is unsafe for autonomous runs [V].
7. **Exit codes in JSON mode don't reflect failure.** Parse `stopReason` and `agent_settled`, and use the task verifier for success [V].
8. **Nondeterminism and catalog drift.** The runtime model-catalog overlay, cache warming and auto-retry all affect cost and behavior. Use `--offline`, and consider `cacheWarming: "off"` and fixed retry settings for comparability [I].
9. **The optimizer has a lot of power.** Extensions run in-process with full permissions and can change scoring-relevant behavior (e.g. skip the verifier, or rewrite tool results). Guard by keeping the eval and verifier outside the harness dir the optimizer can edit [I].
10. **Timing is incomplete in the session file.** Tool durations live only in the event stream, so keep `events.jsonl` alongside the session JSONL [V].

---

## Key file and URL index

- Repo: https://github.com/earendil-works/pi (old: https://github.com/badlogic/pi-mono). Docs: https://pi.dev/docs/latest
- `packages/coding-agent/docs/`: `how-pi-works.md`, `extensions.md`, `sdk.md`, `rpc.md`, `rpc-commands.md`, `json.md`, `cli.md`, `cli-integration.md`, `session-format.md`, `message-types.md`, `compaction.md`, `settings.md`, `configuration.md`, `skills.md`, `prompt-templates.md`, `custom-provider.md`, `models.md`, `security.md`, `containerization.md`
- `packages/coding-agent/src/core/extensions/types.ts` (event and API types), `src/core/system-prompt.ts` (sections), `src/core/session-manager.ts`, `src/core/usage-totals.ts`
- `packages/coding-agent/examples/extensions/`: `subagent/`, `custom-compaction.ts`, `prompt-customizer.ts`, `permission-gate.ts`, `handoff.ts`, `tool-override.ts`, `dynamic-tools.ts`, `trigger-compact.ts`, `structured-output.ts`
- `packages/coding-agent/examples/sdk/01..14`, `examples/rpc-client.ts`
- `packages/durable/README.md` (subagents, child tasks, task graph, usage)
- Harbor: `src/harbor/agents/installed/pi.py`, `openclaw.py`; https://harborframework.com/docs
- Third-party: https://pypi.org/project/harbor-pi-sdk-agent/, https://pi.dev/packages/@tintinweb/pi-subagents, https://pi.dev/packages/pi-agents-team
