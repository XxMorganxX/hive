# Framework adapters (Deep Agents, Strands) and extensions in the contract

## Goal

1. Explain the agent harness contract as it stands. Only Pi has a real adapter.
2. Make **extensions** a first-class, framework-neutral part of the contract. In Pi, extensions are TypeScript modules hooking the agent loop. They are a primary lever for harness evolution, but today they exist only inside the Pi adapter.
3. Add adapters for **LangChain Deep Agents** and **AWS Strands**, so the same candidate/suite/overseer loop runs on three frameworks.

## Approach

1. **Explore** (workflow, 4 parallel subagents):
   - explain the current contract;
   - scout a Deep Agents adapter against the installed package;
   - scout a Strands adapter against the installed package;
   - map Pi extensions, Deep Agents middleware and Strands hooks onto neutral lifecycle points.
2. **Design** how extensions fit the contract. Weigh minimal vs. rich designs; record the decision here and in `docs/FOUNDING_PRINCIPLES.md` §5.
3. **Implement:** I make the contract change myself, then fixers fan out on disjoint files (one adapter each, plus controller and Pi alignment). All work is test-first and offline, using fake models.
4. **Review** adversarially, then verify with real runs if credentials allow.

## Decisions

- **Extensions are first-class in the contract; the code stays native.** The extension map showed a portable API would shrink to the lowest common denominator. The three frameworks give the "same" hook different semantics:
  - context edits apply to one call (Pi) vs. persist (LangChain);
  - a retry discards the first reply (Strands) vs. keeps it (LangChain);
  - only Pi has a continue-run hook.

  So the contract shares the shape around the code, not the code:
  - **`EXTENSIONS_DIR`:** one directory name in every candidate.
  - **`ExtensionSpec`:** each adapter declares its language, file suffixes and entry convention, the pinned framework version, and a catalog over 10 neutral `LIFECYCLE_POINTS`, each mapped to its native API with support level and caveats. It also lists forbidden APIs (P3) and the pinned API reference text. The overseer's sandbox can't read the installed framework, so the definitions travel in the guide.
  - **`LoadReport`:** returned in every `AgentResult` when the candidate has extensions. **The controller** (`hive/extensions.load_problem`) enforces it. A file not reported loaded makes the trial `harness_load_error`; extension code that throws mid-run makes it the new `extension_error`. Previously only the Pi adapter checked, using a negative stderr regex.
  - **The guide is rendered by the controller** (`render_extension_guide`) into `OverseerBrief.extension_guide`, so it reads the same for every framework.
  - Rejected: a portable `policies.toml` extension DSL. Revisit once the overseer is seen writing the same extension shapes repeatedly.
- **Deep Agents and Strands run one subprocess per trial, in their own pinned venvs** (`.venvs/<framework>`, built by `scripts/setup-framework-venvs.sh`). The controller stays dependency-free; the frameworks pin conflicting `openai` majors (3.x vs 2.x); candidate code never runs in the controller; timeouts can kill the run.
- **A shared subprocess helper** (`hive/agents/process.py`):
  - **Output goes to files and the run ends on process exit.** A detached grandchild inheriting stdout made pipe-based `communicate()` hang until timeout; a test reproduced this.
  - **Shell children are reaped after every trial.** Both frameworks start shells in their own sessions, so a group kill misses them; the runner lists their pgids and the helper kills them.
- **A shared runner protocol** (`hive/agents/runner_protocol.py`): one place classifies every Python runner's output.
  - **Usage and the reported model come from `trace.jsonl`,** one line per model call. A run killed on timeout still has its spend counted.
  - **`result.json` carries status, text and the load report.**
  - **Cost comes from a required `prices` table** in `[agent]`, since neither framework computes dollars. `gpt-5-nano` costs $0.05/$0.40 per million input/output tokens, matching Pi's own cost on the earlier real run exactly.
  - **The model check accepts dated snapshots** (`gpt-5-nano-2025-08-07`).

## Progress log

- 2026-10-19: Launched the exploration workflow.
- 2026-10-19: Exploration landed (the explorers' full reports were working notes, not kept in the repo). Contract gaps confirmed:
  - **Extensions aren't in the neutral contract.** Only the Pi adapter knows about them.
  - **Detection is negative-only.**
  - **An extension tool missing from `tools` is silently inactive.**
  - **The valve catalog is free-form markdown.**

  The scouts verified Deep Agents 0.7.28 (60 packages) and Strands 1.61.1 (54 packages) offline. Neither ships a tool-capable fake model, so the runners take a scripted test model. Strands defaults to Bedrock and silently probes AWS, so its runner must refuse Bedrock outright. Both leave orphan shell processes after a group kill.
- 2026-10-19: Wrote the contract additions, `hive/extensions.py`, `hive/agents/process.py` and `hive/agents/runner_protocol.py` myself, test-first. Added a `_path` suffix to the path-setting rule, and `scripts/setup-framework-venvs.sh`.

- 2026-10-19: Implementation fan-out landed: 4 implementers on disjoint files, all test-first.
  - **Controller:**
    - `evaluate` applies `load_problem` once per trial. `model_mismatch` outranks everything; a failed load replaces any other class but keeps usage and wall clock; a runtime throw replaces `completed`/`agent_error` but not timeout, quota or provider errors.
    - **Refinement (accepted):** a *missing* report only fails a completed run. A timeout kill or crash leaves no `result.json`, and relabelling it "no load report" would point the overseer at the wrong cause. Both score 0, so it can't be gamed.
    - `TrialOutcome` records the load report, so the overseer's evidence shows what loaded.
    - The overseer gets `context/extensions.md`, and a prompt section (`extensions_prompt.md`) invites an extension when traces show a behavioral failure, still as one isolated change.
  - **Pi:**
    - The `ExtensionSpec` covers all 10 points: `after_model_call`, `wrap_model_call` and `after_run` are partial, the rest are supported. `EXTENSIONS_API.md` is about 400 lines excerpted from `types.d.ts`.
    - **Load report by construction:** Pi 0.75.3 exits before the session header if any `-e` file fails, so "header seen ⇒ all loaded". Checked in `main.js` and on the binary.
    - Runtime `Extension error (path)` lines are now `extension_error`, not `harness_load_error`.
    - `registered` comes from static analysis of the `.ts` source.
    - `validate` rejects a literal extension tool name missing from `[valves].tools`. This closes the silently-inactive-tool gap.
  - **Deep Agents** (`deepagents==0.7.28`, `langchain-openai==1.7.2`):
    - **Entry convention:** `def extension()` returns a list of `AgentMiddleware`/`BaseTool`.
    - **HIVE guards** (call limit, tool allowlist, `ToolErrorMiddleware`) sit inside the extension middleware.
    - **The general-purpose subagent is off by default,** because user middleware doesn't reach it. When the valve enables it, it gets HIVE's guards.
    - **Usage** is summed per call by callback, since subagent tokens are missing from the final state.
    - Shell children are reaped through a `LocalShellBackend` subclass.
  - **Strands** (`strands-agents[openai]==1.61.1`, `openai==2.54.0`, no `strands-agents-tools`):
    - **Entry convention:** `def extension()` returns a `strands.plugins.Plugin` named after its file stem.
    - **Bedrock is refused outright,** and the AWS environment is scrubbed.
    - **A recording OpenAI client** records the model the provider reported, which Strands ignores.
    - **A model guard hook** catches `agent.model` swaps.
    - **Hook throws** are attributed to their file and re-raised.
    - **`validate` statically rejects obvious P3 breaches,** such as importing model SDKs or assigning `.model`.
  - **Integration (me):**
    - Registered `deepagents` and `strands` as built-ins.
    - Fixed a bug both adapter builders hit independently: `Path.resolve()` on `python_path` followed the venv's `bin/python` symlink to the bare base interpreter, so the configs would have run without the framework installed. Path settings are now absolute but keep symlinks (test-first).
    - **Observed:** 1098 tests pass, including real offline framework runs through the runners with the scripted provider.
- 2026-10-19: Launched an adversarial review (5 lenses, two skeptics per finding).

- 2026-10-19: Adapter review: 5 lenses, 81 agents. 34 findings confirmed, 4 refuted. The highs share one root cause: the provider key, the model check and the spend count all live in the same process as candidate extension code, so they were *detection*, not a boundary.
  - **The key leaks:** the agent's shell reads it from the runner's exec environment (`ps eww $PPID`), and it lands in traces the overseer reads.
  - **Calls escape the checks:** an extension can call another model or change effort through the client object.
  - **Spend can be erased:** an extension can rewrite `trace.jsonl` so the run reads $0.

  The 20 findings that don't depend on the fix are being fixed in parallel: key via stdin, call limits, name and tool collisions, load-report precedence, attribution, validators, Pi compaction usage, setsid escapes.
- 2026-10-19: **Decision (user): build a controller-side model gateway now.** **Decision (user): no real Deep Agents / Strands runs by me;** the user will export `OPENAI_API_KEY`.

- 2026-10-19: Gateway built and attacked. Tests use a fake upstream and real OpenAI SDKs (openai 2.x/3.x, langchain-openai).
  - **It held against 14 attack families:** auth, path desync, duplicate keys, model/effort lookalikes, limit races, revocation, key echo, disconnect billing.
  - **Two gaps, fixed by me test-first:**
    - Hosted tools (web/file search, code interpreter) bill outside token usage. Only function and custom tools are allowed now.
    - One trial could pin unbounded gateway threads. Now there is a per-trial in-flight cap and a 30 s client timeout.
  - **The builder's own additions:** a `background` refusal, forced `include_usage`, billing of incomplete/failed responses, `usage_missing` flags, and key scrubbing of relayed bytes.
- 2026-10-19: Independent review fixes landed and were rechecked.
  - **Key:** sent on stdin (never in the runner's environment); shell children get a minimal environment.
  - **Call limits:** counted per provider request in the callback, shared with subagents. The subagent recursion limit is bound.
  - **Collisions:** extension name and tool collisions are rejected.
  - **Load reports:** the report is omitted until the load phase has run.
  - **Trial marker:** a per-trial marker kills `setsid` escapes.
  - **Pi:** `before_provider_request` is rejected statically, and compaction usage is estimated.
  - **Limits the recheck confirmed that only containers close** (recorded, not fixed):
    - On macOS, system binaries hide their environment, so a `setsid` escape via `/bin/*` can survive the marker sweep.
    - On the host, any same-user process can read any other's exec environment (`ps eww`). The recheck saw a real OpenAI key in two unrelated long-running processes on this machine, and the user was told. So the controller's own `OPENAI_API_KEY` is readable by agent shells too. The gateway keeps the key out of trial processes, but not out of reach of the host. The real boundary is running trials in containers or as a separate OS user.

- 2026-10-19: Gateway integrated (4 integrators + end-to-end recheck against a fake upstream; no real API).
  - **Controller:** `evaluate` opens/closes a grant per trial, and the ledger is authoritative for usage and cost. A violation or a mismatching reported model makes the trial `model_mismatch`. An exhausted call limit on an incomplete run makes it `agent_error`; an upstream quota code makes it `quota`. The ledger goes to `gateway.json`, and `TrialOutcome.gateway` holds a summary.
  - **Config:** `[model] prices` and `upstream_url`; `[evaluation] max_model_calls_per_trial`. A missing key fails before the run dir exists.
  - **Pi:** a per-run `models.json` points its provider at the endpoint, and there is no `auth.json`. Verified with the real binary against a dead local URL.
  - **Deep Agents / Strands:** the endpoint goes on stdin. Strands now gets a fresh `HOME`.
  - **The recheck verified, per framework:**
    - report cost equals the ledger;
    - an extension's hidden calls are billed;
    - wrong model or effort gives `model_mismatch`;
    - the limit holds;
    - no key or token in any file;
    - the key is absent from trial environments.
- 2026-10-19: **Decision: `[model] thinking` is required for real providers, and limited to `minimal|low|medium|high`.** The recheck found "unset" and "off" meant different things to the adapters (fallback "low", Pi clamping "off" to "minimal") and to the gateway (expects no effort), so every trial became `model_mismatch`. One shared set of levels removes the ambiguity (test-first in `test_config.py`).

## Model gateway (spec)

**Why.** Model access is where P3 (frozen model), P12 (spend) and key secrecy are decided. Today all three are decided inside the trial process, next to candidate code, so they can only be detected. Moving them into the controller makes them a boundary, the same for every framework.

**What.** `hive/gateway/` is trusted, stdlib-only, controller-side code. It is a threaded HTTP server on `127.0.0.1:<ephemeral>` that speaks the OpenAI-compatible API.
- **Per-trial grants.**
  - `open_trial()` mints a random per-trial token and returns a `ModelEndpoint(base_url, api_key=<token>)`.
  - `close_trial()` revokes the token and returns the trial's `TrialLedger`.
  - The real provider key is read once from the controller's environment (`OPENAI_API_KEY` for provider `openai`) and never leaves the gateway.
- **Each request** (`POST /v1/chat/completions`, `POST /v1/responses`; anything else is refused):
  - **Auth:** an unknown or revoked token gets 401.
  - **Model and effort (P3):** `model` must equal the experiment's model id. The effort (`reasoning_effort` for chat, `reasoning.effort` for responses) must equal `[model] thinking`; if absent, the gateway sets it. A mismatch gets 403 and is recorded as a violation.
  - **Cost-changing parameters:** `service_tier` other than default/auto and `n > 1` get 403 and a violation.
  - **Call limit:** past `max_model_calls_per_trial`, requests get 403 and `limit_reached` is set.
  - **Forwarding:** requests go upstream with the real key. Streaming is passed through as it arrives, and usage is parsed on the way: chat `include_usage` is injected if missing and the final chunk read; responses use the `response.completed` event.
  - **Billing:** the gateway reads the upstream response to the end even if the client disconnected (bounded), so a killed trial is still billed.
  - **Recording:** every call is logged with the requested and provider-reported model, normalized usage (`input` excludes cache reads), the HTTP status and any error code (e.g. `insufficient_quota`).
- **The controller.** `evaluate()` opens a grant per trial, passes the endpoint to `Agent.run(..., endpoint)`, closes the grant after the run, and applies the ledger.
  - **Usage and cost come from the ledger** (`[model] prices`), not from the adapter.
  - **Any violation or reported-model mismatch** makes the trial `model_mismatch`.
  - **The call limit** makes it `agent_error`, with the reason stated.
  - **An upstream quota code** makes an errored run `quota`.
  - The ledger is written to `trial_dir/gateway.json` as evidence.
- **Contract.**
  - `Agent.run(task, workdir, trace_dir, endpoint: ModelEndpoint | None)`. Every adapter must send ALL model traffic to the endpoint when one is given.
  - `ModelSpec` is unchanged.
  - Prices move from `[agent]` to `[model] prices`, because cost is a property of the model and is now computed in one place.
  - `[evaluation] max_model_calls_per_trial`.
- **Adapters.**
  - **Pi:** a `models.json` in its per-run agent dir points the provider at the endpoint, with the token as key.
  - **Deep Agents:** `ChatOpenAI(base_url=…, api_key=token)`.
  - **Strands:** `AsyncOpenAI(base_url=…, api_key=token)`.
  - The token travels on stdin, as the credential convention already does.
- **Offline / test.** Provider `scripted` gets no gateway (`endpoint=None`). Gateway tests use a fake upstream server, a local HTTP server that streams canned SSE.
- **Limits that stay.** The gateway can't stop a trial from opening its own connection to the internet with a key it doesn't have. Without the key, though, there is nothing to bill, and the in-process checks remain as a second line of detection.

## Open questions

- Credentials. Pi's only credential is an OpenAI key in `~/.pi/agent/auth.json`. Deep Agents (`langchain-openai`) and Strands (OpenAI provider) read `OPENAI_API_KEY` from the environment, so a real smoke run needs that variable set.

## Out of scope

- A portable extension API, where one extension file runs on every framework. Pending the mapping result.
