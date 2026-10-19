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
- 2026-10-19: Exploration landed; scratch reports are under scratchpad/explore. Contract gaps confirmed:
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

## Open questions

- Credentials. Pi's only credential is an OpenAI key in `~/.pi/agent/auth.json`. Deep Agents (`langchain-openai`) and Strands (OpenAI provider) read `OPENAI_API_KEY` from the environment, so a real smoke run needs that variable set.

## Out of scope

- A portable extension API, where one extension file runs on every framework. Pending the mapping result.
