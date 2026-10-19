# Harbor, Meta-Harness, and the Mismanaged Geniuses Hypothesis

Research notes for HIVE (CS 4701): the evaluation and harness-optimization foundations.
Compiled 2026-10-19.

**Evidence labels.** Every claim carries one of these:

- **[V]** Verified. I read the source code, the docs in the repo, or the paper's LaTeX source myself.
- **[R]** Reported. It comes from a web page or third-party summary I did not check against code.
- **[M]** From memory. It is prior knowledge of a well-known paper that I did not re-read this session.
- **[I]** Inferred. It is my own conclusion or recommendation.

**Sources read directly**

- Harbor repo, shallow clone of `main` at commit `357c5e7` (2026-10-18), version `0.24.0`: https://github.com/harbor-framework/harbor. The old URL `laude-institute/harbor` redirects there. Docs: https://harborframework.com/docs
- Meta-Harness paper, LaTeX source of arXiv 2603.28052 v1: https://arxiv.org/abs/2603.28052
- Meta-Harness code: https://github.com/stanford-iris-lab/meta-harness (MIT, commit from 2026-10-01). The optimized TB2 harness is at https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact
- MGH blog post: https://alexzhang13.github.io/blog/2026/mgh/

---

## A. Harbor

### A.1 What it is

Harbor comes from the team that built Terminal-Bench. It is the official harness for Terminal-Bench 2.0. The README describes it as "a framework ... for evaluating and optimizing agents and language models": you evaluate arbitrary agents, build benchmarks, run thousands of sandboxes in parallel, and "generate rollouts for RL optimization." [V]

Core concepts, from `docs-mintlify/core-concepts/index.mdx` [V]:

| Concept | Meaning |
|---|---|
| **Task** | One or more instructions, plus a sandbox environment and a verifier. Stored as a directory. |
| **Dataset** | A collection of tasks, usually one benchmark. Can be shared via Harbor Hub / the registry. |
| **Agent** | A program that completes tasks. Implements `BaseAgent`. |
| **Sandbox / "environment"** | An isolated runtime that implements `BaseEnvironment`. The CLI flag is `-e`. |
| **Verifier** | Scores the agent's work and writes the reward. |
| **Trial** | One agent's attempt at one task. |
| **Job** | A set of trials, run in parallel, across datasets × agents × models. |
| **Trajectory** | The agent/user interaction history, in the ATIF format. |

### A.2 Task format [V]

```
my-task/
├── instruction.md          # the prompt given to the agent
├── task.toml               # config + metadata (schema_version = "1.3")
├── environment/
│   └── Dockerfile          # or docker-compose.yaml (multi-container), Apptainer.def (HPC)
├── solution/solve.sh       # oracle solution (used by the `oracle` agent for sanity checks)
└── tests/test.sh           # verifier entrypoint (test.bat on Windows)
```

- **Reward contract.** `tests/test.sh` must write `/logs/verifier/reward.txt` (one number, usually 0/1) or `/logs/verifier/reward.json` (labeled numeric metrics, e.g. `{"correctness": 1, "style": 0.5}`). When both exist, `reward.json` wins.
- **`task.toml` sections.** `[task]` holds name/authors/keywords. `[metadata]` is free-form. `[verifier]` sets `timeout_sec` (default 600), `env`, `user`, `network_mode`, and `environment_mode = "shared" | "separate"`. `[agent]` sets `timeout_sec`, `user`, and `network_mode`. `[environment]` sets `docker_image`, `cpus`, `memory_mb`, `storage_mb`, `network_mode` (`public`/`allowlist`/`no-network`), `allowed_hosts`, and `workdir`. `[artifacts]` lists files to collect.
- **Timing.** `tests/` is uploaded to `/tests/` only after the agent finishes. By default the verifier runs in the same container as the agent (`shared`). `separate` starts a fresh verifier container: the agent's filesystem changes are not carried over, and only declared artifacts are copied in. That gives a stronger anti-tampering boundary, and it also enables `harbor regrade`.
- **Multi-step tasks.** These use a `steps/step-N/{instruction.md,tests/,solution/,workdir/setup.sh}` layout with `multi_step_reward_strategy = "mean" | "final"`. Agents that support `resume` continue their native session between steps. This is useful for testing memory and continual learning.
- **Task resources.** Tasks can supply **skills** (copied to `/harbor/skills`) and **MCP servers**. Agents opt in through the `skills` and `mcp_servers` capabilities.
- **RewardKit** (`harbor-rewardkit`) is a helper package for programmatic criteria and LLM/agent-as-judge verifiers.
- **Paths inside the sandbox** (`EnvironmentPaths`): `/logs/agent`, `/logs/verifier`, `/logs/artifacts`, `/tests`, `/solution`, `/harbor/skills`.

### A.3 Agent interface (exact shape) [V]

Source: `src/harbor/agents/base.py`.

```python
class BaseAgent(ABC):
    capabilities: ClassVar[AgentCapabilities] = AgentCapabilities()
    options_model: ClassVar[type[AgentOptions] | None] = None   # pydantic model for --ak kwargs
    MODEL_CONNECTION: ClassVar[ModelConnectionSpec | None] = None

    def __init__(self, logs_dir: Path, model_name: str | None = None, logger=None,
                 mcp_servers: list[MCPServerConfig] | None = None,
                 skills_dir: str | None = None, *, extra_env=None,
                 model_base_url=None, model_api_key_name=None, model_api_format=None,
                 skip_model_parsing=False, load_trajectory=None,
                 environment_logs_dir=None, **kwargs): ...

    @staticmethod
    @abstractmethod
    def name() -> str: ...
    @abstractmethod
    def version(self) -> str | None: ...
    @abstractmethod
    async def setup(self, environment: BaseEnvironment) -> None: ...
    @abstractmethod
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None: ...      # mutate `context` in place

    # optional
    async def resume(self, instruction, environment, context) -> None
    async def load(self, instruction, environment, context) -> None
    @classmethod
    def handoff(cls, trial_dir, cwd) -> list[str]
    def populate_context_post_run(self, context: AgentContext) -> None
    @classmethod
    def preflight(cls, kwargs=None, env=None) -> None
```

`run()` returns `None`. The agent reports results by filling the `AgentContext` it is handed, ideally while it runs so a timeout still leaves partial data. Source: `src/harbor/models/agent/context.py`.

```python
class AgentContext(BaseModel):
    n_input_tokens: int | None        # includes cache
    n_cache_tokens: int | None
    n_output_tokens: int | None
    cost_usd: float | None
    model_usage: dict[str, ModelUsage] | None   # per model, incl. subagents; backfilled from ATIF
    rollout_details: list[RolloutDetail] | None # token ids, logprobs, loss masks for RL
    metadata: dict[str, Any] | None
```

`RolloutDetail` holds `prompt_token_ids`, `completion_token_ids`, and `logprobs` per turn, plus `extra`. Several `RolloutDetail` entries can represent non-linear histories such as subagents or summarization. This is the RL hook. Terminus-2 has a `collect_rollout_details` option.

`AgentCapabilities` is a frozen pydantic model. Its fields are `atif`, `resume`, `load_native_trajectory`, `load_atif_trajectory`, `handoff`, `native_config`, `windows`, `skills`, `mcp_servers`, `mcp_http_headers`, and `bridges`.

**The two ways to integrate** (`docs-mintlify/agents/custom-agents.mdx`) [V]:

1. **Installed agent: `BaseInstalledAgent(BaseAgent)`.** This is the preferred route. You implement `install(environment)` (use `exec_as_root` for system packages and `exec_as_agent` for user installs) and `run()`, which usually launches a CLI via `self.exec_as_agent(environment, command=...)`. Decorate `run` with `@with_prompt_template`, which applies a Jinja `prompt_template_path` containing `{{ instruction }}`. Then override `populate_context_post_run` or `convert_trajectory(logs_dir) -> Trajectory` to parse native logs into ATIF and token counts. Shared options come from `InstalledAgentOptions`: `version`, `prompt_template_path`, and `config` (native config path or inline JSON). Declarative `CLI_FLAGS`/`ENV_VARS` map options to flags.
2. **External agent: subclass `BaseAgent` directly.** The agent loop runs in the Harbor host process and acts on the sandbox only through `environment.exec(...)`, plus file upload and download. Terminus-2 and the Meta-Harness TB2 candidates work this way.

**Running a custom agent** [V]. Use `harbor run -t <task> -a module.path:ClassName -m provider/model --ak key=val --ae ENV=val`. In a job config, set `agents[].import_path`. The module must be importable from the Harbor process.

**The result on disk** [V]. Each trial directory contains `agent/` (native logs plus `trajectory.json`), `verifier/` (reward files, test output), `artifacts/`, and `result.json`. `result.json` is a `TrialResult` with `task_name`, `config`, `agent_info`, `agent_result: AgentContext`, `verifier_result.rewards: dict[str, float]`, `exception_info`, timing for each phase (env setup, agent setup, agent execution, verifier), and `step_results` for multi-step tasks. `JobResult` aggregates `cost_usd`.

### A.4 ATIF trajectory format [V]

The Agent Trajectory Interchange Format is specified in `rfcs/0001-trajectory-format.md`. Pydantic models live in `harbor.models.trajectories`. The current version is **ATIF-v1.7**, and Harbor accepts v1.0 through v1.7. The file is `agent/trajectory.json`.

- Root fields: `schema_version`, `session_id`, `trajectory_id`, `agent{name, version, model_name, tool definitions}`, `steps[]`, `final_metrics`, `subagent_trajectories[]` (embedded ATIF, new in 1.7), and `extra`.
- A step has `step_id` (sequential from 1), `source` (`system`/`user`/`agent`), `message`, optional reasoning, `tool_calls[{tool_call_id, function_name, arguments}]`, `observation.results[{source_call_id, content}]`, and per-step `metrics`.
- Validate with `python -m harbor.utils.trajectory_validator path/to/trajectory.json`.
- **Why this matters for HIVE [I].** ATIF is the agent-neutral trace format we want the optimizer to read. Since v1.7 it can embed **subagent trajectories**, so it can represent multi-agent topologies natively.

### A.5 Built-in agents [V]

The `AgentName` enum and `src/harbor/agents/installed/` list these agents: `oracle`, `nop`, `acp` (any agent from the ACP registry), `claude-code`, `codex`, `openhands`, `openhands-sdk`, **`pi`**, **`strands`**, **`langgraph`**, `terminus-2`, `mini-swe-agent`, `swe-agent`, `aider`, `gemini-cli`, `cursor-cli`, `copilot-cli`, `opencode`, `goose`, `cline-cli`, `kimi-cli`, `kimi-code`, `qwen-coder`, `deerflow`, `dspy-rlm`, `computer-1`, `devin`, `hermes`, `junie`, `openclaw`, `trae-agent`, `eve`, `vibe`, `fx`, `mimo`, `mcode`, `muse-code`, `nemo-agent`, `grok-build`, `antigravity-cli`/`-sdk`, `rovodev-cli`, `cortex-code`, and `code-puppy`.

These matter most for HIVE:

- **`pi`** (`installed/pi.py`, about 1100 lines) [V]. It installs `@earendil-works/pi-coding-agent` via npm/nvm and runs `pi --print --mode json --session-dir ... --model ... [--extension ...] <instruction>`. Its capabilities are `atif`, `resume`, `skills`, and `mcp_servers`. It converts Pi's session branch to ATIF v1.7. It writes two extensions of its own: the MCP adapter and a `max-turns.ts` turn limiter. **It has no hook for user-supplied Pi extensions or a native config** (`native_config=False`). Task skills are copied to `~/.agents/skills`. `prompt_template_path` is supported.
  - **Implication [I].** To optimize Pi *extensions*, *tools*, and *system prompts*, HIVE should subclass `harbor.agents.installed.pi.Pi`. The subclass would upload a candidate's extension directory and settings, then append `--extension` flags. Alternatively, it could bake them into a custom install. The base class gives ATIF conversion for free.
- **`strands`** [V]. Options are `factory="module:create_agent"`, `factory_path` (a local file or directory uploaded to the sandbox), `factory_kwargs`, and `model_kwargs`. Your own Strands agent factory gets installed and run, with conversion to ATIF. That is a direct fit for the AWS Strands alternative.
- **`langgraph`** [V]. Options are `project_path` (a local LangGraph project with `langgraph.json`), `graph` (the graph id), `model_kwargs`, and `configurable`. It also supports JS graphs. Deep Agents is a LangGraph graph, and the unit tests reference a `"deepagent"` graph id, so Deep Agents can run through this adapter.
  - LangChain also ships an external `deepagents_harbor:DeepAgentsWrapper`. It is referenced in `examples/configs/environments/langsmith-deepagents-job.yaml` [V] and lives in the `langchain-ai/deepagents` repo under `libs/harbor` [R, per the LangChain blog: https://www.blog.langchain.com/evaluating-deepagents-cli-on-terminal-bench-2-0].
- **`claude-code`, `codex`** [V]. These have the richest capability sets: ATIF, resume, load ATIF/native trajectory, native config (`--ak config=settings.json`), and HTTP MCP headers. `claude-code` also supports handoff.
- **`terminus-2`** [V]. Harbor's reference external agent (`BaseAgent`, tmux-driven). It is the base class for the Meta-Harness TB2 candidates.
- **`dspy-rlm`** [V]. A Recursive-LM agent, which is directly relevant to MGH.

ATIF support covers about 33 agents, including `pi` and `strands`. `langgraph` is **not** in the ATIF capability list in the docs table [V]. Verify this before relying on LangGraph traces. It may emit LangSmith traces instead (it forwards `LANGSMITH_*` env vars).

### A.6 Datasets and registry [V]

`registry.json` lists 80 dataset versions. Run `harbor datasets list` to browse them, and use `-d name@version` to select one. Highlights:

- **Agentic coding / terminal:** `terminal-bench@2.0` (89 tasks), `terminal-bench-sample@2.0` (10), `terminal-bench-pro@1.0` (200), `openthoughts-tblite@2.0` (100), `termigen-environments` (3566), `seta-env` (1376).
- **SWE:** `swebench-verified@1.0` (500), `swebenchpro` (731), `swebench_multilingual` (300), `swesmith` (100), `swe-gen-js` (1000), `swe-lancer-diamond` (463 total; `ic` 198, `manager` 265), `swtbench-verified` (433), `featurebench` (200, lite 30).
- **Code:** `aider-polyglot` (225), `humanevalfix` (164), `livecodebench@6.0` (100), `code-contests` (9644), `bigcodebench-hard-complete` (145), `usaco` (304), `ds-1000`, `quixbugs` (80).
- **Reasoning / QA / other:** `aime` (60), `gpqa-diamond` (198), `arc_agi_2` (167), `gaia` (165), `simpleqa`, `bfcl` (3641), `reasoning-gym-easy`/`-hard` (288 each), `mlgym-bench`, `ml-dev-bench`, `replicationbench`, `spider2-dbt`, `bird-bench`, `medagentbench` (300), `financeagent`, `cooperbench` (652), `gso` (102), `scale-ai/swe-atlas-*`.
- There is also `hello-world` (1 task) for smoke tests.
- `harbor init`, adapters (`harbor adapters`), and `datasets/create-a-dataset.mdx` cover building our own suites.

### A.7 Local vs cloud execution and parallelism [V]

- **Local runtimes:** `docker` (default), `podman`, `apple-container`, `singularity`, and `sbx`.
- **Remote sandboxes:** `daytona`, `modal`, `e2b`, `runloop`, `langsmith`, `blaxel`, `beam`, `novita`, `tensorlake`, `vercel`, `prime`, `gke`, `ec2`, `skypilot`, `islo`, `sail`, `runta`, `cwsandbox`, `hf-sandbox`, `openshift`, `ack`, and more. Install them with `uv tool install "harbor[daytona]"`, or `harbor[cloud]` for all of them.
- **Concurrency:** `-n/--n-concurrent` (default 4), set in the config as `n_concurrent_trials`. You can also limit per agent with `agents[].n_concurrent` and `concurrency_group`. The README example runs TB2 with `-n 100 --env daytona`.
- **Other job knobs:** `n_attempts` (repeat each trial; use it for pass@k and variance), `retry.{max_retries, include/exclude_exceptions, backoff}`, `timeout_multiplier` (and per-phase variants), `metrics`, `artifacts`, `extra_instructions`, and `source_jobs`. The Python API is `Job.create(JobConfig(...))` followed by `await job.run()`.
- **Hosted Harbor / Harbor Hub** offer hosted jobs, leaderboards, and an upload/share flow (`docs-mintlify/hosted-harbor/*`).
- **Results viewer:** `harbor view jobs` serves a local UI showing reward, cost, tokens, timing, trajectory, and files.

### A.8 Metrics and reward aggregation [V]

- The default aggregate is the mean reward over tasks, and **a missing reward counts as 0**. Multi-dimensional rewards are averaged per key, and missing keys also count as 0.
- For a custom aggregate, add `metric.py` to the dataset. It is called with `-i rewards.jsonl -o metric.json`, and it handles pass@k, min, or weighted scores.
- Cost: `AgentContext.cost_usd` and `model_usage` per model roll up into `TrialResult` and `JobResult.cost_usd`. Error classification (`jobs/error-classification.mdx`) separates infra/API failures from agent failures. `BaseInstalledAgent` defines typed errors such as `ApiRateLimitError`, `ContextWindowExceededError`, and `AgentSafetyRefusalError`.

### A.9 Optimization and RL support [V]

- **RL:** `rollout_details` carries token ids and logprobs. The Harbor Cookbook has a **Thinking Machines Tinker RL** recipe that uses Harbor tasks as RL environments, and a **GEPA** recipe that optimizes an agent harness on MedAgentBench (`docs-mintlify/news/harbor-cookbook.mdx`; https://github.com/harbor-framework/harbor-cookbook).
- **`harbor sweeps run`:** repeated sweeps with `--trials-per-task`, `--hint`/`--hints-file`, and export of success/failure splits to a HF DatasetDict. It suits data generation, not harness search [V from CLI source; purpose I].
- **`harbor traces export`:** exports trajectories, including to OTEL.
- **`harbor analyze`:** runs an *agent* over trial directories against a rubric. The default rubric (`src/harbor/analyze/prompts/analyze-rubric.toml`) checks **`reward_hacking`**: edits to `tests/`, writes to `/logs/verifier/reward.*`, or reading `solution/`. It also checks **`task_specification`**: whether the instructions are insufficient for the tests. Use it as an automatic audit of every candidate harness [I].
- **`job_diff.py` / `harbor jobs` diff:** compares jobs. This is useful for an optimizer's "what changed between candidates" CLI [I; not inspected in detail].

### A.10 Cost reference points

- Meta-Harness TB2 search: Opus 4.6, 89 tasks × 2 trials, concurrency 50. About **4–6 h and ~$500 per iteration** [V, `reference_examples/terminal_bench_2/README.md`].
- The Meta-Harness Harbor pilot ran small suites (3–10 tasks) with `gpt-5.4-nano` as the child model and `eval_budget = 4–8` [V].
- **[I]** For a class budget, expect to use small suites (10–30 tasks) and a cheap child model, and to save full TB2 runs for the final comparisons.

---

## B. Meta-Harness (Lee, Nair, Zhang, K. Lee, Khattab, Finn; arXiv 2603.28052, 2026-03-30)

### B.1 Method [V]

- **Objective.** Hold the model *M* fixed. Find the harness *H* (a stateful program that decides what context the model sees at each step) that maximizes E[r(τ, x)]. When there are several objectives, such as accuracy vs context tokens, keep a **Pareto frontier**.
- **Loop (Algorithm 1).**
  1. Seed a population with known harnesses and evaluate each one.
  2. For each of N iterations, a **coding-agent proposer** (Claude Code with Opus 4.6) queries the filesystem *D* and proposes k new harnesses.
  3. Each candidate must pass **interface validation**. It is then evaluated, and its code, scores, and traces are added to *D*.
  4. Return the Pareto frontier. The final test-set evaluation is run only on the frontier.
- **Key design choice: full history in a filesystem.** Each candidate gets a directory holding its source code, its scores, and **raw execution traces** (prompts, tool calls, model outputs, state updates). The proposer navigates with `grep`/`cat`; nothing is packed into one prompt. A single evaluation can produce up to about 10M tokens of diagnostics, roughly 1000× the feedback budget of prior text optimizers. Their table puts MTok/iter at OPRO 0.002, TextGrad 0.015, AlphaEvolve 0.022, GEPA 0.008, TTT-Discover 0.026, and Meta-Harness 10.0.
- **Minimal outer loop by design.** There is **no parent-selection rule**, no archive heuristics, and no mutation operators. The proposer decides what to inspect, which failures to address, and whether to make a local edit or a rewrite. The stated rationale is that the loop "can improve automatically as coding agents become more capable."
- **Search space.** Arbitrary code. In practice each harness is a single-file Python program covering prompting, retrieval, memory, and orchestration. For TB2 that means any subclass of Harbor's `Terminus2`, overriding `_call_llm_with_tools`, `_parse_tool_calls`, `_execute_commands`, `_run_agent_loop`, `_get_completion_confirmation_message`, `_get_prompt_template_path`, `_summarize_context`, and so on. Discovered harnesses run 100–1000 LOC.
- **Proposer steering.** A "minimal domain-specific **skill**" describes where to write candidates, how to inspect history, and what may and may not be modified.
- **Scale.** A typical run is about 60 harnesses over 20 iterations, finishing in "a few hours."
- **The proposer never sees test-set results.**

### B.2 What the proposer sees and why [V]

- In the TB2 run, the proposer read a **median of 82 files per iteration** (range 69–99). The split was about 41% source code, 40% execution traces, 6% score summaries, and 13% other. It referenced 20+ prior candidates per step. So the access pattern is **non-Markovian**: it does not just look at the last parent.
- **Ablation (text classification, Table "history ablation"):**

  | Proposer input | Median acc | Best acc | Runs > zero-shot |
  |---|---|---|---|
  | Scores + code | 34.6 | 41.3 | 26 |
  | Scores + code + LLM summaries | 34.9 | 38.7 | 23 |
  | **Scores + code + raw traces (full)** | **50.0** | **56.7** | **39** |

  "Summaries do not recover the missing signal, and may even hurt by compressing away diagnostically useful details." The median full-interface candidate beats the *best* candidate from either ablation.
- In the reference implementation, the state files the proposer reads are `frontier_val.json` (best agent per task plus overall `_best`), `evolution_summary.jsonl` (each candidate's hypothesis, changes, scores, and `rollout_metrics` such as tokens, turns, and cost), and the raw Harbor `jobs/` and `logs/` directories [V, `meta_harness.py`, SKILL.md].

### B.3 Results [V]

- **Online text classification** (GPT-OSS-120B; LawBench, Symptom2Disease, USPTO-50k): 48.6% vs ACE 40.9 (**+7.7**) and MCE (+8.6), using **11.4K context tokens vs 50.8K** for ACE. On 9 unseen OOD datasets it averages 73.1% vs ACE's 70.2%, best on 6/9.
- **Compute-matched comparison to other optimizers** (same proposer and evaluation budget; median/best): GEPA 32.6/40.2, Best-of-N 34.0/44.2, OpenEvolve 39.1/43.3, TTT-Discover 34.1/45.6, **Meta-Harness 50.0/56.7**. It matches OpenEvolve and TTT-Discover's final accuracy within **4 evaluations**, about 10× fewer.
- **Retrieval-augmented IMO math:** one harness (BM25 over 500K+ decontaminated solved problems) was found on a 250-problem search set with GPT-OSS-20B. It gives **+4.7 pts on average across 5 held-out models** (GPT-5.4-nano/mini, Gemini-3.1-Flash-Lite, Gemini-3-Flash, GPT-OSS-20B) on 200 unseen IMO-level problems, and +1.3 over plain BM25.
- **TerminalBench-2:** **76.4%** on Opus 4.6, vs Terminus-KIRA 74.7 and Claude Code 58.0. It ranks #2; ForgeCode reports 81.8 but the authors could not reproduce it. On Haiku 4.5 it reaches **37.6%**, ranked #1, vs Goose 35.5. The winning change was **environment bootstrapping**: one shell command before the first LLM call collects pwd, `/app` listing, languages, package managers, and memory, and injects them into the prompt. It is about 80 LOC over KIRA, guarded by a 15 s timeout. It gained on 7/89 tasks, mostly ones that need non-obvious tooling.

### B.4 Qualitative lessons from the TB2 search log [V]

- Iterations 1–2 bundled structural bug fixes with a prompt rewrite, and both regressed by 5–7 pp. In iteration 3 the proposer **identified the shared prompt change as the confound** and isolated the structural fixes.
- Iterations 4–6 kept editing the completion and control flow and kept regressing. The lesson it drew was that "prompt and completion-flow edits are high-risk."
- Iteration 7 tried a **purely additive** change (the env snapshot) and got the best result. Iteration 8 composed two orthogonal fixes. Iteration 10 transferred a lesson from an earlier run.
- **Take-away [I].** Encode "one mechanism per candidate," "prefer additive changes," and "attribute regressions before compounding" in the proposer skill. The reference skill already requires "One mechanism per candidate" and "Mechanism-first" [V].

### B.5 Pitfalls and how they handled them

- **Overfitting / leakage [V].** On TB2, search and evaluation used *the same 89 tasks*, framed as a "discovery problem." They mitigated with manual inspection plus **regex audits for task-specific string leakage**. The skill's anti-overfitting rules say: no task names in code, prompts, or comments; no task-specific hints; and the test is "would this advice help a human on MANY unfamiliar tasks?" The text-classification and math results *do* use held-out test sets and models. The paper argues code-space overfitting is "more inspectable" because brittle if-chains and hard-coded mappings are visible.
- **Harbor pilot trust boundary [V]** (`experimental/harbor_meta_harness`):
  - Each task declares `[meta_harness].forbidden_references` (task name, I/O filenames).
  - A universal denylist (`/tests`, `test_outputs`, `verifier`, `/solution`, `task.toml`) is checked against candidate source.
  - Candidate code is **never imported by the outer agent or controller**. A short-lived controller subprocess stages it, and only the child Harbor job imports it.
  - A frozen `EvaluationState` enforces an `eval_budget` and keeps an append-only history.
  - Aggregates are `mean`, `min`, or `fraction_solved`.
- **Follow-up showing that overfitting is real [R].** RRSI (arXiv 2609.24972, co-author Yoonho Lee) reports that unregularized harness evolution "overfit[s] by memorizing the training tasks." It adds a shrinking edit budget, a critic that screens out benchmark-specific edits, and a pruner. The result is up to +4.3 on 5 OOD benchmarks and about 30% fewer policy tokens.
- **Reward hacking [I].** The paper does not discuss reward hacking explicitly. Harbor's default shared verifier container lets an agent touch `/logs/verifier`. Use `environment_mode = "separate"` for our suites and run `harbor analyze` with the reward-hacking rubric.
- **Practical tips (Appendix, [V]).**
  - The **skill text is the strongest lever**. Constrain outputs and safety, not the diagnosis procedure. Debug the skill with 3–5 iteration mini-runs.
  - Pick a search set that is **hard for the baseline**, sized for about 50 full evaluations.
  - Log in navigable JSON with consistent names.
  - Add a small **CLI** for frontier, top-k, and diffs.
  - Run cheap **validation** (import, instantiate, tiny run) before expensive evaluations.
  - Keep **evaluation automated outside the proposer**. The proposer must not run benchmarks itself.
- **Limitations they state [V].** They tested only one strong proposer (Claude Code). They name co-evolving harness and weights as future work, which WHALE (arXiv 2609.00196) does [R].

### B.6 Released code [V]

- `stanford-iris-lab/meta-harness` (MIT) contains:
  - `ONBOARDING.md`, a flow that produces a `domain_spec.md` for a new domain.
  - `reference_examples/text_classification/` and `reference_examples/terminal_bench_2/`. The latter includes `meta_harness.py` (about 960 LOC outer loop), `claude_wrapper.py` (runs `claude --dangerously-skip-permissions --output-format stream-json` and logs proposer sessions), the proposer skill at `.claude/skills/meta-harness-terminal-bench-2/SKILL.md`, and `agents/baseline_kira.py`.
  - **`experimental/harbor_meta_harness/`**, a pilot where the outer loop is *itself a Harbor task* (`tasks/select-harness`). An outer agent edits `/app/harness.py` and calls a bounded `evaluate_harness` tool backed by `controller.py`. Suites cover TB2-easy, HumanEvalFix-lite, and CodePDE.
- The README says it was "cleaned up ... not tested beyond verifying that it runs."
- Community ports: SuperagenticAI/metaharness (Codex), 001TMF/harness-forge (a Claude Code skill), meta-harness-on-islo, and Tencent VideoHarness-RSI.

---

## C. The Mismanaged Geniuses Hypothesis (Alex Zhang, Zhening Li, Omar Khattab; blog, April 2026)

URL: https://alexzhang13.github.io/blog/2026/mgh/ [R; read via web fetch, quotes as returned]

- **Core claim.** "Existing frontier language models are severely underutilized due to sub-optimal use of individual language model calls." The bottleneck is *management*, meaning how a task is decomposed into LM calls, not raw model capability. "You and I are not good managers."
- **On scaffolds.** "Nearly all modern agent scaffolds are human-engineered, task-specific decomposition strategies that use language models," and they "are often brittle with respect to different models and different problems." Orchestrator-subagent coding agents are a first step toward models decomposing tasks themselves: "the model does not need to know the exact solution to a problem to outline how it may go about decomposing it."
- **Decomposition space.** "The first step would be to define the *space of decompositions* the LM has access to." "How we define 'decomposition' has an exponentially large impact (with respect to depth)." Recursive Language Models (code execution plus recursive sub-calls, e.g. a `for` loop over sub-calls) expand that space beyond tool-call subagents.
- **Composition beats OOD.** If each sub-call stays in-distribution and the decomposition step is itself in-distribution, then the composition can solve OOD problems such as very long contexts.
- **Evidence.** On MRCRv2 (1M context, 8 needles), RLM(Qwen3-4B-Instruct) scores about 0%. After RL on a much simpler setting (32K context, 1 needle) it scores **100%**. No algorithm details or ablations are given; it is a single result in a blog post.
- **Prescription.** Train the "operator to compose LMs" with RL and bootstrapping instead of scaling base models. "A key unknown is the right general scaffold to train over."

**Implications for HIVE [I]**

1. Treat the **topology and coordination rules as the thing being optimized**, not fixed scaffolding. Fixed human-designed baselines (hierarchical, peer-to-peer, manager-worker) are exactly the "human managers" MGH predicts will underperform. That makes them the right baselines and supports the project's hypothesis.
2. **Expand the decomposition action space** available to the harness: spawn subagents dynamically, recurse, and loop over sub-calls in code (RLM-style), rather than only choosing among K fixed topologies. One candidate family should be "let the manager decide its own decomposition at runtime," compared against "the optimizer picks a static topology."
3. MGH is about *training* the decomposer. We optimize in *harness space* with a frozen model, so our analogue is to let the optimizer search over decomposition policies (prompts plus code). That is Meta-Harness's lever, applied to topology.
4. Keep each sub-call in-distribution: short, focused contexts per subagent. Measure whether optimized topologies win by **context isolation** (fewer tokens per call) rather than by more total compute, and report tokens and cost alongside reward.

---

## D. Related work: what to borrow

| Work | Idea | What to borrow for HIVE |
|---|---|---|
| **ADAS / Meta Agent Search**, Hu, Lu, Clune 2024, https://arxiv.org/abs/2408.08435 [M] | A meta-agent writes new agents as Python code `forward()` functions, conditioned on an archive of prior agents and scores. | Define agents in code, keep a growing archive, and transfer across domains and models. Meta-Harness is essentially ADAS with full traces and a coding agent as proposer. |
| **AFlow**, Zhang et al. 2024, https://arxiv.org/abs/2410.10762 [M] | MCTS over code-represented workflows built from reusable *operators* (Ensemble, Review/Revise, Programmer...), with experience-driven expansion. | A small **operator library** (debate, critique, vote, delegate) gives the optimizer composable topology primitives and makes search cheaper. Tree-structured search and backtracking are an option if the free-form proposer stalls. |
| **Darwin Gödel Machine**, Zhang, Hu, Lu, Lange, Clune 2025, https://arxiv.org/abs/2505.22954 [M] | A self-modifying coding agent with an **open-ended archive**: it samples parents from the archive, not just the best. SWE-bench went from about 20% to 50%. They documented **objective hacking**, e.g. the agent removed the markers used to detect hallucinated tool use. | Keep stepping-stone candidates, not just the champion. Run independent audits, since the optimizer *will* game weak metrics. Sandbox the self-edits. |
| **GEPA**, Agrawal et al. 2025, https://arxiv.org/abs/2507.19457 [M] | Reflective prompt evolution: an LLM reflects on rollout traces in natural language and keeps a **per-instance Pareto frontier**, so candidates that win on different subsets survive. It beat GRPO with far fewer rollouts. There is a Harbor cookbook recipe [V]. | Per-task Pareto selection (Meta-Harness's `frontier_val.json` also tracks best-per-task). A cheap baseline optimizer to compare against our Claude Code optimizer. |
| **MASS**, Zhou et al. 2025 (ICLR 2026), https://arxiv.org/abs/2502.02533 [V abstract; details M] | Three stages: block-level prompt optimization, then topology optimization over a pruned space, then workflow-level prompt optimization. They find **prompts matter as much as or more than topology**, and only a few topology choices help. | Order the search: optimize per-role prompts first, then topology, then joint prompts. Prune the topology space to the influential building blocks. Compare "topology-only" vs "prompt-only" ablations. |
| **RRSI**, Xia et al. 2026, https://arxiv.org/abs/2609.24972 [R] | Regularized harness evolution: edit budget, a benchmark-specificity critic, and a pruner. | An overfitting-control recipe for our held-out generalization claim. |
| **MAST / "Why do multi-agent LLM systems fail?"**, Cemri et al. 2025, https://arxiv.org/abs/2503.13657 [M] | A failure taxonomy for MAS: specification, inter-agent misalignment, and verification failures. | Use it as the rubric for `harbor analyze`-style trace labeling of multi-agent failures, which gives the optimizer structured diagnostics in addition to raw traces. |

---

## E. Recommendations for the HIVE optimizer loop [I]

1. **Use Harbor as the evaluation substrate, with each harness as a Harbor agent.**
   - The swappable contract is `BaseAgent.run(instruction, environment, context)`.
   - For Pi, subclass `harbor.agents.installed.pi.Pi`, inject candidate extensions, prompts, and skills, and pass the extra `--extension` flags.
   - For Strands, use `-a strands --ak factory_path=... --ak factory=module:create_agent`.
   - For Deep Agents, use `-a langgraph --ak project_path=...` or `deepagents_harbor:DeepAgentsWrapper`.
   - Multi-agent topologies live *inside* one Harbor agent and emit ATIF v1.7 with `subagent_trajectories`. Then every topology, single or multi-agent, gives the same result shape: `result.json` with rewards, `AgentContext` tokens and cost, and `trajectory.json`.
2. **Define HIVE's own `HarnessSpec` above Harbor.** It is the directory the optimizer edits: prompts, tools, extensions, `topology.yaml`, coordination rules, and `config`. A thin adapter for each framework (Pi, Strands, Deep Agents) turns a `HarnessSpec` into a Harbor agent. Only the adapter knows about the framework, which keeps the optimizer framework-agnostic.
3. **Copy the Meta-Harness filesystem layout.**
   - Each candidate is `runs/<id>/{harness/, scores.json, per_task.json, jobs/<harbor job dir>, proposer_session/, hypothesis.md}`.
   - Global `frontier.json` (overall plus per-task) and `evolution_summary.jsonl`.
   - **Expose raw ATIF traces, not summaries.** This is their biggest ablation effect (+15 median points).
   - Add a small CLI: `hive frontier`, `hive top -k`, `hive diff A B` (code diff plus per-task reward diff), and `hive failures <id>`.
4. **Keep the optimizer out of evaluation.** A trusted controller validates candidates (import, schema, leakage denylist, 1-task smoke test), runs Harbor, writes results, and enforces the eval budget. The outer Claude Code session only reads files and writes the new candidate plus `pending_eval.json`.
5. **The proposer skill is the main lever.** Enforce one mechanism per candidate, require a falsifiable hypothesis and prediction, prefer additive changes, and ban task names and task-specific hints. Do short 3–5 iteration runs to debug the skill before long runs.
6. **Splits and anti-overfitting.**
   - Search set: hard-for-baseline, 15–30 tasks, n_attempts ≥ 2 to tame variance.
   - Separate validation set for selection.
   - Untouched held-out test set, plus ideally a held-out *dataset* (e.g. search on TB2 subsets, test on terminal-bench-pro or SWE-bench subsets) and a held-out *model*.
   - Add regex leakage audits, an RRSI-style edit budget or critic, and `harbor analyze` with the reward-hacking rubric on winners.
   - Use `verifier.environment_mode = "separate"` for our own tasks.
7. **Baselines on the same axis.** Compare a single agent, fixed hierarchical, fixed peer-to-peer, and fixed manager-worker, all as `HarnessSpec`s under the same model, budget, and suite. Also compare optimizer variants: scores-only, scores plus summaries, and full traces (replicating the Meta-Harness ablation on multi-agent harnesses would itself be a contribution), plus a GEPA or Best-of-N control at equal evaluation budget.
8. **Report a Pareto frontier**, not only mean reward: reward vs cost and tokens, and reward vs wall-clock. Per MGH, check whether multi-agent wins come from context isolation or just more compute.
9. **Budget.** Develop on `docker` locally with `terminal-bench-sample@2.0` or `hello-world`. Scale on Modal or Daytona with a cheap child model. Full TB2 with Opus costs about $500 per iteration, which is out of scope except for a final run.

**Open questions to resolve**

- Does the `langgraph` adapter emit ATIF? The docs table says no. Inspect `langgraph.py` or run it once.
- What is the cleanest way to load custom Pi extensions? Subclass, or contribute a `extensions` option upstream?
- Should the optimizer itself run as a Harbor task, as in the `experimental/harbor_meta_harness` pilot? That would make the optimizer reproducible and comparable, at the cost of complexity.
