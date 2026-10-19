# HIVE: Founding Principles

**Status:** v0.1 draft, 2026-10-19. This is the constitution for the codebase. Change it on purpose, in its own commit, and log why in `tasks/`.

**Sources:** the project proposal and the research notes in [`docs/research/`](research/). Each principle links to the evidence behind it.

---

## 1. What HIVE is

HIVE is a loop that improves agent harnesses while the model stays fixed.

1. A **harness** (prompts, tools, extensions, multi-agent topology) runs a **suite** of tasks.
2. An **evaluator** turns each run into a score and a trace.
3. An **overseer** agent (Claude Code, headless) reads the scores and the raw traces and proposes edited harnesses.
4. A trusted **controller** checks each proposal, scores it, and keeps it only if it is better beyond noise.

**Research question (from the proposal):** can an evaluator-guided optimizer improve the topology and coordination rules of a Pi-based multi-agent coding system, compared with single-agent Pi and fixed multi-agent baselines (hierarchical, peer-to-peer, manager-worker)?

The bet behind it is the Mismanaged Geniuses Hypothesis. Models are already capable enough, and the bottleneck is how we decompose work and feed them context. The harness is the lever. ([MGH](https://alexzhang13.github.io/blog/2026/mgh/), [Meta-Harness](https://arxiv.org/abs/2603.28052))

## 2. Vocabulary

Use these words in code and docs with exactly these meanings.

| Term | Meaning |
|---|---|
| **Framework** | The agent runtime a harness is built on: Pi, Deep Agents, Strands. |
| **Harness** | Everything around the model that shapes behavior: prompts, tools, extensions/middleware, context management, topology. Not the model. |
| **Harness spec** | A directory that fully describes one harness. It is content-hashed, and that hash is the **candidate id**. |
| **Candidate** | One harness spec under evaluation, plus its lineage (parents, origin, rationale). |
| **Seed** | The starting candidate for a search. Usually the stock single-agent baseline. |
| **Adapter** | The only code that knows a framework's native format. It validates a candidate and builds it into a runnable agent behind the `Agent` contract. |
| **Topology** | The roles in a multi-agent harness, who can talk to whom, and the coordination rules. A first-class part of the harness spec. |
| **Valve** | One editable aspect of a harness that the overseer may change (a prompt section, a hook, a role, a fan-out limit). |
| **Suite** | A set of tasks for one use case, plus its scoring rule. A suite is split into **search**, **validation** and **held-out** sets. |
| **Task** | One unit of evaluation: an instruction plus optional workspace files. The suite that owns the task holds its grader, and the agent never sees answers or hidden tests. |
| **Trial** | One run of one candidate on one task. A task is run **k** times. |
| **Outcome** | The scored result of a task for one candidate: status, reward, cost, tokens, trace pointer. |
| **Experience store** | The append-only directory the overseer reads: every candidate's spec, scores and raw traces. |
| **Overseer** | The agent that proposes improved candidates: Claude Code, run headless. Also called the optimizer or proposer. It gets the current candidate config, the evaluation metric and the suite description, and edits an isolated copy of the candidate. |
| **Controller** | Trusted code that validates, schedules, scores and selects. It is never edited by the overseer. |

## 3. Principles

Each principle has a rule and the reason for it. The reasons come from the Meta-Harness paper, the framework and Harbor docs, and basic statistics.

### P1. Measure the noise before you optimize

**Rule:** before any search, run the seed k times on the search set and record the noise floor: the standard deviation of one candidate's score, and the minimum detectable effect (MDE). Do not claim any improvement smaller than the MDE.

**Why:** agent runs are stochastic.
- With one trial per task, the same harness scores differently on identical reruns. On a few dozen tasks the swing is several tasks.
- A loop that keeps the best of many noisy candidates sees its running maximum rise by chance alone (the winner's curse).
- A search-set gain smaller than the noise floor can't be told apart from luck.

### P2. The evaluator is the product

**Rule:** the controller, the graders and the statistics get the most care in the codebase. Any change to them is a deliberate experiment change and is versioned. A grader is trusted only after it passes three checks: the reference solution scores 1.0, a no-op scores 0.0, and repeated judge samples agree.

**Why:** score bugs are invisible from the scores.
- A misconfigured judge, or an infrastructure failure counted as a task failure, shifts every number the same way.
- The overseer will then faithfully climb the wrong hill.

### P3. The model is frozen; the harness is the variable

**Rule:** inside one experiment, the model id, its parameters and the provider are fixed and recorded. The overseer may not change them. Comparing models is a separate experiment with its own seed.

**Why:** otherwise we can't tell a harness gain from a model gain, and the research question is about harnesses.

### P4. A harness is data

**Rule:** a candidate is a directory: a manifest plus framework-native files. Its id is the content hash. The loop never interprets what a valve means. It hashes, builds, runs and reads back outcomes. Only the adapter understands the files.

**Why:** content addressing makes it easy to deduplicate, diff and resume candidates, and one loop can drive any framework. A re-proposed candidate is never paid for twice.

### P5. Three swap boundaries, and HIVE owns them

**Rule:**
- Every role that can be swapped has one small contract in `hive/contracts.py`:
  - **`AgentAdapter` / `Agent`:** frameworks (Pi, Strands, Deep Agents, …).
  - **`Suite`:** evaluation use cases.
  - **`Overseer`:** the improvement strategy (Claude Code, a null control, …).
- Implementations are chosen by name in the experiment config through one registry.
- Framework-specific code stays inside its adapter, and nothing outside an adapter imports a framework.
- Harbor plugs in **behind** these contracts: as a `Suite` that runs Harbor datasets in Docker, and as an execution backend for agents.

**Why:**
- Evaluation tasks must be abstract and universal. A 10-line exact-answer check and a Terminal-Bench container task should both satisfy the same suite contract.
- Harbor's `BaseAgent` is execution-shaped. It would tie every suite to containers and every task to Harbor's directory layout.
- Keeping Harbor as an implementation keeps everything it offers (sandboxing, the dataset registry, ATIF traces, built-in Pi/Strands/LangGraph agents) without making it the definition of a task. ([harbor](research/harbor-and-meta-harness.md))

### P6. Traces are raw, complete, and never scored

**Rule:**
- Every trial keeps its full native trace (Pi session JSONL plus the event stream) and an ATIF `trajectory.json`.
- In multi-agent runs, **every child agent's trace is kept** and linked to its parent.
- A missing trace becomes an explicit `.MISSING` marker, never silence.
- Traces feed the overseer. They never feed the score.

**Why:**
- The Meta-Harness ablation: raw traces scored 50.0 median accuracy, against 34.9 for scores plus summaries. The overseer needs the evidence, not a précis of it.
- Pi's reference subagent extension drops child traces by default (`--no-session`). For topology research, that's the data we care about most. ([harbor](research/harbor-and-meta-harness.md), [pi-harness](research/pi-harness.md))

### P7. Splits are walls

**Rule:**
- Every suite has disjoint **search**, **validation** and **held-out** sets. The controller asserts the sets don't overlap before spending anything.
- Held-out results never enter the experience store.
- The held-out set is scored once per experiment, at the end.
- Verifiers run in a separate container (`environment_mode="separate"`).

**Why:**
- The Meta-Harness paper's Terminal-Bench-2 result used the same tasks for search and test.
- A follow-up paper (RRSI) shows that evolved harnesses overfit.
- A verifier that shares the agent's container can be tampered with. ([harbor](research/harbor-and-meta-harness.md))

### P8. Errors are not failures

**Rule:**
- Every outcome has one status: `passed`, `failed`, `errored`, `noop` or `incomplete`. An unmeasured outcome also has an error class (`INFRA_TIMEOUT`, `INFRA_QUOTA`, `INFRA_OOM`, `INFRA_MISSING`, `MEASUREMENT_LOST`).
- The reported pass rate leaves errors out of the denominator.
- **Selection** still uses all requested tasks as the denominator, so a candidate can't improve its rank by pushing hard tasks into "errored".
- Never trust an exit code. Count completed trials. Pi's JSON mode doesn't signal failure through its exit code, so parse `stopReason`.

**Why:**
- Quotas, timeouts and out-of-memory kills belong to the infrastructure, not the harness. Counting them as failures punishes a candidate for bad luck.
- Leaving them out of the selection denominator would reward a candidate for failing loudly. ([pi-harness](research/pi-harness.md))

### P9. Selection is conservative

**Rule:**
- Score each task as its **mean reward over k trials**, never pass@k.
- Rank candidates by the 95% Wilson lower bound over tasks × k.
- A child is promoted only if it beats the incumbent by more than the promotion margin, and never if it scores below the seed.
- Accuracy comes first. Cost only breaks ties among statistically tied candidates.
- Every search also runs a **null-overseer control**: the same loop with no edits.

**Why:**
- pass@k pushes flaky tasks to the ceiling and hides variance.
- A lower bound penalizes candidates measured on too few trials.
- The null control shows directly how much "improvement" the selection rule produces from pure noise.

### P10. The overseer reads everything and writes one thing

**Rule:**
- The overseer runs headless in a scratch workspace. It reads a copy of the evidence and edits an isolated copy of the candidate. The controller harvests only that copy and a rationale.
- Each proposed candidate makes **one isolated change** and contains no task names, ids or answers.
- Each one carries a rationale: "observed …; changed …; expected to transfer because …".
- The controller validates (schema, typecheck, a real load check), runs a smoke test, then scores.
- The overseer never touches the controller, the graders or the suites.

**Why:**
- With one change per candidate, credit goes to the right valve.
- The proposer is the part most likely to reward-hack, so it gets the narrowest write access.
- The Meta-Harness paper uses this shape: the proposer reads a filesystem of prior candidates and writes new ones, and evaluation runs outside it. ([harbor](research/harbor-and-meta-harness.md))

### P11. The search space is written down, and topology is in it

**Rule:**
- Each adapter publishes a `describe_schema()` text that lists every valve the overseer may edit. That text goes into the proposer prompt verbatim, and changes to it are versioned like code.
- Topology valves (roles, delegation edges, fan-out, depth, coordination rules) are listed first.
- The controller enforces **valve diversity** across generations, so the overseer can't keep circling one idea.

**Why:**
- A coding-agent proposer explores what it's told exists. Left alone, it tends to keep refining the first idea that worked.
- The Meta-Harness paper's lessons (one mechanism per candidate; additive changes beat rewrites) assume the proposer knows the menu.
- Topology, HIVE's research axis, is the least obvious valve, so it's listed first. ([harbor](research/harbor-and-meta-harness.md))

### P12. Budget is a first-class constraint

**Rule:**
- Every run has a cost estimate before it starts and a hard dollar cap.
- Every result reports reward **and** cost (dollars, tokens, wall-clock), and the frontier is plotted as reward against cost.
- Multi-agent runs also report tokens per agent call, so we can tell gains from context isolation apart from gains from simply spending more.

**Why:**
- Agent evaluation is expensive. The Meta-Harness paper reports about $500 per iteration on full Terminal-Bench 2.
- Multi-agent topologies multiply tokens, so a topology can look better just by spending more. ([harbor](research/harbor-and-meta-harness.md))

### P13. Every run can be reproduced

**Rule:**
- Pin exact versions of the framework (Pi, Deep Agents), Harbor and the model id.
- Run Pi with `--offline` so its model catalog (and with it the cost numbers) can't change between runs.
- Every run writes a `run_meta.json` with versions, the candidate id, the suite split, k, seeds and cost, written even when the run crashes.
- Resuming a run reloads paid evaluations by candidate id.

**Why:** Pi shipped about 50 breaking-change sections in 10 months, and Deep Agents breaks on minor releases. Without pins, a candidate's score changes underneath it. ([pi-harness](research/pi-harness.md), [deep agents](research/langchain-deep-agents.md))

### P14. Pick suites with headroom

**Rule:** choose the suite and model pair so the seed scores roughly 30–70%. Avoid suites where the seed is near 0% or near 100%.

**Why:** near the ceiling or the floor, a real improvement can't move the score enough to clear the noise floor. The experiment would be uninformative however good the overseer is.

## 4. Framework decision

**Pi is the primary framework.** A second framework comes in after the Pi loop runs end to end. Its only job is to prove the interface is portable.

| | **Pi** (`@earendil-works/pi` 1.3.0) | **Deep Agents** (`deepagents` 0.7.28) | **Strands** (`strands-agents` 1.61.1) |
|---|---|---|---|
| Language | TypeScript | Python (JS port exists) | Python (TS SDK exists) |
| License | MIT | MIT | Apache-2.0 |
| Valve surface | Widest: SYSTEM/AGENTS.md, skills, settings, about 30 extension hooks (context transforms, tool gating, loop continuation, compaction) | Prompt, model profile (YAML), subagent defs, skills, middleware | Prompt, tools, skills, hooks, middleware, conversation managers |
| Valves as files vs. code | Files plus TypeScript that loads without a build step | Mostly declarative, middleware is code | Mostly Python code |
| Multi-agent | DIY: a reference subagent extension spawns child `pi` processes | Manager-worker built in (`task` tool); anything else is LangGraph code | Richest: agents-as-tools, Graph, Swarm, A2A |
| Harbor support | Built-in `pi` agent with ATIF; needs a thin subclass to inject our harness | Through the generic `langgraph` agent; no ATIF | Built-in `strands` agent with a custom factory and ATIF; inner Graph/Swarm turns may be lost |
| Packaged optimizer | None | `examples/better-harness` | Strands Labs `harness-optimizer` |
| Churn | High (about 50 breaking-change sections in 10 months) | High (pre-1.0) | High (weekly minor releases) |

**Why Pi:**
1. The proposal commits to it, and the team knows it.
2. It has the widest valve surface, and most valves are files the overseer can edit without a build.
3. Harbor's Pi adapter emits ATIF natively, so traces come for free.
4. It runs headless (`--mode json`, `--mode rpc`) and records per-message tokens and cost in its session files.

**Pi's costs, and how we handle them:**
- **Multi-agent is DIY.** That's acceptable, because topology is the research variable and we want to own it. We start from the reference subagent extension and make it keep child traces.
- **Churn:** pin one exact version for the semester.
- **Untyped extension edits:** run `tsc --noEmit` and a real load check before every scored run.

**Second framework: deferred to week 6.** Default lean is **Strands**, because of its factory hook, its ATIF output and its topology primitives. **Deep Agents** is the alternative if we want to stay with the proposal or reuse `better-harness` as an optimizer baseline. Either way, it plugs in as one adapter, and the controller doesn't change. If it does need to change, the interface is wrong, and we fix the interface.

**The controller and adapters are written in Python,** because Harbor is Python. Pi harness content stays TypeScript and Markdown inside the harness spec.

Full comparisons: [pi-harness](research/pi-harness.md), [langchain-deep-agents](research/langchain-deep-agents.md), [aws-strands](research/aws-strands.md).

## 5. The standard interface

Plugging in a new framework, use case or improvement strategy means writing **one class** and registering it by name. Nothing else changes. `hive/contracts.py` is the source of truth; this section explains it.

```
experiments/<name>.toml ──► registry ──► AgentAdapter      Suite        Overseer
                                             │               │              │
controller (trusted)                         ▼               ▼              │
  seed candidate ──► evaluate(candidate × suite split × k trials) ──► EvalReport
        ▲                                                                 │
        │   OverseerBrief: candidate config, metric, suite description,   ▼
        └── child ◄── valve catalog, selection rule, report, traces ── decide()
```

### 5.1 Task: the universal unit of evaluation

```python
@dataclass(frozen=True)
class Task:
    id: str
    split: Split                          # "search" | "validation" | "heldout"
    instruction: str                      # everything the agent is told
    workspace_files: Mapping[str, str]    # relative path -> text, written into the trial workdir first
```

- **Grading data never appears on a `Task`.** The suite keeps reference answers and hidden tests to itself, so neither an agent nor the overseer can read them.
- **One shape covers every task style.** A Q&A task is an instruction alone. A coding task adds stub files. A Harbor task maps its `instruction.md` and environment onto the same shape.

### 5.2 Agent contract (framework seam)

```python
class AgentAdapter(Protocol):
    name: str
    def validate(self, candidate: Candidate) -> list[str]: ...       # problems; [] = valid; never raises
    def build(self, candidate: Candidate, model: ModelSpec,
              settings: Mapping[str, Any]) -> Agent: ...
    def describe_valves(self) -> str: ...                            # every editable valve, for the overseer

class Agent(Protocol):
    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult: ...

@dataclass(frozen=True)
class AgentResult:
    status: Literal["completed", "errored"]   # finished its run; says nothing about correctness
    final_text: str
    usage: Usage                               # tokens + cost_usd
    wall_clock_s: float
    error: str | None = None
    error_class: str | None = None             # agent_error | timeout | provider_error | quota | harness_load_error
```

- **`run` never raises for agent-side failures.** It returns `status="errored"`, and it writes native traces into `trace_dir`.

**Pi adapter obligations:**
- **Isolate from the user's global Pi config.** Pass `--no-extensions --no-skills --no-prompt-templates --no-context-files`, then pass the candidate's own files explicitly (`--system-prompt`, `--append-system-prompt`, `-e`, `--skill`).
- **Freeze the model catalog** with `--offline`.
- **Check the pinned Pi version.**
- **Ignore the exit code.** JSON mode exits 0 on provider errors. Status comes from the last assistant `message_end.stopReason`, and usage is summed over every assistant `message_end`.
- **Catch failed extension loads.** An extension error on stderr makes the trial `errored`, with class `harness_load_error`.

### 5.3 Suite contract (use-case seam)

```python
class Suite(Protocol):
    name: str
    def describe(self) -> str: ...                    # markdown for the overseer; never answers
    def metric(self) -> MetricSpec: ...               # name, how a task's reward in [0, 1] is computed
    def tasks(self, split: Split) -> list[Task]: ...  # splits are disjoint
    def grade(self, task: Task, result: AgentResult, workdir: Path) -> Grade: ...

@dataclass(frozen=True)
class Grade:
    reward: float     # in [0, 1]
    detail: str       # what the agent produced and whether it passed; never the expected value
```

- **Each use case is one suite.** Its metric is defined in `grade` and `metric()`, and nowhere else.
- **Swapping suites is a one-line config change.**

### 5.4 Overseer contract (improvement seam)

```python
@dataclass(frozen=True)
class OverseerBrief:
    incumbent: Candidate          # the current best candidate: its config is what gets edited
    metric: MetricSpec            # what "better" means
    suite_description: str        # what the tasks look like and how they are graded
    valves: str                   # the agent adapter's describe_valves()
    selection_rule: str           # the controller's actual promotion rule, in words
    generation: int
    incumbent_report: Path        # search-split EvalReport (JSON)
    evidence_dir: Path            # the incumbent's search-split traces and workdirs
    history: Sequence[Mapping]    # earlier generations: ids, scores, decisions, rationales

class Overseer(Protocol):
    name: str
    proposes_changes: bool        # False only for the null control
    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal: ...
```

The **Claude Code overseer** runs `claude -p` with its working directory set to a scratch workspace:

```
workspace/
  candidate/        # editable copy of the incumbent: the ONLY thing harvested
  context/          # suite.md, metric.md, valves.md, selection_rule.md, history.json,
                    # report.json, evidence/ (search-split traces + workdirs)
  RATIONALE.md      # written by the overseer: observed / changed / why it should transfer
```

- **Tool limits:** Read, Grep, Glob, Edit and Write only (no Bash).
- **No user settings,** so no user hooks fire.
- **A dollar cap per proposal.**
- **Held-out results are never copied into the workspace.**

The **null overseer** returns an unchanged copy, so the loop measures pure noise (P9).

### 5.5 Candidate: what the overseer edits

```
<candidate>/
  candidate.toml    # schema_version, agent (adapter name), description, [valves] (adapter-defined knobs)
  ...               # adapter-owned files. Pi: SYSTEM.md, APPEND_SYSTEM.md, extensions/*.ts, skills/<name>/SKILL.md
  lineage.json      # parents, origin, rationale: excluded from the content hash
```

- **The candidate id is a content hash** of every file except `lineage.json`, so identical edits deduplicate (P4).
- **The model is not in the candidate (P3).**
- **Planned: `topology.yaml`,** compiled by each adapter into its framework's multi-agent primitives. Schema v0:

```yaml
kind: manager-worker        # single | hierarchical | manager-worker | peer
entry: manager
edges:
  - { from: manager, to: coder,    mode: delegate }   # call and await result
  - { from: coder,   to: reviewer, mode: message }    # peer channel
limits: { max_depth: 2, max_fanout: 3, max_agents_total: 8 }
coordination: { handoff: summary, parallel_children: true, termination: entry_decides }
```

### 5.6 Results

- **`TrialOutcome`:** one agent run on one task. It is either `scored` (it has a reward) or `errored` (it has an `error_class`), and it carries usage, wall-clock time and a trace pointer.
- **`TaskOutcome`:** a task's k trials.
- **`EvalReport`:** one candidate on one split. It reports:
  - `measured_mean`: the mean reward over scored trials, for reporting;
  - `selection_score`: the mean over all tasks × k, with errored trials counted as 0, so a candidate can't hide hard tasks behind errors (P8);
  - `stderr`, cost and tokens.

## 6. The optimizer loop

```
measure noise floor (seed × k on search)                         ← P1, once per experiment
for generation in 1..G:
    optimizer reads experience store → writes N proposals         ← P10
    controller: validate → smoke (1 task) → score on search × k   ← P2, P8
    controller: classify → rank (Wilson LB) → promote if margin   ← P9
    controller: append everything to experience store             ← P6
    stop on: dollar cap | ≥50% quota/infra errors | seed unmeasurable
score final best + seed on validation (choose) and held-out (report once)   ← P7
```

### 6.1 Experience store layout

Modeled on the Meta-Harness paper's proposer filesystem: every prior candidate's code, scores and raw traces, which the optimizer reads selectively with grep and cat.

```
experience/<experiment>/
  noise_floor.json                       # seed SD, MDE, k used
  gen-NNN/
    candidates/<id>/                     # full harness spec + lineage.json
    scores/<id>.json                     # per-task Outcomes + aggregates
    traces/<id>/<task>/trial-<k>/        # trajectory.json, native session, load_report
    flips/<id>.json                      # per-task change vs. parent and vs. seed
  frontier.json
  evolution.jsonl                        # one row per generation
```

- **`flips/`** records, per task, what changed versus the parent and versus the seed. Aggregate pass rates hide which valve did what.
- Held-out results go to `runs/<experiment>/heldout/`. They never go here.

### 6.2 The optimizer contract

The optimizer runs headless: `claude -p` with Read, Grep, Glob and Write, scoped to the experience store, and a proposals file as its only output.

Its prompt contains:
- the adapter's `describe_schema()`;
- the experience store layout;
- **the real selection rule**;
- the fact that a hidden held-out set exists;
- the noise floor;
- the valves explored so far, so it can be pushed toward unexplored ones (P11).

### 6.3 Arms every experiment runs

| Arm | What it is | Why |
|---|---|---|
| **Single-agent seed** | Stock Pi plus our adapter | The baseline the proposal names |
| **Fixed topologies** | Hand-written hierarchical, peer-to-peer and manager-worker specs | The "human managers" MGH predicts the overseer should beat |
| **Self-decomposing** | One agent that can spawn sub-agents at runtime with no fixed roles | MGH's "let the model manage itself" contrast |
| **HIVE overseer** | The loop above, starting from the seed | The thing under test |
| **Null overseer** | The same loop, but proposals are unchanged copies | Shows how much "improvement" pure noise produces (P9) |

Other optimizers worth comparing against at the same evaluation budget, if time allows: scores-only (no traces), GEPA, and the `better-harness` or Strands `harness-optimizer` approaches.

## 7. Metrics

The proposal compares arms on task performance, cost, latency and consistency. Every suite reports all four, and multi-agent runs add a coordination block.

| Metric | Definition | Role |
|---|---|---|
| **Reward** | The suite's `classify` reward, as the mean over k trials and then the mean over tasks | **Primary**: the only thing selection optimizes |
| **Selection score** | 95% Wilson lower bound on reward, over all requested tasks × k | Ranking and promotion |
| **Cost** | USD per task, and in total (eval plus overseer) | Tie-breaker; Pareto axis |
| **Latency** | Wall-clock time per trial | Reported |
| **Consistency** | Per-task standard deviation of reward across k trials; the share of tasks that flip between trials | Reported; drives the choice of k |
| **Coordination** (multi-agent only) | Agents spawned, delegation depth, tokens per agent call, share of tokens spent in coordination vs. work | Separates "better organization" from "more compute" (P12) |

**Reward composition is per suite.**
- Tasks with a binary verifier use the verifier's reward directly.
- Suites that need graded quality may add LLM judges. A judge that can't score abstains, and that's recorded as null, never as 0.
- Every judge must pass the P2 checks before its scores count.

## 8. Repository layout

Organized by role, so the trust boundary is visible in the tree.

```
hive/
  controller/        # TRUSTED: loop, validation gate, selection, stats, experience writer
  optimizer/         # optimizer prompt template + headless invocation
adapters/
  pi/                # Harness impl, HivePi Harbor agent, topology → Pi extension compiler
suites/
  <use-case>/        # Suite impl, Harbor dataset ref, splits, judges
harnesses/           # hand-written seeds: single, hierarchical, peer, manager-worker, self-decomposing
experiments/
  <name>.toml        # ONE file per experiment: model, suite, splits, k, G, N, margin, $ cap
experience/          # gitignored: the optimizer's read-only world
runs/  jobs/         # gitignored: raw Harbor output
docs/  tasks/
```

- **Single source of truth.** Every tunable (model id, k, generations, promotion margin, dollar cap, τ) lives in `experiments/<name>.toml`. Every prompt the controller sends lives in `hive/optimizer/`. Nothing is inlined in loop code.
- **Write fences.** The optimizer may write only inside proposed candidates. `hive/controller/` and `suites/` are off-limits to it, and the controller enforces this.

## 9. Known pitfalls (checklist)

Each item is documented behavior of the tools or a standard failure mode of evaluation loops. Each one should be guarded by a controller check or a test.

- [ ] Exit codes don't mean success. **Count completed trials.** Pi `--mode json` doesn't signal failure through its exit code, so **parse `stopReason`**.
- [ ] Pi silently skips project `.pi/` resources in headless mode. **Install into `PI_CODING_AGENT_DIR` and check `load_report.json`.**
- [ ] Pi's reference subagent extension runs children with `--no-session`. **Turn on child sessions and link them in ATIF.**
- [ ] Pi refreshes its model catalog at runtime, which can change cost numbers. **Run with `--offline`.**
- [ ] Extension TypeScript isn't typechecked when it loads. **Run `tsc --noEmit` and a real load check in `validate`.**
- [ ] Harbor forces an old Pi (0.85.1) when MCP servers are configured. **Avoid MCP in harness specs, or override the pin.**
- [ ] Usage caps and quotas look like task failures. **Classify them as `INFRA_QUOTA`, and stop the generation if ≥50% hit it.**
- [ ] A broken judge (bad key, model outage) quietly collapses rewards. **Run a judge health check before each generation; an abstaining judge is null, never 0.**
- [ ] A task that never ran can look like a cheap failure. **Flag trials with zero tokens or zero cost.**
- [ ] A coding-agent overseer fixates on one idea. **Track explored valves and push toward unexplored ones.**
- [ ] Best-of-population rises with noise alone. **Use the null-overseer arm and promotion margins.**

## 10. Budget model

```
attempts ≈ (1 + (G − 1) · N) · tasks · k        # overseer arm
         + arms_fixed · tasks · k               # seed, fixed topologies, self-decomposing, null
cost     ≈ attempts · $/attempt · topology_multiplier
```

- **Worked example.** Use a 20-task search set, k = 5, G = 6 generations, N = 3 proposals each. The overseer arm alone is 16 candidates × 100 = 1,600 attempts.
- **At frontier-model prices this adds up fast.** The Meta-Harness paper reports about $500 per iteration on full Terminal-Bench 2. A class budget needs a cheap base model, low reasoning effort, and small slices.
- **Multi-agent topologies multiply cost** by the number of agents involved.
- **Measure before committing.** Calibrate `$/attempt` from the seed's noise-floor run before committing to G and N.

## 11. Open decisions

These are for the team. Each has a default, so work can go ahead until someone overrides it.

| Decision | Default until decided | Why it matters |
|---|---|---|
| **Use case and suite** (proposal week 1) | Coding: a 30–40 task slice of Terminal-Bench 2.0, split by difficulty into search, validation and held-out | P14 headroom; the Harbor registry has it ready |
| **Base model** | The cheapest current Claude or OpenAI model that puts the single-agent seed at 30–70% on the slice | P3, P14, budget |
| **Total budget** | Set a dollar cap per experiment and per semester before the first run | Sets G, N, k |
| **Second framework** | Strands (default lean) or Deep Agents (matches the proposal); decide in week 6 | Portability proof only |
| **Reward for coding suites** | The verifier's binary reward first; add LLM judges only if the reward is too coarse | Judges add cost and variance |

## 12. References

- Project proposal: `CS 4701_ Project Proposal (1).pdf`
- Meta-Harness paper: <https://arxiv.org/abs/2603.28052>, code at stanford-iris-lab/meta-harness
- Mismanaged Geniuses Hypothesis: <https://alexzhang13.github.io/blog/2026/mgh/>
- Harbor: <https://github.com/harbor-framework/harbor>
- Pi: <https://github.com/earendil-works/pi>, <https://pi.dev/docs/latest>
- Research notes: [harbor-and-meta-harness](research/harbor-and-meta-harness.md) · [pi-harness](research/pi-harness.md) · [langchain-deep-agents](research/langchain-deep-agents.md) · [aws-strands](research/aws-strands.md)
