# HIVE: Founding Principles

**Status:** v0.1 draft, 2026-10-19. This is the constitution for the codebase. Change it on purpose, in its own commit, and log why in `tasks/`.

**Sources:** the project proposal and the research notes in [`docs/research/`](research/). Each principle links to the evidence behind it.

---

## 1. What HIVE is

HIVE is a loop that improves agent harnesses while the model stays fixed.

1. A **harness** (prompts, tools, extensions, multi-agent topology) runs a **suite** of tasks.
2. An **evaluator** turns each run into a score and a trace.
3. An **optimizer** agent (Claude Code, headless) reads the scores and the raw traces and proposes edited harnesses.
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
| **Adapter** | The only code that knows a framework's native format. It turns a harness spec into a runnable Harbor agent. |
| **Topology** | The roles in a multi-agent harness, who can talk to whom, and the coordination rules. A first-class part of the harness spec. |
| **Valve** | One editable aspect of a harness that the optimizer may change (a prompt section, a hook, a role, a fan-out limit). |
| **Suite** | A set of tasks for one use case, plus its scoring rule. A suite is split into **search**, **validation** and **held-out** sets. |
| **Task** | One Harbor task: instruction, environment, verifier. |
| **Trial** | One run of one candidate on one task. A task is run **k** times. |
| **Outcome** | The scored result of a task for one candidate: status, reward, cost, tokens, trace pointer. |
| **Experience store** | The append-only directory the optimizer reads: every candidate's spec, scores and raw traces. |
| **Optimizer** | The agent that proposes new candidates (also called the proposer). It reads the experience store and writes proposals only. |
| **Controller** | Trusted code that validates, schedules, scores and selects. It is never edited by the optimizer. |

## 3. Principles

Each principle has a rule and the reason for it. The reasons come from the Meta-Harness paper, the framework and Harbor docs, and basic statistics.

### P1. Measure the noise before you optimize

**Rule:** before any search, run the seed k times on the search set and record the noise floor: the standard deviation of one candidate's score, and the minimum detectable effect (MDE). Do not claim any improvement smaller than the MDE.

**Why:** agent runs are stochastic.
- With one trial per task, the same harness scores differently on identical reruns. On a few dozen tasks the swing is several tasks.
- An optimizer keeps the best of many noisy candidates, so its running maximum rises by chance alone (the winner's curse).
- A search-set gain smaller than the noise floor can't be told apart from luck.

### P2. The evaluator is the product

**Rule:** the controller, the graders and the statistics get the most care in the codebase. Any change to them is a deliberate experiment change and is versioned. A grader is trusted only after it passes three checks: the reference solution scores 1.0, a no-op scores 0.0, and repeated judge samples agree.

**Why:** score bugs are invisible from the scores.
- A misconfigured judge, or an infrastructure failure counted as a task failure, shifts every number the same way.
- The optimizer will then faithfully climb the wrong hill.

### P3. The model is frozen; the harness is the variable

**Rule:** inside one experiment, the model id, its parameters and the provider are fixed and recorded. The optimizer may not change them. Comparing models is a separate experiment with its own seed.

**Why:** otherwise we can't tell a harness gain from a model gain, and the research question is about harnesses.

### P4. A harness is data

**Rule:** a candidate is a directory: a manifest plus framework-native files. Its id is the content hash. The loop never interprets what a valve means. It hashes, builds, runs and reads back outcomes. Only the adapter understands the files.

**Why:** content addressing makes it easy to deduplicate, diff and resume candidates, and one loop can drive any framework. A re-proposed candidate is never paid for twice.

### P5. One swap boundary, and it's Harbor's

**Rule:** every framework enters through a Harbor agent (`BaseAgent` / `BaseInstalledAgent`), loaded by `module:Class`. Framework-specific code stays inside its adapter. Nothing outside an adapter imports a framework.

**Why:** Harbor already provides sandboxing, concurrency, verifiers, the dataset registry and ATIF traces, and it has built-in agents for Pi, Strands and LangGraph. The framework research converged on this boundary independently. Building our own would mean rebuilding Harbor. ([harbor](research/harbor-and-meta-harness.md))

### P6. Traces are raw, complete, and never scored

**Rule:**
- Every trial keeps its full native trace (Pi session JSONL plus the event stream) and an ATIF `trajectory.json`.
- In multi-agent runs, **every child agent's trace is kept** and linked to its parent.
- A missing trace becomes an explicit `.MISSING` marker, never silence.
- Traces feed the optimizer. They never feed the score.

**Why:**
- The Meta-Harness ablation: raw traces scored 50.0 median accuracy, against 34.9 for scores plus summaries. The optimizer needs the evidence, not a précis of it.
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
- Every search also runs a **null-optimizer control**: the same loop with no edits.

**Why:**
- pass@k pushes flaky tasks to the ceiling and hides variance.
- A lower bound penalizes candidates measured on too few trials.
- The null control shows directly how much "improvement" the selection rule produces from pure noise.

### P10. The optimizer reads everything and writes one thing

**Rule:**
- The optimizer runs headless with read-only access to the experience store. It writes only a proposals file.
- Each proposed candidate makes **one isolated change** and contains no task names, ids or answers.
- Each one carries a rationale: "observed …; changed …; expected to transfer because …".
- The controller validates (schema, typecheck, a real load check), runs a smoke test, then scores.
- The optimizer never touches the controller, the graders or the suites.

**Why:**
- With one change per candidate, credit goes to the right valve.
- The proposer is the part most likely to reward-hack, so it gets the narrowest write access.
- The Meta-Harness paper uses this shape: the proposer reads a filesystem of prior candidates and writes new ones, and evaluation runs outside it. ([harbor](research/harbor-and-meta-harness.md))

### P11. The search space is written down, and topology is in it

**Rule:**
- Each adapter publishes a `describe_schema()` text that lists every valve the optimizer may edit. That text goes into the proposer prompt verbatim, and changes to it are versioned like code.
- Topology valves (roles, delegation edges, fan-out, depth, coordination rules) are listed first.
- The controller enforces **valve diversity** across generations, so the optimizer can't keep circling one idea.

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

**Why:** near the ceiling or the floor, a real improvement can't move the score enough to clear the noise floor. The experiment would be uninformative however good the optimizer is.

