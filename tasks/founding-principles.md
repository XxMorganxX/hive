# Founding principles + framework selection

## Goal

Pick the base agent framework for HIVE and define the standard interface that lets agents and harnesses be plugged in or swapped. Capture both in a founding principles document (`docs/FOUNDING_PRINCIPLES.md`) that the team and the optimizer agent build against.

## Approach

1. Read the project proposal (`CS 4701_ Project Proposal (1).pdf`).
2. Fan out four research subagents in parallel. Each writes full notes to `docs/research/`:
   - `pi-harness.md`: Pi coding harness (primary candidate in the proposal)
   - `langchain-deep-agents.md`: alternative named in the proposal
   - `aws-strands.md`: alternative raised by the team
   - `harbor-and-meta-harness.md`: Harbor eval framework, the Meta-Harness paper, the MGH blog, related work
3. Synthesize the findings into the founding principles doc: framework decision, interface contract, eval and optimizer-loop rules.

## Decisions

- **The doc lives in the repo** (`docs/`), not as an external doc, because it is the codebase's constitution and should version with the code.
- **Research notes are kept** under `docs/research/` so the framework decision can be audited later.
- **Many small commits.** Each research note and each doc section lands in its own commit. `runs/`, `jobs/` and `experience/` are gitignored because raw traces are large.

## Progress log

- 2026-10-19: Read the proposal. Key constraints: Pi primary, Deep Agents possible alternative, Harbor for eval. Research question: can an evaluator-guided optimizer improve the topology and coordination rules of a Pi multi-agent coding system, compared with single-agent Pi and fixed multi-agent baselines? Launched 4 research agents.
- 2026-10-19: Pi research landed (`docs/research/pi-harness.md`).
  - **Package:** now `@earendil-works/pi` (v1.3.0, MIT, TypeScript).
  - **Valves:** declarative files plus about 30 extension hooks.
  - **Headless:** runs via `--mode json` or `--mode rpc`. The session JSONL records prompt sections, tools, and token/cost per message.
  - **Harbor:** has a built-in `pi` agent, but it can't inject a custom harness, so we need a thin `HivePi` subclass.
  - **Multi-agent:** DIY; the reference subagent extension spawns child `pi` processes.
  - **Risks that shaped the principles:** API churn (pin the exact version), extension TypeScript isn't typechecked (gate on `tsc`), the exit code doesn't signal failure (parse `stopReason`), and child-agent traces are dropped by default.
- 2026-10-19: Deep Agents research landed (`docs/research/langchain-deep-agents.md`).
  - **Package:** `deepagents` 0.7.28 (Python, MIT, built on LangGraph).
  - **`examples/better-harness`:** an outer agent edits an inner agent's prompt, tools and middleware, gated by train and holdout evals. LangChain reports Terminal-Bench 2.0 going from 52.8% to 66.5% through harness edits alone (self-reported).
  - **Multi-agent:** manager-worker is built in; peer and arbitrary graphs need LangGraph code.
  - **Harbor:** no native adapter (it goes through the generic `langgraph` agent) and no ATIF export. Pre-1.0 churn.
- 2026-10-19: Harbor / Meta-Harness / MGH research landed (`docs/research/harbor-and-meta-harness.md`).
  - **Harbor:** v0.24.0, now at `harbor-framework/harbor`. The agent interface is `BaseAgent` (`setup`, `run`, which fills an `AgentContext`), loaded by `module:Class`. Traces are ATIF v1.7 and can nest `subagent_trajectories`. Built-in adapters exist for `pi`, `strands` (custom factory) and `langgraph`.
  - **Meta-Harness ablation:** raw traces 50.0 vs scores-plus-summaries 34.9 median accuracy, so the optimizer gets raw traces.
  - **Meta-Harness weakness:** the TB2 result used the same tasks for search and test, and a follow-up (RRSI) shows evolved harnesses overfit. So use separate splits and a separate verifier container.
  - **MGH:** fixed topologies are the "human managers"; add a self-decomposing candidate as a contrast.
- 2026-10-19: **Decision: the swap boundary is Harbor's `BaseAgent`, not a HIVE-invented interface.** The Deep Agents and Harbor researchers converged on it independently. Rejected alternative: a custom protocol that wraps every framework ourselves. It duplicates Harbor's sandboxing and trace capture, and we'd lose the registry datasets.
- 2026-10-19: Strands research landed (`docs/research/aws-strands.md`).
  - **Package:** the SDKs are now consolidated in the `strands-agents/harness-sdk` repo; the Python package is `strands-agents` 1.61.1 (Apache-2.0).
  - **Harbor fit:** Harbor's `strands` adapter accepts a user factory and records ATIF. But a Graph or Swarm has to be wrapped in one `Agent`, and inner agents' turns may be lost.
  - **Topology:** the richest built-in primitives (agents-as-tools, Graph, Swarm, A2A).
  - **Prior art:** a Strands Labs `harness-optimizer` exists, which is a useful comparison baseline.
  - **Downsides:** Python only, Bedrock is the default everywhere, and churn is high.
- 2026-10-19: **Decision: Pi is the primary framework.**
  - **Why:** it's in the proposal, the team knows it, it has the richest valve surface, and Harbor ships a native ATIF adapter for it.
  - **Controller language:** Python, because Harbor and every adapter are Python. Pi harness content stays TypeScript and Markdown.
  - **Secondary framework:** deferred to week 6, after the Pi loop runs end to end. Default lean is Strands; Deep Agents is the alternative. Recorded as an open decision in the doc.
- 2026-10-19: Wrote `docs/FOUNDING_PRINCIPLES.md` v0.1 and a root `README.md`. Interface decisions made while writing:
  - **The model is not in the harness spec.** It lives in the experiment config, so P3 can't be bypassed through a candidate.
  - **`load_report.json` is an adapter obligation.** It guards against Pi silently skipping resources.
  - **A `flips/` dir in the experience store** records per-task changes so credit goes to the right valve.
  - **A neutral `topology.yaml` (v0)** is compiled by adapters. This is the one deliberate new abstraction, justified because topology is the research variable compared across arms and frameworks.
- **Next step:** scaffold the agent contract, two swappable suites, and the improvement loop. Tracked in `tasks/scaffold-contracts-and-loop.md`.

## Open questions

- Application domain and task suite (proposal week 1). Coding / terminal tasks via Harbor are the default assumption.

## Out of scope

- Implementation code. This pass produces the spec only.
