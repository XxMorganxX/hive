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

