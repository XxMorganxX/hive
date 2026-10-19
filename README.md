# HIVE

Self-improving agent harnesses (Cornell CS 4701).

An agent runs a suite of tasks, and the suite grades it. An **overseer** (Claude Code, headless) then reads three things and proposes an edited harness:
- the current candidate config;
- the evaluation metric;
- the suite description, along with the raw traces.

A trusted controller keeps the edit only if the score says it is better.

- **Start here:** [docs/FOUNDING_PRINCIPLES.md](docs/FOUNDING_PRINCIPLES.md) covers the principles, the framework decision, the plug-in contracts and the loop.
- **Evidence:** [docs/research/](docs/research/) holds the framework evaluations and the related work.
- **Working logs:** [tasks/](tasks/).

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                                        # the whole suite; no network, no model calls
uv run hive run experiments/dryrun-exact-answer.toml    # the full loop with a scripted agent: $0
uv run hive run experiments/dryrun-python-functions.toml
```

Runs land in `experience/<experiment>/<run-id>/`, which is gitignored. Its layout is in the principles doc §6.1.

### Real runs (cost money)

```bash
uv run hive eval experiments/pi-exact-answer.toml       # score the stock Pi seed once
uv run hive run  experiments/pi-exact-answer.toml       # seed → overseer generations → held-out
```

These need:
- **Pi 0.75.3** on `PATH` (`pi --version`), with credentials for the provider in the config. The demo configs use `openai/gpt-5-nano`; log in with `pi` and `/login`, or set `OPENAI_API_KEY`.
- **Claude Code** on `PATH`, logged in. It is the overseer.

Every tunable lives in the experiment file, including the model, trials per task, generations, promotion margin, dollar cap and overseer model. Change the file, not the code.

## Plugging something in

Each role has one contract in [`hive/contracts.py`](hive/contracts.py). A new implementation is one class. Name it in a config, either by a built-in short name from [`hive/registry.py`](hive/registry.py) or by an import path such as `my_package.module:MyClass`. The controller does not change.

| Role | Implement | Selected by | Built-ins |
|---|---|---|---|
| Agent framework | `AgentAdapter` (validate, build → `Agent`, describe_valves) | the candidate's `candidate.toml`: `agent = "..."` | `pi`, `scripted` (test double) |
| Evaluation suite | `Suite` (describe, metric, tasks, grade) | the experiment's `[suite] name` | `exact_answer`, `python_functions` |
| Overseer | `Overseer` (propose from an `OverseerBrief`) | the experiment's `[overseer] kind` | `claude_code`, `null` (noise control) |

**Swapping suites** is a one-line change in the experiment file. For example, `experiments/pi-exact-answer.toml` and `experiments/pi-python-functions.toml` differ only in `[suite] name` and a timeout.

```bash
uv run hive validate harnesses/pi-baseline   # check a candidate against its adapter
```

## Status

The scaffold runs end to end. Deliberately deferred work, each tied to a principle, is listed in [tasks/scaffold-contracts-and-loop.md](tasks/scaffold-contracts-and-loop.md):
- container isolation for trials (Harbor);
- multi-agent topology;
- lower-bound selection;
- the noise-floor run.

Until container isolation lands, trials run on the host, so don't trust headline numbers from them.
