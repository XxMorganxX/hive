# Improve an agent harness (generation $generation)

You are improving an agent harness. An agent built from this harness runs a suite of
tasks, and the suite grades every run. Your job is to propose one change to the harness
that should raise its score on tasks like these, including tasks you have not seen.

The model behind the agent is fixed for this experiment. It is not part of the harness,
and you cannot change it. Improve the harness around it.

## The current candidate

This is `candidate/candidate.toml`, the manifest of the harness you are editing:

```toml
$candidate_toml
```

The rest of the harness lives in the other files under `candidate/`.

## The evaluation metric: $metric_name

$metric_description

## The evaluation suite

$suite_description

## How your proposal will be judged

$selection_rule

## Evidence in this workspace

Everything under `context/` is read-only evidence. Read what you need:

- `context/valves.md`: every part of the harness you may edit, and how. Read this first.
- `context/report.json`: the current candidate's results on the search split, per task and per trial.
- `context/evidence/`: the raw trace and the final working directory of every one of those trials.
- `context/history.json`: earlier generations: what was tried, how it scored, and whether it was kept.
- `context/suite.md`, `context/metric.md`, `context/selection_rule.md`: the sections above, as files.

## What to do

1. Read `context/report.json` and find the tasks that failed or errored.
2. For those tasks, read the raw traces and working directories in `context/evidence/` and
   diagnose the root cause. Look for a pattern shared across tasks, not a one-off.
3. Check `context/history.json` so you do not repeat a change that was already tried and rejected.
4. Make ONE isolated change to files under `candidate/` that fixes that root cause and should
   generalize to unseen tasks of the same kind.

Rules:

- Never hard-code task ids, questions, answers or test inputs into the harness.
- Never edit anything outside `candidate/`, apart from writing `RATIONALE.md` below.
  Nothing else you write is used.
- Keep the `agent` field in `candidate/candidate.toml` exactly as it is, and keep the manifest valid TOML.
- Do not edit `candidate/lineage.json`.

## Finish with RATIONALE.md

Write `RATIONALE.md` in the workspace root (next to `candidate/`, not inside it) with exactly
these three headed parts:

```markdown
## Observed
What the evidence showed, with the failure pattern and how many tasks it affected.

## Changed
The one change you made, and which files it touched.

## Why it should transfer
Why this change should help on unseen tasks of the same kind, not only on these ones.
```
