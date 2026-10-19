# Scripted agent valves

This agent is a **test double**, not a real agent. It calls no model and does no
reasoning: it replays canned text and files. It exists so the loop can run end to
end with zero model spend.

`candidate.toml` must set `agent = "scripted"` and must not define any `[valves]`.
It runs no extensions: a replay has no agent loop to hook, so files in `extensions/`
are ignored.

## The only valve: `responses.toml`

All three keys are optional.

```toml
default = "ANSWER: unknown"      # final text for any task with no entry below ("" if omitted)

[responses]                      # task id -> final text the agent returns
"task-a" = "ANSWER: 4"

[files."task-b"]                 # task id -> {relative path -> file content}
"solution.py" = "def f(): return 1"
```

- Files are written into the task's working directory before the agent returns,
  overwriting any file already at that path.
- File paths must be relative and stay inside the working directory: no absolute
  paths and no `..`.
- Values must be strings. Unknown top-level keys are rejected, so a typo cannot
  silently fall back to the default.

To change the score, change the text in `[responses]` or the files in `[files]`
for the task ids that fail.
