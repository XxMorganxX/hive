# Pi candidate: what you may edit

A Pi candidate is a directory. HIVE runs it with the Pi coding agent (`pi --mode json -p`), one
fresh session per task, with the task's working directory as Pi's working directory. Pi's
global user configuration is switched off for every run: the only extensions, skills and
prompts the agent gets are the ones in this directory.

Every file in the directory is part of the candidate's identity, so anything not listed below
is rejected. Make one isolated change per proposal.

## Valves in `candidate.toml` (`[valves]` table)

| Valve | Type | Default | What it does in Pi |
|---|---|---|---|
| `thinking` | string | `"low"` | Reasoning effort, passed as `--thinking`. One of `off`, `minimal`, `low`, `medium`, `high`, `xhigh`. Pi clamps it to what the model supports. Higher levels usually cost more tokens. |
| `tools` | list of strings | `["read", "bash", "edit", "write"]` | The tool allowlist, passed as `--tools`. Built-in tools are `read`, `bash`, `edit`, `write`, `grep`, `find` and `ls`. A tool registered by one of your extensions is active **only if its name is also listed here**. An empty list (`[]`) disables every tool. Names must be non-empty and contain no commas. |

No other valve names are accepted. `schema_version` and `agent = "pi"` must stay as they are;
`description` is free text and may be updated to describe the change.

## Files

| Path | What it does in Pi | Constraints |
|---|---|---|
| `SYSTEM.md` | **Replaces** Pi's default system prompt (passed as `--system-prompt`). Pi still adds `APPEND_SYSTEM.md`, the skills list, the date and the working directory after it, but **not** its default tool list and usage guidelines: the model still receives the tool schemas, but any advice on using them must now come from your prompt. | Optional. Must not be empty: delete the file to get Pi's stock prompt back. |
| `APPEND_SYSTEM.md` | Text **appended** to the system prompt, whichever base prompt is in use (passed as `--append-system-prompt`). The least disruptive way to add guidance. | Optional. |
| `extensions/<name>.ts` or `extensions/<name>.js` | A Pi extension, loaded with `-e`. Extensions can register tools, react to session events (tool calls, turns, compaction), gate or rewrite tool calls, and transform context. Each file must default-export the extension factory, as Pi's extension API expects. | Optional. Files must sit directly in `extensions/`; subdirectories and other suffixes are rejected, so keep each extension in one file. An extension that fails to load or throws while running marks the trial as `harness_load_error` (counted as 0), so test the logic carefully. |
| `skills/<name>/SKILL.md` | A Pi skill, passed with `--skill`. Pi lists each skill's name and description in the system prompt, and the agent reads the full `SKILL.md` (with the `read` tool) when it decides the skill applies. Other files in the skill directory can be referenced from `SKILL.md`. | Optional. Every directory under `skills/` must contain a `SKILL.md`; loose files directly in `skills/` are rejected. Pi lists skills in the prompt only when `read` is in `tools`. |
| `README.md` | Notes for humans. Ignored by Pi. | Optional. |

## Not editable

- **The model** (provider, id and parameters). It is fixed by the experiment so that a score
  change reflects the harness, not the model. Do not try to change it through any file.
- **Run settings**: the Pi version, the time limit per task, offline mode and the session
  directory. These belong to the experiment.
- `lineage.json`, which the controller writes.
- Anything outside this directory: the suite, its grader and the controller.
