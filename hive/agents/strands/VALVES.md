# Strands candidate: what you may edit

A Strands candidate is a directory. HIVE runs it with Strands Agents (the `strands-agents` Python
SDK, version pinned by the experiment), one fresh agent per task, in its own process, with the
task's working directory as the process's working directory. The run gets only what this
directory says: the system prompt, the tools, the extensions and the skills listed below.
Strands loads nothing else (no tool directory, no global config, no AWS defaults).

Every file in the directory is part of the candidate's identity, so anything not listed below is
rejected. Make one isolated change per proposal.

## Valves in `candidate.toml` (`[valves]` table)

| Valve | Type | Default | What it does in Strands |
|---|---|---|---|
| `tools` | list of strings | `["shell", "file_editor"]` | The tool allowlist. Vended tools: `shell` (runs one `sh` command in the working directory, fresh shell each call, model-chosen timeout, default 120 s), `file_editor` (view, create, str_replace and insert on files; **absolute paths only**) and `python_repl` (runs Python in a fresh process each call). A tool registered by one of your extensions is offered to the model **only if its name is also listed here**; an unlisted one is dropped and is missing from the load report. Tool names must be unique: an extension tool named like a vended tool or another extension's tool makes the trial `harness_load_error`, and a tool registered once the run has started is refused (`extension_error`). A listed name that no vended tool or extension provides makes the trial `harness_load_error`. An empty list (`[]`) gives the agent no tools. |
| `conversation_window` | integer >= 1 | `40` | How many messages Strands keeps: `SlidingWindowConversationManager(window_size=...)` drops the oldest messages (and truncates large tool results) when the conversation grows past it or overflows the context. |

No other valve names are accepted. `schema_version` and `agent = "strands"` must stay as they are;
`description` is free text and may be updated to describe the change.

## Files

| Path | What it does in Strands | Constraints |
|---|---|---|
| `SYSTEM.md` | The agent's system prompt. Strands has no default prompt of its own, so without this file the model gets only HIVE's one-line note naming the working directory. HIVE always appends that note after your text, because `file_editor` rejects relative paths. | Optional. Must not be empty: delete the file to run without one. |
| `extensions/<name>.py` | A Strands extension: hooks on the agent loop and new tools. **Each file must define a top-level `def extension()`** with no arguments, returning an instance of a `strands.plugins.Plugin` subclass whose `name` is `<name>`. How extensions work, where they can hook in, what you must not touch and the pinned API are in the **extension guide** that accompanies this file. | Optional. Files sit directly in `extensions/` and end in `.py`; subdirectories and other files are rejected. Each file is imported on its own, so extensions cannot import each other. A file that fails to import, returns the wrong thing, or whose `init_agent` raises makes the trial `harness_load_error`; a hook that raises during the run ends it as `extension_error`. Both count as 0. A tool that raises does not: the model gets an error result and the run goes on. |
| `skills/<name>/SKILL.md` | A Strands skill (`AgentSkills`): its name and description are listed in the system prompt and a `skills` tool lets the agent read the full `SKILL.md` and its other files when it decides the skill applies. The `skills` tool is added automatically whenever there is a skill; do not list it in `tools`. | Optional. Every directory under `skills/` must contain a `SKILL.md` starting with a `---` frontmatter block holding `name:` (equal to the directory name) and a non-empty `description:`. Names are 1-64 lowercase letters, digits and single hyphens. Loose files directly in `skills/` are rejected. |
| `README.md` | Notes for humans. Ignored by Strands. | Optional. |

## How a run is judged

A trial counts as completed only when Strands' loop ends normally (`end_turn`) on the model's final
reply; its text is the final answer. A run that hits the step limit (the experiment's `max_turns`
model turns), crashes, is cancelled, or is killed at the time limit is errored and counts as 0.
Throttling that persists after HIVE's short retries, and billing failures (`insufficient_quota`), are
`quota`; other provider failures are `provider_error`.

## Not editable

- **The model**: provider, id, parameters, and its thinking level (reasoning effort). They are fixed
  by the experiment so that a score change reflects the harness, not the model. HIVE records the
  model the provider reports on every response and checks the agent's model object before and after
  every call: a run that answers with another model, replaces `agent.model`, changes its config, or
  tries to build any other model (Strands falls back to Bedrock, which HIVE disables) is marked
  `model_mismatch` and counted as 0.
- **Run settings**: the Strands version, the time limit, the step limit (`max_turns`), the retry
  policy, the environment and the price table. These belong to the experiment.
- `lineage.json`, which the controller writes.
- Anything outside this directory: the suite, its grader and the controller.
