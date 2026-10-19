# Deep Agents candidate: what you may edit

A Deep Agents candidate is a directory. HIVE runs it with LangChain Deep Agents (deepagents
0.7.28, pinned), one fresh agent per task, in a subprocess of its own whose working directory is
the task's working directory. The agent gets real file and shell tools rooted there: `/` in the
file tools is the working directory, and HIVE adds a short note with its absolute path to the
system prompt. Nothing is read from the user's or the machine's configuration: the only prompt,
extensions and skills the agent gets are the ones in this directory.

Every file in the directory is part of the candidate's identity, so anything not listed below is
rejected. Make one isolated change per proposal.

## Valves in `candidate.toml` (`[valves]` table)

| Valve | Type | Default | What it does |
|---|---|---|---|
| `tools` | list of strings | `["ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"]` | The tool allowlist over built-in and extension tools alike. A tool not listed is hidden from the model, and a call to it is refused. The built-ins are the eight defaults (`execute` runs a shell command in the working directory) and `task` (only with `general_purpose_subagent = true`). A tool an extension registers must be listed here, or that extension fails to load. An empty list disables every tool. Names must be unique. |
| `general_purpose_subagent` | true / false | `false` | Adds deepagents' general-purpose subagent and its `task` tool (list `task` in `tools` too), so the model can hand a subtask to a fresh agent with the same model and tools. **Caveat:** your extension middleware does not reach the subagent: deepagents applies main-agent middleware only to the main agent, so hooks that gate or rewrite tool calls are bypassed for work the model delegates. HIVE's own tool allowlist and tool-error handling do apply to it, the step limit applies to each of its runs, and its model calls count toward the one model-call limit of the trial, shared with the main agent. |

No other valve names are accepted. `schema_version` and `agent = "deepagents"` must stay as they
are; `description` is free text and may be updated to describe the change.

## Files

| Path | What it does | Constraints |
|---|---|---|
| `SYSTEM.md` | The system prompt (`create_deep_agent(system_prompt=...)`). HIVE appends its working-directory note; deepagents adds its own sections after that (the skills list, and with skills a note on shell vs. virtual paths). Without the file, the model gets only those. | Optional. Must not be empty: delete it to have no prompt of your own. |
| `extensions/<name>.py` | Python code that hooks the agent loop: LangChain `AgentMiddleware` and tools. Each file must define `def extension():` (sync, no arguments) returning a list of middleware and/or tools; see the extension guide for the hooks, what each can change, and the pinned API. | Optional. Files sit directly in `extensions/` and end in `.py`; each is imported on its own, so it can't import another extension file. `validate` parses each file (it never runs it) and rejects a missing or async `extension`, and any use of APIs that build or switch the model. |
| `skills/<name>/SKILL.md` | A deepagents skill. The skill's name and description are listed in the system prompt, and the agent reads the full `SKILL.md` (with `read_file`, under `/skills/`) when it decides the skill applies. Other files in the skill directory can be referenced from `SKILL.md`. | Optional. Every directory under `skills/` needs a `SKILL.md` starting with a `---` frontmatter block holding `name:` (equal to the directory name: lowercase letters, digits and single hyphens) and a non-empty `description:`; deepagents silently skips a skill without them. The agent works on a copy, so it cannot change your skill files. |
| `README.md` | Notes for humans. Ignored by the agent. | Optional. |

## Extensions

Extensions are first-class: the controller renders an extension guide for you (where each hook
sits, what it can change, its caveats, what is forbidden, and the pinned API reference). In short,
each `extensions/<name>.py` defines `def extension():` returning `AgentMiddleware` instances and/or
`BaseTool` objects. The controller checks every run: a file that fails to import, lacks
`extension()`, returns anything else, has an async-only hook, registers a tool missing from
`tools`, or returns middleware named like an entry of deepagents' middleware stack or a HIVE guard
(only `SummarizationMiddleware` may be taken, by one extension) makes the trial
`harness_load_error`; an exception thrown by extension code during the run
makes it `extension_error`. Both score 0. An exception thrown by a tool itself does **not**: it
becomes an error result the model sees, and the run goes on.

## How a run is judged

A trial counts as completed only when the agent's run finishes normally and its last message is a
model reply with no tool calls. Hitting the model-call limit or the step (recursion) limit, a
crash, or a run that ends any other way is `agent_error`; a provider rate limit or billing failure
is `quota`; any other provider failure is `provider_error`. All score 0.

## Not editable

- **The model**: provider, id and its thinking level (reasoning effort) are fixed by the experiment
  so that a score change reflects the harness, not the model. HIVE checks every request made
  through the agent's model and every reply it traces: a run that sends a request with another
  model or reasoning effort, or gets a reply from another model, is marked `model_mismatch` and
  counted as 0. Extension code runs in the same process as the agent, so these checks catch
  mistakes and plain violations; they are not a sandbox. Getting around them, for example by
  calling the provider client behind the model directly, is forbidden.
- **Run settings**: the deepagents version, the time limit per task, the model-call and step
  limits, the shell command timeout and the provider retry policy. These belong to the experiment.
- `lineage.json`, which the controller writes.
- Anything outside this directory: the suite, its grader and the controller.
