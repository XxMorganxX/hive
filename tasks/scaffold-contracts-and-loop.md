# Scaffold: agent contract, two swappable suites, bare-bones improvement loop

## Goal

Stand up the smallest runnable version of HIVE's outer loop:

1. **Agent contract.** Any agent framework plugs in through one adapter.
2. **Suite contract,** with **two very simple suites** that swap by changing one config line.
3. **Improvement loop.** It evaluates the incumbent and asks the **overseer** (Claude Code, headless) for an improved candidate. The overseer gets three inputs: the current candidate config, the evaluation metric, and the suite description. The loop then evaluates the child and keeps it only if the score says it's better.

Spec lives in `docs/FOUNDING_PRINCIPLES.md` §5–§8. The code-level source of truth for the contracts is `hive/contracts.py`.

## Approach

- **Python 3.12 with uv.** No runtime dependencies (stdlib only); pytest for tests.
- **One package, `hive/`.**
  - Core: contracts, candidate, registry, config, evaluate, selection, loop, CLI.
  - Plug-ins are subpackages: `hive/agents/{pi,scripted}`, `hive/suites/{exact_answer,python_functions}`, `hive/overseers/{claude_code,null}`.
- **Order of work.** I write the contracts and candidate model myself first. Then a workflow fans out implementation by module, test-first, followed by an adversarial review, and I verify by running everything.

## Decisions

- **Revised: HIVE owns the swap boundaries; Harbor moves behind them.** v0.1 made Harbor's `BaseAgent` the single boundary.
  - The user asked for evaluation tasks to be "abstract and universal" and to fit the plug-in boundary. Harbor's agent interface is execution-shaped and ties every suite to containers.
  - So HIVE defines three small contracts: `AgentAdapter`/`Agent`, `Suite` and `Overseer`. Harbor comes back later as one `Suite` implementation (Harbor datasets, Docker sandboxes) and as an execution backend for agents.
  - Rejected: keeping Harbor as the only boundary. It needs Docker for every task, even a 10-line exact-answer check, and couples task format to Harbor's layout.
- **Task shape is universal:** an instruction plus optional workspace files. The suite owns grading, and the agent sees neither reference answers nor hidden tests. This one shape covers Q&A, code-in-a-directory, and later Harbor tasks.
- **The overseer edits an isolated copy.** Claude Code runs in a scratch workspace holding:
  - `candidate/`: an editable copy of the incumbent;
  - `context/`: suite description, metric, valve catalog, selection rule, the incumbent's search-split report and traces, and history.

  Only `candidate/` and `RATIONALE.md` are harvested afterwards. This is a real write fence that doesn't depend on permission-flag behavior. Tools are limited to Read, Grep, Glob, Edit and Write (no Bash), and user settings are not loaded, so no user hooks fire.
- **The Pi adapter isolates Pi from the user's global config.** It passes `--no-extensions --no-skills --no-prompt-templates --no-context-files`, and only the candidate's files are passed in explicitly (`--system-prompt`, `--append-system-prompt`, `-e`, `--skill`). ~~It keeps the default agent dir so auth still works. `--offline` freezes the model catalog.~~ Superseded after review: the user's `settings.json` / `models.json` still applied. It now runs against a per-run temp agent dir (symlinked `auth.json` + `bin/`, pinned `settings.json`, no `models.json`). `--offline` only skips startup network calls.
- **The Pi exit code is ignored.** JSON mode exits 0 even on provider errors (verified in `dist/modes/print-mode.js` for 0.75.3). Status comes from the last assistant `message_end.stopReason`, and usage is summed over every assistant `message_end`.
- **The model lives in the experiment config, not the candidate (P3).** The thinking level and tool allowlist are candidate valves.
- **Bare-bones selection.** Promotion requires the child's conservative score to beat the incumbent's by more than `promotion_margin`, and never to fall below the seed's. The conservative score counts errored trials as 0 over all tasks × k. The Wilson/lower-bound rule, the noise-floor run and valve-diversity pressure are deferred (see Out of scope).
- **Grader feedback never reveals answers.** `Grade.detail` says what the agent produced and whether it passed, never the expected value or hidden test contents, because the overseer reads it.
- **Two demo suites:**
  - `exact_answer`: short questions; the agent ends with `ANSWER: <x>`; normalized exact match; reward 0/1.
  - `python_functions`: implement a function in `solution.py` in the workdir; hidden tests run in a subprocess with a timeout; reward = fraction of tests passed.
- **A scripted agent for dry runs.** It's a test double that replays canned responses from the candidate, so the loop can be exercised end to end with zero model spend.

## Progress log

- 2026-10-19: Verified the CLIs locally. Pi 0.75.3 is at `/opt/homebrew/bin/pi`; the flags confirmed via `--help` and the JSON event and usage shapes via installed type defs. Claude Code is 2.1.303, with `-p --output-format json --tools --permission-mode acceptEdits --setting-sources --max-budget-usd --no-session-persistence`. Pi's only credential is OpenAI (`auth.json`), so the demo experiments use `openai/gpt-5-nano`.
- 2026-10-19: Revised the principles doc to match:
  - **Terminology:** the "optimizer" role is now the **overseer**.
  - **P5:** HIVE owns three contracts, and Harbor sits behind them.
  - **Rewritten sections:** §5 (contracts), §6 (loop + experience store) and §8 (layout).
- 2026-10-19: Wrote `hive/contracts.py` and `hive/candidate.py` myself, since every module depends on them. Candidate was test-first: 13 tests, red then green. Added a contract clause: `Agent.run` may be called concurrently with distinct workdirs, because the evaluator uses a thread pool.
- 2026-10-19: Launched wave 1 as a workflow: seven implementers in parallel on disjoint files, test-first. The modules are evaluate+selection, the Pi adapter, the scripted agent + null overseer, the `exact_answer` suite, the `python_functions` suite, the Claude Code overseer, and registry+config. Implementers don't commit; I commit per module after checking their tests myself.
- 2026-10-19: Wave 1 landed. **Observed:** 456 tests pass in the full suite. Committed one module per commit. What the implementers found in Pi 0.75.3's source that the docs had wrong:
  - **A failed extension load** exits 1 at startup with `Failed to load extension`. It does not continue, so both forms map to `harness_load_error`.
  - **Global SYSTEM.md / APPEND_SYSTEM.md leak in** unless both flags are always passed. The adapter passes an empty string when the candidate has none.
  - **Instructions starting with `-` or `@` get parsed as flags,** so they go on stdin.
  - **Pi reads piped stdin,** so stdin must be `DEVNULL` or it hangs.
- 2026-10-19: Fixed a hole the overseer implementer flagged. `Candidate.copy_to` copied hidden files that the content hash ignores, so an overseer could ship a hidden payload without changing the id. Now copy == hashed set + lineage (test-first).
- 2026-10-19: **Decision: the overseer runs in a temp workspace outside any git checkout,** then the workspace is copied into the run dir for the record. Why: Claude Code walks up from its working directory loading CLAUDE.md and project settings. Inside the HIVE repo, the overseer would read instructions written for HIVE's developers.
- 2026-10-19: Known limitation, recorded rather than fixed. Pi trials run on the host, and the `python_functions` hidden cases live in the repo, so an agent with bash could read them. This is fine for a scaffold, but it must close before real experiments (Docker / Harbor verifier, P7).
- 2026-10-19: Launched wave 2 (loop, CLI, experiment configs, dry-run seeds) with an independent checker agent that tries to break it.
- 2026-10-19: Wave 2 landed. **Observed:** 534 tests pass, and both dry runs exit 0. The checker found two medium issues, fixed test-first:
  - **The overseer was briefed with the live incumbent snapshot,** so it could forge held-out scores. Now it gets copies in temp dirs, and every evaluation re-hashes its snapshot first.
  - **The cap was checked only between generations** (it overshot by 60%). Now it is estimate-gated from the seed's cost per trial.

  Also fixed:
  - `*_file` / `*_dir` settings now resolve against the config file.
  - `Candidate.load` rejects symlinks.
  - `run_meta` records `git_dirty`.
  - A temp dir inside a git checkout is refused.
- 2026-10-19: Adversarial review: 6 lenses, 74 agents, and every finding had to survive 2 skeptics. 25 confirmed, 9 refuted, and no prior-work references found.
  - **Highs:**
    - The overseer's `--allowedTools Edit,Write` had no path scope, so it could write and read anywhere, including held-out answers. The report it got also carried absolute paths into the repo.
    - The thinking level was a candidate valve. That breaks P3, because more reasoning buys accuracy with spend.
    - `pi --version` prints to stderr, so the version pin would fail every real run.
    - Ctrl-C mid-evaluation kept paying for queued trials, and the spend went unrecorded.
  - **Decision: thinking moves into `ModelSpec` / `[model] thinking`.** The Pi adapter checks every response's provider and model (`model_mismatch`).
  - **Decision: an adapter's `name` must equal the string candidates use in `agent`.** The registry enforces this.
  - Fixes fanned out to four fixers on disjoint files, with an independent recheck. I reconciled the spec myself.

## Open questions

- Is the overseer model `sonnet` or `opus`? The default is `sonnet` in the experiment config, which is cheap to change.

## Out of scope (deferred, tracked against principles)

- Harbor suite and Docker isolation for agent-written code. The `python_functions` grader runs code in a local subprocess with a timeout, which is not a sandbox.
- Multi-agent topology (`topology.yaml` compiler for Pi).
- Wilson lower-bound selection, a noise-floor run, valve diversity, `flips/` diffs, ATIF export.
- Extension typecheck (`tsc --noEmit`) in Pi `validate`.
