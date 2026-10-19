## Extensions: code that hooks the agent loop

This agent's framework runs extensions: files under `candidate/extensions/` that execute
inside the agent loop. They are one of the strongest levers you have. A prompt can only ask
the model to behave; an extension can make the loop enforce it. Read `context/extensions.md`
before writing one. It says what each file must define, where an extension can hook in and
what it can change there, which APIs are forbidden, and the pinned API reference.

Consider an extension when the traces show a behavioural failure that the loop itself could
catch or correct, for example:

- **gate** a tool call that should never run, or should run only after a check;
- **rewrite** a tool's arguments or result, or the context before a model call;
- **retry** a model reply that misses a required format or step;
- **verify** the final answer, or the work left in the working directory, before the run ends.

An extension still counts as your one change: add or edit one extension, not an extension
plus a prompt rewrite. Every extension file must load, and extension code that throws during
a run errors that trial; both score 0, so keep it small and defensive. Each trial's
`outcome.json` in `context/evidence/` has a `load_report` saying which extensions loaded and
what they registered.
