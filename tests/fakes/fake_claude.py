"""A stand-in for the `claude` CLI, so overseer tests never call a real model.

It records how it was invoked, optionally edits the workspace the way Claude Code
would, and prints a result object shaped like `claude -p --output-format json`.

Environment:
- FAKE_CLAUDE_RECORD: path of a JSON file to write {"argv": [...], "cwd": "..."} to.
- FAKE_CLAUDE_MODE: which behaviour to simulate (default "edit"); see MODES below.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

RATIONALE = "## Observed\nAnswers lacked the final marker.\n\n## Changed\nAppended a reminder.\n\n## Why it should transfer\nEvery task needs the marker.\n"
RESULT_TEXT = "I appended a reminder to the system prompt."
COST_USD = 0.42


def result_object(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": RESULT_TEXT,
        "total_cost_usd": COST_USD,
    }
    result.update(overrides)
    return result


def edit_candidate(workspace: Path) -> None:
    (workspace / "candidate" / "APPEND_SYSTEM.md").write_text("Always end with ANSWER: <x>.\n")


def write_rationale(workspace: Path) -> None:
    (workspace / "RATIONALE.md").write_text(RATIONALE)


def main() -> int:
    workspace = Path.cwd()
    record_path = os.environ.get("FAKE_CLAUDE_RECORD")
    if record_path:
        Path(record_path).write_text(json.dumps({"argv": sys.argv[1:], "cwd": str(workspace)}))

    mode = os.environ.get("FAKE_CLAUDE_MODE", "edit")
    print("fake claude stderr line", file=sys.stderr)

    if mode == "edit":
        edit_candidate(workspace)
        write_rationale(workspace)
        print(json.dumps(result_object()))
    elif mode == "noop":
        print(json.dumps(result_object()))
    elif mode == "no_rationale":
        edit_candidate(workspace)
        print(json.dumps(result_object()))
    elif mode == "no_cost":
        edit_candidate(workspace)
        write_rationale(workspace)
        result = result_object()
        del result["total_cost_usd"]
        print(json.dumps(result))
    elif mode == "not_json":
        print("this is not json")
    elif mode == "is_error":
        print(json.dumps(result_object(is_error=True, subtype="error_max_budget_usd", result="budget exceeded")))
    elif mode == "exit_nonzero":
        print(json.dumps(result_object(total_cost_usd=0.1)))
        return 3
    elif mode == "sleep":
        time.sleep(30)
        print(json.dumps(result_object()))
    elif mode == "break_candidate":
        (workspace / "candidate" / "candidate.toml").write_text("this is = = not toml")
        write_rationale(workspace)
        print(json.dumps(result_object()))
    elif mode == "change_agent":
        manifest = workspace / "candidate" / "candidate.toml"
        manifest.write_text(manifest.read_text().replace('agent = "scripted"', 'agent = "other"'))
        write_rationale(workspace)
        print(json.dumps(result_object()))
    else:
        print(f"unknown FAKE_CLAUDE_MODE {mode!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
