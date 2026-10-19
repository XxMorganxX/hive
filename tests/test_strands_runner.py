"""Layer 2 for the Strands adapter: the real runner in the real Strands venv, offline.

Every run uses the test-only "scripted" provider: the runner replays model turns from a
JSON script through a minimal strands Model subclass, so a full framework run (real event
loop, real vended shell and file_editor in the workdir, real candidate extensions) needs
no network and no key. Skipped unless scripts/setup-framework-venvs.sh strands has built
.venvs/strands.
"""

import json
import os
import signal
import time
from pathlib import Path

import pytest

from hive.agents.strands import StrandsAdapter
from hive.candidate import Candidate
from hive.contracts import ModelSpec, Task
from hive.extensions import expected_extension_files, load_problem

REPO_ROOT = Path(__file__).resolve().parents[1]
STRANDS_PYTHON = REPO_ROOT / ".venvs" / "strands" / "bin" / "python"
MODEL = ModelSpec(provider="scripted", id="scripted-model")
PRICES = {"input": 0.05, "output": 0.40, "cache_read": 0.005}
TASK = Task(id="t1", split="search", instruction="Write solution.py, then answer.")

pytestmark = pytest.mark.skipif(not STRANDS_PYTHON.exists(), reason="the Strands venv is not built (.venvs/strands)")


def write_candidate(root: Path, valves: str = "", files: dict[str, str] | None = None) -> Candidate:
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text(f'schema_version = 1\nagent = "strands"\ndescription = "test"\n\n[valves]\n{valves}')
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Candidate.load(root)


def text(value: str) -> dict:
    return {"text": value}


def tool(name: str, **arguments) -> dict:
    return {"tool": name, "input": arguments}


def run_script(tmp_path: Path, turns: list, candidate: Candidate | None = None, **settings):
    """Run one trial with the scripted provider; returns (result, workdir, trace_dir)."""
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"turns": turns}))
    candidate = candidate or write_candidate(tmp_path / "candidate")
    agent = StrandsAdapter().build(
        candidate,
        MODEL,
        {"python_path": str(STRANDS_PYTHON), "strands_version": "1.61.1", "prices": PRICES, "timeout_s": 60,
         "script_file": str(script), **settings},
    )
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(exist_ok=True)
    return agent.run(TASK, workdir, trace_dir), workdir.resolve(), trace_dir


def trace_lines(trace_dir: Path, kind: str | None = None) -> list[dict]:
    lines = [json.loads(line) for line in (trace_dir / "trace.jsonl").read_text().splitlines() if line.strip()]
    return [line for line in lines if kind is None or line.get("type") == kind]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(status) and not status.startswith("Z")


def wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


# --- a plain run ------------------------------------------------------------------------


def test_a_run_that_writes_solution_py_with_file_editor_completes(tmp_path):
    # file_editor needs absolute paths; the workdir is known before the run (tmp_path/work).
    solution = (tmp_path / "work").resolve() / "solution.py"
    turns = [
        [tool("file_editor", command="create", path=str(solution), file_text="def add(a, b):\n    return a + b\n")],
        [text("Wrote solution.py. ANSWER: done")],
    ]
    result, workdir, _ = run_script(tmp_path, turns)
    assert result.status == "completed", result.error
    assert result.final_text == "Wrote solution.py. ANSWER: done"
    assert (workdir / "solution.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_the_shell_tool_runs_in_the_workdir(tmp_path):
    turns = [[tool("shell", command="echo hello > out.txt")], [text("done")]]
    result, workdir, _ = run_script(tmp_path, turns)
    assert result.status == "completed", result.error
    assert (workdir / "out.txt").read_text().strip() == "hello"


def test_every_model_call_is_one_usage_line_in_the_trace(tmp_path):
    turns = [
        {"blocks": [tool("shell", command="true")], "usage": {"input_tokens": 120, "output_tokens": 7, "cache_read_tokens": 20}},
        {"blocks": [text("done")], "usage": {"input_tokens": 200, "output_tokens": 9, "cache_read_tokens": 0}},
    ]
    result, _, trace_dir = run_script(tmp_path, turns)
    calls = trace_lines(trace_dir, "model_call")
    assert [call["model"] for call in calls] == ["scripted-model", "scripted-model"]
    assert [call["usage"]["input_tokens"] for call in calls] == [100, 200]  # cache reads excluded
    assert result.usage.input_tokens == 300
    assert result.usage.output_tokens == 16
    assert result.usage.cache_read_tokens == 20


def test_a_step_limit_hit_is_agent_error(tmp_path):
    turns = [[tool("shell", command="true")]] * 5
    result, _, _ = run_script(tmp_path, turns, max_turns=2)
    assert result.status == "errored"
    assert result.error_class == "agent_error"
    assert "max_turns=2" in result.error


def test_a_tool_outside_the_allowlist_is_not_offered(tmp_path):
    candidate = write_candidate(tmp_path / "candidate", valves='tools = ["file_editor"]\n')
    turns = [[tool("shell", command="echo hi > out.txt")], [text("done")]]
    result, workdir, _ = run_script(tmp_path, turns, candidate=candidate)
    assert result.status == "completed", result.error
    assert not (workdir / "out.txt").exists()


# --- extensions -------------------------------------------------------------------------

GOOD_EXTENSION = '''
from pathlib import Path

from strands import tool
from strands.hooks import BeforeToolCallEvent
from strands.plugins import Plugin, hook


class Guard(Plugin):
    name = "guard"

    @hook
    def block_rm(self, event: BeforeToolCallEvent) -> None:
        Path("guard-fired.txt").write_text(event.tool_use["name"])
        if "rm " in str(event.tool_use["input"].get("command", "")):
            event.cancel_tool = "rm is blocked by the guard"

    @tool
    def word_count(self, text: str) -> int:
        """Count the words in text."""
        return len(text.split())


def extension():
    return Guard()
'''


def test_a_good_extension_loads_and_its_hook_fires(tmp_path):
    candidate = write_candidate(
        tmp_path / "candidate",
        valves='tools = ["shell", "file_editor", "word_count"]\n',
        files={"extensions/guard.py": GOOD_EXTENSION},
    )
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "keep.txt").write_text("x")
    turns = [[tool("shell", command="rm keep.txt")], [tool("word_count", text="a b c")], [text("done")]]
    result, workdir, _ = run_script(tmp_path, turns, candidate=candidate)
    assert result.status == "completed", result.error
    load = result.load_report.extensions[0]
    assert (load.path, load.loaded, load.error) == ("extensions/guard.py", True, None)
    assert set(load.registered) == {"before_tool_call", "register_tool", "tool:word_count"}
    assert (workdir / "guard-fired.txt").exists()
    assert (workdir / "keep.txt").exists()  # the guard cancelled the rm
    assert load_problem(expected_extension_files(candidate, StrandsAdapter().extension_spec()), result.load_report) is None


def test_an_extension_tool_missing_from_the_allowlist_is_not_registered(tmp_path):
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/guard.py": GOOD_EXTENSION})
    turns = [[tool("word_count", text="a b c")], [text("done")]]
    result, _, trace_dir = run_script(tmp_path, turns, candidate=candidate)
    assert result.status == "completed", result.error
    assert "tool:word_count" not in result.load_report.extensions[0].registered
    assert trace_lines(trace_dir, "tool_result")[0]["status"] == "error"  # the model could not call it


def test_an_extension_that_fails_to_import_is_harness_load_error(tmp_path):
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/broken.py": "oops()\n\ndef extension():\n    return None\n"})
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.error_class == "harness_load_error"
    load = result.load_report.extensions[0]
    assert (load.path, load.loaded) == ("extensions/broken.py", False)
    assert "NameError" in load.error


def test_an_extension_returning_the_wrong_type_is_harness_load_error(tmp_path):
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/broken.py": "def extension():\n    return 42\n"})
    result, _, trace_dir = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.error_class == "harness_load_error"
    assert "Plugin" in result.load_report.extensions[0].error
    assert trace_lines(trace_dir, "model_call") == []  # the run never started


def test_a_plugin_named_differently_from_its_file_is_harness_load_error(tmp_path):
    source = GOOD_EXTENSION.replace('name = "guard"', 'name = "other"')
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/guard.py": source})
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.error_class == "harness_load_error"
    assert "guard" in result.load_report.extensions[0].error


def test_one_broken_extension_does_not_hide_a_good_one(tmp_path):
    candidate = write_candidate(
        tmp_path / "candidate",
        files={"extensions/guard.py": GOOD_EXTENSION, "extensions/broken.py": "def extension():\n    return 42\n"},
    )
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    loads = {load.path: load.loaded for load in result.load_report.extensions}
    assert loads == {"extensions/broken.py": False, "extensions/guard.py": True}


def test_the_api_reference_example_extension_loads_and_runs(tmp_path):
    reference = (REPO_ROOT / "hive" / "agents" / "strands" / "EXTENSIONS_API.md").read_text()
    example = reference.split("```python\n", 1)[1].split("```", 1)[0]
    candidate = write_candidate(
        tmp_path / "candidate",
        valves='tools = ["shell", "file_editor", "word_count"]\n',
        files={"extensions/guard.py": example},
    )
    assert StrandsAdapter().validate(candidate) == []
    turns = [[tool("shell", command="rm -rf nothing")], [tool("word_count", text="a b")], [text("done")]]
    result, _, trace_dir = run_script(tmp_path, turns, candidate=candidate)
    assert result.status == "completed", result.error
    assert set(result.load_report.extensions[0].registered) == {
        "before_tool_call", "after_tool_call", "register_tool", "tool:word_count"}  # fmt: skip
    results = trace_lines(trace_dir, "tool_result")
    assert results[0]["status"] == "error" and "rm -rf is blocked" in results[0]["content"]


THROWING_EXTENSION = '''
from strands.hooks import AfterToolCallEvent
from strands.plugins import Plugin, hook


class Flaky(Plugin):
    name = "flaky"

    @hook
    def explode(self, event: AfterToolCallEvent) -> None:
        raise ValueError("flaky hook bug")


def extension():
    return Flaky()
'''


def test_an_extension_that_throws_mid_run_is_extension_error(tmp_path):
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/flaky.py": THROWING_EXTENSION})
    turns = [[tool("shell", command="true")], [text("done")]]
    result, _, _ = run_script(tmp_path, turns, candidate=candidate)
    assert result.status == "errored"
    assert result.error_class == "extension_error"
    assert result.load_report.extensions[0].loaded
    assert any("extensions/flaky.py" in error and "flaky hook bug" in error for error in result.load_report.runtime_errors)


INIT_FAILING_EXTENSION = '''
from strands.plugins import Plugin


class Bad(Plugin):
    name = "bad"

    def init_agent(self, agent):
        raise RuntimeError("init failed")


def extension():
    return Bad()
'''


def test_an_extension_whose_init_agent_raises_is_harness_load_error(tmp_path):
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/bad.py": INIT_FAILING_EXTENSION})
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.error_class == "harness_load_error"
    load = result.load_report.extensions[0]
    assert not load.loaded and "init failed" in load.error


def test_an_extension_registering_with_add_hook_in_init_agent_is_recorded(tmp_path):
    source = '''
from strands.hooks import AfterInvocationEvent
from strands.plugins import Plugin


def note(event: AfterInvocationEvent) -> None:
    pass


class Watcher(Plugin):
    name = "watcher"

    def init_agent(self, agent):
        agent.add_hook(note)


def extension():
    return Watcher()
'''
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/watcher.py": source})
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.status == "completed", result.error
    assert result.load_report.extensions[0].registered == ("after_run",)


SWAPPING_EXTENSION = '''
from strands.hooks import BeforeModelCallEvent
from strands.plugins import Plugin, hook


class Swap(Plugin):
    name = "swap"

    @hook
    def swap(self, event: BeforeModelCallEvent) -> None:
        setattr(event.agent, "mod" + "el", object())


def extension():
    return Swap()
'''


def test_an_extension_that_swaps_the_model_is_model_mismatch(tmp_path):
    # setattr with a computed name gets past validate's static check; the runtime guard still sees it.
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/swap.py": SWAPPING_EXTENSION})
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.error_class == "model_mismatch"


def test_bedrock_can_never_be_constructed(tmp_path):
    source = '''
from strands import Agent
from strands.plugins import Plugin


class Sub(Plugin):
    name = "sub"

    def init_agent(self, agent):
        Agent()  # no model: Strands would default to Bedrock


def extension():
    return Sub()
'''
    candidate = write_candidate(tmp_path / "candidate", files={"extensions/sub.py": source})
    result, _, _ = run_script(tmp_path, [[text("done")]], candidate=candidate)
    assert result.status == "errored"
    assert "Bedrock" in result.load_report.extensions[0].error


# --- skills -----------------------------------------------------------------------------


def test_a_skill_is_offered_through_the_skills_tool(tmp_path):
    skill = "---\nname: plan\ndescription: Use when planning a multi-step change.\n---\n\n# Steps\n1. Think.\n"
    candidate = write_candidate(tmp_path / "candidate", files={"skills/plan/SKILL.md": skill})
    turns = [[tool("skills", skill_name="plan")], [text("done")]]
    result, _, trace_dir = run_script(tmp_path, turns, candidate=candidate)
    assert result.status == "completed", result.error
    assert "skills" in trace_lines(trace_dir, "harness")[0]["tools"]
    skill_result = trace_lines(trace_dir, "tool_result")[0]
    assert skill_result["status"] == "success" and "Think." in skill_result["content"]


# --- provider failures ------------------------------------------------------------------


def test_insufficient_quota_is_quota(tmp_path):
    turns = [{"error": {"kind": "rate_limit", "code": "insufficient_quota", "message": "You exceeded your current quota"}}]
    result, _, _ = run_script(tmp_path, turns)
    assert result.error_class == "quota"


def test_a_rate_limit_that_persists_after_retries_is_quota(tmp_path):
    error = {"error": {"kind": "rate_limit", "code": "rate_limit_exceeded", "message": "Rate limit reached"}}
    started = time.monotonic()
    result, _, _ = run_script(tmp_path, [error, error, error], max_model_attempts=2)
    assert result.error_class == "quota"
    assert time.monotonic() - started < 30  # a fast retry strategy, not Strands' 6 attempts with 240 s backoff


def test_a_server_error_is_provider_error(tmp_path):
    result, _, _ = run_script(tmp_path, [{"error": {"kind": "server_error", "message": "upstream exploded"}}], max_model_attempts=1)
    assert result.error_class == "provider_error"


def test_a_reported_model_other_than_the_experiments_is_model_mismatch(tmp_path):
    result, _, _ = run_script(tmp_path, [{"blocks": [text("done")], "model": "gpt-4o"}])
    assert result.error_class == "model_mismatch"


def test_a_script_that_runs_out_is_agent_error(tmp_path):
    result, _, _ = run_script(tmp_path, [[tool("shell", command="true")]])
    assert result.error_class == "agent_error"


# --- child processes and timeout --------------------------------------------------------


def test_a_background_shell_child_is_reaped_after_the_run(tmp_path):
    turns = [[tool("shell", command="sleep 300 > /dev/null 2>&1 & echo $! > sleep.pid")], [text("done")]]
    result, workdir, trace_dir = run_script(tmp_path, turns)
    assert result.status == "completed", result.error
    assert (trace_dir / "child_pgids.txt").read_text().split()
    pid = int((workdir / "sleep.pid").read_text())
    try:
        assert wait_dead(pid)
    finally:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_a_timeout_kills_the_run_and_its_shell_children(tmp_path):
    turns = [[tool("shell", command="echo $$ > shell.pid; sleep 60")], [text("done")]]
    started = time.monotonic()
    result, workdir, _ = run_script(tmp_path, turns, timeout_s=4, term_grace_s=0.5)
    assert result.error_class == "timeout"
    assert time.monotonic() - started < 20
    pid = int((workdir / "shell.pid").read_text())
    try:
        assert wait_dead(pid)
    finally:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
