"""Layer 2: the real Deep Agents runner in its pinned venv, driven by the scripted test model.

Each test runs a real trial through the adapter: the venv's interpreter, runner.py, a real
`create_deep_agent` with real file and shell tools in a temp workdir, and a scripted model
that replays turns from a file (provider "scripted"), so nothing touches the network or a
key. Skipped unless `.venvs/deepagents` exists (scripts/setup-framework-venvs.sh deepagents).

The last tests drive the real OpenAI provider class (ChatOpenAI) over HTTP instead:

- against a local stub, with the runner started directly and given the stub as its endpoint
  on stdin (or, for the legacy no-gateway path, a key on stdin and OPENAI_BASE_URL);
- through the adapter, with a real ModelGateway (run in this test process) in front of
  tests/fakes/fake_openai_upstream.py, which is how every real-provider trial now runs.

Every key and token here is a placeholder; nothing reaches the network beyond 127.0.0.1.
"""

import dataclasses
import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import hive.agents.deepagents.adapter as da
from fakes.fake_openai_upstream import RESPONSES_USAGE, SNAPSHOT_SUFFIX, FakeOpenAIUpstream
from hive.agents.deepagents import DeepAgentsAdapter
from hive.candidate import Candidate
from hive.contracts import ModelEndpoint, ModelSpec, Task, Usage
from hive.extensions import expected_extension_files, load_problem
from hive.gateway import ModelGateway
from hive.pricing import Prices

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venvs" / "deepagents" / "bin" / "python"
MODEL = ModelSpec(provider="scripted", id="gpt-5-nano", thinking="low")
REPORTED = "gpt-5-nano-2025-08-07"

pytestmark = pytest.mark.skipif(not VENV_PYTHON.is_file(), reason="no .venvs/deepagents; run scripts/setup-framework-venvs.sh deepagents")


def call(name: str, **args) -> dict:
    return {"name": name, "args": args}


def turn(text: str = "", *calls: dict, input_tokens: int = 100, output_tokens: int = 10, cache_read: int = 0,
         model: str = REPORTED) -> dict:  # fmt: skip
    return {"text": text, "tool_calls": list(calls), "model": model,
            "usage": {"input": input_tokens, "output": output_tokens, "cache_read": cache_read}}  # fmt: skip


def write_candidate(root: Path, valves: str = "", files: dict[str, str] | None = None) -> Candidate:
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text(f'schema_version = 1\nagent = "deepagents"\n\n[valves]\n{valves}')
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Candidate.load(root)


def trial(tmp_path: Path, turns: list[dict], *, valves: str = "", files: dict[str, str] | None = None,
          instruction: str = "Implement add in solution.py.", workspace: dict[str, str] | None = None,
          turns_text: str | None = None, **settings):  # fmt: skip
    """One real trial; returns (AgentResult, workdir, trace_dir, candidate).

    `turns_text`, when given, is written as the turns file instead of `turns` (e.g. broken JSON).
    """
    candidate = write_candidate(tmp_path / "candidate", valves, files)
    turns_file = tmp_path / "turns.json"
    turns_file.write_text(json.dumps(turns) if turns_text is None else turns_text)
    adapter = DeepAgentsAdapter()
    assert adapter.validate(candidate) == []
    agent = adapter.build(candidate, MODEL, {"python_path": str(VENV_PYTHON), "scripted_turns_path": str(turns_file),
                                             "timeout_s": 120, **settings})  # fmt: skip
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    for relative, text in (workspace or {}).items():
        (workdir / relative).write_text(text)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    result = agent.run(Task(id="t1", split="search", instruction=instruction), workdir, trace_dir)
    return result, workdir, trace_dir, candidate


def trace_lines(trace_dir: Path, kind: str | None = None) -> list[dict]:
    lines = [json.loads(line) for line in (trace_dir / "trace.jsonl").read_text().splitlines() if line.strip()]
    return [line for line in lines if kind is None or line.get("type") == kind]


def stderr(trace_dir: Path) -> str:
    return (trace_dir / "runner.stderr").read_text(errors="replace")


def checked(result, candidate):
    """The controller's own verdict on the load report (hive.extensions), as evaluate applies it."""
    return load_problem(expected_extension_files(candidate, DeepAgentsAdapter().extension_spec()), result.load_report)


# --- a normal run ---------------------------------------------------------------------------


def test_a_run_that_writes_solution_py_in_the_workdir_completes(tmp_path):
    turns = [
        turn("", call("write_file", file_path="/solution.py", content="def add(a, b):\n    return a + b\n")),
        turn("", call("execute", command="python3 -c 'import solution; print(solution.add(2, 3))'")),
        turn("Implemented add."),
    ]
    result, workdir, trace_dir, _ = trial(tmp_path, turns)
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert result.final_text == "Implemented add."
    assert (workdir / "solution.py").read_text() == "def add(a, b):\n    return a + b\n"
    tool_ends = [line for line in trace_lines(trace_dir, "tool_end") if line["name"] == "execute"]
    assert "5" in tool_ends[0]["output"]


def test_trace_has_one_usage_line_per_model_call_with_the_reported_model(tmp_path):
    turns = [turn("", call("ls", path="/"), input_tokens=120, output_tokens=7, cache_read=20), turn("done", input_tokens=200)]
    result, _, trace_dir, _ = trial(tmp_path, turns)
    calls = trace_lines(trace_dir, "model_call")
    assert [line["model"] for line in calls] == [REPORTED, REPORTED]
    # input_tokens excludes cache reads (the provider's count includes them).
    assert calls[0]["usage"] == {"input_tokens": 100, "output_tokens": 7, "cache_read_tokens": 20, "cache_write_tokens": 0}
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (300, 17, 20)
    assert result.usage.cost_usd == 0.0  # dollars come from the gateway's ledger, never from the trace


def test_shell_commands_start_in_the_workdir(tmp_path):
    result, workdir, trace_dir, _ = trial(tmp_path, [turn("", call("execute", command="pwd > where.txt")), turn("ok")])
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert Path((workdir / "where.txt").read_text().strip()).resolve() == workdir.resolve()


def test_file_tools_accept_the_full_workdir_path(tmp_path):
    def turns_for(workdir: Path):
        return [turn("", call("write_file", file_path=f"{workdir.resolve()}/full.txt", content="x")), turn("ok")]

    # The workdir is known before the trial: trial() creates tmp_path / "workdir".
    result, workdir, trace_dir, _ = trial(tmp_path, turns_for(tmp_path / "workdir"))
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert (workdir / "full.txt").read_text() == "x"


def test_the_system_prompt_and_workdir_note_reach_the_model(tmp_path):
    result, workdir, trace_dir, _ = trial(tmp_path, [turn("ok")], files={"SYSTEM.md": "You are terse."})
    request = trace_lines(trace_dir, "model_request")[0]
    assert request["system_prompt"].startswith("You are terse.")
    assert str(workdir.resolve()) in request["system_prompt"]


def test_skills_are_listed_in_the_system_prompt_from_a_copy(tmp_path):
    files = {"skills/plan-first/SKILL.md": "---\nname: plan-first\ndescription: Use when planning work.\n---\nPlan.\n"}
    turns = [turn("", call("read_file", file_path="/skills/plan-first/SKILL.md")), turn("ok")]
    result, _, trace_dir, candidate = trial(tmp_path, turns, files=files)
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert "plan-first" in trace_lines(trace_dir, "model_request")[0]["system_prompt"]
    assert "Plan." in [line for line in trace_lines(trace_dir, "tool_end") if line["name"] == "read_file"][0]["output"]
    assert str(candidate.root) not in trace_lines(trace_dir, "model_request")[0]["system_prompt"]


# --- tools valve and subagent ---------------------------------------------------------------


def test_the_tools_valve_limits_what_the_model_sees(tmp_path):
    result, _, trace_dir, _ = trial(tmp_path, [turn("ok")], valves='tools = ["read_file", "write_file"]\n')
    assert trace_lines(trace_dir, "model_request")[0]["tools"] == ["read_file", "write_file"]


def test_a_call_to_a_tool_outside_the_valve_is_refused(tmp_path):
    turns = [turn("", call("execute", command="touch ran.txt")), turn("ok")]
    result, workdir, trace_dir, _ = trial(tmp_path, turns, valves='tools = ["read_file"]\n')
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert not (workdir / "ran.txt").exists()


def test_the_general_purpose_subagent_is_off_by_default(tmp_path):
    _, _, trace_dir, _ = trial(tmp_path, [turn("ok")])
    assert "task" not in trace_lines(trace_dir, "model_request")[0]["tools"]


def test_the_general_purpose_subagent_valve_adds_the_task_tool(tmp_path):
    valves = 'general_purpose_subagent = true\ntools = ["ls", "read_file", "task"]\n'
    turns = [turn("", call("task", description="List the files.", subagent_type="general-purpose")),
             turn("Two files."), turn("The subagent says two files.")]  # fmt: skip
    result, _, trace_dir, _ = trial(tmp_path, turns, valves=valves)
    assert result.status == "completed", (result.error, stderr(trace_dir))
    requests = trace_lines(trace_dir, "model_request")
    assert "task" in requests[0]["tools"]
    # The subagent's calls are traced and paid for too (callbacks reach subagents).
    assert len(trace_lines(trace_dir, "model_call")) == 3
    assert {line["agent"] for line in requests} >= {"general-purpose"}
    assert "task" not in [r for r in requests if r["agent"] == "general-purpose"][0]["tools"]


# --- extensions -----------------------------------------------------------------------------

FIRING_EXTENSION = '''
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool


class Marker(AgentMiddleware):
    def before_agent(self, state, runtime):
        Path("fired.txt").write_text("before_agent")
        return None


@tool
def shout(text: str) -> str:
    """Return the text in capitals."""
    return text.upper()


def extension():
    return [Marker(), shout]
'''


def test_a_good_extension_loads_fires_and_reports_what_it_registered(tmp_path):
    valves = 'tools = ["read_file", "shout"]\n'
    turns = [turn("", call("shout", text="hi")), turn("HI")]
    result, workdir, trace_dir, candidate = trial(tmp_path, turns, valves=valves, files={"extensions/marker.py": FIRING_EXTENSION})
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert (workdir / "fired.txt").read_text() == "before_agent"
    [load] = result.load_report.extensions
    assert (load.path, load.loaded, load.error) == ("extensions/marker.py", True, None)
    assert set(load.registered) == {"before_run", "register_tool", "tool:shout"}
    assert [line["output"] for line in trace_lines(trace_dir, "tool_end") if line["name"] == "shout"] == ["HI"]
    assert checked(result, candidate) is None


def test_the_documented_example_extension_loads_and_runs(tmp_path):
    api = (REPO_ROOT / "hive/agents/deepagents/EXTENSIONS_API.md").read_text()
    example = re.search(r"```python\n(.*?)```", api, re.DOTALL).group(1)
    valves = 'tools = ["read_file", "execute", "word_count"]\n'
    turns = [turn("", call("word_count", text="a b c")), turn("", call("execute", command="rm -rf nothing && touch ran.txt")),
             turn("3 words.")]  # fmt: skip
    result, workdir, trace_dir, candidate = trial(tmp_path, turns, valves=valves, files={"extensions/careful.py": example})
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert not (workdir / "ran.txt").exists()
    [load] = result.load_report.extensions
    assert set(load.registered) == {"wrap_model_call", "before_tool_call", "after_tool_call", "register_tool", "tool:word_count"}
    assert "run your code once" in trace_lines(trace_dir, "model_request")[0]["system_prompt"]


def test_an_extension_that_fails_to_import_is_a_harness_load_error_and_every_file_is_reported(tmp_path):
    files = {"extensions/a_broken.py": "raise RuntimeError('import-time failure')\n\ndef extension():\n    return []\n",
             "extensions/b_good.py": "def extension():\n    return []\n"}  # fmt: skip
    result, _, trace_dir, candidate = trial(tmp_path, [turn("ok")], files=files)
    assert (result.status, result.error_class) == ("errored", "harness_load_error")
    loads = {load.path: load for load in result.load_report.extensions}
    assert loads["extensions/a_broken.py"].loaded is False
    assert "RuntimeError: import-time failure" in loads["extensions/a_broken.py"].error
    assert loads["extensions/b_good.py"].loaded is True
    assert trace_lines(trace_dir, "model_call") == []  # the run never reached the model
    assert checked(result, candidate)[0] == "harness_load_error"


def test_an_extension_returning_the_wrong_type_fails_to_load(tmp_path):
    files = {"extensions/wrong.py": "def extension():\n    return ['not middleware']\n"}
    result, _, _, _ = trial(tmp_path, [turn("ok")], files=files)
    assert result.error_class == "harness_load_error"
    assert "AgentMiddleware" in result.load_report.extensions[0].error


def test_an_extension_whose_entry_function_raises_fails_to_load(tmp_path):
    files = {"extensions/raises.py": "def extension():\n    raise KeyError('missing')\n"}
    result, _, _, _ = trial(tmp_path, [turn("ok")], files=files)
    assert result.error_class == "harness_load_error"
    assert "KeyError" in result.load_report.extensions[0].error


def test_an_async_only_hook_fails_to_load(tmp_path):
    source = (
        "from langchain.agents.middleware import AgentMiddleware\n\n"
        "class Late(AgentMiddleware):\n    async def abefore_model(self, state, runtime):\n        return None\n\n"
        "def extension():\n    return [Late()]\n"
    )
    result, _, _, _ = trial(tmp_path, [turn("ok")], files={"extensions/late.py": source})
    assert result.error_class == "harness_load_error"
    assert "abefore_model" in result.load_report.extensions[0].error


def test_an_extension_tool_missing_from_the_tools_valve_fails_to_load(tmp_path):
    result, _, _, _ = trial(tmp_path, [turn("ok")], files={"extensions/marker.py": FIRING_EXTENSION})
    assert result.error_class == "harness_load_error"
    assert "shout" in result.load_report.extensions[0].error and "tools" in result.load_report.extensions[0].error


def test_an_extension_tool_reusing_a_built_in_name_fails_to_load(tmp_path):
    source = (
        "from langchain_core.tools import tool\n\n"
        "@tool\ndef execute(command: str) -> str:\n    \"\"\"Pretend to run a command.\"\"\"\n    return 'ok'\n\n"
        "def extension():\n    return [execute]\n"
    )
    result, _, _, _ = trial(tmp_path, [turn("ok")], files={"extensions/shadow.py": source})
    assert result.error_class == "harness_load_error"
    assert "built-in" in result.load_report.extensions[0].error


def test_duplicate_middleware_names_are_a_harness_load_error(tmp_path):
    source = (
        "from langchain.agents.middleware import AgentMiddleware\n\n"
        "class Same(AgentMiddleware):\n    def before_model(self, state, runtime):\n        return None\n\n"
        "def extension():\n    return [Same()]\n"
    )
    files = {"extensions/one.py": source, "extensions/two.py": source}
    result, _, _, _ = trial(tmp_path, [turn("ok")], files=files)
    assert result.error_class == "harness_load_error"
    assert "duplicate" in result.error.lower()


def test_extension_code_that_throws_mid_run_is_an_extension_error(tmp_path):
    source = (
        "from langchain.agents.middleware import AgentMiddleware\n\n"
        "class Boom(AgentMiddleware):\n    def before_model(self, state, runtime):\n        raise ValueError('boom')\n\n"
        "def extension():\n    return [Boom()]\n"
    )
    result, _, trace_dir, candidate = trial(tmp_path, [turn("ok")], files={"extensions/boom.py": source})
    assert (result.status, result.error_class) == ("errored", "extension_error"), (result.error, stderr(trace_dir))
    [runtime_error] = result.load_report.runtime_errors
    assert runtime_error.startswith("extensions/boom.py:") and "ValueError: boom" in runtime_error
    assert result.load_report.extensions[0].loaded is True
    assert checked(result, candidate)[0] == "extension_error"


def test_a_tool_hook_that_throws_is_an_extension_error(tmp_path):
    source = (
        "from langchain.agents.middleware import wrap_tool_call\n\n"
        "@wrap_tool_call\ndef gate(request, handler):\n    raise PermissionError('gate bug')\n\n"
        "def extension():\n    return [gate]\n"
    )
    result, _, trace_dir, _ = trial(tmp_path, [turn("", call("ls", path="/")), turn("ok")], files={"extensions/gate.py": source})
    assert result.error_class == "extension_error", (result.error, stderr(trace_dir))
    assert "PermissionError: gate bug" in result.load_report.runtime_errors[0]


def test_an_extension_raising_keyboard_interrupt_at_load_fails_to_load(tmp_path):
    files = {"extensions/interrupt.py": "def extension():\n    raise KeyboardInterrupt()\n"}
    result, _, trace_dir, candidate = trial(tmp_path, [turn("ok")], files=files)
    assert (result.status, result.error_class) == ("errored", "harness_load_error"), (result.error, stderr(trace_dir))
    [load] = result.load_report.extensions
    assert load.loaded is False and "KeyboardInterrupt" in load.error
    assert checked(result, candidate)[0] == "harness_load_error"


def test_an_extension_raising_keyboard_interrupt_mid_run_is_an_extension_error(tmp_path):
    source = (
        "from langchain.agents.middleware import AgentMiddleware\n\n"
        "class Stop(AgentMiddleware):\n    def before_model(self, state, runtime):\n        raise KeyboardInterrupt()\n\n"
        "def extension():\n    return [Stop()]\n"
    )
    result, _, trace_dir, candidate = trial(tmp_path, [turn("ok")], files={"extensions/stop.py": source})
    assert (result.status, result.error_class) == ("errored", "extension_error"), (result.error, stderr(trace_dir))
    [runtime_error] = result.load_report.runtime_errors
    assert runtime_error.startswith("extensions/stop.py:") and "KeyboardInterrupt" in runtime_error
    assert checked(result, candidate)[0] == "extension_error"


TAIL_SLOT_EXTENSION = '''
from langchain.agents.middleware import AgentMiddleware


class Inner(AgentMiddleware):
    @property
    def name(self):
        return "{name}"

    def wrap_tool_call(self, request, handler):
        raise ValueError("hook bug")


def extension():
    return [Inner()]
'''


@pytest.mark.parametrize("name", ["UnsupportedContentMiddleware", "FilesystemMiddleware", "HiveToolErrors"])
def test_an_extension_middleware_named_like_a_stack_entry_or_guard_fails_to_load(tmp_path, name):
    files = {"extensions/slot.py": TAIL_SLOT_EXTENSION.replace("{name}", name)}
    result, _, trace_dir, candidate = trial(tmp_path, [turn("", call("ls", path="/")), turn("done")], files=files)
    assert (result.status, result.error_class) == ("errored", "harness_load_error"), (result.error, stderr(trace_dir))
    [load] = result.load_report.extensions
    assert load.loaded is False and name in load.error
    assert trace_lines(trace_dir, "model_call") == []


def test_an_extension_may_take_the_summarization_slot(tmp_path):
    source = (
        "from langchain.agents.middleware import ContextEditingMiddleware\n\n"
        "class Trim(ContextEditingMiddleware):\n    @property\n    def name(self):\n"
        "        return 'SummarizationMiddleware'\n\n"
        "def extension():\n    return [Trim()]\n"
    )
    result, _, trace_dir, candidate = trial(tmp_path, [turn("ok")], files={"extensions/trim.py": source})
    assert result.status == "completed", (result.error, stderr(trace_dir))
    [load] = result.load_report.extensions
    assert load.loaded is True and "compaction" in load.registered


def test_two_extensions_cannot_both_take_the_summarization_slot(tmp_path):
    source = (
        "from langchain.agents.middleware import ContextEditingMiddleware\n\n"
        "class Trim(ContextEditingMiddleware):\n    @property\n    def name(self):\n"
        "        return 'SummarizationMiddleware'\n\n"
        "def extension():\n    return [Trim()]\n"
    )
    files = {"extensions/one.py": source, "extensions/two.py": source}
    result, _, trace_dir, _ = trial(tmp_path, [turn("ok")], files=files)
    assert result.error_class == "harness_load_error", (result.error, stderr(trace_dir))
    loads = {load.path: load for load in result.load_report.extensions}
    assert loads["extensions/one.py"].loaded is True
    assert loads["extensions/two.py"].loaded is False and "extensions/one.py" in loads["extensions/two.py"].error


def test_the_reserved_names_cover_every_middleware_deepagents_builds(tmp_path):
    # A pin check against the installed deepagents: every name in the stacks it hands to
    # create_agent (main agent and subagent, with and without skills) is reserved or a HIVE guard.
    script = '''
import importlib.util, json, sys, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location("runner", sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import deepagents.graph, deepagents.middleware.subagents
seen = set()
def capture(original):
    def create_agent(model, **kwargs):
        seen.update(m.name for m in kwargs["middleware"])
        return original(model, **kwargs)
    return create_agent
deepagents.graph.create_agent = capture(deepagents.graph.create_agent)
deepagents.middleware.subagents.create_agent = capture(deepagents.middleware.subagents.create_agent)
fw = runner.Framework()
model = runner.scripted_model_class(fw)(model_name="gpt-5-nano", reasoning={"effort": "low"}, turns=[{"text": "ok"}])
root = Path(tempfile.mkdtemp())
(root / "skills/x").mkdir(parents=True)
(root / "skills/x/SKILL.md").write_text("---\\nname: x\\ndescription: y\\n---\\n")
for subagent in (False, True):
    for skills in (None, str(root / "skills")):
        trial = runner.Trial({"workdir": str(root), "trace_file": str(root / "t.jsonl"), "extensions": [],
                              "tools": ["ls", "task"], "child_pgids_file": str(root / "p"), "skills_dir": skills,
                              "limits": {"shell_timeout_s": 5, "max_model_calls": 3, "recursion_limit": 10},
                              "general_purpose_subagent": subagent, "system_prompt": None, "workdir_note": "x"})
        trial.build_agent(fw, model, [], [])
        trial.cleanup()
print(json.dumps({"seen": sorted(seen), "stack": sorted(runner.DEEPAGENTS_STACK_NAMES),
                  "guards": sorted(runner.hive_middleware_classes(fw))}))
'''
    completed = subprocess.run([str(VENV_PYTHON), "-I", "-c", script, str(da.RUNNER)], capture_output=True, text=True,
                               timeout=120, cwd=tmp_path)  # fmt: skip
    assert completed.returncode == 0, completed.stderr
    names = json.loads(completed.stdout.strip().splitlines()[-1])
    assert set(names["seen"]) - set(names["guards"]) <= set(names["stack"])


def test_the_runner_writes_no_bytecode_into_the_candidate(tmp_path):
    result, _, trace_dir, candidate = trial(tmp_path, [turn("ok")], files={"extensions/marker.py": FIRING_EXTENSION},
                                            valves='tools = ["shout"]\n')  # fmt: skip
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert not list(candidate.root.rglob("__pycache__")) and not list(candidate.root.rglob("*.pyc"))


def test_a_runner_failure_before_the_load_phase_reports_no_extensions(tmp_path):
    files = {"extensions/marker.py": FIRING_EXTENSION}
    result, _, trace_dir, _ = trial(tmp_path, [], files=files, valves='tools = ["shout"]\n', turns_text="not json")
    assert (result.status, result.error_class) == ("errored", "agent_error"), (result.error, stderr(trace_dir))
    written = json.loads((trace_dir / "result.json").read_text())
    assert "extensions" not in written
    assert "JSONDecodeError" in written["error"]


def test_a_tool_that_raises_becomes_an_error_result_and_the_run_goes_on(tmp_path):
    source = (
        "from langchain_core.tools import tool\n\n"
        "@tool\ndef flaky(x: str) -> str:\n    \"\"\"Always fails.\"\"\"\n    raise RuntimeError('tool exploded')\n\n"
        "def extension():\n    return [flaky]\n"
    )
    turns = [turn("", call("flaky", x="1")), turn("It failed.")]
    result, _, trace_dir, _ = trial(tmp_path, turns, valves='tools = ["flaky"]\n', files={"extensions/flaky.py": source})
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert result.load_report.runtime_errors == ()
    flaky = [line for line in trace_lines(trace_dir) if line.get("name") == "flaky" and line["type"].startswith("tool_")]
    assert any("tool exploded" in json.dumps(line) for line in flaky)


# --- limits and failures --------------------------------------------------------------------


def test_hitting_the_model_call_limit_is_an_agent_error(tmp_path):
    result, _, trace_dir, _ = trial(tmp_path, [turn("", call("ls", path="/"))], max_model_calls=3)
    assert (result.status, result.error_class) == ("errored", "agent_error"), stderr(trace_dir)
    assert "model call limit" in result.error.lower()
    assert len(trace_lines(trace_dir, "model_call")) == 3


RETRYING_EXTENSION = '''
from langchain.agents.middleware import AgentMiddleware


class Retry(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        for _ in range(4):
            response = handler(request)
        return response


def extension():
    return [Retry()]
'''


def test_every_model_call_inside_wrap_model_call_counts_toward_the_limit(tmp_path):
    turns = [turn("", call("ls", path="/"))] * 7 + [turn("done")]
    result, _, trace_dir, _ = trial(tmp_path, turns, files={"extensions/retry.py": RETRYING_EXTENSION}, max_model_calls=2)
    assert (result.status, result.error_class) == ("errored", "agent_error"), (result.error, stderr(trace_dir))
    assert "model call limit (2)" in result.error
    assert len(trace_lines(trace_dir, "model_call")) == 2


def test_an_extension_that_swallows_the_model_call_limit_still_ends_errored(tmp_path):
    source = (
        "from langchain.agents.middleware import AgentMiddleware\n"
        "from langchain_core.messages import AIMessage\n\n"
        "class Swallow(AgentMiddleware):\n    def wrap_model_call(self, request, handler):\n"
        "        try:\n            return handler(request)\n"
        "        except Exception:\n            return AIMessage(content='done anyway')\n\n"
        "def extension():\n    return [Swallow()]\n"
    )
    result, _, trace_dir, _ = trial(tmp_path, [turn("", call("ls", path="/"))], files={"extensions/swallow.py": source},
                                    max_model_calls=2)  # fmt: skip
    assert (result.status, result.error_class) == ("errored", "agent_error"), (result.error, stderr(trace_dir))
    assert "model call limit (2)" in result.error
    assert len(trace_lines(trace_dir, "model_call")) == 2


def test_the_model_call_limit_is_shared_by_the_main_agent_and_its_subagents(tmp_path):
    valves = 'general_purpose_subagent = true\ntools = ["ls", "read_file", "task"]\n'
    tasks = [call("task", description=f"Look at file {index}.", subagent_type="general-purpose") for index in range(6)]
    turns = [turn("", *tasks)] + [turn("sub done")] * 6 + [turn("done")]
    result, _, trace_dir, _ = trial(tmp_path, turns, valves=valves, max_model_calls=2)
    assert (result.status, result.error_class) == ("errored", "agent_error"), (result.error, stderr(trace_dir))
    assert "model call limit (2)" in result.error
    assert len(trace_lines(trace_dir, "model_call")) == 2


def test_the_recursion_limit_reaches_the_subagent(tmp_path):
    valves = 'general_purpose_subagent = true\ntools = ["ls", "read_file", "task"]\n'
    turns = [turn("", call("task", description="Loop.", subagent_type="general-purpose")), turn("", call("ls", path="/"))]
    result, _, trace_dir, _ = trial(tmp_path, turns, valves=valves, recursion_limit=12)
    assert (result.status, result.error_class) == ("errored", "agent_error"), (result.error, stderr(trace_dir))
    assert "recursion limit: 12" in result.error
    assert len(trace_lines(trace_dir, "model_call")) < 12


def test_hitting_the_recursion_limit_is_an_agent_error(tmp_path):
    result, _, trace_dir, _ = trial(tmp_path, [turn("", call("ls", path="/"))], recursion_limit=6)
    assert (result.status, result.error_class) == ("errored", "agent_error"), stderr(trace_dir)
    assert "recursion limit" in result.error.lower()


def test_a_scripted_rate_limit_is_quota(tmp_path):
    result, _, _, _ = trial(tmp_path, [{"raise": "rate_limit", "message": "Rate limit reached"}])
    assert (result.status, result.error_class) == ("errored", "quota")


def test_insufficient_quota_is_quota(tmp_path):
    result, _, _, _ = trial(tmp_path, [{"raise": "insufficient_quota", "message": "You exceeded your current quota"}])
    assert result.error_class == "quota"


def test_another_provider_failure_is_a_provider_error(tmp_path):
    result, _, _, _ = trial(tmp_path, [{"raise": "provider", "message": "upstream 500"}])
    assert (result.status, result.error_class) == ("errored", "provider_error")


def test_a_reply_from_another_model_is_a_model_mismatch(tmp_path):
    result, _, _, _ = trial(tmp_path, [turn("ok", model="gpt-4o-2024-08-06")])
    assert result.error_class == "model_mismatch"


def test_a_request_with_another_reasoning_effort_is_a_model_mismatch(tmp_path):
    # `**{...}` dodges validate()'s static scan on purpose: the runner's request check must still catch it.
    source = (
        "from langchain.agents.middleware import wrap_model_call\n\n"
        "@wrap_model_call\ndef effort(request, handler):\n"
        "    return handler(request.override(**{'model_' + 'settings': {'reasoning': {'effort': 'high'}}}))\n\n"
        "def extension():\n    return [effort]\n"
    )
    result, _, trace_dir, _ = trial(tmp_path, [turn("ok")], files={"extensions/effort.py": source})
    assert (result.status, result.error_class) == ("errored", "model_mismatch"), (result.error, stderr(trace_dir))
    assert "high" in result.error
    assert trace_lines(trace_dir, "model_call") == []  # stopped before the request went out


def test_background_shell_children_are_listed_and_reaped(tmp_path):
    turns = [turn("", call("execute", command="sleep 300 & echo $! > sleeper.pid")), turn("ok")]
    started = time.monotonic()
    result, workdir, trace_dir, _ = trial(tmp_path, turns)
    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert time.monotonic() - started < 60  # execute returned without waiting for the background job
    assert (trace_dir / "child_pgids.txt").read_text().split()
    sleeper = int((workdir / "sleeper.pid").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(sleeper):
        time.sleep(0.05)
    try:
        assert not _alive(sleeper)
    finally:
        try:
            os.kill(sleeper, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _alive(pid: int) -> bool:
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


# --- the real OpenAI provider class, against a local stub -----------------------------------


class _StubOpenAI(BaseHTTPRequestHandler):
    """Answers POST /v1/responses like the Responses API; the test sets `reply` per server, or
    `replies`, a list answered in order before falling back to `reply`."""

    def do_POST(self):  # noqa: N802 (http.server's naming)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({"body": body, "authorization": self.headers.get("Authorization")})
        status, payload = self.server.replies.pop(0) if self.server.replies else self.server.reply
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


RESPONSE = {
    "id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": REPORTED,
    "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "ANSWER: 42", "annotations": []}]}],
    "usage": {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1000}, "output_tokens": 40,
              "output_tokens_details": {"reasoning_tokens": 32}, "total_tokens": 1240},
    "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
}  # fmt: skip
QUOTA = {"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota", "code": "insufficient_quota"}}


@pytest.fixture
def stub_openai():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubOpenAI)
    server.requests = []
    server.replies = []
    server.reply = (200, RESPONSE)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


PLACEHOLDER_KEY = "sk-offline-placeholder-not-a-key"
PLACEHOLDER_TOKEN = "hive-trial-token-placeholder-not-a-key"


def openai_trial(tmp_path: Path, server, files: dict[str, str] | None = None, access: str = "endpoint"):
    """The runner started directly (not through the adapter), with the stub as its model server.

    access "endpoint": the stub is the trial's endpoint, sent on stdin with a placeholder token,
    exactly as the adapter sends a gateway grant. access "legacy": the no-gateway path, a key on
    stdin and OPENAI_BASE_URL in the environment (the adapter itself never sets it).
    """
    candidate = write_candidate(tmp_path / "candidate", files=files)
    workdir, trace_dir = tmp_path / "workdir", tmp_path / "trace"
    workdir.mkdir()
    trace_dir.mkdir()
    layout = da.read_layout(candidate.root)
    spec = da.DeepAgentsAgent(
        python_path=str(VENV_PYTHON), candidate_root=candidate.root, system_prompt=None,
        extensions=tuple((p.relative_to(candidate.root).as_posix(), str(p)) for p in layout.extensions),
        skills_dir=None, tools=da.DEFAULT_TOOLS, general_purpose_subagent=False,
        model=ModelSpec(provider="openai", id="gpt-5-nano", thinking="low"), reasoning_effort="low",
        limits={"max_model_calls": 10, "recursion_limit": 100, "shell_timeout_s": 30, "provider_timeout_s": 30,
                "max_retries": 0},
        timeout_s=60, term_grace_s=1, scripted_turns_file=None, workdir_prompt="Workdir: {workdir}",
    ).run_spec(Task(id="t", split="search", instruction="What is 6 * 7?"), workdir.resolve(), trace_dir)  # fmt: skip
    (trace_dir / "run_spec.json").write_text(json.dumps(spec))
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    env = {"PATH": os.environ["PATH"], "HOME": str(trace_dir), "LANG": "C.UTF-8"}
    if access == "endpoint":
        stdin = json.dumps({"endpoint": {"base_url": base_url, "api_key": PLACEHOLDER_TOKEN}}) + "\n"
    else:
        env["OPENAI_BASE_URL"] = base_url
        stdin = json.dumps({"credentials": {"OPENAI_API_KEY": PLACEHOLDER_KEY}}) + "\n"
    completed = subprocess.run([str(VENV_PYTHON), "-I", str(da.RUNNER), str(trace_dir / "run_spec.json")], cwd=workdir,
                               env=env, input=stdin, text=True, timeout=120, capture_output=True)  # fmt: skip
    (trace_dir / "runner.stderr").write_text(completed.stderr)
    return json.loads((trace_dir / "result.json").read_text()), trace_dir


@pytest.mark.parametrize("access, secret", [("endpoint", PLACEHOLDER_TOKEN), ("legacy", PLACEHOLDER_KEY)])
def test_openai_usage_and_reported_model_are_traced(tmp_path, stub_openai, access, secret):
    result, trace_dir = openai_trial(tmp_path, stub_openai, access=access)
    assert result["status"] == "completed" and result["final_text"] == "ANSWER: 42"
    [line] = trace_lines(trace_dir, "model_call")
    assert line["model"] == REPORTED
    assert line["usage"] == {"input_tokens": 200, "output_tokens": 40, "cache_read_tokens": 1000, "cache_write_tokens": 0}
    request = stub_openai.requests[0]
    assert request["body"]["model"] == "gpt-5-nano" and request["body"]["reasoning"] == {"effort": "low"}
    assert request["authorization"] == f"Bearer {secret}"


def test_openai_insufficient_quota_is_quota(tmp_path, stub_openai):
    stub_openai.reply = (429, QUOTA)
    result, _ = openai_trial(tmp_path, stub_openai)
    assert (result["status"], result["error_class"]) == ("errored", "quota")


@pytest.mark.parametrize("stdin", [
    {"endpoint": {"base_url": "http://127.0.0.1:9/v1"}},
    {"endpoint": {"base_url": "file:///etc/passwd", "api_key": PLACEHOLDER_TOKEN}},
    {"endpoint": {"base_url": "http://127.0.0.1:9/v1", "api_key": PLACEHOLDER_TOKEN}, "credentials": {}},
])  # fmt: skip
def test_a_malformed_endpoint_on_stdin_fails_before_any_model_call(tmp_path, stub_openai, stdin):
    spec_path = tmp_path / "run_spec.json"
    trial_spec = {"result_file": str(tmp_path / "result.json"), "trace_file": str(tmp_path / "trace.jsonl")}
    spec_path.write_text(json.dumps(trial_spec))
    completed = subprocess.run([str(VENV_PYTHON), "-I", str(da.RUNNER), str(spec_path)], cwd=tmp_path, input=json.dumps(stdin),
                               env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)}, text=True, timeout=60,
                               capture_output=True)  # fmt: skip
    result = json.loads((tmp_path / "result.json").read_text())
    assert (result["status"], result["error_class"]) == ("errored", "agent_error"), completed.stderr
    assert "endpoint" in result["error"] and PLACEHOLDER_TOKEN not in result["error"]
    assert stub_openai.requests == []


@pytest.mark.parametrize("access", ["endpoint", "legacy"])
def test_extensions_cannot_read_the_provider_key_from_the_environment(tmp_path, stub_openai, access):
    source = "import os\n\ndef extension():\n    open('key_seen.txt', 'w').write(os.environ.get('OPENAI_API_KEY', 'absent'))\n    return []\n"
    result, _ = openai_trial(tmp_path, stub_openai, files={"extensions/peek.py": source}, access=access)
    assert result["status"] == "completed"
    assert (tmp_path / "workdir" / "key_seen.txt").read_text() == "absent"


@pytest.mark.parametrize("access, secret", [("endpoint", PLACEHOLDER_TOKEN), ("legacy", PLACEHOLDER_KEY)])
def test_the_agent_shell_cannot_read_the_provider_key(tmp_path, stub_openai, access, secret):
    # `ps eww -p $PPID` prints the runner's exec-time environment, which os.environ.pop never clears.
    command = "env; ps eww -o command= -p $PPID"
    # Its own response id: messages sharing an id replace each other in the agent's state.
    execute = {**RESPONSE, "id": "resp_0", "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "execute",
                                       "status": "completed", "arguments": json.dumps({"command": command})}]}  # fmt: skip
    stub_openai.replies = [(200, execute)]
    result, trace_dir = openai_trial(tmp_path, stub_openai, access=access)
    assert result["status"] == "completed", (result, stderr(trace_dir))
    [tool_end] = [line for line in trace_lines(trace_dir, "tool_end") if line["name"] == "execute"]
    assert "PATH=" in tool_end["output"] and "runner.py" in tool_end["output"]  # both commands ran
    assert stub_openai.requests[0]["authorization"] == f"Bearer {secret}"  # the secret did reach the model server
    leaked = [str(path) for path in trace_dir.rglob("*") if path.is_file() and secret in path.read_text(errors="replace")]
    assert leaked == []


# --- the real gateway in front of the fake OpenAI upstream ------------------------------------

GATEWAY_MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
GATEWAY_PRICES = Prices(input=0.05, output=0.40, cache_read=0.005)
# A made-up stand-in for the provider key, held only by the gateway. Never a real key.
UPSTREAM_KEY = "sk-fake-upstream-key-for-deepagents-tests-0123456789"


def gateway_trial(tmp_path: Path, gateway: ModelGateway, endpoint: ModelEndpoint):
    """One trial through the adapter, as the controller runs it: endpoint passed as a keyword."""
    candidate = write_candidate(tmp_path / "candidate")
    agent = DeepAgentsAdapter().build(candidate, GATEWAY_MODEL, {"python_path": str(VENV_PYTHON), "timeout_s": 120,
                                                                 "max_retries": 0})  # fmt: skip
    workdir, trace_dir = tmp_path / "workdir", tmp_path / "trace"
    workdir.mkdir()
    trace_dir.mkdir()
    result = agent.run(Task(id="t1", split="search", instruction="Say hello."), workdir, trace_dir, endpoint=endpoint)
    return result, trace_dir


def test_a_real_run_goes_through_the_gateway_which_records_its_calls_and_usage(tmp_path, monkeypatch):
    # The controller's environment has no provider key at all: only the gateway holds one.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with FakeOpenAIUpstream() as upstream, ModelGateway(model=GATEWAY_MODEL, prices=GATEWAY_PRICES, upstream_url=upstream.url,
                                                        upstream_key=UPSTREAM_KEY, max_calls_per_trial=10) as gateway:  # fmt: skip
        trial_key, endpoint = gateway.open_trial()
        result, trace_dir = gateway_trial(tmp_path, gateway, endpoint)
        ledger = gateway.close_trial(trial_key)

    assert result.status == "completed", (result.error, stderr(trace_dir))
    assert result.final_text == "Hello world"

    # The ledger, the authority for usage and cost (P12) and for model violations (P3).
    assert ledger.violations == () and not ledger.limit_reached
    [call_record] = ledger.calls
    assert (call_record.path, call_record.status) == ("/v1/responses", 200)
    assert call_record.model_requested == "gpt-5-nano"
    assert call_record.model_reported == "gpt-5-nano" + SNAPSHOT_SUFFIX
    cached = RESPONSES_USAGE["input_tokens_details"]["cached_tokens"]
    tokens = Usage(input_tokens=RESPONSES_USAGE["input_tokens"] - cached, output_tokens=RESPONSES_USAGE["output_tokens"],
                   cache_read_tokens=cached)  # fmt: skip
    assert call_record.usage == dataclasses.replace(tokens, cost_usd=GATEWAY_PRICES.cost(tokens))

    # The upstream saw the provider key and the frozen effort; the trial saw only its token.
    [request] = upstream.requests
    assert request.headers["authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert request.json["model"] == "gpt-5-nano" and request.json["reasoning"]["effort"] == "low"

    # The runner's own trace agrees, as the second line: same tokens, no dollars.
    [line] = trace_lines(trace_dir, "model_call")
    assert line["model"] == call_record.model_reported
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (
        tokens.input_tokens, tokens.output_tokens, tokens.cache_read_tokens)
    assert result.usage.cost_usd == 0.0

    # Neither the trial token nor the provider key is written anywhere under the trace dir.
    for secret in (endpoint.api_key, UPSTREAM_KEY):
        leaked = [str(path) for path in trace_dir.rglob("*") if path.is_file() and secret in path.read_text(errors="replace")]
        assert leaked == [], secret


def test_a_revoked_token_gets_no_model_call_through(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with FakeOpenAIUpstream() as upstream, ModelGateway(model=GATEWAY_MODEL, prices=GATEWAY_PRICES, upstream_url=upstream.url,
                                                        upstream_key=UPSTREAM_KEY, max_calls_per_trial=10) as gateway:  # fmt: skip
        trial_key, endpoint = gateway.open_trial()
        gateway.close_trial(trial_key)
        result, trace_dir = gateway_trial(tmp_path, gateway, endpoint)
    assert (result.status, result.error_class) == ("errored", "provider_error"), (result.error, stderr(trace_dir))
    assert upstream.requests == []
