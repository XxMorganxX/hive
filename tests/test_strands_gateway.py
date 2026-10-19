"""Layer 2 for the Strands adapter behind the model gateway: the real runner, the real OpenAI
SDK in the Strands venv, a real ModelGateway, and tests/fakes/fake_openai_upstream.py as the
provider. Offline: the gateway forwards to the fake on 127.0.0.1 with a placeholder key.

These pin the integration the spec asks for (tasks/framework-adapters-and-extensions.md,
"Model gateway (spec)"): every model request of a Strands trial reaches the gateway with the
per-trial token, the gateway's ledger bills it, the provider key never enters the trial, and
the gateway refuses a model or effort the experiment did not freeze. The runner's in-process
recording stays as a second line and must agree with the ledger.

Skipped unless scripts/setup-framework-venvs.sh strands has built .venvs/strands.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from fakes.fake_openai_upstream import CHAT_USAGE, OTHER_MODEL, REPLY_TEXT, SNAPSHOT_SUFFIX, FakeOpenAIUpstream
from hive.agents.strands import StrandsAdapter
from hive.candidate import Candidate
from hive.contracts import AgentResult, ModelSpec, Task, Usage
from hive.gateway import ModelGateway, TrialLedger
from hive.pricing import Prices

REPO_ROOT = Path(__file__).resolve().parents[1]
STRANDS_PYTHON = REPO_ROOT / ".venvs" / "strands" / "bin" / "python"
MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
PRICES = Prices(input=0.05, output=0.40, cache_read=0.005)
# Made-up stand-ins. Never a real key.
UPSTREAM_KEY = "sk-fake-upstream-key-for-strands-tests-0123456789"
CONTROLLER_KEY = "sk-PLACEHOLDER-controller-key-strands-4701"
TASK = Task(id="t1", split="search", instruction="Say hello.")

pytestmark = pytest.mark.skipif(not STRANDS_PYTHON.exists(), reason="the Strands venv is not built (.venvs/strands)")


@pytest.fixture(autouse=True)
def controller_key(monkeypatch):
    """The controller holds a provider key, as in a real run; only the gateway may ever use it."""
    monkeypatch.setenv("OPENAI_API_KEY", CONTROLLER_KEY)


def write_candidate(root: Path) -> Candidate:
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text('schema_version = 1\nagent = "strands"\ndescription = "test"\n\n[valves]\n')
    return Candidate.load(root)


@contextmanager
def gateway_in_front_of(upstream: FakeOpenAIUpstream, model: ModelSpec = MODEL) -> Iterator[ModelGateway]:
    with ModelGateway(model=model, prices=PRICES, upstream_url=upstream.url, upstream_key=UPSTREAM_KEY,
                      max_calls_per_trial=10) as gateway:  # fmt: skip
        yield gateway


@dataclass(frozen=True)
class Trial:
    result: AgentResult
    ledger: TrialLedger
    trace_dir: Path
    token: str


def run_trial(tmp_path: Path, gateway: ModelGateway, agent_model: ModelSpec = MODEL) -> Trial:
    """One trial the way the controller runs it: open a grant, run with its endpoint, close it."""
    agent = StrandsAdapter().build(write_candidate(tmp_path / "candidate"), agent_model,
                                   {"python_path": str(STRANDS_PYTHON), "strands_version": "1.61.1", "timeout_s": 60,
                                    "max_model_attempts": 1})  # fmt: skip
    workdir, trace_dir = tmp_path / "work", tmp_path / "trace"
    workdir.mkdir()
    trace_dir.mkdir()
    trial_key, endpoint = gateway.open_trial()
    try:
        result = agent.run(TASK, workdir, trace_dir, endpoint=endpoint)
    finally:
        ledger = gateway.close_trial(trial_key)
    return Trial(result=result, ledger=ledger, trace_dir=trace_dir, token=endpoint.api_key)


def secrets_under(trace_dir: Path, token: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, secret in (("upstream_key", UPSTREAM_KEY), ("controller_key", CONTROLLER_KEY), ("trial_token", token)):
        hits = [str(path.relative_to(trace_dir)) for path in trace_dir.rglob("*")
                if path.is_file() and secret in path.read_text(errors="replace")]  # fmt: skip
        if hits:
            found[name] = hits
    return found


def traced_calls(trace_dir: Path) -> list[dict]:
    lines = [json.loads(line) for line in (trace_dir / "trace.jsonl").read_text().splitlines() if line.strip()]
    return [line for line in lines if line.get("type") == "model_call"]


def test_a_strands_run_goes_through_the_gateway_and_the_ledger_bills_it(tmp_path):
    with FakeOpenAIUpstream() as upstream, gateway_in_front_of(upstream) as gateway:
        trial = run_trial(tmp_path, gateway)
    result, ledger = trial.result, trial.ledger

    assert result.status == "completed", result.error
    assert result.final_text == REPLY_TEXT
    assert ledger.violations == () and not ledger.limit_reached
    [call] = ledger.calls
    assert (call.path, call.status, call.streamed) == ("/v1/chat/completions", 200, True)
    assert call.model_reported == MODEL.id + SNAPSHOT_SUFFIX
    cached = CHAT_USAGE["prompt_tokens_details"]["cached_tokens"]
    tokens = Usage(input_tokens=CHAT_USAGE["prompt_tokens"] - cached, output_tokens=CHAT_USAGE["completion_tokens"],
                   cache_read_tokens=cached)  # fmt: skip
    assert ledger.usage == Usage(input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens,
                                 cache_read_tokens=cached, cost_usd=PRICES.cost(tokens))  # fmt: skip

    [request] = upstream.requests
    assert request.headers["authorization"] == f"Bearer {UPSTREAM_KEY}"  # the gateway's key, not the trial's
    assert request.json["model"] == MODEL.id
    assert request.json["reasoning_effort"] == "low"


def test_the_in_process_record_agrees_with_the_ledger(tmp_path):
    with FakeOpenAIUpstream() as upstream, gateway_in_front_of(upstream) as gateway:
        trial = run_trial(tmp_path, gateway)
    result, ledger, trace_dir = trial.result, trial.ledger, trial.trace_dir

    [traced] = traced_calls(trace_dir)
    assert traced["model"] == ledger.calls[0].model_reported
    usage = ledger.usage
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens)  # fmt: skip
    assert result.usage.cost_usd == 0.0  # dollars are the ledger's alone


def test_no_key_or_token_is_written_under_the_trace_dir(tmp_path):
    with FakeOpenAIUpstream() as upstream, gateway_in_front_of(upstream) as gateway:
        trial = run_trial(tmp_path, gateway)
    result, trace_dir = trial.result, trial.trace_dir
    assert result.status == "completed", result.error
    assert secrets_under(trace_dir, trial.token) == {}


def test_a_provider_reporting_another_model_is_caught_by_both_lines(tmp_path):
    with FakeOpenAIUpstream(scenario="other_model") as upstream, gateway_in_front_of(upstream) as gateway:
        trial = run_trial(tmp_path, gateway)
    result, ledger = trial.result, trial.ledger
    assert ledger.violations, "the gateway records the reported-model mismatch"
    assert ledger.calls[0].model_reported == OTHER_MODEL
    assert result.error_class == "model_mismatch"


@pytest.mark.parametrize(
    "agent_model",
    [ModelSpec(provider="openai", id="gpt-5-mini", thinking="low"),
     ModelSpec(provider="openai", id="gpt-5-nano", thinking="high")],
    ids=["other_model", "other_effort"],
)  # fmt: skip
def test_the_gateway_refuses_a_model_or_effort_the_experiment_did_not_freeze(tmp_path, agent_model):
    # The trial asks for something else than the gateway's model: whatever the trial process
    # does, the request is refused at the gateway and never reaches the provider.
    with FakeOpenAIUpstream() as upstream, gateway_in_front_of(upstream) as gateway:
        trial = run_trial(tmp_path, gateway, agent_model=agent_model)
    result, ledger = trial.result, trial.ledger
    assert upstream.requests == []
    assert ledger.violations
    assert [call.status for call in ledger.calls] == [403]
    assert result.status == "errored"


def test_an_out_of_credit_provider_is_quota_and_the_ledger_has_its_code(tmp_path):
    with FakeOpenAIUpstream(scenario="quota") as upstream, gateway_in_front_of(upstream) as gateway:
        trial = run_trial(tmp_path, gateway)
    result, ledger = trial.result, trial.ledger
    assert result.error_class == "quota", result.error
    assert [call.error_code for call in ledger.calls] == ["insufficient_quota"]  # not retried
