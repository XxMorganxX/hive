"""The model gateway: per-trial grants, P3 enforcement, the call limit, forwarding and billing.

Everything runs offline against tests/fakes/fake_openai_upstream.py. The last section drives
the real OpenAI SDKs (2.x in the Strands venv, 3.x and langchain-openai in the Deep Agents
venv) at the gateway in a subprocess; those tests skip when a venv is missing.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from fakes.fake_openai_upstream import CHAT_USAGE, OTHER_MODEL, REPLY_TEXT, RESPONSES_USAGE, FakeOpenAIUpstream
from hive.contracts import ModelEndpoint, ModelSpec, Usage
from hive.gateway import GatewayCall, ModelGateway, TrialLedger
from hive.pricing import Prices, parse_prices

MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
PRICES = Prices(input=0.05, output=0.40, cache_read=0.005)
# A made-up stand-in for the provider key. Never a real key.
UPSTREAM_KEY = "sk-fake-upstream-key-for-tests-0123456789abcdef"
CHAT = "/v1/chat/completions"
RESPONSES = "/v1/responses"
PROJECT = Path(__file__).resolve().parent.parent


def priced(input_tokens: int, output_tokens: int, cache_read: int) -> Usage:
    tokens = Usage(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read)
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read,
                 cost_usd=PRICES.cost(tokens))  # fmt: skip


# Normalised from the fake's canned usage: input excludes cache reads.
CHAT_BILL = priced(1000 - 200, 300, 200)
RESPONSES_BILL = priced(1200 - 400, 500, 400)


# --- harness -----------------------------------------------------------------------------------


@pytest.fixture
def upstream():
    with FakeOpenAIUpstream() as fake:
        yield fake


@contextmanager
def make_gateway(upstream_url: str, *, model: ModelSpec = MODEL, max_calls: int = 50, **options: Any):
    with ModelGateway(model=model, prices=PRICES, upstream_url=upstream_url, upstream_key=UPSTREAM_KEY,
                      max_calls_per_trial=max_calls, **options) as gateway:  # fmt: skip
        yield gateway


@pytest.fixture
def gateway(upstream):
    with make_gateway(upstream.url) as gw:
        yield gw


def chat_body(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL.id, "messages": [{"role": "user", "content": "hi"}], **extra}


def responses_body(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL.id, "input": "hi", **extra}


def connect(endpoint: ModelEndpoint, timeout: float = 10) -> http.client.HTTPConnection:
    parts = urlsplit(endpoint.base_url)
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)


def send(
    endpoint: ModelEndpoint,
    path: str,
    body: Any = None,
    *,
    method: str = "POST",
    token: str | None = "trial",
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One request to the gateway. `token="trial"` means the endpoint's own token."""
    all_headers = {"Content-Type": "application/json"}
    if token is not None:
        all_headers["Authorization"] = f"Bearer {endpoint.api_key if token == 'trial' else token}"
    all_headers.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    connection = connect(endpoint)
    try:
        connection.request(method, path, body=data, headers=all_headers)
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        connection.close()


def sse_data(raw: bytes) -> list[Any]:
    """The JSON payloads of an SSE body, in order ([DONE] kept as the string)."""
    out = []
    for line in raw.decode().splitlines():
        if line.startswith("data: "):
            text = line[len("data: ") :]
            out.append(text if text == "[DONE]" else json.loads(text))
    return out


def only_call(ledger: TrialLedger) -> GatewayCall:
    assert len(ledger.calls) == 1, ledger.calls
    return ledger.calls[0]


def refusal_code(raw: bytes) -> str:
    error = json.loads(raw)["error"]
    assert error["type"] == "hive_gateway"
    assert isinstance(error["message"], str) and error["message"]
    return error["code"]


# --- pricing -------------------------------------------------------------------------------------


def test_pricing_parses_a_table_and_prices_usage_per_million_tokens():
    prices = parse_prices({"input": 0.05, "output": 0.4, "cache_read": 0.005})
    assert prices == Prices(input=0.05, output=0.4, cache_read=0.005, cache_write=0.0)
    assert prices.cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == pytest.approx(0.45)


@pytest.mark.parametrize("table", [None, {"input": 1}, {"input": 1, "output": -1}, {"input": 1, "output": 1, "x": 1}])
def test_pricing_rejects_bad_tables(table):
    with pytest.raises(ValueError):
        parse_prices(table)


# --- lifecycle --------------------------------------------------------------------------------------


def test_base_url_is_loopback_on_an_ephemeral_port_ending_in_v1(gateway):
    _, endpoint = gateway.open_trial()
    parts = urlsplit(endpoint.base_url)
    assert parts.scheme == "http" and parts.hostname == "127.0.0.1" and parts.port and parts.path == "/v1"


def test_the_server_stops_when_the_context_exits(upstream):
    with make_gateway(upstream.url) as gw:
        _, endpoint = gw.open_trial()
        assert send(endpoint, CHAT, chat_body())[0] == 200
    with pytest.raises(OSError):
        send(endpoint, CHAT, chat_body())


def test_each_trial_gets_a_distinct_unguessable_token(gateway):
    keys_and_endpoints = [gateway.open_trial() for _ in range(3)]
    tokens = {endpoint.api_key for _, endpoint in keys_and_endpoints}
    assert len(tokens) == 3 and all(len(token) >= 40 for token in tokens)
    # The controller's handle is not the secret the trial holds.
    assert all(key != endpoint.api_key for key, endpoint in keys_and_endpoints)


def test_open_trial_needs_a_running_gateway(upstream):
    gw = ModelGateway(model=MODEL, prices=PRICES, upstream_url=upstream.url, upstream_key=UPSTREAM_KEY,
                      max_calls_per_trial=5)  # fmt: skip
    with pytest.raises(RuntimeError):
        gw.open_trial()


@pytest.mark.parametrize(
    "options",
    [
        {"upstream_url": "ftp://example.com"},
        {"upstream_url": "https://api.openai.com/v1"},  # the path is appended; a /v1 base would double it
        {"upstream_key": ""},
        {"max_calls_per_trial": -1},
    ],
)
def test_constructor_rejects_bad_settings(options):
    settings = {"model": MODEL, "prices": PRICES, "upstream_url": "https://api.openai.com",
                "upstream_key": UPSTREAM_KEY, "max_calls_per_trial": 5} | options  # fmt: skip
    with pytest.raises(ValueError):
        ModelGateway(**settings)


def test_closing_an_unknown_trial_raises(gateway):
    with pytest.raises(KeyError):
        gateway.close_trial("no-such-trial")


# --- auth ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token, headers",
    [
        (None, {}),
        ("not-a-real-token", {}),
        (None, {"api-key": "trial"}),  # Azure's header: no SDK HIVE uses sends it, so it isn't accepted
        (None, {"Authorization": "Basic Zm9vOmJhcg=="}),
    ],
)
def test_requests_without_a_valid_bearer_token_get_401_and_are_not_recorded(gateway, upstream, token, headers):
    key, endpoint = gateway.open_trial()
    headers = {name: (endpoint.api_key if value == "trial" else value) for name, value in headers.items()}
    status, _, raw = send(endpoint, CHAT, chat_body(), token=token, headers=headers)
    assert status == 401
    assert json.loads(raw)["error"]["type"] == "hive_gateway"
    assert upstream.requests == []
    assert gateway.close_trial(key).calls == ()


def test_close_trial_revokes_the_token(gateway, upstream):
    key, endpoint = gateway.open_trial()
    assert send(endpoint, CHAT, chat_body())[0] == 200
    ledger = gateway.close_trial(key)
    assert send(endpoint, CHAT, chat_body())[0] == 401
    assert len(ledger.calls) == 1 and len(upstream.requests) == 1
    with pytest.raises(KeyError):
        gateway.close_trial(key)


# --- forwarding and billing -------------------------------------------------------------------------


def test_chat_non_streaming_is_relayed_and_billed(gateway, upstream):
    key, endpoint = gateway.open_trial()
    status, headers, raw = send(endpoint, CHAT, chat_body())
    assert status == 200 and headers["content-type"] == "application/json"
    reply = json.loads(raw)
    assert reply["choices"][0]["message"]["content"] == REPLY_TEXT and reply["usage"] == CHAT_USAGE
    assert headers.get("x-request-id") == "req_fake123"

    call = only_call(gateway.close_trial(key))
    assert call == GatewayCall(path=CHAT, model_requested="gpt-5-nano", model_reported="gpt-5-nano-2025-08-07",
                               status=200, usage=CHAT_BILL, error_code=None, streamed=False)  # fmt: skip
    assert upstream.requests[0].path == CHAT


def test_chat_streaming_injects_include_usage_and_bills_the_final_chunk(gateway, upstream):
    key, endpoint = gateway.open_trial()
    status, headers, raw = send(endpoint, CHAT, chat_body(stream=True))
    assert status == 200 and headers["content-type"].startswith("text/event-stream")
    events = sse_data(raw)
    assert events[-1] == "[DONE]"
    usage_chunk = events[-2]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"] == CHAT_USAGE  # relayed unchanged
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if isinstance(e, dict) and e["choices"])
    assert text == REPLY_TEXT

    assert upstream.requests[0].json["stream_options"] == {"include_usage": True}
    call = only_call(gateway.close_trial(key))
    assert call.streamed and call.status == 200 and call.usage == CHAT_BILL
    assert call.model_reported == "gpt-5-nano-2025-08-07"


def test_chat_streaming_forces_include_usage_on_and_keeps_other_stream_options(gateway, upstream):
    key, endpoint = gateway.open_trial()
    body = chat_body(stream=True, stream_options={"include_usage": False, "include_obfuscation": False})
    assert send(endpoint, CHAT, body)[0] == 200
    assert upstream.requests[0].json["stream_options"] == {"include_usage": True, "include_obfuscation": False}
    assert only_call(gateway.close_trial(key)).usage == CHAT_BILL


def test_responses_non_streaming_is_relayed_and_billed(gateway, upstream):
    key, endpoint = gateway.open_trial()
    status, _, raw = send(endpoint, RESPONSES, responses_body())
    assert status == 200 and json.loads(raw)["usage"] == RESPONSES_USAGE
    call = only_call(gateway.close_trial(key))
    assert call.path == RESPONSES and not call.streamed and call.usage == RESPONSES_BILL
    assert call.model_reported == "gpt-5-nano-2025-08-07"


def test_responses_streaming_bills_the_completed_event(gateway, upstream):
    key, endpoint = gateway.open_trial()
    status, _, raw = send(endpoint, RESPONSES, responses_body(stream=True))
    assert status == 200
    events = sse_data(raw)
    assert [e["type"] for e in events][0] == "response.created"
    assert events[-1]["type"] == "response.completed" and events[-1]["response"]["usage"] == RESPONSES_USAGE
    assert "event: response.completed" in raw.decode()  # event lines are relayed too
    call = only_call(gateway.close_trial(key))
    assert call.streamed and call.usage == RESPONSES_BILL and call.model_reported == "gpt-5-nano-2025-08-07"


def test_a_responses_stream_that_ends_incomplete_is_still_billed(gateway, upstream):
    key, endpoint = gateway.open_trial()
    assert send(endpoint, RESPONSES, responses_body(stream=True, input="scenario=incomplete"))[0] == 200
    assert only_call(gateway.close_trial(key)).usage == RESPONSES_BILL


@pytest.mark.parametrize("path, body", [(CHAT, chat_body()), (CHAT, chat_body(stream=True)), (RESPONSES, responses_body())])
def test_a_success_without_reported_usage_is_flagged(gateway, upstream, path, body):
    key, endpoint = gateway.open_trial()
    upstream.scenario = "no_usage"
    assert send(endpoint, path, body)[0] == 200
    call = only_call(gateway.close_trial(key))
    assert call.status == 200 and call.usage == Usage() and call.error_code == "usage_missing"


def test_streams_are_relayed_as_they_arrive(gateway, upstream):
    _, endpoint = gateway.open_trial()
    upstream.scenario = "slow_stream"
    connection = connect(endpoint)
    body = json.dumps(chat_body(stream=True)).encode()
    connection.request("POST", CHAT, body=body, headers={"Authorization": f"Bearer {endpoint.api_key}",
                                                          "Content-Type": "application/json"})  # fmt: skip
    response = connection.getresponse()
    first = response.readline()
    assert first.startswith(b"data: ")
    assert not upstream.requests[0].finished.is_set(), "the first event should arrive before the stream ends"
    response.read()
    connection.close()


# --- secrets ---------------------------------------------------------------------------------------------


def test_upstream_gets_the_provider_key_and_never_the_trial_token(gateway, upstream):
    key, endpoint = gateway.open_trial()
    extra = {"OpenAI-Organization": "org-elsewhere", "OpenAI-Project": "proj-elsewhere", "X-Custom": "1"}
    for path, body in [(CHAT, chat_body()), (CHAT, chat_body(stream=True)), (RESPONSES, responses_body(stream=True))]:
        assert send(endpoint, path, body, headers=extra)[0] == 200
    gateway.close_trial(key)
    assert len(upstream.requests) == 3
    for request in upstream.requests:
        assert request.headers["authorization"] == f"Bearer {UPSTREAM_KEY}"
        assert endpoint.api_key not in json.dumps(request.headers) and endpoint.api_key.encode() not in request.body
        # Client headers that could route billing elsewhere are not forwarded.
        assert not {"openai-organization", "openai-project", "x-custom"} & set(request.headers)


def test_the_trial_never_sees_the_upstream_key_even_when_the_provider_echoes_it(gateway, upstream):
    key, endpoint = gateway.open_trial()
    upstream.scenario = "echo_auth"
    status, headers, raw = send(endpoint, CHAT, chat_body())
    assert status == 401  # the provider's error passes through...
    assert UPSTREAM_KEY not in raw.decode() and UPSTREAM_KEY not in json.dumps(headers)  # ...without the key
    assert "Incorrect API key provided" in json.loads(raw)["error"]["message"]
    assert only_call(gateway.close_trial(key)).error_code == "invalid_api_key"


# --- P3: model, effort and cost parameters ----------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-5-nano", "gpt-5-nano-2025-08-07", "", None, 7])
def test_any_model_but_the_experiments_is_refused_as_a_violation(gateway, upstream, model):
    key, endpoint = gateway.open_trial()
    body = chat_body(model=model)
    if model is None:
        del body["model"]
    status, _, raw = send(endpoint, CHAT, body)
    assert status == 403 and refusal_code(raw) == "model_not_allowed"
    assert upstream.requests == []
    ledger = gateway.close_trial(key)
    call = only_call(ledger)
    assert call.status == 403 and call.error_code == "model_not_allowed" and call.usage == Usage()
    assert len(ledger.violations) == 1


def test_chat_effort_is_injected_when_absent(gateway, upstream):
    key, endpoint = gateway.open_trial()
    assert send(endpoint, CHAT, chat_body())[0] == 200
    assert send(endpoint, CHAT, chat_body(reasoning_effort=None))[0] == 200
    assert send(endpoint, CHAT, chat_body(reasoning_effort="low"))[0] == 200
    assert [r.json["reasoning_effort"] for r in upstream.requests] == ["low", "low", "low"]
    assert gateway.close_trial(key).violations == ()


def test_responses_effort_is_injected_when_absent_and_other_reasoning_settings_kept(gateway, upstream):
    key, endpoint = gateway.open_trial()
    assert send(endpoint, RESPONSES, responses_body())[0] == 200
    assert send(endpoint, RESPONSES, responses_body(reasoning={"summary": "auto"}))[0] == 200
    assert send(endpoint, RESPONSES, responses_body(reasoning={"effort": "low"}))[0] == 200
    assert [r.json["reasoning"] for r in upstream.requests] == [
        {"effort": "low"}, {"summary": "auto", "effort": "low"}, {"effort": "low"},
    ]  # fmt: skip
    assert gateway.close_trial(key).violations == ()


@pytest.mark.parametrize(
    "path, body",
    [
        (CHAT, chat_body(reasoning_effort="high")),
        (CHAT, chat_body(reasoning_effort="minimal")),
        (RESPONSES, responses_body(reasoning={"effort": "high"})),
        (RESPONSES, responses_body(reasoning="high")),
    ],
)
def test_a_different_effort_is_refused_as_a_violation(gateway, upstream, path, body):
    key, endpoint = gateway.open_trial()
    status, _, raw = send(endpoint, path, body)
    assert status == 403 and refusal_code(raw) == "effort_not_allowed"
    assert upstream.requests == []
    ledger = gateway.close_trial(key)
    assert only_call(ledger).error_code == "effort_not_allowed" and len(ledger.violations) == 1


def test_without_a_fixed_effort_none_is_injected_and_any_explicit_effort_is_refused(upstream):
    with make_gateway(upstream.url, model=ModelSpec(provider="openai", id="gpt-5-nano")) as gw:
        key, endpoint = gw.open_trial()
        assert send(endpoint, CHAT, chat_body())[0] == 200
        assert send(endpoint, RESPONSES, responses_body())[0] == 200
        assert "reasoning_effort" not in upstream.requests[0].json and "reasoning" not in upstream.requests[1].json
        assert send(endpoint, CHAT, chat_body(reasoning_effort="high"))[0] == 403
        assert send(endpoint, RESPONSES, responses_body(reasoning={"effort": "low"}))[0] == 403
        assert len(gw.close_trial(key).violations) == 2


@pytest.mark.parametrize("tier", [None, "auto", "default"])
def test_default_service_tiers_pass(gateway, upstream, tier):
    key, endpoint = gateway.open_trial()
    body = chat_body() if tier is None else chat_body(service_tier=tier)
    assert send(endpoint, CHAT, body)[0] == 200
    assert gateway.close_trial(key).violations == ()


@pytest.mark.parametrize(
    "path, body, code",
    [
        (CHAT, chat_body(service_tier="priority"), "service_tier_not_allowed"),
        (RESPONSES, responses_body(service_tier="flex"), "service_tier_not_allowed"),
        (CHAT, chat_body(n=2), "n_not_allowed"),
        (CHAT, chat_body(n="2"), "n_not_allowed"),
        (RESPONSES, responses_body(background=True), "background_not_allowed"),
        # Hosted tools run at the provider and bill per call, outside the token usage the gateway counts.
        (RESPONSES, responses_body(tools=[{"type": "web_search_preview"}]), "hosted_tool_not_allowed"),
        (RESPONSES, responses_body(tools=[{"type": "file_search", "vector_store_ids": ["vs"]}]), "hosted_tool_not_allowed"),
        (RESPONSES, responses_body(tools=[{"type": "code_interpreter", "container": {"type": "auto"}}]), "hosted_tool_not_allowed"),
        (CHAT, chat_body(tools=[{"type": "web_search"}]), "hosted_tool_not_allowed"),
        (RESPONSES, responses_body(tools="web_search"), "hosted_tool_not_allowed"),
        (RESPONSES, responses_body(tools=[{"name": "no_type"}]), "hosted_tool_not_allowed"),
    ],
)
def test_cost_changing_parameters_are_refused_as_violations(gateway, upstream, path, body, code):
    key, endpoint = gateway.open_trial()
    status, _, raw = send(endpoint, path, body)
    assert status == 403 and refusal_code(raw) == code
    assert upstream.requests == []
    ledger = gateway.close_trial(key)
    assert only_call(ledger).error_code == code and len(ledger.violations) == 1


def test_n_of_one_passes(gateway, upstream):
    key, endpoint = gateway.open_trial()
    assert send(endpoint, CHAT, chat_body(n=1))[0] == 200
    assert gateway.close_trial(key).violations == ()


@pytest.mark.parametrize(
    "method, path",
    [("POST", "/v1/embeddings"), ("GET", "/v1/models"), ("GET", CHAT), ("POST", "/v1/chat/completions/x"),
     ("POST", "/v1/responses?stream=true"), ("POST", "/chat/completions")],
)  # fmt: skip
def test_unknown_paths_are_refused_and_recorded_but_not_violations(gateway, upstream, method, path):
    key, endpoint = gateway.open_trial()
    status, _, raw = send(endpoint, path, chat_body() if method == "POST" else None, method=method)
    assert status == 403 and refusal_code(raw) == "unknown_path"
    assert upstream.requests == []
    ledger = gateway.close_trial(key)
    assert only_call(ledger).status == 403 and only_call(ledger).error_code == "unknown_path"
    assert ledger.violations == ()


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b"\xff\xfe", b""])
def test_bodies_that_are_not_a_json_object_are_refused(gateway, upstream, raw):
    key, endpoint = gateway.open_trial()
    status, _, reply = send(endpoint, CHAT, raw=raw)
    assert status == 403 and refusal_code(reply) == "invalid_body"
    ledger = gateway.close_trial(key)
    assert only_call(ledger).error_code == "invalid_body" and ledger.violations == ()
    assert upstream.requests == []


def test_a_body_over_the_size_cap_is_refused(upstream):
    with make_gateway(upstream.url, max_body_bytes=1024) as gw:
        key, endpoint = gw.open_trial()
        status, _, raw = send(endpoint, CHAT, chat_body(padding="x" * 2048))
        assert status == 403 and refusal_code(raw) == "body_too_large"
        # A declared length over the cap is refused before any of the body is read.
        connection = connect(endpoint)
        connection.putrequest("POST", CHAT)
        connection.putheader("Authorization", f"Bearer {endpoint.api_key}")
        connection.putheader("Content-Length", str(10**9))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 403 and refusal_code(response.read()) == "body_too_large"
        connection.close()
        assert send(endpoint, CHAT, chat_body())[0] == 200
        ledger = gw.close_trial(key)
    assert [call.error_code for call in ledger.calls] == ["body_too_large", "body_too_large", None]
    assert len(upstream.requests) == 1


# --- call limit -----------------------------------------------------------------------------------------------


def test_the_call_after_the_limit_is_refused_and_flagged(upstream):
    with make_gateway(upstream.url, max_calls=2) as gw:
        key, endpoint = gw.open_trial()
        assert send(endpoint, CHAT, chat_body(model="gpt-4o"))[0] == 403  # refusals don't use up the limit
        assert send(endpoint, CHAT, chat_body())[0] == 200
        assert send(endpoint, RESPONSES, responses_body())[0] == 200
        status, _, raw = send(endpoint, CHAT, chat_body())
        assert status == 403 and refusal_code(raw) == "model_call_limit"
        ledger = gw.close_trial(key)
    assert ledger.limit_reached
    assert [call.status for call in ledger.calls] == [403, 200, 200, 403]
    assert ledger.calls[-1].error_code == "model_call_limit"
    assert len(ledger.violations) == 1  # the model refusal; the limit is not a P3 violation
    assert len(upstream.requests) == 2


def test_a_trial_under_the_limit_is_not_flagged(gateway):
    key, endpoint = gateway.open_trial()
    send(endpoint, CHAT, chat_body())
    assert not gateway.close_trial(key).limit_reached


# --- upstream errors --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario, status, code",
    [("quota", 429, "insufficient_quota"), ("rate_limit", 429, "rate_limit_exceeded"), ("server_error", 500, "server_error")],
)
def test_provider_errors_pass_through_with_their_code_recorded(gateway, upstream, scenario, status, code):
    key, endpoint = gateway.open_trial()
    upstream.scenario = scenario
    got, headers, raw = send(endpoint, CHAT, chat_body(stream=True))
    assert got == status and headers["content-type"] == "application/json"
    assert json.loads(raw)["error"]["type"] in {"insufficient_quota", "requests", "server_error"}
    if status == 429:
        assert headers.get("retry-after") == "0"
    call = only_call(gateway.close_trial(key))
    assert call.status == status and call.error_code == code and call.usage == Usage()


def test_an_upstream_timeout_becomes_a_502(upstream):
    upstream.scenario = "hang"
    with make_gateway(upstream.url, upstream_timeout_s=0.5) as gw:
        key, endpoint = gw.open_trial()
        status, _, raw = send(endpoint, CHAT, chat_body())
        assert status == 502 and refusal_code(raw) == "upstream_unreachable"
        call = only_call(gw.close_trial(key))
    assert call.status == 502 and call.error_code == "upstream_unreachable" and call.usage == Usage()


def test_an_unreachable_upstream_becomes_a_502():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with make_gateway(f"http://127.0.0.1:{port}") as gw:
        key, endpoint = gw.open_trial()
        status, _, raw = send(endpoint, RESPONSES, responses_body(stream=True))
        assert status == 502 and refusal_code(raw) == "upstream_unreachable"
        call = only_call(gw.close_trial(key))
    assert call.error_code == "upstream_unreachable" and call.usage == Usage() and call.streamed


@pytest.mark.parametrize("path, body", [(CHAT, chat_body()), (CHAT, chat_body(stream=True)), (RESPONSES, responses_body(stream=True))])
def test_a_different_reported_model_is_a_violation_but_still_billed(gateway, upstream, path, body):
    key, endpoint = gateway.open_trial()
    upstream.scenario = "other_model"
    assert send(endpoint, path, body)[0] == 200
    ledger = gateway.close_trial(key)
    call = only_call(ledger)
    assert call.model_reported == OTHER_MODEL and call.usage.cost_usd > 0
    assert len(ledger.violations) == 1 and OTHER_MODEL in ledger.violations[0]


# --- disconnects and closing --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path, body, bill", [(CHAT, chat_body(stream=True), CHAT_BILL),
                                              (RESPONSES, responses_body(stream=True), RESPONSES_BILL)])  # fmt: skip
def test_a_client_that_disconnects_mid_stream_is_still_billed(gateway, upstream, path, body, bill):
    key, endpoint = gateway.open_trial()
    upstream.scenario = "slow_stream"
    connection = connect(endpoint)
    connection.request("POST", path, body=json.dumps(body).encode(),
                       headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"})  # fmt: skip
    response = connection.getresponse()
    assert response.readline()  # the stream has started
    response.close()  # hang up: the gateway's next writes fail
    connection.close()
    # Closed while the upstream is still streaming: close_trial waits for the call to be recorded.
    assert not upstream.requests[0].finished.is_set()
    ledger = gateway.close_trial(key)
    assert upstream.requests[0].finished.is_set()
    call = only_call(ledger)
    assert call.status == 200 and call.usage == bill


def test_close_trial_waits_only_so_long_for_in_flight_calls(upstream):
    upstream.scenario = "hang"
    upstream.hang_s = 1.5
    with make_gateway(upstream.url, upstream_timeout_s=10, close_timeout_s=0.3) as gw:
        key, endpoint = gw.open_trial()
        thread = threading.Thread(target=lambda: send(endpoint, CHAT, chat_body()))
        thread.start()
        while not upstream.requests:
            time.sleep(0.01)
        started = time.monotonic()
        ledger = gw.close_trial(key)
        assert time.monotonic() - started < 1
        thread.join()
    call = only_call(ledger)
    assert call.status == 0 and call.error_code == "in_flight_at_close" and call.model_requested == MODEL.id


# --- concurrency ---------------------------------------------------------------------------------------------------


def test_concurrent_trials_keep_separate_ledgers(upstream):
    kinds = [(CHAT, chat_body(), CHAT_BILL), (CHAT, chat_body(stream=True), CHAT_BILL),
             (RESPONSES, responses_body(), RESPONSES_BILL), (RESPONSES, responses_body(stream=True), RESPONSES_BILL)]  # fmt: skip
    per_trial = 6
    with make_gateway(upstream.url, max_calls=per_trial) as gw:
        trials = [gw.open_trial() for _ in range(4)]
        errors: list[BaseException] = []

        def drive(index: int, endpoint: ModelEndpoint) -> None:
            try:
                path, body, _ = kinds[index]
                for _ in range(per_trial + index):  # trial i goes i calls over its limit
                    send(endpoint, path, body)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=drive, args=(i, endpoint)) for i, (_, endpoint) in enumerate(trials)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        ledgers = [gw.close_trial(key) for key, _ in trials]

    for index, ledger in enumerate(ledgers):
        path, body, bill = kinds[index]
        assert len(ledger.calls) == per_trial + index
        forwarded = [call for call in ledger.calls if call.status == 200]
        assert len(forwarded) == per_trial
        assert all(call.path == path and call.usage == bill for call in forwarded)
        assert ledger.limit_reached == (index > 0)
        assert ledger.usage.input_tokens == bill.input_tokens * per_trial
    assert len(upstream.requests) == 4 * per_trial


def test_closing_one_trial_leaves_the_others_working(gateway):
    key_a, endpoint_a = gateway.open_trial()
    key_b, endpoint_b = gateway.open_trial()
    send(endpoint_a, CHAT, chat_body())
    gateway.close_trial(key_a)
    assert send(endpoint_a, CHAT, chat_body())[0] == 401
    assert send(endpoint_b, CHAT, chat_body())[0] == 200
    assert len(gateway.close_trial(key_b).calls) == 1


# --- real OpenAI SDKs as clients -----------------------------------------------------------------------------------

SDK_CLIENT = r"""
import json, sys
spec = json.load(sys.stdin)
import openai
client = openai.OpenAI(base_url=spec["base_url"], api_key=spec["api_key"], max_retries=0, timeout=30)
model = spec["model"]
out = {"openai_version": openai.__version__}

reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
out["chat"] = {"text": reply.choices[0].message.content, "prompt_tokens": reply.usage.prompt_tokens}

text, usage = "", None
for chunk in client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], stream=True):
    for choice in chunk.choices:
        text += choice.delta.content or ""
    if chunk.usage is not None:
        usage = chunk.usage.prompt_tokens
out["chat_stream"] = {"text": text, "prompt_tokens": usage}

reply = client.responses.create(model=model, input="hi")
out["responses"] = {"text": reply.output_text, "input_tokens": reply.usage.input_tokens}

text, usage = "", None
for event in client.responses.create(model=model, input="hi", stream=True):
    if event.type == "response.output_text.delta":
        text += event.delta
    elif event.type == "response.completed":
        usage = event.response.usage.input_tokens
out["responses_stream"] = {"text": text, "input_tokens": usage}

if spec.get("langchain"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model, base_url=spec["base_url"], api_key=spec["api_key"], use_responses_api=True,
                     reasoning={"effort": "low"}, max_retries=0, timeout=30)
    message = llm.invoke("hi")
    text = message.text() if callable(message.text) else message.text
    out["langchain"] = {"text": text, "usage": dict(message.usage_metadata or {})}

print(json.dumps(out))
"""


@pytest.mark.parametrize(
    "venv, langchain",
    [("strands", False), ("deepagents", True)],
)
def test_real_openai_sdks_parse_what_the_gateway_relays(upstream, tmp_path, venv, langchain):
    python = PROJECT / ".venvs" / venv / "bin" / "python"
    if not python.exists():
        pytest.skip(f"{python} is missing; build it with scripts/setup-framework-venvs.sh")
    with make_gateway(upstream.url) as gw:
        key, endpoint = gw.open_trial()
        spec = {"base_url": endpoint.base_url, "api_key": endpoint.api_key, "model": MODEL.id, "langchain": langchain}
        # A minimal environment: no provider key and no proxy settings reach the client.
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path)}
        run = subprocess.run([str(python), "-I", "-c", SDK_CLIENT], input=json.dumps(spec), capture_output=True,
                             text=True, env=env, cwd=tmp_path, timeout=120)  # fmt: skip
        ledger = gw.close_trial(key)
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout)

    assert out["chat"] == {"text": REPLY_TEXT, "prompt_tokens": 1000}
    assert out["chat_stream"] == {"text": REPLY_TEXT, "prompt_tokens": 1000}  # the injected usage chunk parsed fine
    assert out["responses"] == {"text": REPLY_TEXT, "input_tokens": 1200}
    assert out["responses_stream"] == {"text": REPLY_TEXT, "input_tokens": 1200}
    expected = [CHAT_BILL, CHAT_BILL, RESPONSES_BILL, RESPONSES_BILL]
    if langchain:
        assert out["langchain"]["text"] == REPLY_TEXT
        assert out["langchain"]["usage"]["input_tokens"] == 1200
        expected.append(RESPONSES_BILL)

    assert ledger.violations == () and not ledger.limit_reached
    assert [call.status for call in ledger.calls] == [200] * len(expected)
    assert [call.usage for call in ledger.calls] == expected
    assert [call.streamed for call in ledger.calls][:4] == [False, True, False, True]
    for request in upstream.requests:
        assert request.headers["authorization"] == f"Bearer {UPSTREAM_KEY}"
        effort = request.json.get("reasoning_effort") or request.json["reasoning"]["effort"]
        assert effort == "low"


@pytest.mark.parametrize(
    "path, body",
    [
        (CHAT, chat_body(tools=[{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}])),
        (RESPONSES, responses_body(tools=[{"type": "function", "name": "calc", "parameters": {"type": "object"}}])),
        (RESPONSES, responses_body(tools=[{"type": "custom", "name": "patch"}])),
    ],
)
def test_client_executed_tools_pass(gateway, upstream, path, body):
    # Function and custom tools run in the trial, so every token they cost comes back as usage.
    key, endpoint = gateway.open_trial()
    assert send(endpoint, path, body)[0] == 200
    assert gateway.close_trial(key).violations == ()


def test_a_trial_cannot_hold_more_than_its_share_of_connections(upstream):
    # A trial that opens connections and never finishes a request must not pin unbounded gateway threads.
    with make_gateway(upstream.url, max_in_flight_per_trial=2, client_timeout_s=1.0) as gateway:
        key, endpoint = gateway.open_trial()
        stalled = []
        for _ in range(4):
            connection = connect(endpoint)
            connection.putrequest("POST", CHAT)
            connection.putheader("Authorization", f"Bearer {endpoint.api_key}")
            connection.putheader("Content-Length", "100000")
            connection.endheaders()  # the body never comes
            stalled.append(connection)
        # Another trial is still served promptly.
        other_key, other = gateway.open_trial()
        started = time.monotonic()
        assert send(other, CHAT, chat_body())[0] == 200
        assert time.monotonic() - started < 5
        refused = [c for c in stalled if c.getresponse().status == 429]
        for connection in stalled:
            connection.close()
        assert len(refused) >= 2  # beyond the per-trial cap, a stalled request is turned away
        gateway.close_trial(other_key)
        gateway.close_trial(key)
