"""A stand-in for the OpenAI API, so gateway tests never touch the network.

`FakeOpenAIUpstream` is a local HTTP server on 127.0.0.1 that records every request it
receives (method, path, headers, body), so a test can check which key and which token
reached the provider, and replies with canned OpenAI-shaped responses:

- chat completions, non-streaming and streaming (the `include_usage` final chunk, with empty
  `choices`, is sent only when the request asked for it, as the real API does);
- responses, non-streaming and streaming (response.created, output_text deltas,
  response.completed carrying the usage).

A scenario changes the reply. It is set for the whole server (`upstream.scenario = "quota"`)
or for one request, by putting `scenario=NAME` anywhere in the request body (e.g. in the
prompt), which lets concurrent requests and real SDK clients pick their own:

    ok            the normal reply for the path and the "stream" flag
    other_model   as ok, but reports OTHER_MODEL instead of the requested model
    quota         429 insufficient_quota
    rate_limit    429 rate_limit_exceeded
    server_error  500 with an OpenAI error body whose code is null
    slow_stream   as ok, with SLOW_DELAY_S between streamed events
    hang          sleeps `hang_s` before replying (to trip the gateway's upstream timeout)
    echo_auth     401 whose message repeats the Authorization header it received
    no_usage      as ok, but the provider reports no usage at all
    incomplete    responses stream that ends in response.incomplete (still carrying usage)

Canned usage: CHAT_USAGE and RESPONSES_USAGE, so tests can compute the expected ledger.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SNAPSHOT_SUFFIX = "-2025-08-07"
OTHER_MODEL = "gpt-4o-2024-08-06"
REPLY_TEXT = "Hello world"
SLOW_DELAY_S = 0.25

CHAT_USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 300,
    "total_tokens": 1300,
    "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 100, "audio_tokens": 0},
}
RESPONSES_USAGE = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 400},
    "output_tokens": 500,
    "output_tokens_details": {"reasoning_tokens": 200},
    "total_tokens": 1700,
}

_SCENARIO_MARKER = re.compile(rb"scenario=([a-z_]+)")


@dataclass
class ReceivedRequest:
    method: str
    path: str
    headers: dict[str, str]  # lower-cased names
    body: bytes
    json: Any = None
    finished: threading.Event = field(default_factory=threading.Event)  # set once the reply was fully written


class FakeOpenAIUpstream:
    def __init__(self, *, scenario: str = "ok", hang_s: float = 3.0) -> None:
        self.scenario = scenario
        self.hang_s = hang_s
        self.requests: list[ReceivedRequest] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None, "use FakeOpenAIUpstream as a context manager"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeOpenAIUpstream:
        upstream = self

        class Handler(_FakeHandler):
            fake = upstream

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self._server is not None and self._thread is not None
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def record(self, request: ReceivedRequest) -> None:
        with self._lock:
            self.requests.append(request)


class _FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fake: FakeOpenAIUpstream

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    # --- dispatch --------------------------------------------------------------------------

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        request = ReceivedRequest(
            method=self.command,
            path=self.path,
            headers={name.lower(): value for name, value in self.headers.items()},
            body=body,
            json=parsed,
        )
        self.fake.record(request)
        try:
            self._reply(request, parsed if isinstance(parsed, dict) else {})
        except (BrokenPipeError, ConnectionResetError):
            return
        request.finished.set()

    def _reply(self, request: ReceivedRequest, body: dict[str, Any]) -> None:
        marker = _SCENARIO_MARKER.search(request.body)
        scenario = marker.group(1).decode() if marker else self.fake.scenario
        requested = str(body.get("model", ""))
        model = OTHER_MODEL if scenario == "other_model" else requested + SNAPSHOT_SUFFIX
        stream = body.get("stream") is True
        delay = SLOW_DELAY_S if scenario == "slow_stream" else 0.0

        if scenario == "hang":
            time.sleep(self.fake.hang_s)
        if scenario == "quota":
            return self._json(429, _error("You exceeded your current quota.", "insufficient_quota", "insufficient_quota"))
        if scenario == "rate_limit":
            return self._json(429, _error("Rate limit reached.", "requests", "rate_limit_exceeded"))
        if scenario == "server_error":
            return self._json(500, _error("The server had an error.", "server_error", None))
        if scenario == "echo_auth":
            message = f"Incorrect API key provided: {self.headers.get('Authorization')}"
            return self._json(401, _error(message, "invalid_request_error", "invalid_api_key"))

        usage_wanted = scenario != "no_usage"
        if request.path == "/v1/chat/completions":
            if not stream:
                return self._json(200, _chat_completion(model, usage_wanted))
            options = body.get("stream_options")
            include_usage = usage_wanted and isinstance(options, dict) and options.get("include_usage") is True
            return self._sse([(None, chunk) for chunk in _chat_chunks(model, include_usage)] + [(None, "[DONE]")], delay)
        if request.path == "/v1/responses":
            if not stream:
                return self._json(200, _response(model, "completed", usage_wanted))
            terminal = "incomplete" if scenario == "incomplete" else "completed"
            return self._sse(_response_events(model, terminal, usage_wanted), delay)
        return self._json(404, _error(f"Unknown path {request.path}", "invalid_request_error", "unknown_url"))

    # --- writing -----------------------------------------------------------------------------

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("x-request-id", "req_fake123")
        if status == 429:
            self.send_header("retry-after", "0")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _sse(self, events: list[tuple[str | None, Any]], delay: float) -> None:
        """Stream server-sent events with chunked transfer encoding, as the real API does."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("x-request-id", "req_fake123")
        self.end_headers()
        for index, (name, data) in enumerate(events):
            if index and delay:
                time.sleep(delay)
            text = data if isinstance(data, str) else json.dumps(data)
            frame = (f"event: {name}\n" if name else "") + f"data: {text}\n\n"
            payload = frame.encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


# --- canned payloads ---------------------------------------------------------------------------


def _error(message: str, type_: str, code: str | None) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def _chat_completion(model: str, with_usage: bool) -> dict[str, Any]:
    completion: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1760000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY_TEXT, "refusal": None, "annotations": []},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "service_tier": "default",
        "system_fingerprint": None,
    }
    if with_usage:
        completion["usage"] = CHAT_USAGE
    return completion


def _chat_chunks(model: str, include_usage: bool) -> list[dict[str, Any]]:
    def chunk(choices: list[dict[str, Any]], usage: Any = None) -> dict[str, Any]:
        out: dict[str, Any] = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 1760000000,
                               "model": model, "service_tier": "default", "system_fingerprint": None,
                               "choices": choices}  # fmt: skip
        if include_usage:
            out["usage"] = usage
        return out

    def delta(content: dict[str, Any], finish: str | None = None) -> list[dict[str, Any]]:
        return [{"index": 0, "delta": content, "logprobs": None, "finish_reason": finish}]

    first, second = REPLY_TEXT.split(" ", 1)
    chunks = [
        chunk(delta({"role": "assistant", "content": "", "refusal": None})),
        chunk(delta({"content": first})),
        chunk(delta({"content": " " + second})),
        chunk(delta({}, "stop")),
    ]
    if include_usage:
        chunks.append(chunk([], CHAT_USAGE))
    return chunks


def _message_item(status: str) -> dict[str, Any]:
    content = [{"type": "output_text", "text": REPLY_TEXT, "annotations": [], "logprobs": []}] if status == "completed" else []
    return {"id": "msg_fake", "type": "message", "status": status, "role": "assistant", "content": content}


def _response(model: str, status: str, with_usage: bool, *, finished: bool = True) -> dict[str, Any]:
    return {
        "id": "resp_fake",
        "object": "response",
        "created_at": 1760000000,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [_message_item("completed")] if finished else [],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "low", "summary": None},
        "service_tier": "default",
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": RESPONSES_USAGE if (with_usage and finished) else None,
        "metadata": {},
    }


def _response_events(model: str, terminal: str, with_usage: bool) -> list[tuple[str | None, Any]]:
    first, second = REPLY_TEXT.split(" ", 1)
    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {"type": "response.created", "response": _response(model, "in_progress", False, finished=False)}),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": _message_item("in_progress")}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_fake", "output_index": 0,
                                        "content_index": 0, "delta": first, "logprobs": []}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_fake", "output_index": 0,
                                        "content_index": 0, "delta": " " + second, "logprobs": []}),
        ("response.output_text.done", {"type": "response.output_text.done", "item_id": "msg_fake", "output_index": 0,
                                       "content_index": 0, "text": REPLY_TEXT, "logprobs": []}),
        ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": _message_item("completed")}),
        (f"response.{terminal}", {"type": f"response.{terminal}", "response": _response(model, terminal, with_usage)}),
    ]  # fmt: skip
    for number, (_, event) in enumerate(events):
        event["sequence_number"] = number
    return list(events)
