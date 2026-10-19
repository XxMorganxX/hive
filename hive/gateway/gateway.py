"""The model gateway: a loopback HTTP server that is the only way a trial reaches a model.

See tasks/framework-adapters-and-extensions.md, "Model gateway (spec)". The controller opens a
grant per trial; the trial gets a `ModelEndpoint` whose key is a per-trial token, never the
provider key. Each request is authenticated, checked against the experiment's model, effort
and cost rules (hive/gateway/policy.py), counted against the call limit, forwarded upstream
with the real key, relayed back as it arrives, and recorded with the usage the provider
reported (hive/gateway/usage.py). `close_trial` revokes the token and returns the ledger.

Stdlib only: this is trusted controller code and must not depend on any agent framework.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import secrets
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from hive.agents.runner_protocol import model_matches
from hive.contracts import ModelEndpoint, ModelSpec, Usage
from hive.gateway import policy
from hive.gateway.usage import StreamObserver, parse_body
from hive.pricing import Prices

DEFAULT_UPSTREAM_TIMEOUT_S = 600.0  # per socket read: a non-streamed reasoning call can think for minutes
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024  # long contexts with inline images still fit
DEFAULT_CLIENT_TIMEOUT_S = 30.0  # a trial that stops sending or reading is let go after this
DEFAULT_MAX_IN_FLIGHT_PER_TRIAL = 8  # concurrent requests one trial may hold (subagents run in parallel)

# _admit's answer when the trial already holds its share of connections.
_BUSY = object()
_REDACTED = b"[redacted]"
# Upstream response headers the trial may see. Everything else (organization, project, rate
# limit internals) stays in the gateway.
_RELAYED_HEADERS = ("x-request-id", "retry-after", "retry-after-ms")
_MAX_RECORDED_PATH = 256


@dataclass(frozen=True)
class GatewayCall:
    """One model request that reached the gateway, whether or not it was forwarded."""

    path: str  # "/v1/chat/completions" | "/v1/responses" | anything refused
    model_requested: str
    model_reported: str | None  # what the provider said it ran; None if never forwarded or not reported
    status: int  # HTTP status returned to the trial (403 for a refused request; 0 if still running at close)
    usage: Usage  # tokens from the provider's response, cost from the experiment's prices
    error_code: str | None = None  # provider error code (e.g. "insufficient_quota") or the gateway's refusal reason
    streamed: bool = False


@dataclass(frozen=True)
class TrialLedger:
    calls: tuple[GatewayCall, ...]
    violations: tuple[str, ...]  # P3 / cost-parameter refusals and reported-model mismatches, human-readable
    limit_reached: bool

    @property
    def usage(self) -> Usage:
        total = Usage()
        for call in self.calls:
            total = total + call.usage
        return total


@dataclass
class _InFlight:
    """A request being handled, so close_trial can wait for it and name it if it can't."""

    path: str
    model_requested: str = ""
    streamed: bool = False


@dataclass
class _Trial:
    key: str
    token_digest: bytes
    calls: list[GatewayCall] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    limit_reached: bool = False
    forwarded: int = 0
    in_flight: dict[int, _InFlight] = field(default_factory=dict)


class ModelGateway:
    def __init__(
        self,
        *,
        model: ModelSpec,
        prices: Prices,
        upstream_url: str,
        upstream_key: str,
        max_calls_per_trial: int,
        upstream_timeout_s: float = DEFAULT_UPSTREAM_TIMEOUT_S,
        close_timeout_s: float | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        max_in_flight_per_trial: int = DEFAULT_MAX_IN_FLIGHT_PER_TRIAL,
        client_timeout_s: float = DEFAULT_CLIENT_TIMEOUT_S,
    ) -> None:
        """`upstream_url` is the provider's origin (e.g. "https://api.openai.com"), optionally with a
        path prefix; the request path ("/v1/...") is appended to it, so it must not end in /v1.

        `upstream_timeout_s` bounds each read from the provider. `close_timeout_s` bounds how long
        close_trial waits for a trial's in-flight calls (default: the upstream timeout plus 5 s).
        `max_in_flight_per_trial` and `client_timeout_s` bound what one trial can hold: a trial that
        opens connections and never finishes its requests can't pin the gateway's threads.
        """
        parts = urlsplit(upstream_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"upstream_url must be an http(s) URL, got {upstream_url!r}")
        if parts.query or parts.fragment:
            raise ValueError("upstream_url must not have a query or fragment")
        prefix = parts.path.rstrip("/")
        if prefix.endswith("/v1"):
            raise ValueError(f"upstream_url is the origin; the path /v1/... is appended, so drop the trailing /v1 from {upstream_url!r}")
        if not isinstance(upstream_key, str) or not upstream_key:
            raise ValueError("upstream_key is required")
        if isinstance(max_calls_per_trial, bool) or not isinstance(max_calls_per_trial, int) or max_calls_per_trial < 0:
            raise ValueError(f"max_calls_per_trial must be an int >= 0, got {max_calls_per_trial!r}")
        if upstream_timeout_s <= 0 or max_body_bytes <= 0 or (close_timeout_s is not None and close_timeout_s < 0):
            raise ValueError("timeouts and the body cap must be positive")
        if max_in_flight_per_trial < 1 or client_timeout_s <= 0:
            raise ValueError("max_in_flight_per_trial must be >= 1 and client_timeout_s positive")

        self.model = model
        self.prices = prices
        self.max_calls_per_trial = max_calls_per_trial
        self.upstream_timeout_s = upstream_timeout_s
        self.close_timeout_s = upstream_timeout_s + 5 if close_timeout_s is None else close_timeout_s
        self.max_body_bytes = max_body_bytes
        self.max_in_flight_per_trial = max_in_flight_per_trial
        self.client_timeout_s = client_timeout_s
        self._upstream_https = parts.scheme == "https"
        self._upstream_host = parts.hostname
        self._upstream_port = parts.port
        self._upstream_prefix = prefix
        self._upstream_key = upstream_key
        self._changed = threading.Condition()  # guards every _Trial and both indexes below
        self._trials_by_key: dict[str, _Trial] = {}
        self._trials_by_digest: dict[bytes, _Trial] = {}
        self._server: _GatewayServer | None = None
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:  # never show the key
        return f"ModelGateway(model={self.model.qualified!r}, upstream={self._upstream_host!r})"

    # --- lifecycle ---------------------------------------------------------------------------

    def __enter__(self) -> ModelGateway:
        if self._server is not None:
            raise RuntimeError("the gateway is already running")
        self._server = _GatewayServer(self)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="hive-model-gateway", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        server, thread = self._server, self._thread
        if server is None or thread is None:
            return
        server.shutdown()
        server.server_close()
        thread.join()
        self._server = self._thread = None
        with self._changed:
            self._trials_by_digest.clear()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("the gateway is not running; use it as a context manager")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def open_trial(self) -> tuple[str, ModelEndpoint]:
        """A new grant: (the controller's handle for close_trial, the endpoint the trial uses)."""
        base_url = self.base_url
        token = secrets.token_urlsafe(32)
        trial = _Trial(key=f"trial-{secrets.token_hex(8)}", token_digest=_digest(token))
        with self._changed:
            self._trials_by_key[trial.key] = trial
            self._trials_by_digest[trial.token_digest] = trial
        return trial.key, ModelEndpoint(base_url=base_url, api_key=token)

    def close_trial(self, trial_key: str) -> TrialLedger:
        """Revoke the trial's token, wait (bounded) for its in-flight calls, and return its ledger.

        A call still running when the wait ends is listed with status 0 and error_code
        "in_flight_at_close", so the gap in billing is visible rather than silent.
        """
        deadline = time.monotonic() + self.close_timeout_s
        with self._changed:
            trial = self._trials_by_key.pop(trial_key)
            self._trials_by_digest.pop(trial.token_digest, None)
            while trial.in_flight and (remaining := deadline - time.monotonic()) > 0:
                self._changed.wait(remaining)
            unfinished = [
                GatewayCall(path=f.path, model_requested=f.model_requested, model_reported=None, status=0,
                            usage=Usage(), error_code="in_flight_at_close", streamed=f.streamed)
                for f in trial.in_flight.values()
            ]  # fmt: skip
            return TrialLedger(calls=tuple(trial.calls + unfinished), violations=tuple(trial.violations),
                               limit_reached=trial.limit_reached)  # fmt: skip

    # --- bookkeeping, called from request threads -------------------------------------------

    def _admit(self, token: str | None, path: str) -> tuple[_Trial, int, _InFlight] | object | None:
        """The trial a token belongs to, with this request registered as in flight; None if the
        token is unknown, _BUSY if the trial already has its maximum number of requests open."""
        if not token:
            return None
        digest = _digest(token)
        with self._changed:
            trial = self._trials_by_digest.get(digest)
            if trial is None or not hmac.compare_digest(trial.token_digest, digest):
                return None
            if len(trial.in_flight) >= self.max_in_flight_per_trial:
                return _BUSY
            flight = _InFlight(path=path[:_MAX_RECORDED_PATH])
            flight_id = id(flight)
            trial.in_flight[flight_id] = flight
            return trial, flight_id, flight

    def _release(self, trial: _Trial, flight_id: int) -> None:
        with self._changed:
            trial.in_flight.pop(flight_id, None)
            self._changed.notify_all()

    def _reserve_call(self, trial: _Trial) -> bool:
        with self._changed:
            if trial.forwarded >= self.max_calls_per_trial:
                trial.limit_reached = True
                return False
            trial.forwarded += 1
            return True

    def _record(self, trial: _Trial, call: GatewayCall, violation: str | None = None) -> None:
        with self._changed:
            trial.calls.append(call)
            if violation is not None:
                trial.violations.append(violation)

    def _priced(self, tokens: Usage | None) -> Usage:
        if tokens is None:
            return Usage()
        return Usage(input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens,
                     cache_read_tokens=tokens.cache_read_tokens, cache_write_tokens=tokens.cache_write_tokens,
                     cost_usd=self.prices.cost(tokens))  # fmt: skip

    def _model_violation(self, path: str, reported: str | None) -> str | None:
        if reported is None or model_matches(reported, self.model):
            return None
        return f"{path}: the provider reported model {reported!r} for a request for {self.model.id!r}"

    def _open_upstream(self) -> http.client.HTTPConnection:
        connection_class = http.client.HTTPSConnection if self._upstream_https else http.client.HTTPConnection
        return connection_class(self._upstream_host, self._upstream_port, timeout=self.upstream_timeout_s)

    def _upstream_headers(self, body_length: int, stream: bool) -> dict[str, str]:
        # Built fresh: no trial header (its token, organization, project) ever goes upstream.
        return {
            "Authorization": f"Bearer {self._upstream_key}",
            "Content-Type": "application/json",
            "Content-Length": str(body_length),
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "hive-model-gateway",
        }

    def _scrub(self, data: bytes) -> bytes:
        """Remove the provider key from anything relayed to the trial, in case the provider echoes it."""
        return data.replace(self._upstream_key.encode(), _REDACTED)


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway
        super().__init__(("127.0.0.1", 0), _GatewayHandler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind does a reverse DNS lookup of the host, which can stall; skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # a trial dropping its connection is routine; anything else is recorded per call


class _GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _GatewayServer

    def setup(self) -> None:
        self.timeout = self.server.gateway.client_timeout_s  # applied to the socket by StreamRequestHandler
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # requests are recorded in the ledger; nothing here may print a header

    def do_POST(self) -> None:
        self._handle()

    do_GET = do_PUT = do_PATCH = do_DELETE = do_POST

    # --- one request -----------------------------------------------------------------------------

    def _handle(self) -> None:
        gateway = self.server.gateway
        admitted = gateway._admit(self._bearer_token(), self.path)
        if admitted is None:
            self.close_connection = True  # the body, if any, is never read
            self._send_json(401, _error_body("invalid_api_key", "unknown or revoked gateway token"), close=True)
            return
        if admitted is _BUSY:
            self.close_connection = True
            self._send_json(429, _error_body("too_many_in_flight",
                                             f"this trial already has {gateway.max_in_flight_per_trial} requests open"),
                            close=True)  # fmt: skip
            return
        trial, flight_id, flight = admitted
        try:
            self._serve(gateway, trial, flight)
        except Exception as error:  # noqa: BLE001  (a gateway bug must still be recorded, not lost)
            gateway._record(trial, GatewayCall(path=flight.path, model_requested=flight.model_requested,
                                               model_reported=None, status=500, usage=Usage(),
                                               error_code="gateway_error", streamed=flight.streamed))  # fmt: skip
            self.close_connection = True
            try:
                self._send_json(500, _error_body("gateway_error", f"gateway error: {type(error).__name__}"), close=True)
            except OSError:
                pass
        finally:
            gateway._release(trial, flight_id)

    def _bearer_token(self) -> str | None:
        scheme, _, token = (self.headers.get("Authorization") or "").strip().partition(" ")
        return (token.strip() or None) if scheme.lower() == "bearer" else None

    def _serve(self, gateway: ModelGateway, trial: _Trial, flight: _InFlight) -> None:
        def refuse(refusal: policy.Refusal, *, body_unread: bool = False) -> None:
            gateway._record(trial, GatewayCall(path=flight.path, model_requested=flight.model_requested,
                                               model_reported=None, status=403, usage=Usage(),
                                               error_code=refusal.code, streamed=flight.streamed),
                            refusal.message if refusal.violation else None)  # fmt: skip
            if body_unread:
                self.close_connection = True
            self._send_json(403, _error_body(refusal.code, refusal.message), close=body_unread)

        if self.command != "POST" or self.path not in policy.ALLOWED_PATHS:
            refuse(policy.Refusal("unknown_path", f"{self.command} {flight.path} is not served; only POST "
                                  f"{policy.CHAT_PATH} and POST {policy.RESPONSES_PATH}", False), body_unread=True)  # fmt: skip
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0 or "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            refuse(policy.Refusal("invalid_body", "a JSON body with a Content-Length is required", False), body_unread=True)
            return
        if length > gateway.max_body_bytes:
            refuse(policy.Refusal("body_too_large", f"the body is over {gateway.max_body_bytes} bytes", False),
                   body_unread=True)  # fmt: skip
            return
        raw = self.rfile.read(length)
        if len(raw) < length:
            self.close_connection = True
            return  # the trial hung up mid-body: nothing to answer, nothing forwarded
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            refuse(policy.Refusal("invalid_body", "the body must be a JSON object", False))
            return
        flight.model_requested = body["model"] if isinstance(body.get("model"), str) else ""
        flight.streamed = body.get("stream") is True

        prepared = policy.prepare(self.path, body, gateway.model)
        if isinstance(prepared, policy.Refusal):
            refuse(prepared)
            return
        if not gateway._reserve_call(trial):
            refuse(policy.Refusal("model_call_limit",
                                  f"this trial has used its {gateway.max_calls_per_trial} model calls", False))  # fmt: skip
            return
        self._forward(gateway, trial, flight, json.dumps(prepared).encode())

    # --- forwarding ------------------------------------------------------------------------------

    def _forward(self, gateway: ModelGateway, trial: _Trial, flight: _InFlight, data: bytes) -> None:
        def record(status: int, reported: str | None, tokens: Usage | None, error_code: str | None) -> None:
            if status < 300 and tokens is None and error_code is None:
                error_code = "usage_missing"  # a success the provider didn't bill us for is worth seeing
            gateway._record(trial, GatewayCall(path=flight.path, model_requested=flight.model_requested,
                                               model_reported=reported, status=status, usage=gateway._priced(tokens),
                                               error_code=error_code, streamed=flight.streamed),
                            gateway._model_violation(flight.path, reported))  # fmt: skip

        connection = gateway._open_upstream()
        try:
            try:
                connection.request("POST", gateway._upstream_prefix + self.path, body=data,
                                   headers=gateway._upstream_headers(len(data), flight.streamed))  # fmt: skip
                response = connection.getresponse()
            except (OSError, http.client.HTTPException):
                record(502, None, None, "upstream_unreachable")
                self._send_json(502, _error_body("upstream_unreachable", "the model provider could not be reached"))
                return
            content_type = response.getheader("Content-Type") or "application/json"
            relayed = {name: response.getheader(name) for name in _RELAYED_HEADERS if response.getheader(name)}
            if response.status < 300 and content_type.startswith("text/event-stream"):
                self._relay_stream(gateway, response, content_type, relayed, record)
                return
            try:
                raw = response.read()
            except (OSError, http.client.HTTPException):
                record(502, None, None, "upstream_unreachable")
                self._send_json(502, _error_body("upstream_unreachable", "the model provider stopped responding"))
                return
            reported, tokens, error_code = parse_body(self.path, raw)
            if response.status >= 400:
                error_code = error_code or f"http_{response.status}"
            record(response.status, reported, tokens, error_code)
            try:
                self._send(response.status, content_type, gateway._scrub(raw), relayed)
            except OSError:
                pass  # the trial is gone; the call is already recorded
        finally:
            connection.close()

    def _relay_stream(self, gateway: ModelGateway, response: http.client.HTTPResponse, content_type: str,
                      relayed: dict[str, str], record: Any) -> None:  # fmt: skip
        """Pass SSE lines through as they arrive, and read to the end even if the trial hangs up."""
        self.close_connection = True  # the stream ends when the connection does
        observer = StreamObserver(self.path)
        client_alive = True
        try:
            self.send_response(response.status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            for name, value in relayed.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.flush()
        except OSError:
            client_alive = False
        upstream_error = None
        while True:
            try:
                line = response.readline()
            except (OSError, http.client.HTTPException):
                upstream_error = "upstream_unreachable"
                break
            if not line:
                break
            observer.feed(line)
            if client_alive:
                try:
                    self.wfile.write(gateway._scrub(line))
                    self.wfile.flush()
                except OSError:
                    client_alive = False  # keep reading: the provider bills the whole response
        observer.finish()
        record(response.status, observer.model, observer.tokens, upstream_error or observer.error_code)

    # --- writing to the trial ------------------------------------------------------------------

    def _send_json(self, status: int, payload: dict[str, Any], *, close: bool = False) -> None:
        self._send(status, "application/json", json.dumps(payload).encode(), {}, close=close)

    def _send(self, status: int, content_type: str, data: bytes, headers: dict[str, str], *, close: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if close:
            self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "hive_gateway", "param": None, "code": code}}
