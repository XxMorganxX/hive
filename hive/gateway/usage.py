"""Reading what the provider reported: the model it ran and the tokens it billed.

Usage is normalised to HIVE's `Usage`: `input_tokens` excludes cache reads, which are
counted separately in `cache_read_tokens`.

- chat:      input = prompt_tokens - prompt_tokens_details.cached_tokens, output = completion_tokens
- responses: input = input_tokens - input_tokens_details.cached_tokens,   output = output_tokens

Streams are read line by line with `StreamObserver`: chat usage arrives in the last chunk
(the one with empty `choices`, sent because the gateway asks for `include_usage`); responses
usage arrives in the terminal event (`response.completed`, or `response.incomplete` /
`response.failed`, which are billed too).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from hive.contracts import Usage
from hive.gateway.policy import CHAT_PATH

_RESPONSE_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


def tokens_from(path: str, usage: Any) -> Usage | None:
    """Normalised token counts (no cost yet) from a provider `usage` object, or None if there is none."""
    if not isinstance(usage, Mapping):
        return None
    if path == CHAT_PATH:
        total_in, details, output = usage.get("prompt_tokens"), usage.get("prompt_tokens_details"), usage.get("completion_tokens")
    else:
        total_in, details, output = usage.get("input_tokens"), usage.get("input_tokens_details"), usage.get("output_tokens")
    cached = _count(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
    total_in = _count(total_in)
    cached = min(cached, total_in)
    return Usage(input_tokens=total_in - cached, output_tokens=_count(output), cache_read_tokens=cached)


def error_code_from(payload: Any) -> str | None:
    """The provider's error code (`error.code`, else `error.type`) from an OpenAI-style error body."""
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if not isinstance(error, Mapping):
        return None
    for key in ("code", "type"):
        if isinstance(error.get(key), str) and error[key]:
            return error[key]
    return None


def parse_body(path: str, raw: bytes) -> tuple[str | None, Usage | None, str | None]:
    """(reported model, tokens, provider error code) from a whole JSON response body."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None, None, None
    if not isinstance(payload, Mapping):
        return None, None, None
    return _model(payload), tokens_from(path, payload.get("usage")), error_code_from(payload)


class StreamObserver:
    """Watches an SSE stream go by, one line at a time, and keeps the model, usage and any error."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.model: str | None = None
        self.tokens: Usage | None = None
        self.error_code: str | None = None
        self._data: list[str] = []

    def feed(self, line: bytes) -> None:
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if text == "":
            self._dispatch()
        elif text.startswith("data:"):
            self._data.append(text[5:].removeprefix(" "))

    def finish(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        data, self._data = "\n".join(self._data), []
        if not data or data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except ValueError:
            return
        if not isinstance(event, Mapping):
            return
        self.error_code = error_code_from(event) or self.error_code
        if self.path == CHAT_PATH:
            self.model = _model(event) or self.model
            self.tokens = tokens_from(self.path, event.get("usage")) or self.tokens
            return
        response = event.get("response")
        if isinstance(response, Mapping):
            self.model = _model(response) or self.model
            if event.get("type") in _RESPONSE_TERMINAL_EVENTS:
                self.tokens = tokens_from(self.path, response.get("usage")) or self.tokens
                failed = response.get("error")
                if isinstance(failed, Mapping) and isinstance(failed.get("code"), str):
                    self.error_code = failed["code"]
        if event.get("type") == "error" and isinstance(event.get("code"), str):
            self.error_code = event["code"]


def _model(payload: Mapping[str, Any]) -> str | None:
    model = payload.get("model")
    return model if isinstance(model, str) and model else None


def _count(value: Any) -> int:
    # Only trust non-negative ints: a bogus count must not shrink or inflate the spend.
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
