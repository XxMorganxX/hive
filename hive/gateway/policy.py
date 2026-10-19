"""What the gateway lets through: the request rules for P3 (frozen model and effort) and spend.

`prepare` takes one parsed request body and either returns the body to forward (with the
experiment's effort filled in and, for chat streams, usage reporting switched on) or a
`Refusal` that says why not. It has no I/O, so every rule is checked in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hive.contracts import ModelSpec

CHAT_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"
ALLOWED_PATHS = frozenset({CHAT_PATH, RESPONSES_PATH})

# Tool types the trial executes itself, so every token they cost comes back as usage. Hosted tools
# (web_search, file_search, code_interpreter, ...) run at the provider and bill per call, outside
# the token usage the gateway counts, so they are refused (P12).
_CLIENT_EXECUTED_TOOL_TYPES = frozenset({"function", "custom"})

# Service tiers that bill at the standard rate. "flex", "priority" and "scale" change the price.
_DEFAULT_SERVICE_TIERS = (None, "auto", "default")


@dataclass(frozen=True)
class Refusal:
    code: str  # the error code the trial sees and the ledger records
    message: str
    violation: bool  # a P3 or cost-parameter breach (goes in TrialLedger.violations), not just a bad request


def prepare(path: str, body: dict[str, Any], model: ModelSpec) -> dict[str, Any] | Refusal:
    """The body to forward for an allowed `path`, or why the request is refused."""
    requested = body.get("model")
    if not isinstance(requested, str) or requested != model.id:
        return Refusal("model_not_allowed",
                       f"{path}: requested model {requested!r}; this experiment's model is {model.id!r}", True)  # fmt: skip

    refusal = _check_effort(path, body, model)
    if refusal is not None:
        return refusal

    tier = body.get("service_tier")
    if tier not in _DEFAULT_SERVICE_TIERS:
        return Refusal("service_tier_not_allowed", f"{path}: service_tier {tier!r} changes the price; use the default", True)
    n = body.get("n")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n != 1):
        return Refusal("n_not_allowed", f"{path}: n={n!r} multiplies the spend; only one choice per call is allowed", True)
    # Background responses run upstream after the reply, and their usage never comes back through here.
    if body.get("background") not in (None, False):
        return Refusal("background_not_allowed", f"{path}: background responses can't be billed by the gateway", True)
    refusal = _check_tools(path, body.get("tools"))
    if refusal is not None:
        return refusal

    forwarded = dict(body)
    if model.thinking is not None:
        if path == CHAT_PATH and forwarded.get("reasoning_effort") is None:
            forwarded["reasoning_effort"] = model.thinking
        if path == RESPONSES_PATH and _responses_effort(forwarded) is None:
            forwarded["reasoning"] = {**(forwarded.get("reasoning") or {}), "effort": model.thinking}
    if path == CHAT_PATH and forwarded.get("stream") is True:
        # Without it, a streamed chat call reports no usage and would go unbilled.
        options = forwarded.get("stream_options")
        forwarded["stream_options"] = {**(options if isinstance(options, dict) else {}), "include_usage": True}
    return forwarded


def _check_tools(path: str, tools: Any) -> Refusal | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        return Refusal("hosted_tool_not_allowed", f"{path}: tools must be a list of function tools", True)
    for tool in tools:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind not in _CLIENT_EXECUTED_TOOL_TYPES:
            return Refusal("hosted_tool_not_allowed",
                           f"{path}: tool type {kind!r} runs at the provider and is billed outside token usage; "
                           "only function and custom tools are allowed", True)  # fmt: skip
    return None


def _check_effort(path: str, body: dict[str, Any], model: ModelSpec) -> Refusal | None:
    """Every effort field present must be the experiment's; with no fixed effort, none may be set.

    Both spellings are checked on both paths, so a provider that accepts either can't be steered
    through the one the path doesn't normally use.
    """
    reasoning = body.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        return Refusal("effort_not_allowed", f"{path}: reasoning must be an object, got {reasoning!r}", True)
    for field, value in (("reasoning_effort", body.get("reasoning_effort")), ("reasoning.effort", _responses_effort(body))):
        if value is not None and value != model.thinking:
            wanted = repr(model.thinking) if model.thinking is not None else "unset (the provider default)"
            return Refusal("effort_not_allowed", f"{path}: {field} {value!r}; this experiment's effort is {wanted}", True)
    return None


def _responses_effort(body: dict[str, Any]) -> Any:
    reasoning = body.get("reasoning")
    return reasoning.get("effort") if isinstance(reasoning, dict) else None
