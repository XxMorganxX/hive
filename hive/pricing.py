"""Model pricing: the one place a token count becomes dollars (P12).

Cost is a property of the model, not of the agent framework, so prices live here, where the
model gateway (hive/gateway) and the controller both read them. `[model] prices` in the
experiment config is parsed with `parse_prices`.

Transition note: hive/agents/runner_protocol.py still carries its own copy of `Prices` and
`parse_prices` until the adapters import from here. The two are identical; keep them so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hive.contracts import Usage

_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
_REQUIRED_PRICES = ("input", "output")


@dataclass(frozen=True)
class Prices:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * self.cache_write
        ) / 1_000_000


def parse_prices(table: Any) -> Prices:
    """A `prices` table from the experiment config. Required: without it, spend would read as $0."""
    if not isinstance(table, Mapping):
        raise ValueError("prices must be a table of USD per million tokens, e.g. { input = 0.05, output = 0.40 }")
    unknown = sorted(set(table) - set(_PRICE_FIELDS))
    if unknown:
        raise ValueError(f"unknown price keys {unknown}; allowed: {list(_PRICE_FIELDS)}")
    missing = [name for name in _REQUIRED_PRICES if name not in table]
    if missing:
        raise ValueError(f"prices must include {missing}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"price {name!r} must be a finite number >= 0, got {value!r}")
    return Prices(**{name: float(value) for name, value in table.items()})
