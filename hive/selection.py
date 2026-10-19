"""The promotion rule: whether a child candidate replaces the incumbent (P9).

Bare-bones version: compare `selection_score` (errored trials count as 0) on the
search split with a fixed margin, and never promote below the seed. The overseer is
told this rule in plain words by `describe_rule`, whose text lives in
`selection_rule.md`; when `decide` changes, that file changes in the same commit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from string import Template

from hive.evaluate import EvalReport

# The rule is only ever applied to search-split scores; validation and held-out stay walls (P7).
SELECTION_SPLIT = "search"

_RULE_TEMPLATE = Path(__file__).with_name("selection_rule.md")

# Scores are float means, so `incumbent + margin` can round below its decimal value
# (0.7 + 0.1 == 0.7999999999999999) and let a gain of exactly the margin promote. The
# tolerance is far below any real score step (1/(tasks * trials)) and errs toward the
# incumbent, as the stated rule does: "a gain of the margin or less keeps the incumbent".
_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Decision:
    promoted: bool
    reason: str


def decide(child: EvalReport, incumbent: EvalReport, seed: EvalReport, margin: float) -> Decision:
    """Promote iff child > incumbent + margin and child >= seed, all on selection_score."""
    _check_margin(margin)
    _check_comparable(child, incumbent, seed)

    child_score = child.selection_score
    incumbent_score = incumbent.selection_score
    seed_score = seed.selection_score
    bar = incumbent_score + margin
    beats_incumbent = child_score - incumbent_score > margin + _TIE_TOLERANCE
    holds_seed = child_score >= seed_score

    comparison = (
        f"child {child_score:.4f} {'>' if beats_incumbent else '<='} "
        f"incumbent {incumbent_score:.4f} + margin {margin:.4f} = {bar:.4f}; "
        f"child {child_score:.4f} {'>=' if holds_seed else '<'} seed {seed_score:.4f}"
    )
    if beats_incumbent and holds_seed:
        return Decision(promoted=True, reason=f"promoted: {comparison}")
    failed = []
    if not beats_incumbent:
        failed.append("does not beat the incumbent by more than the margin")
    if not holds_seed:
        failed.append("scores below the seed")
    return Decision(promoted=False, reason=f"rejected (child {' and '.join(failed)}): {comparison}")


def describe_rule(margin: float) -> str:
    """The promotion rule in plain language, for the overseer brief."""
    _check_margin(margin)
    return Template(_RULE_TEMPLATE.read_text()).substitute(margin=f"{margin:.4f}")


def _check_margin(margin: float) -> None:
    # NaN would make every comparison False and silently block all promotions.
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"promotion margin must be a finite number >= 0, got {margin!r}")


def _check_comparable(child: EvalReport, incumbent: EvalReport, seed: EvalReport) -> None:
    reports = {"child": child, "incumbent": incumbent, "seed": seed}
    for name, report in reports.items():
        if report.split != SELECTION_SPLIT:
            raise ValueError(f"{name} report is on split {report.split!r}; selection uses only {SELECTION_SPLIT!r}")
    for field in ("suite", "split", "model", "trials_per_task"):
        values = {name: getattr(report, field) for name, report in reports.items()}
        if len(set(values.values())) > 1:
            raise ValueError(f"reports are not comparable: {field} differs {values}")
    task_ids = {name: tuple(task.task_id for task in report.tasks) for name, report in reports.items()}
    if len(set(task_ids.values())) > 1:
        raise ValueError("reports are not comparable: they cover different tasks")
