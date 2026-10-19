import math
from dataclasses import replace

import pytest

from hive.contracts import Usage
from hive.evaluate import EvalReport, TaskOutcome, TrialOutcome
from hive.selection import Decision, decide, describe_rule


def report_scoring(*task_rewards: float, candidate_id: str = "c", **overrides) -> EvalReport:
    """A one-trial-per-task search report whose selection_score is the mean of task_rewards."""
    tasks = tuple(
        TaskOutcome(
            task_id=f"t{index}",
            trials=(
                TrialOutcome(
                    task_id=f"t{index}",
                    trial=1,
                    status="scored",
                    reward=reward,
                    error_class=None,
                    detail="",
                    usage=Usage(),
                    wall_clock_s=1.0,
                    trial_dir=f"trials/t{index}/1",
                ),
            ),
        )
        for index, reward in enumerate(task_rewards)
    )
    report = EvalReport(
        candidate_id=candidate_id,
        suite="fake-suite",
        split="search",
        model="fake/model-1",
        trials_per_task=1,
        tasks=tasks,
    )
    return replace(report, **overrides)


SEED = report_scoring(0.0, 1.0, 0.0, 0.0, candidate_id="seed")  # 0.25
INCUMBENT = report_scoring(1.0, 1.0, 0.0, 0.0, candidate_id="inc")  # 0.5


def test_child_beating_incumbent_by_more_than_margin_is_promoted():
    child = report_scoring(1.0, 1.0, 1.0, 0.0, candidate_id="child")  # 0.75
    assert decide(child, INCUMBENT, SEED, margin=0.1).promoted


def test_child_beating_incumbent_by_exactly_the_margin_is_not_promoted():
    child = report_scoring(1.0, 1.0, 1.0, 0.0, candidate_id="child")  # 0.75
    assert not decide(child, INCUMBENT, SEED, margin=0.25).promoted


def test_child_tying_incumbent_with_zero_margin_is_not_promoted():
    child = report_scoring(1.0, 0.0, 1.0, 0.0, candidate_id="child")  # 0.5
    assert not decide(child, INCUMBENT, SEED, margin=0.0).promoted


def test_child_below_seed_is_not_promoted_even_if_it_beats_incumbent():
    weak_incumbent = report_scoring(0.0, 0.0, 0.0, 0.0, candidate_id="inc")  # 0.0
    child = report_scoring(0.0, 0.0, 0.0, 0.5, candidate_id="child")  # 0.125 < seed 0.25
    assert not decide(child, weak_incumbent, SEED, margin=0.0).promoted


def test_child_equal_to_seed_that_beats_incumbent_is_promoted():
    weak_incumbent = report_scoring(0.0, 0.0, 0.0, 0.0, candidate_id="inc")
    child = report_scoring(0.0, 1.0, 0.0, 0.0, candidate_id="child")  # 0.25 == seed
    assert decide(child, weak_incumbent, SEED, margin=0.0).promoted


def test_promotion_reason_states_the_scores_and_margin():
    child = report_scoring(1.0, 1.0, 1.0, 0.0, candidate_id="child")
    reason = decide(child, INCUMBENT, SEED, margin=0.1).reason
    for number in ("0.7500", "0.5000", "0.1000", "0.2500"):
        assert number in reason


def test_rejection_reason_states_the_scores_and_margin():
    child = report_scoring(1.0, 0.0, 1.0, 0.0, candidate_id="child")
    reason = decide(child, INCUMBENT, SEED, margin=0.05).reason
    for number in ("0.5000", "0.0500", "0.5500"):
        assert number in reason


def test_decide_returns_a_decision():
    assert isinstance(decide(INCUMBENT, INCUMBENT, SEED, margin=0.0), Decision)


@pytest.mark.parametrize(
    "field, value",
    [("suite", "other-suite"), ("split", "validation"), ("trials_per_task", 3), ("model", "other/model")],
)
def test_reports_that_differ_in_setup_are_not_comparable(field, value):
    child = replace(report_scoring(1.0, 1.0, 1.0, 1.0), **{field: value})
    with pytest.raises(ValueError):
        decide(child, INCUMBENT, SEED, margin=0.0)


def test_reports_over_different_task_ids_are_not_comparable():
    child = report_scoring(1.0, 1.0, 1.0)  # three tasks instead of four
    with pytest.raises(ValueError):
        decide(child, INCUMBENT, SEED, margin=0.0)


def test_reports_on_a_split_other_than_search_are_rejected():
    child, incumbent, seed = (replace(r, split="heldout") for r in (INCUMBENT, INCUMBENT, SEED))
    with pytest.raises(ValueError):
        decide(child, incumbent, seed, margin=0.0)


@pytest.mark.parametrize("margin", [-0.01, math.nan, math.inf])
def test_invalid_margin_is_rejected_by_decide(margin):
    with pytest.raises(ValueError):
        decide(INCUMBENT, INCUMBENT, SEED, margin=margin)


@pytest.mark.parametrize("margin", [-0.01, math.nan])
def test_invalid_margin_is_rejected_by_describe_rule(margin):
    with pytest.raises(ValueError):
        describe_rule(margin)


def test_rule_description_states_the_margin():
    assert "0.0500" in describe_rule(0.05)


def test_rule_description_says_errored_trials_count_as_zero():
    assert "errored trial counts as 0" in describe_rule(0.05).lower().replace("\n", " ")


def test_rule_description_says_only_the_search_split_is_used():
    assert "search split" in describe_rule(0.05).lower()


def test_rule_description_says_held_out_split_is_hidden_and_scored_once_at_the_end():
    text = describe_rule(0.05).lower()
    assert "held-out" in text and "once" in text and "end" in text


def test_rule_description_says_the_child_must_not_fall_below_the_seed():
    assert "seed" in describe_rule(0.05).lower()


def test_gain_exactly_equal_to_a_decimal_margin_is_not_promoted_despite_float_rounding():
    # 0.7 + 0.1 == 0.7999999999999999 in binary floating point, so a naive bar lets 0.8 through.
    incumbent = report_scoring(*([1.0] * 7 + [0.0] * 3), candidate_id="inc")  # 0.7
    child = report_scoring(*([1.0] * 8 + [0.0] * 2), candidate_id="child")  # 0.8
    decision = decide(child, incumbent, incumbent, margin=0.1)
    assert not decision.promoted
    assert "<=" in decision.reason


def test_gain_clearly_above_a_decimal_margin_is_still_promoted():
    incumbent = report_scoring(*([1.0] * 7 + [0.0] * 3), candidate_id="inc")  # 0.7
    child = report_scoring(*([1.0] * 9 + [0.0] * 1), candidate_id="child")  # 0.9
    assert decide(child, incumbent, incumbent, margin=0.1).promoted
