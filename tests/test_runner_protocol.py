import json
from pathlib import Path

import pytest

from hive.agents.process import ProcessOutcome
from hive.agents.runner_protocol import (
    RESULT_FILE,
    TRACE_FILE,
    Prices,
    interpret_runner,
    model_matches,
    parse_prices,
)
from hive.contracts import LoadReport, ModelSpec, Usage

MODEL = ModelSpec(provider="openai", id="gpt-5-nano", thinking="low")
PRICES = Prices(input=0.05, output=0.40, cache_read=0.005, cache_write=0.0)


def finished(returncode: int = 0, stderr: bytes = b"") -> ProcessOutcome:
    return ProcessOutcome(b"", stderr, returncode, False, None)


def write_trace(trace_dir: Path, *calls: dict) -> None:
    lines = [json.dumps({"type": "model_call", **call}) for call in calls]
    (trace_dir / TRACE_FILE).write_text("\n".join(lines) + "\n")


def write_result(trace_dir: Path, **fields) -> None:
    result = {"status": "completed", "final_text": "ANSWER: 4", "error": None, "error_class": None,
              "extensions": [], "runtime_errors": []}  # fmt: skip
    result.update(fields)
    (trace_dir / RESULT_FILE).write_text(json.dumps(result))


def call(model: str = "gpt-5-nano-2025-08-07", input_tokens=1000, output_tokens=500, cache_read=0, cache_write=0) -> dict:
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
             "cache_read_tokens": cache_read, "cache_write_tokens": cache_write}  # fmt: skip
    return {"model": model, "usage": usage}


def interpret(tmp_path: Path, outcome: ProcessOutcome | None = None):
    return interpret_runner(tmp_path, outcome or finished(), model=MODEL, prices=PRICES, wall_clock_s=2.0)


# --- prices ----------------------------------------------------------------------------------


def test_prices_parse_from_settings():
    assert parse_prices({"input": 0.05, "output": 0.4}) == Prices(input=0.05, output=0.4, cache_read=0.0, cache_write=0.0)


@pytest.mark.parametrize(
    "table", [None, {}, {"input": 0.05}, {"input": -1, "output": 1}, {"input": True, "output": 1}, {"input": 1, "output": 1, "other": 2}]
)
def test_invalid_prices_are_rejected(table):
    with pytest.raises(ValueError):
        parse_prices(table)


# --- model matching ----------------------------------------------------------------------------


@pytest.mark.parametrize("reported", ["gpt-5-nano", "gpt-5-nano-2025-08-07", "openai/gpt-5-nano", "gpt-5-nano-20250807"])
def test_reported_model_matches_the_spec_or_a_dated_snapshot_of_it(reported):
    assert model_matches(reported, MODEL)


@pytest.mark.parametrize("reported", ["gpt-5-mini", "gpt-5-nano-pro", "gpt-5", "", "anthropic/gpt-5-nano"])
def test_other_models_do_not_match(reported):
    assert not model_matches(reported, MODEL)


# --- interpreting a finished run -------------------------------------------------------------


def test_completed_run_reports_text_usage_and_cost_from_the_trace(tmp_path):
    write_trace(tmp_path, call(input_tokens=1000, output_tokens=500), call(input_tokens=2000, output_tokens=100, cache_read=800))
    write_result(tmp_path)
    result = interpret(tmp_path)
    assert result.status == "completed"
    assert result.final_text == "ANSWER: 4"
    assert result.usage == Usage(
        input_tokens=3000, output_tokens=600, cache_read_tokens=800, cache_write_tokens=0,
        cost_usd=pytest.approx((3000 * 0.05 + 600 * 0.40 + 800 * 0.005) / 1e6),
    )  # fmt: skip
    assert result.wall_clock_s == 2.0


def test_errored_runner_result_keeps_its_class_and_error(tmp_path):
    write_trace(tmp_path, call())
    write_result(tmp_path, status="errored", final_text="", error="rate limited", error_class="quota")
    result = interpret(tmp_path)
    assert (result.status, result.error_class, result.error) == ("errored", "quota", "rate limited")
    assert result.usage.cost_usd > 0  # spend is counted even when the run failed


def test_unknown_runner_error_class_becomes_agent_error(tmp_path):
    write_result(tmp_path, status="errored", error="x", error_class="made_up")
    result = interpret(tmp_path)
    assert result.error_class == "agent_error"
    assert "made_up" in result.error


def test_a_call_to_another_model_is_a_model_mismatch_even_if_the_run_completed(tmp_path):
    write_trace(tmp_path, call(), call(model="gpt-5-mini"))
    write_result(tmp_path)
    result = interpret(tmp_path)
    assert result.error_class == "model_mismatch"
    assert "gpt-5-mini" in result.error
    assert result.usage.output_tokens == 1000  # the mismatched call's spend still counts


def test_load_report_comes_from_the_runner_result(tmp_path):
    write_trace(tmp_path, call())
    write_result(
        tmp_path,
        extensions=[{"path": "extensions/guard.py", "loaded": True, "error": None, "registered": ["before_tool_call"]}],
        runtime_errors=["extensions/guard.py: KeyError"],
    )
    report = interpret(tmp_path).load_report
    assert report.extensions[0].path == "extensions/guard.py"
    assert report.extensions[0].registered == ("before_tool_call",)
    assert report.runtime_errors == ("extensions/guard.py: KeyError",)


def test_a_result_without_an_extensions_key_has_no_load_report(tmp_path):
    # The runner never reached its load phase, so it has nothing to report about extensions.
    write_trace(tmp_path, call())
    result = {"status": "errored", "final_text": "", "error": "the runner failed: ImportError",
              "error_class": "agent_error", "runtime_errors": []}  # fmt: skip
    (tmp_path / RESULT_FILE).write_text(json.dumps(result))
    interpreted = interpret(tmp_path)
    assert interpreted.error_class == "agent_error"
    assert interpreted.load_report is None


def test_an_empty_extensions_list_is_a_load_report_with_no_entries(tmp_path):
    write_result(tmp_path, extensions=[])
    assert interpret(tmp_path).load_report == LoadReport()


# --- runs that did not finish normally ---------------------------------------------------------


def test_timeout_is_reported_with_the_spend_recorded_so_far(tmp_path):
    write_trace(tmp_path, call())
    result = interpret(tmp_path, ProcessOutcome(b"", b"", None, True, None))
    assert result.error_class == "timeout"
    assert result.usage.cost_usd > 0


def test_launch_failure_is_an_agent_error(tmp_path):
    result = interpret(tmp_path, ProcessOutcome(b"", b"", None, False, FileNotFoundError("no python")))
    assert result.error_class == "agent_error"
    assert "no python" in result.error


def test_crash_without_a_result_file_is_an_agent_error_with_the_stderr_tail(tmp_path):
    write_trace(tmp_path, call())
    result = interpret(tmp_path, finished(returncode=1, stderr=b"Traceback ...\nImportError: no module deepagents"))
    assert result.error_class == "agent_error"
    assert "ImportError" in result.error
    assert result.usage.cost_usd > 0


def test_killed_by_a_signal_says_so(tmp_path):
    result = interpret(tmp_path, finished(returncode=-9))
    assert result.error_class == "agent_error"
    assert "SIGKILL" in result.error


def test_malformed_result_file_is_an_agent_error(tmp_path):
    (tmp_path / RESULT_FILE).write_text("{not json")
    assert interpret(tmp_path).error_class == "agent_error"


def test_malformed_trace_lines_are_skipped(tmp_path):
    (tmp_path / TRACE_FILE).write_text("garbage\n" + json.dumps({"type": "model_call", **call()}) + "\n{\n")
    write_result(tmp_path)
    assert interpret(tmp_path).usage.input_tokens == 1000


def test_negative_or_non_integer_token_counts_in_the_trace_are_ignored_not_trusted(tmp_path):
    write_trace(tmp_path, call(input_tokens=-5), call(input_tokens="12"), call(input_tokens=7))
    write_result(tmp_path)
    assert interpret(tmp_path).usage.input_tokens == 7


# --- prices live in hive.pricing; the gateway's ledger sets cost ----------------------------


def test_prices_are_the_shared_pricing_types():
    import hive.pricing

    assert Prices is hive.pricing.Prices
    assert parse_prices is hive.pricing.parse_prices


def test_interpret_runner_without_prices_counts_tokens_at_zero_cost(tmp_path):
    write_trace(tmp_path, call(input_tokens=1000, output_tokens=500))
    write_result(tmp_path)
    result = interpret_runner(tmp_path, finished(), model=MODEL, wall_clock_s=2.0)
    assert result.status == "completed"
    assert result.usage == Usage(input_tokens=1000, output_tokens=500, cost_usd=0.0)
