"""The `hive` command line: run an experiment, evaluate one candidate, or validate one.

    hive run CONFIG [--run-id ID] [--output-dir DIR]
    hive eval CONFIG [--candidate DIR] [--split SPLIT] [--trials K] [--output-dir DIR]
    hive validate CANDIDATE_DIR

Exit codes: 0 success; 1 `validate` found problems; 2 a user error (bad config,
arguments, candidate or plug-in name), reported as one line on stderr with no
traceback. Anything else is a bug and keeps its traceback.

The CLI holds no tunables of its own: every number comes from the experiment config,
and the flags only override where outputs go or which candidate and split to score.

For a real provider, `run` and `eval` read the provider's key from its environment variable
(OPENAI_API_KEY for openai) and route every trial through one model gateway; a missing key
is a user error, reported before any output directory is created.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from hive.candidate import Candidate, CandidateError
from hive.config import ConfigError, ExperimentConfig, load_experiment
from hive.contracts import SPLITS
from hive.evaluate import REPORT_FILE, EvalReport, EvaluationError, evaluate
from hive.loop import LOGGER, RUN_ID_FORMAT, ExperimentSummary, LoopError, model_gateway, provider_key, run_experiment
from hive.registry import RegistryError, make_agent_adapter, make_suite

USER_ERRORS = (ConfigError, LoopError, RegistryError, CandidateError, EvaluationError)

EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_USER_ERROR = 2

EVAL_DIR_PREFIX = "eval-"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_request:  # argparse exits on bad arguments and on --help
        return exit_request.code if isinstance(exit_request.code, int) else EXIT_USER_ERROR
    if args.command is None:
        parser.print_usage(sys.stderr)
        return EXIT_USER_ERROR

    # Bound to the current stderr on every call and removed afterwards, so repeated calls
    # (tests, or a caller embedding the CLI) never stack handlers or write to a stale stream.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    previous_level = LOGGER.level
    LOGGER.setLevel(logging.INFO)
    try:
        return args.handler(args)
    except USER_ERRORS as error:
        print(f"hive: error: {_one_line(error)}", file=sys.stderr)
        return EXIT_USER_ERROR
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(previous_level)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hive", description="Self-improving agent harnesses.")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run an experiment: seed, improvement loop, held-out")
    run.add_argument("config", type=Path, help="experiment config (experiments/*.toml)")
    run.add_argument("--run-id", help="run directory name (default: UTC timestamp)")
    run.add_argument("--output-dir", type=Path, help="override the config's output_dir")
    run.set_defaults(handler=_run_command)

    eval_ = commands.add_parser("eval", help="evaluate one candidate once on one split")
    eval_.add_argument("config", type=Path, help="experiment config (experiments/*.toml)")
    eval_.add_argument("--candidate", type=Path, help="candidate directory (default: the config's seed)")
    eval_.add_argument("--split", choices=SPLITS, default="search", help="split to score (default: search)")
    eval_.add_argument("--trials", type=int, help="trials per task (default: the config's trials_per_task)")
    eval_.add_argument("--output-dir", type=Path, help="override the config's output_dir")
    eval_.set_defaults(handler=_eval_command)

    validate = commands.add_parser("validate", help="check that a candidate loads and its adapter accepts it")
    validate.add_argument("candidate", type=Path, help="candidate directory")
    validate.set_defaults(handler=_validate_command)
    return parser


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_experiment(args.config)
    if args.output_dir is not None:
        config = dataclasses.replace(config, output_dir=args.output_dir.expanduser().resolve())
    return config


# --- hive run --------------------------------------------------------------------------------


def _run_command(args: argparse.Namespace) -> int:
    summary = run_experiment(_load_config(args), run_id=args.run_id)
    print(_format_summary(summary))
    return EXIT_OK


def _format_summary(summary: ExperimentSummary) -> str:
    promotions = sum(1 for row in summary.generations if row["promoted"])
    lines = [
        f"search: seed {summary.seed_search_score:.4f} ({summary.seed_id}) -> "
        f"final {summary.final_search_score:.4f} ({summary.final_id})",
        f"held-out: {_format_heldout(summary.heldout)}",
        f"promotions: {promotions} of {len(summary.generations)} generations"
        + (f" ({_format_statuses(summary)})" if summary.generations else ""),
        f"cost: ${summary.total_cost_usd:.2f} (stopped: {summary.stopped_reason})",
        f"run dir: {summary.run_dir}",
    ]
    return "\n".join(lines)


def _format_heldout(heldout) -> str:
    parts = []
    for label in ("seed", "final"):
        entry = heldout.get(label)
        if entry is not None:
            same = " (same candidate as the seed)" if entry.get("same_report_as_seed") else ""
            parts.append(f"{label} {entry['selection_score']:.4f}{same}")
    if heldout.get("skipped_reason"):
        parts.append(f"skipped: {heldout['skipped_reason']}")
    return ", ".join(parts)


def _format_statuses(summary: ExperimentSummary) -> str:
    counts: dict[str, int] = {}
    for row in summary.generations:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return ", ".join(f"{status} {count}" for status, count in counts.items())


# --- hive eval -------------------------------------------------------------------------------


def _eval_command(args: argparse.Namespace) -> int:
    config = _load_config(args)
    candidate = Candidate.load(args.candidate if args.candidate is not None else config.seed_candidate)
    adapter = make_agent_adapter(candidate.agent)
    problems = adapter.validate(candidate)
    if problems:
        raise CandidateError(f"{candidate.root} is not a valid {candidate.agent!r} candidate: " + "; ".join(problems))
    suite = make_suite(config.suite_name, config.suite_settings)
    key = provider_key(config)  # a missing key is a user error, before anything is written

    timestamp = datetime.datetime.now(datetime.UTC).strftime(RUN_ID_FORMAT)
    out_dir = config.output_dir / config.name / f"{EVAL_DIR_PREFIX}{timestamp}"
    if out_dir.exists():
        raise EvaluationError(f"{out_dir} already exists; wait a second and run again")
    with model_gateway(config, key) as gateway:
        del key  # only the gateway holds it from here on
        report = evaluate(
            candidate,
            adapter,
            suite,
            args.split,
            model=config.model,
            agent_settings=config.agent_settings,
            trials_per_task=args.trials if args.trials is not None else config.trials_per_task,
            out_dir=out_dir,
            concurrency=config.concurrency,
            gateway=gateway,
        )
    print(_format_report(report, out_dir / REPORT_FILE))
    return EXIT_OK


def _format_report(report: EvalReport, report_path: Path) -> str:
    width = max([len("task")] + [len(task.task_id) for task in report.tasks])
    lines = [
        f"candidate {report.candidate_id} on {report.suite}/{report.split}, model {report.model}, "
        f"{report.trials_per_task} trial(s) per task",
        "",
        f"{'task':<{width}}  reward  measured  scored  errored",
    ]
    for task in report.tasks:
        measured = f"{task.measured_mean:.4f}" if task.measured_mean is not None else "-"
        lines.append(
            f"{task.task_id:<{width}}  {task.selection_reward:.4f}  {measured:<8}  {task.n_scored:<6}  {task.n_errored}"
        )
    measured_mean = f"{report.measured_mean:.4f}" if report.measured_mean is not None else "-"
    lines += [
        "",
        f"selection_score {report.selection_score:.4f}  measured_mean {measured_mean}  stderr {report.stderr:.4f}",
        f"trials {report.n_trials}  errored {report.n_errored}  cost ${report.cost_usd:.4f}",
        f"report: {report_path}",
    ]
    return "\n".join(lines)


# --- hive validate ---------------------------------------------------------------------------


def _validate_command(args: argparse.Namespace) -> int:
    candidate = Candidate.load(args.candidate)
    problems = make_agent_adapter(candidate.agent).validate(candidate)
    if not problems:
        print("OK")
        return EXIT_OK
    for problem in problems:
        print(f"- {problem}")
    return EXIT_PROBLEMS


def _one_line(error: Exception) -> str:
    if isinstance(error, ConfigError):
        return f"{error.source}: " + "; ".join(error.problems)
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return "; ".join(lines) or type(error).__name__
