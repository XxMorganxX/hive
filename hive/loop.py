"""The improvement loop and the experience-store writer (docs/FOUNDING_PRINCIPLES.md §6).

This is trusted controller code. It resolves every plug-in by name through
`hive.registry`, so it never imports a concrete agent, suite or overseer (P5). The
experiment config names the suite, model and overseer; the seed candidate names its
own agent adapter.

One run, in order:

1. Check the configuration before spending anything: split walls (P7), a non-empty
   search split, a valid seed, a fresh run directory, a temp dir outside any git
   checkout. Problems raise `LoopError`; a bad seed directory or plug-in name raises
   `CandidateError` or `RegistryError` from the layer that found it. Nothing has run yet
   in any of these cases.
2. Evaluate the seed on the search split. It becomes the first incumbent, and its cost
   calibrates the per-trial cost estimate used by every budget check after it (P12).
3. Each generation: stop if the next search evaluation is not expected to fit under the
   cap; brief the overseer with *copies* of the incumbent and its evidence, in temporary
   directories outside the repository; reject empty, invalid, unchanged or duplicate
   proposals without spending on them; skip the child's evaluation if the overseer's own
   spend left too little budget; otherwise evaluate the child on search and apply
   `hive.selection.decide` (P9).
4. Score the seed and the final incumbent on the held-out split, once, at the very end,
   if the estimate fits. Nothing held-out is ever placed in a brief or a workspace (P7).

Every evaluation first re-hashes the candidate snapshot it is about to score. A snapshot
that changed on disk raises `LoopError` instead of being scored under a stale id (P4).

Spend is recorded trial by trial as `evaluate` reports it, so an evaluation cut short by
Ctrl-C or a crash still shows its real cost in run_meta.json.

Trial workdirs are agent-controlled, so the evidence copied into a brief skips anything that
is not a regular file, directory or symlink, or that can't be read, and lists what it
skipped in the evidence's SKIPPED_EVIDENCE_FILE instead of failing the run.

The cap is estimate-gated, not exact: the loop never *starts* work it expects to push the
spend past `max_usd`, but a run can still overshoot by the estimate's error, and the
overseer's own spend is bounded only by the overseer's per-proposal limit.

Run directory, `<output_dir>/<experiment name>/<run id>/`:

    run_meta.json              config snapshot, versions, git commit, times, stop reason or
                               error, spend; written at the start and again in a finally block
    candidates/<id>/           snapshot of every evaluated candidate, with lineage.json
    evals/gen-000/             the seed on search: report.json + trials/<task>/<n>/
    evals/gen-<NNN>/           the child evaluated in generation NNN (absent when not evaluated)
    overseer/gen-<NNN>/        workspace/ (copy of the overseer's workspace) + proposal.json
    heldout/seed/              the seed on held-out
    heldout/final/             the final incumbent on held-out (absent when it is the seed)
    generations.jsonl          one row per generation, appended as it happens
    summary.json               the ExperimentSummary

Evaluations are stored per generation, not per candidate id as §6.1 sketches, because
the null control re-measures the same id every generation and `evaluate` refuses to
reuse an output directory.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import platform
import re
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hive.candidate import Candidate, CandidateError, content_hash
from hive.config import ExperimentConfig, snapshot
from hive.contracts import SPLITS, AgentAdapter, Overseer, OverseerBrief, Proposal, Split, Suite
from hive.evaluate import REPORT_FILE, TRIALS_DIR, EvalReport, evaluate
from hive.registry import make_agent_adapter, make_overseer, make_suite
from hive.selection import decide, describe_rule

LOGGER = logging.getLogger("hive")

RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
WORKSPACE_PREFIX = "hive-overseer-"
BRIEF_PREFIX = "hive-brief-"

RUN_META_FILE = "run_meta.json"
GENERATIONS_FILE = "generations.jsonl"
SUMMARY_FILE = "summary.json"
PROPOSAL_FILE = "proposal.json"
CANDIDATES_DIR = "candidates"
EVALS_DIR = "evals"
OVERSEER_DIR = "overseer"
HELDOUT_DIR = "heldout"
WORKSPACE_DIR = "workspace"
# Written into the brief's evidence dir when some trial files could not be copied. The space
# keeps it from colliding with a task directory: safe_task_dirname never produces one.
SKIPPED_EVIDENCE_FILE = "SKIPPED FILES.txt"

SEED_ORIGIN = "seed"

STOPPED_COMPLETED = "completed"
STOPPED_BUDGET = "budget"

# What happened to a generation's proposal. Only "evaluated" spends on evaluation.
STATUS_EVALUATED = "evaluated"
STATUS_NO_PROPOSAL = "no_proposal"
STATUS_INVALID = "invalid"
STATUS_UNCHANGED = "unchanged"
STATUS_DUPLICATE = "duplicate"
STATUS_OVER_BUDGET = "over_budget"  # a valid child left unevaluated because the budget would not cover it

# The run id becomes a directory name, so it must not be able to climb out of the experiment dir.
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9._-]+")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GIT_TIMEOUT_S = 10


class LoopError(Exception):
    """The experiment can't start, or can't continue without recording something false."""


@dataclass(frozen=True)
class ExperimentSummary:
    run_dir: Path
    seed_id: str
    final_id: str
    seed_search_score: float
    final_search_score: float
    # {"seed": entry | None, "final": entry | None, "skipped_reason": str | None}; an entry
    # has candidate_id, selection_score, measured_mean and report (path).
    heldout: Mapping[str, Any]
    generations: tuple[Mapping[str, Any], ...]
    total_cost_usd: float
    stopped_reason: str  # STOPPED_COMPLETED | STOPPED_BUDGET

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "seed_id": self.seed_id,
            "final_id": self.final_id,
            "seed_search_score": self.seed_search_score,
            "final_search_score": self.final_search_score,
            "heldout": dict(self.heldout),
            "generations": [dict(row) for row in self.generations],
            "total_cost_usd": self.total_cost_usd,
            "stopped_reason": self.stopped_reason,
        }


@dataclass(frozen=True)
class _Evaluated:
    """A candidate snapshot with one search-split evaluation of it."""

    candidate: Candidate
    report: EvalReport
    eval_dir: Path
    generation: int


def run_experiment(config: ExperimentConfig, *, run_id: str | None = None) -> ExperimentSummary:
    """Run the whole loop for one experiment config and return its summary.

    Configuration problems raise LoopError, CandidateError or RegistryError before anything
    is evaluated. Any other exception, including a LoopError for a snapshot changed on disk,
    propagates after run_meta.json has recorded it.
    """
    if run_id is None:
        run_id = _utc_now().strftime(RUN_ID_FORMAT)
    if not _SAFE_RUN_ID.fullmatch(run_id) or run_id in (".", ".."):
        raise LoopError(f"run id must be a single path segment of letters, digits, '.', '_' or '-', got {run_id!r}")

    suite = make_suite(config.suite_name, config.suite_settings)
    overseer = make_overseer(config.overseer_kind, config.overseer_settings)
    seed = Candidate.load(config.seed_candidate)
    adapter = make_agent_adapter(seed.agent)
    _check_splits(suite)
    _check_temp_dir_outside_git()
    problems = adapter.validate(seed)
    if problems:
        listing = "; ".join(problems)
        raise LoopError(f"seed candidate {config.seed_candidate} is not valid for agent {seed.agent!r}: {listing}")

    run_dir = (Path(config.output_dir) / config.name / run_id).resolve()
    if run_dir.exists():
        raise LoopError(f"run directory {run_dir} already exists; choose another run id so runs never mix")
    run_dir.mkdir(parents=True)

    experiment = _Experiment(config, run_dir, suite, overseer, adapter)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "config": snapshot(config),
        "seed_id": seed.id,
        "started_at": _utc_now().isoformat(),
        "ended_at": None,
        "stopped_reason": None,
        "error": None,
        "spent_usd": 0.0,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }
    # Written now as well as at the end, so even a killed process leaves a record (P13).
    _write_json(run_dir / RUN_META_FILE, meta)
    try:
        summary = experiment.run(seed)
        meta["stopped_reason"] = summary.stopped_reason
        return summary
    except BaseException as error:
        meta["error"] = repr(error)
        raise
    finally:
        meta["ended_at"] = _utc_now().isoformat()
        meta["spent_usd"] = experiment.spent_usd
        _write_json(run_dir / RUN_META_FILE, meta)


class _Experiment:
    """The mutable state of one run: the incumbent, the spend and what has been evaluated."""

    def __init__(self, config: ExperimentConfig, run_dir: Path, suite: Suite, overseer: Overseer, adapter: AgentAdapter):
        self.config = config
        self.run_dir = run_dir
        self.suite = suite
        self.overseer = overseer
        self.adapter = adapter
        self.spent_usd = 0.0
        # Calibrated from the seed's search evaluation; 0.0 until then (P12).
        self.cost_per_trial_estimate = 0.0
        self.budget_exhausted = False
        self.rows: list[dict[str, Any]] = []
        # Candidate id -> its latest search evaluation, for duplicate detection.
        self.evaluated: dict[str, _Evaluated] = {}

    def run(self, seed: Candidate) -> ExperimentSummary:
        config = self.config
        LOGGER.info("run %s: suite %s, model %s, overseer %s", self.run_dir, config.suite_name,
                    config.model.qualified, self.overseer.name)  # fmt: skip
        seed_snapshot = self._snapshot_seed(seed)
        seed_eval = self._evaluate_search(seed_snapshot, generation=0)
        if seed_eval.report.n_trials:
            self.cost_per_trial_estimate = seed_eval.report.cost_usd / seed_eval.report.n_trials
        LOGGER.info("seed %s: search score %.4f", seed_snapshot.id, seed_eval.report.selection_score)

        incumbent = seed_eval
        stopped_reason = STOPPED_COMPLETED
        for generation in range(1, config.generations + 1):
            if not self._fits(self._estimate("search")):
                LOGGER.info("generation %d: another search evaluation (~$%.4f) would not fit under the cap "
                            "($%.4f spent of $%.4f); stopping", generation, self._estimate("search"),
                            self.spent_usd, config.max_usd)  # fmt: skip
                stopped_reason = STOPPED_BUDGET
                break
            incumbent = self._generation(generation, incumbent, seed_eval)
            if self.budget_exhausted:
                stopped_reason = STOPPED_BUDGET
                break

        heldout = self._heldout(seed_eval.candidate, incumbent.candidate)
        summary = ExperimentSummary(
            run_dir=self.run_dir,
            seed_id=seed_eval.candidate.id,
            final_id=incumbent.candidate.id,
            seed_search_score=seed_eval.report.selection_score,
            final_search_score=incumbent.report.selection_score,
            heldout=heldout,
            generations=tuple(self.rows),
            total_cost_usd=self.spent_usd,
            stopped_reason=stopped_reason,
        )
        _write_json(self.run_dir / SUMMARY_FILE, summary.to_dict())
        LOGGER.info("done: %s, final %s, spent $%.4f", stopped_reason, incumbent.candidate.id, self.spent_usd)
        return summary

    # --- candidates and evaluation -------------------------------------------------------

    def _snapshot_seed(self, seed: Candidate) -> Candidate:
        copy = seed.copy_to(self.run_dir / CANDIDATES_DIR / seed.id)
        if copy.lineage() is None:
            copy.write_lineage(parents=[], origin=SEED_ORIGIN, rationale=seed.description)
        return copy

    def _snapshot_child(self, child: Candidate, parent: Candidate, rationale: str) -> Candidate:
        destination = self.run_dir / CANDIDATES_DIR / child.id
        if destination.exists():
            # Same id, same content: keep the first lineage, which says where it really came from.
            return Candidate.load(destination)
        copy = child.copy_to(destination)
        copy.write_lineage(parents=[parent.id], origin=self.overseer.name, rationale=rationale)
        return copy

    def _estimate(self, split: Split) -> float:
        """Expected cost of one evaluation on `split`, from the seed's measured cost per trial."""
        return self.cost_per_trial_estimate * len(self.suite.tasks(split)) * self.config.trials_per_task

    def _fits(self, expected_cost: float) -> bool:
        return self.spent_usd < self.config.max_usd and self.spent_usd + expected_cost <= self.config.max_usd

    def _evaluate(self, candidate: Candidate, split: Split, out_dir: Path) -> EvalReport:
        _verify_intact(candidate)
        report = evaluate(
            candidate,
            self.adapter,
            self.suite,
            split,
            model=self.config.model,
            agent_settings=self.config.agent_settings,
            trials_per_task=self.config.trials_per_task,
            out_dir=out_dir,
            concurrency=self.config.concurrency,
            # Per trial, not from the finished report: an aborted evaluation has still spent.
            on_trial_cost=self._record_trial_cost,
        )
        return report

    def _record_trial_cost(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

    def _evaluate_search(self, candidate: Candidate, generation: int) -> _Evaluated:
        eval_dir = self.run_dir / EVALS_DIR / _generation_dirname(generation)
        report = self._evaluate(candidate, "search", eval_dir)
        evaluated = _Evaluated(candidate=candidate, report=report, eval_dir=eval_dir, generation=generation)
        self.evaluated[candidate.id] = evaluated
        return evaluated

    # --- one generation ------------------------------------------------------------------

    def _generation(self, generation: int, incumbent: _Evaluated, seed: _Evaluated) -> _Evaluated:
        record_dir = self.run_dir / OVERSEER_DIR / _generation_dirname(generation)
        record_dir.mkdir(parents=True)
        LOGGER.info("generation %d: asking overseer %s to improve %s", generation, self.overseer.name,
                    incumbent.candidate.id)  # fmt: skip
        # The overseer gets copies, never the run's own files: nothing it writes can change a
        # snapshot that is scored later or the evidence later generations are briefed with.
        with tempfile.TemporaryDirectory(prefix=BRIEF_PREFIX, ignore_cleanup_errors=True) as bundle:
            bundle_dir = Path(bundle).resolve()
            shutil.copy2(incumbent.eval_dir / REPORT_FILE, bundle_dir / REPORT_FILE)
            evidence = incumbent.eval_dir / TRIALS_DIR
            if evidence.is_dir():
                skipped = _copy_evidence(evidence, bundle_dir / TRIALS_DIR)
                if skipped:
                    LOGGER.warning("generation %d: skipped %d evidence entries the brief can't carry", generation,
                                   len(skipped))  # fmt: skip
                    (bundle_dir / TRIALS_DIR).mkdir(exist_ok=True)
                    note = "Not copied into this evidence (paths relative to this directory):\n\n"
                    (bundle_dir / TRIALS_DIR / SKIPPED_EVIDENCE_FILE).write_text(
                        note + "".join(f"{line}\n" for line in skipped)
                    )
            brief = OverseerBrief(
                incumbent=incumbent.candidate.copy_to(bundle_dir / "incumbent"),
                metric=self.suite.metric(),
                suite_description=self.suite.describe(),
                valves=self.adapter.describe_valves(),
                selection_rule=describe_rule(self.config.promotion_margin),
                generation=generation,
                incumbent_report=bundle_dir / REPORT_FILE,
                evidence_dir=bundle_dir / TRIALS_DIR,
                evidence_origin=incumbent.eval_dir / TRIALS_DIR,
                history=[dict(row) for row in self.rows],
            )
            proposal, archived_candidate = self._propose(brief, record_dir)
        overseer_cost = _checked_cost(proposal, self.overseer.name)
        self.spent_usd += overseer_cost

        row: dict[str, Any] = {
            "generation": generation,
            "overseer": self.overseer.name,
            "status": None,
            "child_id": None,
            "child_selection_score": None,
            "child_measured_mean": None,
            "incumbent_before": incumbent.candidate.id,
            "incumbent_score_before": incumbent.report.selection_score,
            "promoted": False,
            "reason": "",
            "rationale": proposal.rationale,
            "error": proposal.error,
            "overseer_cost_usd": overseer_cost,
            "eval_cost_usd": 0.0,
            "spent_usd": None,
        }
        next_incumbent = incumbent
        status, child, reason = self._classify(proposal, archived_candidate, incumbent.candidate)
        row["status"], row["reason"] = status, reason
        if child is not None:
            row["child_id"] = child.id
        if status == STATUS_DUPLICATE:
            earlier = self.evaluated[child.id]
            row["child_selection_score"] = earlier.report.selection_score
            row["child_measured_mean"] = earlier.report.measured_mean
        elif status == STATUS_EVALUATED and not self._fits(self._estimate("search")):
            row["status"] = STATUS_OVER_BUDGET
            row["reason"] = (
                f"a search evaluation (~${self._estimate('search'):.4f}) would not fit under the cap "
                f"(${self.spent_usd:.4f} spent of ${self.config.max_usd:.4f}); not evaluated"
            )
            self.budget_exhausted = True
        elif status == STATUS_EVALUATED:
            snapshot_candidate = self._snapshot_child(child, incumbent.candidate, proposal.rationale)
            spent_before = self.spent_usd
            evaluated = self._evaluate_search(snapshot_candidate, generation)
            decision = decide(evaluated.report, incumbent.report, seed.report, self.config.promotion_margin)
            row.update(
                child_selection_score=evaluated.report.selection_score,
                child_measured_mean=evaluated.report.measured_mean,
                promoted=decision.promoted,
                reason=decision.reason,
                eval_cost_usd=self.spent_usd - spent_before,
            )
            if decision.promoted:
                next_incumbent = evaluated
        row["spent_usd"] = self.spent_usd

        self.rows.append(row)
        with (self.run_dir / GENERATIONS_FILE).open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        LOGGER.info("generation %d: %s%s; %s", generation, row["status"], " (promoted)" if row["promoted"] else "",
                    row["reason"])  # fmt: skip
        return next_incumbent

    def _propose(self, brief: OverseerBrief, record_dir: Path) -> tuple[Proposal, Path | None]:
        """Run the overseer in a fresh temp workspace, archive it, and map the child into the archive.

        The workspace lives outside the repository and the run directory: Claude Code walks up
        from its working directory loading CLAUDE.md and project settings, and inside the HIVE
        repo it would pick up instructions meant for HIVE's own developers.
        """
        workspace = Path(tempfile.mkdtemp(prefix=WORKSPACE_PREFIX)).resolve()
        archive = record_dir / WORKSPACE_DIR
        try:
            proposal = self.overseer.propose(brief, workspace)
            if not isinstance(proposal, Proposal):
                raise TypeError(f"overseer {self.overseer.name!r} returned {type(proposal).__name__}, not Proposal")
            # Resolve against the live workspace before it is removed; afterwards only the archive exists.
            relative = _relative_inside(proposal.candidate_dir, workspace)
        finally:
            # Kept even when the overseer crashed: the partial workspace is the evidence. Anything
            # uncopyable it left behind (a FIFO, an unreadable file) is listed, never fatal.
            skipped = _copy_evidence(workspace, archive)
            if skipped:
                (archive / SKIPPED_EVIDENCE_FILE).write_text("\n".join(skipped) + "\n")
                LOGGER.warning("overseer workspace: %d entries not archived: %s", len(skipped), "; ".join(skipped))
            shutil.rmtree(workspace, ignore_errors=True)

        archived_candidate = archive / relative if relative is not None else None
        _write_json(
            record_dir / PROPOSAL_FILE,
            {
                "candidate_dir": str(proposal.candidate_dir) if proposal.candidate_dir is not None else None,
                "archived_candidate_dir": str(archived_candidate) if archived_candidate is not None else None,
                "rationale": proposal.rationale,
                "cost_usd": proposal.cost_usd,
                "error": proposal.error,
            },
        )
        return proposal, archived_candidate

    def _classify(
        self, proposal: Proposal, archived_candidate: Path | None, incumbent: Candidate
    ) -> tuple[str, Candidate | None, str]:
        """(status, loaded child or None, reason). Nothing here spends on evaluation."""
        if proposal.candidate_dir is None:
            return STATUS_NO_PROPOSAL, None, "the overseer returned no candidate"
        if archived_candidate is None:
            # The write fence is the workspace (P10): a child anywhere else is not the overseer's to propose.
            return STATUS_INVALID, None, f"proposed candidate {proposal.candidate_dir} is outside the overseer workspace"
        try:  # Candidate.load also refuses symlinks, which could reach outside the candidate

            child = Candidate.load(archived_candidate)
        except (CandidateError, OSError) as error:
            return STATUS_INVALID, None, f"candidate does not load: {error}"
        if child.agent != incumbent.agent:
            return STATUS_INVALID, child, f"candidate changed its agent from {incumbent.agent!r} to {child.agent!r}"
        problems = self.adapter.validate(child)
        if problems:
            return STATUS_INVALID, child, "candidate is not valid: " + "; ".join(problems)
        # The null control's copies are identical by design; re-measuring them is how it measures noise.
        if self.overseer.proposes_changes and child.id == incumbent.id:
            return STATUS_UNCHANGED, child, "candidate is identical to the incumbent; not evaluated"
        if self.overseer.proposes_changes and child.id in self.evaluated:
            earlier = self.evaluated[child.id].generation
            return STATUS_DUPLICATE, child, f"candidate was already evaluated in generation {earlier}; not re-evaluated"
        return STATUS_EVALUATED, child, ""

    # --- held-out ------------------------------------------------------------------------

    def _heldout(self, seed: Candidate, final: Candidate) -> dict[str, Any]:
        """Score the seed and the final incumbent on held-out, once. Never shown to an overseer (P7)."""
        result: dict[str, Any] = {"seed": None, "final": None, "skipped_reason": None}
        if not self.suite.tasks("heldout"):
            result["skipped_reason"] = "the suite has no held-out tasks"
        elif not self._fits(self._estimate("heldout")):
            result["skipped_reason"] = (
                f"budget: a held-out evaluation (~${self._estimate('heldout'):.4f}) would not fit under the cap "
                f"(spent ${self.spent_usd:.4f} of ${self.config.max_usd:.4f})"
            )
        else:
            result["seed"] = self._heldout_entry(seed, "seed")
            if final.id == seed.id:
                result["final"] = {**result["seed"], "same_report_as_seed": True}
            elif not self._fits(self._estimate("heldout")):
                result["skipped_reason"] = (
                    f"budget: after the seed's held-out evaluation, another (~${self._estimate('heldout'):.4f}) "
                    f"would not fit under the cap (spent ${self.spent_usd:.4f} of ${self.config.max_usd:.4f}); "
                    "the final incumbent was not scored on held-out"
                )
            else:
                result["final"] = {**self._heldout_entry(final, "final"), "same_report_as_seed": False}
        if result["skipped_reason"]:
            LOGGER.info("held-out: %s", result["skipped_reason"])
        return result

    def _heldout_entry(self, candidate: Candidate, label: str) -> dict[str, Any]:
        out_dir = self.run_dir / HELDOUT_DIR / label
        report = self._evaluate(candidate, "heldout", out_dir)
        LOGGER.info("held-out %s %s: score %.4f", label, candidate.id, report.selection_score)
        return {
            "candidate_id": candidate.id,
            "selection_score": report.selection_score,
            "measured_mean": report.measured_mean,
            "report": str(out_dir / REPORT_FILE),
        }


def _check_splits(suite: Suite) -> None:
    """Split walls (P7): no task id in two splits, and something to search on."""
    seen: dict[str, str] = {}
    overlaps: list[str] = []
    for split in SPLITS:
        for task in suite.tasks(split):
            if task.id in seen and seen[task.id] != split:
                overlaps.append(f"{task.id!r} is in both {seen[task.id]!r} and {split!r}")
            seen.setdefault(task.id, split)
    if overlaps:
        raise LoopError(f"suite {suite.name!r} breaks the split walls: {'; '.join(overlaps)}")
    if not suite.tasks("search"):
        raise LoopError(f"suite {suite.name!r} has no tasks in the search split")


def _check_temp_dir_outside_git() -> None:
    """Overseer workspaces are temp dirs; inside a git checkout, Claude Code would load that repo's CLAUDE.md."""
    temp_root = Path(tempfile.gettempdir()).resolve()
    for directory in (temp_root, *temp_root.parents):
        if (directory / ".git").exists():
            raise LoopError(
                f"the temp dir {temp_root} is inside the git checkout {directory}; overseer workspaces must live "
                "outside any repository, so point TMPDIR somewhere else"
            )


def _verify_intact(candidate: Candidate) -> None:
    """A snapshot is scored under its id, so its files must still hash to that id (P4)."""
    if content_hash(candidate.root) != candidate.id:
        raise LoopError(
            f"candidate snapshot {candidate.id} changed on disk ({candidate.root}); refusing to score it under a stale id"
        )


def _checked_cost(proposal: Proposal, overseer_name: str) -> float:
    cost = proposal.cost_usd
    # A NaN would make every budget comparison False, so the cap would never trigger (P12).
    if isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0:
        raise ValueError(f"overseer {overseer_name!r} reported an invalid cost {cost!r}")
    return float(cost)


def _copy_evidence(source: Path, destination: Path) -> list[str]:
    """Copy trial evidence for a brief; return one "path: reason" line per entry left out.

    Never raises for the evidence's contents: a FIFO, socket or chmod-000 file an agent left
    in its workdir must not decide whether the run continues. Symlinks are copied as links,
    so nothing outside the evidence is read through them. Reasons name paths relative to
    `source`, so the note gives the overseer no absolute path into the run directory.
    """
    skipped: list[str] = []

    def relative(path: str | Path) -> str:
        resolved = Path(path)
        return str(resolved.relative_to(source)) if resolved.is_relative_to(source) else resolved.name

    def without_absolute_paths(reason: str) -> str:
        for root in (source, destination):
            reason = reason.replace(f"{root}{os.sep}", "")
        return reason

    def ignore_special(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(directory) / name
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                skipped.append(f"{relative(path)}: could not stat ({error.strerror or type(error).__name__})")
                ignored.add(name)
                continue
            # Copying a FIFO would block or fail, a socket or device can't be copied at all.
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
                skipped.append(f"{relative(path)}: not a regular file, directory or symlink")
                ignored.add(name)
        return ignored

    try:
        shutil.copytree(source, destination, symlinks=True, ignore=ignore_special)
    except shutil.Error as error:
        # copytree copies everything it can, then reports each failure (e.g. unreadable entries).
        for failed_source, _failed_destination, reason in error.args[0]:
            skipped.append(f"{relative(failed_source)}: {without_absolute_paths(str(reason))}")
    except OSError as error:  # the evidence root itself could not be read
        skipped.append(f".: {error.strerror or type(error).__name__}")
    return skipped


def _relative_inside(path: Path | None, root: Path) -> Path | None:
    """`path` relative to `root` when it lies inside it, else None."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    return resolved.relative_to(root) if resolved.is_relative_to(root) else None


def _generation_dirname(generation: int) -> str:
    return f"gen-{generation:03d}"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _git_dirty() -> bool | None:
    """Whether the HIVE checkout had uncommitted changes, so a commit hash alone can't be trusted (P13)."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
