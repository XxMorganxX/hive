"""The swap boundaries of HIVE.

Three roles can be swapped, and each has one small contract here:

- `AgentAdapter` / `Agent`: an agent framework (Pi, Strands, Deep Agents, a test double).
- `Suite`: one evaluation use case (its tasks, splits, grader and metric).
- `Overseer`: the improvement strategy that proposes better candidates.

Implementations are selected by name in an experiment config (see `hive.registry`).
Nothing in the controller imports a concrete implementation, and nothing in an
implementation imports the controller. See docs/FOUNDING_PRINCIPLES.md §5.

Every plug-in class is constructed as `Cls(settings)`, where `settings` is the mapping
from its section of the experiment config (agent adapters: `Cls()` then `build(...,
settings)`). Unknown settings keys should be rejected, so typos fail loudly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hive.candidate import Candidate

Split = Literal["search", "validation", "heldout"]
SPLITS: tuple[Split, ...] = ("search", "validation", "heldout")

AgentStatus = Literal["completed", "errored"]

# Why an agent run produced no gradeable result. Errored trials are never graded;
# they count as 0 in the selection score and are reported separately (P8).
AGENT_ERROR_CLASSES = (
    "agent_error",  # the agent process crashed or produced no answer
    "timeout",  # the run exceeded its time limit
    "provider_error",  # the model provider returned an error or aborted
    "quota",  # rate limit, usage cap or billing failure
    "harness_load_error",  # the candidate's own files failed to load (e.g. a broken extension)
    "model_mismatch",  # the run used a different model than the experiment fixed (P3)
)


@dataclass(frozen=True)
class ModelSpec:
    """The model under test. Fixed for a whole experiment and never part of a candidate (P3).

    `thinking` is the reasoning-effort level, when the provider has one. It belongs here,
    not in a candidate: more reasoning buys accuracy with spend, which is a model change,
    not a harness change. Adapters map it onto their framework's own setting.
    """

    provider: str
    id: str
    thinking: str | None = None

    @property
    def qualified(self) -> str:
        return f"{self.provider}/{self.id}"


@dataclass(frozen=True)
class Task:
    """One unit of evaluation, in a shape every suite and every agent understands.

    Grading data (reference answers, hidden tests) never appears here: the suite
    keeps it, so neither the agent nor the overseer can read it.
    """

    id: str
    split: Split
    instruction: str
    # Relative path -> text, written as UTF-8 into the trial workdir before the agent runs.
    workspace_files: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class AgentResult:
    """What an agent run produced. `completed` means the run finished, not that it was correct."""

    status: AgentStatus
    final_text: str
    usage: Usage
    wall_clock_s: float
    error: str | None = None
    error_class: str | None = None  # one of AGENT_ERROR_CLASSES when status == "errored"


@runtime_checkable
class Agent(Protocol):
    def run(self, task: Task, workdir: Path, trace_dir: Path) -> AgentResult:
        """Run one task with `workdir` as the working directory.

        Must not raise for agent-side failures: return status="errored" instead.
        Must write its native trace files into `trace_dir`.
        May be called concurrently from several threads, each with its own workdir and trace_dir.
        """
        ...


@runtime_checkable
class AgentAdapter(Protocol):
    """The only code that knows a framework's native format.

    `name` is exactly the string candidates put in `agent = "..."`: the built-in short
    name for a built-in adapter, or the import path ("package.module:Class") for any
    other. The registry resolves that string, and the controller checks the two match.
    """

    name: str

    def validate(self, candidate: Candidate) -> list[str]:
        """Problems that make the candidate unrunnable; [] means valid. Never raises."""
        ...

    def build(self, candidate: Candidate, model: ModelSpec, settings: Mapping[str, Any]) -> Agent:
        """Turn a valid candidate into a runnable agent. `settings` come from the experiment config."""
        ...

    def describe_valves(self) -> str:
        """Markdown listing every valve the overseer may edit, and how. Given to the overseer verbatim."""
        ...


@dataclass(frozen=True)
class MetricSpec:
    """What 'better' means for a suite. A task's reward is always in [0, 1], higher is better."""

    name: str
    description: str


@dataclass(frozen=True)
class Grade:
    reward: float  # in [0, 1]
    detail: str = ""  # what the agent produced and whether it passed; never the expected value


@runtime_checkable
class Suite(Protocol):
    name: str

    def describe(self) -> str:
        """Markdown for the overseer: what the tasks look like and how they are graded. Never answers."""
        ...

    def metric(self) -> MetricSpec: ...

    def tasks(self, split: Split) -> list[Task]:
        """Tasks in one split. Splits are disjoint by task id."""
        ...

    def grade(self, task: Task, result: AgentResult, workdir: Path) -> Grade:
        """Score a completed run. Only called when result.status == "completed".

        May be called concurrently from several threads, each with its own workdir.
        """
        ...


@dataclass(frozen=True)
class OverseerBrief:
    """Everything the overseer gets: the candidate config, the metric and the suite description,
    plus the evidence needed to improve on them. Held-out results are never included (P7)."""

    incumbent: Candidate
    metric: MetricSpec
    suite_description: str
    valves: str
    selection_rule: str
    generation: int
    incumbent_report: Path  # search-split EvalReport as JSON
    evidence_dir: Path  # the incumbent's search-split trial directories (traces + workdirs)
    history: Sequence[Mapping[str, Any]]  # earlier generations: ids, scores, decisions, rationales


@dataclass(frozen=True)
class Proposal:
    candidate_dir: Path | None  # None when the overseer produced nothing usable
    rationale: str
    cost_usd: float = 0.0
    error: str | None = None


@runtime_checkable
class Overseer(Protocol):
    name: str
    proposes_changes: bool  # False only for the null control, whose copies are meant to be identical

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        """Write a proposed child candidate somewhere under `workspace` and describe it.

        Must not raise for overseer-side failures: return a Proposal with candidate_dir=None and an error.
        """
        ...
