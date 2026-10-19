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
    "extension_error",  # a candidate extension loaded but threw while the run was going
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
class ModelEndpoint:
    """Where an agent must send every model request during one trial (the model gateway).

    `api_key` is a per-trial token minted by the controller, not the provider's key: the real
    key never enters the trial process, and the gateway enforces the frozen model (P3), the
    call limit and billing (P12) for every request. OpenAI-compatible: `base_url` ends in /v1.
    """

    base_url: str
    api_key: str


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


# --- extensions ---------------------------------------------------------------------------
#
# Extensions are candidate code that hooks the agent loop: Pi extensions (TypeScript), Deep
# Agents middleware and Strands hooks (Python). They are a primary lever for harness
# evolution, so the contract makes them first-class without pretending they are portable:
# the code stays native to each framework, while the shape around it is shared:
#   - every candidate keeps its extensions in EXTENSIONS_DIR;
#   - every adapter declares how its extensions are written (ExtensionSpec), including a
#     catalog of neutral lifecycle points mapped onto its native API, for the overseer (P11);
#   - every run reports which extension files actually loaded (LoadReport), and the
#     controller, not the adapter, fails a trial whose extensions did not all load (P8).

EXTENSIONS_DIR = "extensions"

# Neutral names for where an extension can hook in. Fixed here so overseer history, rationales
# and analysis use the same words across frameworks; each adapter says how it supports each.
LIFECYCLE_POINTS = (
    "before_run",  # once, before the first model call
    "system_prompt",  # read or rewrite the system prompt
    "before_model_call",  # edit the context/messages for the next model call
    "after_model_call",  # inspect a model response, or ask for a retry
    "wrap_model_call",  # replace or route the model call itself
    "before_tool_call",  # gate a tool call, rewrite its arguments or cancel it
    "after_tool_call",  # rewrite a tool result
    "register_tool",  # add a new tool
    "compaction",  # control how the context is summarized when it grows too long
    "after_run",  # inspect the end of the run, or continue it
)

Support = Literal["supported", "partial", "unsupported"]


@dataclass(frozen=True)
class LifecyclePoint:
    id: str  # one of LIFECYCLE_POINTS
    support: Support
    native_api: str  # the framework's exact name for it, e.g. 'pi.on("tool_call")'
    can_modify: str  # what a handler can change here
    caveats: str = ""


@dataclass(frozen=True)
class ExtensionSpec:
    """How extensions are written for one framework. Declared by its agent adapter."""

    language: str  # "typescript" | "python"
    suffixes: tuple[str, ...]  # files in the candidate's extensions/ directory that are extensions
    entry_convention: str  # what each extension file must define or export
    framework_version: str  # the pinned version this spec was checked against (P13)
    lifecycle: tuple[LifecyclePoint, ...]
    forbidden: tuple[str, ...] = ()  # things an extension must not do (Markdown bullets), e.g. switching the model (P3)
    # The pinned API definitions themselves (e.g. the framework's extension type declarations),
    # as text: the overseer works in a sandbox that can't read the installed framework.
    api_reference: str = ""


@dataclass(frozen=True)
class ExtensionLoad:
    path: str  # relative to the candidate root, e.g. "extensions/guard.ts"
    loaded: bool
    error: str | None = None
    registered: tuple[str, ...] = ()  # what it hooked: lifecycle point ids and "tool:<name>" entries


@dataclass(frozen=True)
class LoadReport:
    """What a run actually loaded. Required whenever the candidate has extension files."""

    extensions: tuple[ExtensionLoad, ...] = ()
    runtime_errors: tuple[str, ...] = ()  # extension code that threw during the run, with its path when known


@dataclass(frozen=True)
class AgentResult:
    """What an agent run produced. `completed` means the run finished, not that it was correct."""

    status: AgentStatus
    final_text: str
    usage: Usage
    wall_clock_s: float
    error: str | None = None
    error_class: str | None = None  # one of AGENT_ERROR_CLASSES when status == "errored"
    load_report: LoadReport | None = None  # required when the candidate has extension files


@runtime_checkable
class Agent(Protocol):
    def run(self, task: Task, workdir: Path, trace_dir: Path, endpoint: ModelEndpoint | None = None) -> AgentResult:
        """Run one task with `workdir` as the working directory.

        When `endpoint` is given (every real provider), ALL model traffic must go to it, with
        its per-trial token: the gateway behind it enforces the frozen model and bills the run.
        The controller passes it as a keyword argument, and only when it is not None.

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

    def extension_spec(self) -> ExtensionSpec | None:
        """How this framework's extensions are written, or None if it has none."""
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
    # Where the evidence was recorded. Native traces name absolute paths under it; an overseer
    # maps those onto evidence_dir so its own workspace never names the run directory (P7).
    evidence_origin: Path | None = None
    # How to write extensions for this candidate's framework, rendered by the controller from the
    # adapter's ExtensionSpec so it reads the same for every framework; "" if it has none.
    extension_guide: str = ""


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
