"""The exact_answer suite: short questions graded by normalized exact match.

It is the smallest suite that still exercises the whole loop: no workspace files,
no subprocesses, a binary reward. Reference answers stay inside this module; a
`Task` carries only the rendered instruction, and grader feedback reports what
the agent said, never what was expected, because the overseer reads it (P2, P10).

The overseer-facing text (DESCRIPTION.md, whose "Metric" section is also the
metric description) and the agent-facing instruction template live in files
next to this module, so each has exactly one source.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hive.contracts import SPLITS, AgentResult, Grade, MetricSpec, Split, Task

_HERE = Path(__file__).resolve().parent
BUNDLED_TASKS_FILE = _HERE / "tasks.jsonl"
DESCRIPTION_FILE = _HERE / "DESCRIPTION.md"
INSTRUCTION_TEMPLATE_FILE = _HERE / "instruction_template.md"

QUESTION_PLACEHOLDER = "{question}"
METRIC_HEADING = "## Metric"
NUMERIC_RELATIVE_TOLERANCE = 1e-6
# Long free-form answers would bloat every report the overseer reads; the prefix is enough evidence.
MAX_DETAIL_ANSWER_CHARS = 200

_ROW_FIELDS = {"id", "split", "question", "answers", "numeric"}
# The marker must open the line, behind nothing but markdown (heading, quote, list,
# bold/italic, code) and whitespace. Anchoring it means a later line of prose that
# merely says "my answer: ..." cannot replace the real answer line. Markup glued to
# the colon ("**ANSWER:**", "_ANSWER:_") is consumed as part of the marker.
_ANSWER_MARKER = re.compile(r"^[\s#>*_`-]*answer[\s*_`]*:[*_`]*", re.IGNORECASE)
# Whitespace is not listed: str.strip() removes all Unicode whitespace, so a no-break
# space between wrappers cannot shield them the way an ASCII-only list would.
_WRAPPER_CHARS = "\"'`*“”‘’"
# Python's float() also accepts "1_000", "inf" and "nan"; a strict pattern keeps
# "a plain decimal number" meaning the same thing in the grader and in DESCRIPTION.md.
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?")
# Commas are only removed when they sit in correct thousands positions, so "1,5" is not 15.
_THOUSANDS_GROUPED = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


@dataclass(frozen=True)
class _AnswerKey:
    accepted: tuple[str, ...]  # normalized
    numeric: bool


def _strip_whitespace_and_wrappers(text: str) -> str:
    """Peel whitespace and wrapper characters in turn until neither end changes."""
    # One pass is not enough when they interleave, e.g. "**\u00a0`x`\u00a0**".
    while True:
        stripped = text.strip().strip(_WRAPPER_CHARS)
        if stripped == text:
            return text
        text = stripped


def normalize_answer(text: str) -> str:
    """Canonical comparison form, applied identically to agent output and accepted answers."""
    text = _strip_whitespace_and_wrappers(text)
    if text.endswith("."):
        text = text[:-1]
    # Strip again so both "**x**." and "**x.**" reduce to "x".
    text = _strip_whitespace_and_wrappers(text)
    return " ".join(text.lower().split())


def parse_number(normalized: str) -> float | None:
    """The value of a plain decimal number, or None when the text is anything else."""
    if _THOUSANDS_GROUPED.fullmatch(normalized):
        normalized = normalized.replace(",", "")
    if not _PLAIN_NUMBER.fullmatch(normalized):
        return None
    value = float(normalized)
    return value if math.isfinite(value) else None


def extract_answer(final_text: str) -> str | None:
    """Text after the marker on the last line that opens with one; None when no line does."""
    for line in reversed(final_text.splitlines()):
        marker = _ANSWER_MARKER.match(line)
        if marker:
            return line[marker.end() :]
    return None


def _numbers_match(given: float, expected: float) -> bool:
    return abs(given - expected) <= NUMERIC_RELATIVE_TOLERANCE * max(1.0, abs(expected))


def _is_correct(normalized: str, key: _AnswerKey) -> bool:
    if not key.numeric:
        return normalized in key.accepted
    given = parse_number(normalized)
    if given is None:
        return False
    return any(_numbers_match(given, parse_number(answer)) for answer in key.accepted)


def _load_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
        rows.append((line_number, row))
    if not rows:
        raise ValueError(f"{path}: no tasks found")
    return rows


def _validate_row(row: dict[str, Any], where: str, seen_ids: set[str]) -> _AnswerKey:
    missing = sorted(_ROW_FIELDS - row.keys())
    if missing:
        raise ValueError(f"{where}: missing field(s) {', '.join(missing)}")
    unknown = sorted(row.keys() - _ROW_FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {', '.join(unknown)}")
    task_id = row["id"]
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"{where}: 'id' must be a non-empty string")
    if task_id in seen_ids:
        raise ValueError(f"{where}: duplicate id {task_id!r}")
    if row["split"] not in SPLITS:
        raise ValueError(f"{where}: 'split' must be one of {SPLITS}, got {row['split']!r}")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"{where}: 'question' must be a non-empty string")
    if not isinstance(row["numeric"], bool):
        raise ValueError(f"{where}: 'numeric' must be true or false")
    answers = row["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"{where}: 'answers' must be a non-empty list")
    if not all(isinstance(answer, str) and normalize_answer(answer) for answer in answers):
        raise ValueError(f"{where}: 'answers' must all be non-empty strings")
    accepted = tuple(normalize_answer(answer) for answer in answers)
    if row["numeric"] and any(parse_number(answer) is None for answer in accepted):
        raise ValueError(f"{where}: every answer of a numeric task must be a plain number")
    return _AnswerKey(accepted=accepted, numeric=row["numeric"])


def _metric_section(description: str) -> str:
    """The body of DESCRIPTION.md's metric section, so the metric text has one source."""
    lines = description.splitlines()
    try:
        start = lines.index(METRIC_HEADING) + 1
    except ValueError as error:
        raise ValueError(f"{DESCRIPTION_FILE} has no '{METRIC_HEADING}' section") from error
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    section = "\n".join(lines[start:end]).strip()
    if not section:
        raise ValueError(f"{DESCRIPTION_FILE}: '{METRIC_HEADING}' section is empty")
    return section


class ExactAnswerSuite:
    name = "exact_answer"

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        settings = settings or {}
        tasks_file = Path(settings.get("tasks_file", BUNDLED_TASKS_FILE))
        template = INSTRUCTION_TEMPLATE_FILE.read_text()
        if template.count(QUESTION_PLACEHOLDER) != 1:
            raise ValueError(f"{INSTRUCTION_TEMPLATE_FILE} must contain {QUESTION_PLACEHOLDER} exactly once")
        self._description = DESCRIPTION_FILE.read_text()
        self._metric = MetricSpec(name="accuracy", description=_metric_section(self._description))

        # Load everything up front so bad data fails at config time, not mid-experiment.
        self._tasks: dict[str, Task] = {}
        self._answer_keys: dict[str, _AnswerKey] = {}
        for line_number, row in _load_rows(tasks_file):
            key = _validate_row(row, f"{tasks_file}:{line_number}", set(self._tasks))
            # str.replace, not str.format, so braces inside a question are never interpreted.
            instruction = template.replace(QUESTION_PLACEHOLDER, row["question"].strip())
            self._tasks[row["id"]] = Task(id=row["id"], split=row["split"], instruction=instruction)
            self._answer_keys[row["id"]] = key

    def describe(self) -> str:
        return self._description

    def metric(self) -> MetricSpec:
        return self._metric

    def tasks(self, split: Split) -> list[Task]:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        return [task for task in self._tasks.values() if task.split == split]

    def grade(self, task: Task, result: AgentResult, workdir: Path) -> Grade:
        key = self._answer_keys.get(task.id)
        if key is None:
            raise ValueError(f"task {task.id!r} is not part of the {self.name} suite")
        raw_answer = extract_answer(result.final_text)
        if raw_answer is None:
            return Grade(reward=0.0, detail='no "ANSWER:" line found')
        normalized = normalize_answer(raw_answer)
        correct = _is_correct(normalized, key)
        shown = normalized
        if len(shown) > MAX_DETAIL_ANSWER_CHARS:
            shown = shown[:MAX_DETAIL_ANSWER_CHARS] + "..."
        verdict = "correct" if correct else "incorrect"
        return Grade(reward=1.0 if correct else 0.0, detail=f'parsed answer "{shown}": {verdict}')
