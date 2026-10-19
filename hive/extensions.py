"""The framework-neutral half of extensions (docs/FOUNDING_PRINCIPLES.md §5.2).

Extension code stays native to each framework; this module holds what is shared and
trusted: which candidate files are extensions, the rule that decides whether a run's
extensions all loaded, and the one rendering of an adapter's ExtensionSpec that every
overseer reads. The controller applies `load_problem` itself, so an adapter that quietly
skips a broken extension can't make a candidate look like it ran as written (P8).
"""

from __future__ import annotations

from collections.abc import Sequence

from hive.candidate import Candidate, content_files
from hive.contracts import EXTENSIONS_DIR, LIFECYCLE_POINTS, ExtensionSpec, LoadReport


def expected_extension_files(candidate: Candidate, spec: ExtensionSpec | None) -> list[str]:
    """Candidate-relative paths of every file that must load as an extension, sorted."""
    if spec is None:
        return []
    return [
        relative.as_posix()
        for relative in content_files(candidate.root)
        if relative.parts[0] == EXTENSIONS_DIR and relative.suffix in spec.suffixes
    ]


def load_problem(expected: Sequence[str], report: LoadReport | None) -> tuple[str, str] | None:
    """(error_class, detail) when the run did not run the candidate's extensions as written."""
    if expected and report is None:
        return "harness_load_error", f"the adapter returned no load report for {len(expected)} extension file(s)"
    if report is None:
        return None
    by_path = {load.path: load for load in report.extensions}
    failures = []
    for path in expected:
        load = by_path.get(path)
        if load is None:
            failures.append(f"{path}: not loaded (absent from the load report)")
        elif not load.loaded:
            failures.append(f"{path}: failed to load: {load.error or 'no reason given'}")
    if failures:
        return "harness_load_error", "; ".join(failures)
    if report.runtime_errors:
        return "extension_error", "extension code threw during the run: " + "; ".join(report.runtime_errors)
    return None


def render_extension_guide(spec: ExtensionSpec | None) -> str:
    """Markdown for the overseer: how to write an extension for this framework, and where it can hook."""
    if spec is None:
        return ""
    suffixes = ", ".join(f"`{suffix}`" for suffix in spec.suffixes)
    lines = [
        "# Extensions",
        "",
        f"Extensions are {spec.language} files in the candidate's `{EXTENSIONS_DIR}/` directory "
        f"(file types: {suffixes}). Each one hooks the agent loop.",
        "",
        f"- **Each file must:** {spec.entry_convention}",
        f"- **Framework version:** {spec.framework_version} (pinned; write against this version's API)",
    ]
    lines += [
        "- **Checked by the controller:** every extension file must load. One that fails to load makes the "
        "trial `harness_load_error`; extension code that throws during the run makes it `extension_error`. "
        "Both score 0.",
        "",
        "## Where an extension can hook in",
        "",
        "| Point | Support | Native API | Can modify | Caveats |",
        "|---|---|---|---|---|",
    ]
    declared = {point.id: point for point in spec.lifecycle}
    for point_id in LIFECYCLE_POINTS:
        point = declared.get(point_id)
        if point is None:
            lines.append(f"| `{point_id}` | not declared | | | |")
        else:
            lines.append(
                f"| `{point_id}` | {point.support} | {_cell(point.native_api)} | {_cell(point.can_modify)} "
                f"| {_cell(point.caveats)} |"
            )
    if spec.forbidden:
        lines += ["", "## Never use", ""]
        lines += [f"- `{api}`" for api in spec.forbidden]
    if spec.api_reference:
        lines += ["", f"## API reference (pinned, {spec.framework_version})", "", spec.api_reference.rstrip()]
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    # Keep a table cell on one row.
    return text.replace("|", "\\|").replace("\n", " ")
