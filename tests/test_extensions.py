from pathlib import Path

from hive.candidate import Candidate
from hive.contracts import LIFECYCLE_POINTS, ExtensionLoad, ExtensionSpec, LifecyclePoint, LoadReport
from hive.extensions import expected_extension_files, load_problem, render_extension_guide

SPEC = ExtensionSpec(
    language="python",
    suffixes=(".py",),
    entry_convention="define `def extension()` returning a hook provider",
    framework_version="1.2.3",
    lifecycle=(
        LifecyclePoint("before_tool_call", "supported", "BeforeToolCallEvent", "cancel or rewrite arguments"),
        LifecyclePoint("compaction", "unsupported", "", ""),
    ),
    forbidden=("agent.model = ...",),
    api_reference="class BeforeToolCallEvent:\n    cancel_tool: bool | str",
)


def candidate_with(tmp_path: Path, files: dict[str, str]) -> Candidate:
    root = tmp_path / "c"
    root.mkdir()
    (root / "candidate.toml").write_text('schema_version = 1\nagent = "x"\n')
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return Candidate.load(root)


# --- expected files --------------------------------------------------------------------------


def test_expected_files_are_the_extension_files_with_the_declared_suffixes(tmp_path):
    candidate = candidate_with(
        tmp_path, {"extensions/b.py": "", "extensions/a.py": "", "extensions/notes.md": "", "SYSTEM.md": ""}
    )
    assert expected_extension_files(candidate, SPEC) == ["extensions/a.py", "extensions/b.py"]


def test_no_spec_means_no_expected_files(tmp_path):
    candidate = candidate_with(tmp_path, {"extensions/a.py": ""})
    assert expected_extension_files(candidate, None) == []


# --- the load check --------------------------------------------------------------------------


def report(*loads: ExtensionLoad, runtime_errors: tuple[str, ...] = ()) -> LoadReport:
    return LoadReport(extensions=loads, runtime_errors=runtime_errors)


def test_nothing_expected_is_never_a_problem():
    assert load_problem([], None) is None


def test_missing_report_when_extensions_are_expected_is_a_load_error():
    error_class, detail = load_problem(["extensions/a.py"], None)
    assert error_class == "harness_load_error"
    assert "no load report" in detail


def test_extension_absent_from_the_report_is_a_load_error():
    error_class, detail = load_problem(["extensions/a.py", "extensions/b.py"], report(ExtensionLoad("extensions/a.py", True)))
    assert error_class == "harness_load_error"
    assert "extensions/b.py" in detail


def test_extension_reported_as_not_loaded_is_a_load_error_with_its_reason():
    error_class, detail = load_problem(["extensions/a.py"], report(ExtensionLoad("extensions/a.py", False, "SyntaxError")))
    assert error_class == "harness_load_error"
    assert "SyntaxError" in detail


def test_runtime_error_after_a_clean_load_is_an_extension_error():
    error_class, detail = load_problem(
        ["extensions/a.py"], report(ExtensionLoad("extensions/a.py", True), runtime_errors=("extensions/a.py: KeyError",))
    )
    assert error_class == "extension_error"
    assert "KeyError" in detail


def test_all_loaded_and_no_runtime_errors_is_fine():
    assert load_problem(["extensions/a.py"], report(ExtensionLoad("extensions/a.py", True))) is None


def test_runtime_errors_count_even_when_no_files_were_expected():
    # An adapter may load its own helper code; a throw there still means the harness misbehaved.
    error_class, _ = load_problem([], report(runtime_errors=("boom",)))
    assert error_class == "extension_error"


# --- the guide for the overseer ----------------------------------------------------------------


def test_guide_names_language_convention_version_and_reference():
    guide = render_extension_guide(SPEC)
    for text in ("python", "`.py`", "def extension()", "1.2.3", "extensions/"):
        assert text in guide


def test_guide_embeds_the_pinned_api_reference_text():
    # The overseer's sandbox can't read the installed framework, so the definitions travel in the guide.
    assert "class BeforeToolCallEvent:\n    cancel_tool: bool | str" in render_extension_guide(SPEC)


def test_guide_lists_every_neutral_lifecycle_point_with_its_support():
    guide = render_extension_guide(SPEC)
    for point in LIFECYCLE_POINTS:
        assert f"`{point}`" in guide
    assert "BeforeToolCallEvent" in guide
    assert "not declared" in guide  # points the adapter did not mention are shown, not hidden


def test_guide_lists_forbidden_apis():
    assert "agent.model = ..." in render_extension_guide(SPEC)


def test_no_spec_gives_an_empty_guide():
    assert render_extension_guide(None) == ""
