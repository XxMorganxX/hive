from pathlib import Path

import pytest

from hive.candidate import Candidate, CandidateError


def write_candidate(root: Path, agent: str = "pi", extra: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.toml").write_text(
        f'schema_version = 1\nagent = "{agent}"\ndescription = "test"\n\n[valves]\nthinking = "low"\n'
    )
    for rel, text in (extra or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_load_reads_manifest_fields(tmp_path):
    candidate = Candidate.load(write_candidate(tmp_path / "c"))
    assert candidate.agent == "pi"
    assert candidate.valves == {"thinking": "low"}
    assert candidate.description == "test"


def test_id_is_stable_for_identical_content(tmp_path):
    a = Candidate.load(write_candidate(tmp_path / "a", extra={"SYSTEM.md": "be careful"}))
    b = Candidate.load(write_candidate(tmp_path / "b", extra={"SYSTEM.md": "be careful"}))
    assert a.id == b.id


def test_id_changes_when_any_file_changes(tmp_path):
    a = Candidate.load(write_candidate(tmp_path / "a", extra={"SYSTEM.md": "be careful"}))
    b = Candidate.load(write_candidate(tmp_path / "b", extra={"SYSTEM.md": "be bold"}))
    assert a.id != b.id


def test_id_changes_when_a_file_is_renamed(tmp_path):
    a = Candidate.load(write_candidate(tmp_path / "a", extra={"x.md": "same"}))
    b = Candidate.load(write_candidate(tmp_path / "b", extra={"y.md": "same"}))
    assert a.id != b.id


def test_id_ignores_lineage_and_os_junk(tmp_path):
    a = Candidate.load(write_candidate(tmp_path / "a"))
    b_root = write_candidate(tmp_path / "b", extra={".DS_Store": "junk", "__pycache__/x.pyc": "junk"})
    b = Candidate.load(b_root)
    b.write_lineage(parents=["p"], origin="overseer", rationale="because")
    assert Candidate.load(b_root).id == a.id


def test_lineage_round_trips(tmp_path):
    candidate = Candidate.load(write_candidate(tmp_path / "c"))
    candidate.write_lineage(parents=["abc"], origin="overseer", rationale="observed x; changed y")
    assert candidate.lineage() == {"parents": ["abc"], "origin": "overseer", "rationale": "observed x; changed y"}


def test_lineage_is_none_when_absent(tmp_path):
    assert Candidate.load(write_candidate(tmp_path / "c")).lineage() is None


def test_copy_to_preserves_id_and_content(tmp_path):
    original = Candidate.load(write_candidate(tmp_path / "a", extra={"extensions/x.ts": "export {}"}))
    copy = original.copy_to(tmp_path / "copy")
    assert copy.id == original.id
    assert (copy.root / "extensions" / "x.ts").read_text() == "export {}"


def test_copy_to_drops_files_that_are_not_part_of_the_id(tmp_path):
    # Whatever ships with a candidate must be covered by its id, or an edit could hide from dedup and diffs.
    root = write_candidate(tmp_path / "a", extra={".hidden/payload.ts": "x", "__pycache__/m.pyc": "x"})
    copy = Candidate.load(root).copy_to(tmp_path / "copy")
    assert not (copy.root / ".hidden").exists()
    assert not (copy.root / "__pycache__").exists()


def test_copy_to_keeps_lineage(tmp_path):
    original = Candidate.load(write_candidate(tmp_path / "a"))
    original.write_lineage(parents=["p"], origin="seed", rationale="r")
    assert original.copy_to(tmp_path / "copy").lineage() == original.lineage()


def test_copy_to_refuses_existing_destination(tmp_path):
    original = Candidate.load(write_candidate(tmp_path / "a"))
    (tmp_path / "taken").mkdir()
    with pytest.raises(CandidateError):
        original.copy_to(tmp_path / "taken")


def test_missing_manifest_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(CandidateError, match="candidate.toml"):
        Candidate.load(tmp_path / "empty")


def test_manifest_without_agent_is_an_error(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "candidate.toml").write_text("schema_version = 1\n")
    with pytest.raises(CandidateError, match="agent"):
        Candidate.load(root)


def test_malformed_toml_is_an_error(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "candidate.toml").write_text("agent = \n")
    with pytest.raises(CandidateError):
        Candidate.load(root)


def test_unsupported_schema_version_is_an_error(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "candidate.toml").write_text('schema_version = 99\nagent = "pi"\n')
    with pytest.raises(CandidateError, match="schema_version"):
        Candidate.load(root)
