from pathlib import Path

from hive.candidate import Candidate
from hive.contracts import MetricSpec, Overseer, OverseerBrief
from hive.overseers.null import NullOverseer


def write_candidate(root: Path) -> Candidate:
    root.mkdir(parents=True)
    (root / "candidate.toml").write_text('schema_version = 1\nagent = "scripted"\n')
    (root / "responses.toml").write_text('default = "ANSWER: unknown"\n')
    (root / "notes").mkdir()
    (root / "notes" / "idea.md").write_text("nested files are copied too")
    return Candidate.load(root)


def make_brief(incumbent: Candidate, tmp_path: Path) -> OverseerBrief:
    # The null overseer ignores everything but the incumbent, so the rest are dummies.
    return OverseerBrief(
        incumbent=incumbent,
        metric=MetricSpec(name="dummy", description="dummy"),
        suite_description="dummy",
        valves="dummy",
        selection_rule="dummy",
        generation=1,
        incumbent_report=tmp_path / "report.json",
        evidence_dir=tmp_path / "evidence",
        history=[],
    )


def test_null_overseer_satisfies_the_overseer_contract():
    overseer = NullOverseer()
    assert isinstance(overseer, Overseer)
    assert overseer.name == "null"
    assert overseer.proposes_changes is False


def test_null_overseer_accepts_settings_argument():
    assert NullOverseer({"anything": 1}).name == "null"


def test_proposal_is_a_copy_in_the_workspace_candidate_dir(tmp_path):
    incumbent = write_candidate(tmp_path / "incumbent")
    workspace = tmp_path / "workspace"
    proposal = NullOverseer().propose(make_brief(incumbent, tmp_path), workspace)
    assert proposal.candidate_dir == (workspace / "candidate").resolve()
    assert (proposal.candidate_dir / "notes" / "idea.md").read_text() == "nested files are copied too"


def test_copy_has_the_same_candidate_id_as_the_incumbent(tmp_path):
    incumbent = write_candidate(tmp_path / "incumbent")
    proposal = NullOverseer().propose(make_brief(incumbent, tmp_path), tmp_path / "workspace")
    assert Candidate.load(proposal.candidate_dir).id == incumbent.id


def test_proposal_has_the_null_control_rationale_zero_cost_and_no_error(tmp_path):
    incumbent = write_candidate(tmp_path / "incumbent")
    proposal = NullOverseer().propose(make_brief(incumbent, tmp_path), tmp_path / "workspace")
    assert proposal.rationale == "null control: unchanged copy of the incumbent"
    assert proposal.cost_usd == 0.0
    assert proposal.error is None


def test_incumbent_is_left_untouched(tmp_path):
    incumbent = write_candidate(tmp_path / "incumbent")
    NullOverseer().propose(make_brief(incumbent, tmp_path), tmp_path / "workspace")
    assert Candidate.load(incumbent.root).id == incumbent.id


def test_existing_candidate_dir_in_workspace_returns_an_error_instead_of_raising(tmp_path):
    incumbent = write_candidate(tmp_path / "incumbent")
    workspace = tmp_path / "workspace"
    (workspace / "candidate").mkdir(parents=True)
    proposal = NullOverseer().propose(make_brief(incumbent, tmp_path), workspace)
    assert proposal.candidate_dir is None
    assert proposal.rationale == ""
    assert proposal.error


def test_vanished_incumbent_returns_an_error_instead_of_raising(tmp_path):
    incumbent = write_candidate(tmp_path / "incumbent")
    for path in sorted(incumbent.root.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    incumbent.root.rmdir()
    proposal = NullOverseer().propose(make_brief(incumbent, tmp_path), tmp_path / "workspace")
    assert proposal.candidate_dir is None
    assert proposal.error
