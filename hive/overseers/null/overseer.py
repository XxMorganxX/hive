"""The null overseer: the noise control from principle P9.

It proposes an unchanged copy of the incumbent. Running the same loop with this
overseer shows how much apparent "improvement" the selection rule produces from
evaluation noise alone, which is the baseline every real overseer must beat.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hive.candidate import CandidateError
from hive.contracts import OverseerBrief, Proposal

CANDIDATE_DIR = "candidate"
RATIONALE = "null control: unchanged copy of the incumbent"


class NullOverseer:
    name = "null"
    # The copy is identical by design, so the controller must not reject it as a duplicate.
    proposes_changes = False

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        # Nothing to configure: the null control is the same in every experiment.
        self.settings = dict(settings or {})

    def propose(self, brief: OverseerBrief, workspace: Path) -> Proposal:
        destination = Path(workspace) / CANDIDATE_DIR
        try:
            copy = brief.incumbent.copy_to(destination)
        except (CandidateError, OSError) as error:
            return Proposal(candidate_dir=None, rationale="", error=f"could not copy the incumbent: {error}")
        return Proposal(candidate_dir=copy.root, rationale=RATIONALE, cost_usd=0.0)
