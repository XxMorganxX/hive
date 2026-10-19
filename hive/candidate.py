"""A candidate is a directory: a manifest plus adapter-owned files (P4).

The controller never interprets a candidate's files. It loads the manifest to learn
which agent adapter owns it, hashes the contents to get an id, and copies it around.
Only the agent adapter understands what the files mean.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST = "candidate.toml"
LINEAGE = "lineage.json"
SUPPORTED_SCHEMA_VERSIONS = (1,)

# Never part of a candidate's identity: lineage is metadata about the candidate,
# and the rest is OS or tool debris that must not create spurious "new" candidates.
_IGNORED_NAMES = {LINEAGE, "__pycache__", "node_modules"}


class CandidateError(Exception):
    """The directory is not a loadable candidate."""


def _is_ignored(relative: Path) -> bool:
    return any(part in _IGNORED_NAMES or part.startswith(".") for part in relative.parts)


def content_files(root: Path) -> list[Path]:
    """Every file that defines the candidate, as sorted paths relative to `root`."""
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and not _is_ignored(relative):
            files.append(relative)
    return sorted(files, key=lambda p: p.as_posix())


def _symlinks_under(root: Path) -> list[str]:
    links = []
    for directory, dirnames, filenames in os.walk(root):  # does not descend into symlinked directories
        for name in dirnames + filenames:
            path = Path(directory) / name
            if path.is_symlink():
                links.append(path.relative_to(root).as_posix())
    return sorted(links)


def content_hash(root: Path) -> str:
    """Hash of every content file's relative path and bytes. Renames and edits both change it."""
    digest = hashlib.sha256()
    for relative in content_files(root):
        path_bytes = relative.as_posix().encode()
        data = (root / relative).read_bytes()
        # Length-prefix both fields so (path, data) boundaries can't be forged by concatenation.
        digest.update(len(path_bytes).to_bytes(8, "big") + path_bytes)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Candidate:
    root: Path
    manifest: Mapping[str, Any]
    id: str

    @classmethod
    def load(cls, root: Path) -> Candidate:
        # A link would let a candidate read or ship files from outside its own directory,
        # and the hash would cover the target's bytes without the target being part of it.
        if Path(root).is_symlink():
            raise CandidateError(f"{root} is a symlink; a candidate must be a real directory")
        root = Path(root).resolve()
        links = _symlinks_under(root)
        if links:
            raise CandidateError(f"{root} contains symlinks: {', '.join(links)}")
        manifest_path = root / MANIFEST
        if not manifest_path.is_file():
            raise CandidateError(f"{root} has no {MANIFEST}")
        try:
            manifest = tomllib.loads(manifest_path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise CandidateError(f"{manifest_path} is not valid TOML: {error}") from error
        schema_version = manifest.get("schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise CandidateError(
                f"{manifest_path}: unsupported schema_version {schema_version!r} "
                f"(supported: {SUPPORTED_SCHEMA_VERSIONS})"
            )
        if not isinstance(manifest.get("agent"), str) or not manifest["agent"]:
            raise CandidateError(f"{manifest_path}: 'agent' must name the agent adapter that runs this candidate")
        if not isinstance(manifest.get("valves", {}), dict):
            raise CandidateError(f"{manifest_path}: [valves] must be a table")
        return cls(root=root, manifest=manifest, id=content_hash(root))

    @property
    def agent(self) -> str:
        return self.manifest["agent"]

    @property
    def description(self) -> str:
        return self.manifest.get("description", "")

    @property
    def valves(self) -> Mapping[str, Any]:
        return self.manifest.get("valves", {})

    def lineage(self) -> dict[str, Any] | None:
        path = self.root / LINEAGE
        return json.loads(path.read_text()) if path.is_file() else None

    def write_lineage(self, parents: Sequence[str], origin: str, rationale: str) -> None:
        lineage = {"parents": list(parents), "origin": origin, "rationale": rationale}
        (self.root / LINEAGE).write_text(json.dumps(lineage, indent=2) + "\n")

    def copy_to(self, dest: Path) -> Candidate:
        """Copy the whole directory (lineage included) to a new location and load the copy."""
        dest = Path(dest)
        if dest.exists():
            raise CandidateError(f"refusing to copy candidate onto existing path {dest}")
        # Copy exactly what the id covers (plus lineage), so nothing can ship without changing the id.
        def not_part_of_candidate(directory: str, names: list[str]) -> set[str]:
            return {name for name in names if name != LINEAGE and _is_ignored(Path(name))}

        shutil.copytree(self.root, dest, ignore=not_part_of_candidate)
        return Candidate.load(dest)
