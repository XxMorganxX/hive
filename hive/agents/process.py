"""Run one agent trial as a subprocess that can always be stopped completely.

Framework adapters run each trial in its own process: candidate code (extensions) never
executes in the controller, and a timeout can actually stop a run. Stopping one is harder
than it looks, because frameworks start shell commands in sessions of their own
(`start_new_session=True`), outside the trial's process group. Killing the group leaves
them running. So:

- the trial process leads a new process group, and a timeout sends SIGTERM to it (so the
  framework's own handler can clean up), waits a grace period, then sends SIGKILL;
- output goes to files, not pipes, and the run ends when the process exits. A detached child
  inherits stdout; waiting for the pipe to close would wait for that child too;
- a runner may list the process-group ids of the detached children it started, one per
  line, in `extra_pgids_file`. Those groups are killed after every run, finished or not,
  because a command the agent left running in the background must not outlive the trial.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessOutcome:
    stdout: bytes  # as written to stdout_path
    stderr: bytes  # as written to stderr_path
    returncode: int | None  # None when the process never started or was killed on timeout
    timed_out: bool
    launch_error: OSError | None


def run_process_group(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    stdout_path: Path,
    stderr_path: Path,
    term_grace_s: float = 5.0,
    stdin_text: str | None = None,
    extra_pgids_file: Path | None = None,
) -> ProcessOutcome:
    stdin = subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=stdin,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # leader of its own group, so the whole group can be signalled
            )
        except OSError as error:
            return ProcessOutcome(b"", b"", None, False, error)
        timed_out = False
        try:
            if stdin_text is not None:
                try:
                    process.stdin.write(stdin_text.encode())
                    process.stdin.close()
                except BrokenPipeError:  # it exited without reading its input; its exit says why
                    pass
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(process, term_grace_s)
        finally:
            if process.poll() is None:  # e.g. KeyboardInterrupt mid-run: never leak a trial process
                _stop_group(process, term_grace_s)
            _kill_group(process.pid)  # plain children the leader left behind
            for pgid in _listed_pgids(extra_pgids_file):
                _kill_group(pgid)
    return ProcessOutcome(
        stdout=stdout_path.read_bytes(),
        stderr=stderr_path.read_bytes(),
        returncode=None if timed_out else process.returncode,
        timed_out=timed_out,
        launch_error=None,
    )


def _stop_group(process: subprocess.Popen, term_grace_s: float) -> None:
    _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        pass
    _kill_group(process.pid)
    process.wait()


def _kill_group(pgid: int) -> None:
    _signal_group(pgid, signal.SIGKILL)


def _signal_group(pgid: int, signum: signal.Signals) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _listed_pgids(path: Path | None) -> list[int]:
    if path is None or not path.is_file():
        return []
    pgids = []
    for line in path.read_text(errors="replace").split():
        # Only positive ids: killpg(0) would signal our own group, and a negative id is nonsense here.
        if line.isdigit() and int(line) > 0:
            pgids.append(int(line))
    return pgids
