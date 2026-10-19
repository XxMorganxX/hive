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
  because a command the agent left running in the background must not outlive the trial;
- a process can also leave every listed group on its own: a command that daemonizes
  (fork + setsid), or a process an extension starts with `start_new_session=True`. So every
  trial carries a marker, `HIVE_TRIAL_ID=<random id>`, in its environment, which children
  inherit, and after the run every process of this user that carries it is SIGKILLed
  (TrialProcesses). A runner that gives its shell commands a minimal environment must copy
  the marker into it.

What the marker cannot catch, and so what can still outlive a trial: a process that clears
its environment (`env -i`) and has left the trial's process tree. On macOS, too, `ps` cannot
read the environment of system binaries (/bin, /usr/bin: `sleep`, `sh`), so those are found
only as descendants of a trial process, sampled every TRACK_INTERVAL_S while the trial runs;
one that is orphaned within a sample interval of starting escapes. Linux reads
/proc/<pid>/environ, which has no such gap.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

TRIAL_MARKER_ENV = "HIVE_TRIAL_ID"
# How often a running trial's process tree is sampled, so a child that loses its parent
# (and whose environment the OS hides) is still known to belong to the trial.
TRACK_INTERVAL_S = 0.5
# Kill rounds after a run: a marked process may fork between a scan and the kill.
REAP_ROUNDS = 5
PS_TIMEOUT_S = 10


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
    mark_trial: bool = True,
) -> ProcessOutcome:
    """Run argv to completion or timeout, then kill everything it left behind.

    With `mark_trial` (the default), the child's environment gets this trial's own
    TRIAL_MARKER_ENV, replacing any value `env` had, and the processes carrying it are killed
    after the run (see the module doc).
    """
    marker = new_trial_marker() if mark_trial else None
    child_env = dict(env)
    if marker is not None:
        child_env[TRIAL_MARKER_ENV] = marker
    stdin = subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=child_env,
                stdin=stdin,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # leader of its own group, so the whole group can be signalled
            )
        except OSError as error:
            return ProcessOutcome(b"", b"", None, False, error)
        tracked = TrialProcesses(process.pid, marker) if marker is not None else None
        if tracked is not None:
            tracked.start()
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
            if tracked is not None:
                tracked.kill_all()
    return ProcessOutcome(
        stdout=stdout_path.read_bytes(),
        stderr=stderr_path.read_bytes(),
        returncode=None if timed_out else process.returncode,
        timed_out=timed_out,
        launch_error=None,
    )


def new_trial_marker() -> str:
    return uuid.uuid4().hex


class TrialProcesses:
    """Every process of one trial that this user can see: those whose environment carries the
    trial's marker, the trial leader, and every descendant of either seen while its parent was
    alive. Start it once the leader is running; kill_all() after the run.

    A process is remembered with its start time, so a pid the OS has since reused for an
    unrelated process is never killed. The controller itself is never killed.
    """

    def __init__(self, leader_pid: int, marker: str):
        self._leader_pid = leader_pid
        self._marker = marker
        self._known: dict[int, str] = {}  # pid -> start time, as the process table reports it
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._track, name=f"trial-processes-{leader_pid}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def kill_all(self) -> None:
        self._stop.set()
        self._thread.join()
        for _ in range(REAP_ROUNDS):
            table = _process_table(self._marker)
            self._update(table)
            alive = [pid for pid, start in self._known.items() if pid in table and table[pid].start == start]
            if not alive:
                return
            for pid in alive:
                _kill_process(pid)

    def _track(self) -> None:
        while True:
            self._update(_process_table(self._marker))
            if self._stop.wait(TRACK_INTERVAL_S):
                return

    def _update(self, table: Mapping[int, ProcessInfo]) -> None:
        if self._leader_pid in table and self._leader_pid not in self._known:
            self._known[self._leader_pid] = table[self._leader_pid].start
        for pid, info in table.items():
            if info.marked:
                self._known[pid] = info.start
        # Descendants of known processes, to a fixed point: a child may be listed before its parent.
        grew = True
        while grew:
            grew = False
            for pid, info in table.items():
                if pid in self._known or info.ppid not in self._known:
                    continue
                parent = table.get(info.ppid)
                if parent is not None and parent.start == self._known[info.ppid]:
                    self._known[pid] = info.start
                    grew = True
        self._known.pop(os.getpid(), None)


@dataclass(frozen=True)
class ProcessInfo:
    ppid: int
    start: str  # opaque; equal for the same process across scans
    marked: bool  # its environment carries the trial marker


def _process_table(marker: str) -> dict[int, ProcessInfo]:
    """This user's processes, except the controller. Empty when they can't be listed."""
    if Path("/proc/self/environ").is_file():
        table = _linux_process_table(marker)
    else:
        table = _ps_process_table(marker)
    table.pop(os.getpid(), None)
    return table


def _linux_process_table(marker: str) -> dict[int, ProcessInfo]:
    wanted = f"{TRIAL_MARKER_ENV}={marker}".encode()
    uid = os.getuid()
    table: dict[int, ProcessInfo] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            stat = Path(entry.path, "stat").read_text()
            environ = Path(entry.path, "environ").read_bytes()
        except OSError:  # it exited, or it isn't ours to read
            continue
        # The command name in parentheses may itself contain spaces or parentheses.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) < 20 or fields[0] == "Z":  # a zombie is already dead
            continue
        table[int(entry.name)] = ProcessInfo(
            ppid=int(fields[1]), start=fields[19], marked=wanted in environ.split(b"\0")
        )
    return table


# pid, ppid, uid, state, then lstart's five fields ("Mon Oct 19 10:00:00 2026"), then the
# command followed by its environment.
_PS_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+\s+\S+\s+\S+\s+\S+\s+\S+)\s?(.*)$")


def _ps_process_table(marker: str) -> dict[int, ProcessInfo]:
    try:
        listing = subprocess.run(
            ["ps", "-axE", "-ww", "-o", "pid=,ppid=,uid=,stat=,lstart=,command="],
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=PS_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    wanted = re.compile(rf"(?:^|\s){TRIAL_MARKER_ENV}={re.escape(marker)}(?=\s|$)")
    uid = os.getuid()
    table: dict[int, ProcessInfo] = {}
    for line in listing.splitlines():
        match = _PS_LINE.match(line)
        if match is None or int(match.group(3)) != uid or match.group(4).startswith("Z"):
            continue
        table[int(match.group(1))] = ProcessInfo(
            ppid=int(match.group(2)), start=match.group(5), marked=wanted.search(match.group(6)) is not None
        )
    return table


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


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
