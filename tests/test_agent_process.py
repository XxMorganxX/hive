import os
import sys
import textwrap
import time
from pathlib import Path

from hive.agents.process import run_process_group


def script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "child.py"
    path.write_text(textwrap.dedent(body))
    return [sys.executable, str(path)]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat "exited but not reaped" as dead.
    status = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(status) and not status.startswith("Z")


def wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_captures_output_and_exit_code(tmp_path):
    argv = script(tmp_path, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    outcome = run_process_group(argv, stdout_path=tmp_path / 'out.txt', stderr_path=tmp_path / 'err.txt', cwd=tmp_path, env=dict(os.environ), timeout_s=10)
    assert outcome.stdout.strip() == b"out"
    assert outcome.stderr.strip() == b"err"
    assert outcome.returncode == 3
    assert not outcome.timed_out and outcome.launch_error is None


def test_runs_in_the_given_working_directory(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    argv = script(tmp_path, "import os; print(os.getcwd())")
    outcome = run_process_group(argv, stdout_path=tmp_path / 'out.txt', stderr_path=tmp_path / 'err.txt', cwd=workdir, env=dict(os.environ), timeout_s=10)
    assert Path(outcome.stdout.decode().strip()).resolve() == workdir.resolve()


def test_passes_stdin_text(tmp_path):
    argv = script(tmp_path, "import sys; print(sys.stdin.read().upper())")
    outcome = run_process_group(argv, stdout_path=tmp_path / 'out.txt', stderr_path=tmp_path / 'err.txt', cwd=tmp_path, env=dict(os.environ), timeout_s=10, stdin_text="hello")
    assert outcome.stdout.strip() == b"HELLO"


def test_timeout_kills_the_process_and_reports_it(tmp_path):
    pid_file = tmp_path / "pid"
    argv = script(tmp_path, f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)")
    started = time.monotonic()
    outcome = run_process_group(argv, stdout_path=tmp_path / 'out.txt', stderr_path=tmp_path / 'err.txt', cwd=tmp_path, env=dict(os.environ), timeout_s=1, term_grace_s=0.5)
    assert outcome.timed_out
    assert time.monotonic() - started < 10
    assert wait_dead(int(pid_file.read_text()))


def test_child_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    pid_file = tmp_path / "pid"
    argv = script(
        tmp_path,
        f"""
        import os, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open({str(pid_file)!r}, 'w').write(str(os.getpid()))
        time.sleep(60)
        """,
    )
    outcome = run_process_group(argv, stdout_path=tmp_path / 'out.txt', stderr_path=tmp_path / 'err.txt', cwd=tmp_path, env=dict(os.environ), timeout_s=1, term_grace_s=0.5)
    assert outcome.timed_out
    assert wait_dead(int(pid_file.read_text()))


def test_a_detached_child_holding_stdout_does_not_hold_up_a_finished_run(tmp_path):
    # Output goes to files, so the run ends when the process exits, not when every inherited
    # copy of its stdout is closed.
    argv = script(
        tmp_path,
        """
        import subprocess, sys
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        print("done")
        """,
    )
    started = time.monotonic()
    outcome = run_process_group(argv, stdout_path=tmp_path / "out.txt", stderr_path=tmp_path / "err.txt", cwd=tmp_path, env=dict(os.environ), timeout_s=20)
    assert time.monotonic() - started < 10
    assert outcome.returncode == 0 and outcome.stdout.strip() == b"done"


def test_listed_detached_children_are_reaped_even_after_a_normal_exit(tmp_path):
    # Frameworks run shell commands with start_new_session=True, outside the runner's group;
    # the runner lists their process-group ids so they can be killed after every trial.
    pgids = tmp_path / "child_pgids.txt"
    sleeper_pid = tmp_path / "sleeper_pid"
    argv = script(
        tmp_path,
        f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        open({str(pgids)!r}, "a").write(f"{{child.pid}}\\n")
        open({str(sleeper_pid)!r}, "w").write(str(child.pid))
        """,
    )
    outcome = run_process_group(argv, stdout_path=tmp_path / 'out.txt', stderr_path=tmp_path / 'err.txt', cwd=tmp_path, env=dict(os.environ), timeout_s=10, extra_pgids_file=pgids)
    assert outcome.returncode == 0
    assert wait_dead(int(sleeper_pid.read_text()))


def test_missing_binary_is_a_launch_error_not_an_exception(tmp_path):
    outcome = run_process_group(["/nonexistent/binary"], stdout_path=tmp_path / "out.txt", stderr_path=tmp_path / "err.txt", cwd=tmp_path, env=dict(os.environ), timeout_s=5)
    assert outcome.launch_error is not None
    assert outcome.returncode is None
