"""Linux ancestor containment for bounded RSI subprocesses (stdlib only)."""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

if __package__:
    from .rsi_process import ProcessError, ProcessResult


def _capture(out_fd: int, err_fd: int, limit: int) -> tuple[str, str]:
    out = os.pread(out_fd, limit + 1, 0)
    err = os.pread(err_fd, limit + 1, 0)
    if len(out) + len(err) > limit:
        raise ValueError(f"exceeded combined output limit of {limit} bytes")
    try:
        return out.decode("utf-8"), err.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("returned invalid UTF-8") from exc


def _tail(fd: int) -> str:
    size = os.fstat(fd).st_size
    text = os.pread(fd, 4096, max(0, size - 4096))
    tail = text.decode("utf-8", errors="replace").strip() or "(no stderr)"
    return ("[last bytes] " if size > 4096 else "") + tail


def _contain() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    # PR_SET_CHILD_SUBREAPER and PR_SET_PDEATHSIG, before any command fork.
    for option, value in ((36, 1), (1, signal.SIGTERM)):
        if libc.prctl(option, value, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))


def _drain(process: subprocess.Popen | None, deadline: float) -> bool:
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        # Unreaped direct children cannot have their PIDs reused. Killing them
        # makes the kernel adopt deeper descendants here, including setsid ones.
        for child in children.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return True
            if pid == 0:
                break
            if process is not None and pid == process.pid:
                process.returncode = os.waitstatus_to_exitcode(status)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def _guardian(owner: int, deadline: float, limit: int, input_fd: int,
              out_fd: int, err_fd: int, result_fd: int,
              cwd: str, command: list[str]) -> int:
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    process = None
    failure = None
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK,
                               {signal.SIGTERM, signal.SIGINT, signal.SIGHUP})
        _contain()
        # PDEATHSIG alone misses death between fork/exec and prctl.
        if os.getppid() != owner:
            stopping = True
        if stopping:
            failure = "cancelled before launch or caller exited"
        elif time.monotonic() >= deadline:
            failure = "timed out before launch"
        else:
            process = subprocess.Popen(
                command, cwd=cwd, stdin=input_fd, stdout=out_fd, stderr=err_fd,
                start_new_session=True, close_fds=True,
            )
        while process is not None and failure is None:
            if stopping:
                failure = "cancelled or caller exited"
            elif os.fstat(out_fd).st_size + os.fstat(err_fd).st_size > limit:
                failure = f"exceeded combined output limit of {limit} bytes"
            elif time.monotonic() >= deadline:
                failure = "timed out"
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.01)
    except BaseException as exc:
        failure = f"could not start or monitor process: {exc}"
    finally:
        try:
            if not _drain(process, time.monotonic() + 4.0):
                failure = (failure or "termination failed") + "; children did not exit after kill"
        except OSError as exc:
            failure = (failure or "termination failed") + f"; cleanup failed: {exc}"
    if failure is None:
        try:
            _capture(out_fd, err_fd, limit)
        except (ValueError, OSError) as exc:
            failure = str(exc)
    with os.fdopen(result_fd, "w", encoding="utf-8") as result:
        json.dump({"returncode": None if process is None else process.returncode,
                   "error": failure}, result)
    return 0


def _stop_guardian(process: subprocess.Popen, deadline: float) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass


def run_guarded(
    command: list[str],
    *,
    cwd: Path,
    input_text: str = "",
    timeout: float,
    artifact_dir: Path,
    cancel: threading.Event | None = None,
    max_output_bytes: int = 1048576,
) -> ProcessResult:
    """Run under a private Linux subreaper, retaining stdout/stderr and status.

    artifact_dir must be new or empty and outside the candidate repository.
    No environment is serialized. Credentials are inherited only by processes.
    Cancellation/death asks the guardian to kill and reap its own subtree.
    """
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or "\0" in arg for arg in command)
            or not command[0]):
        raise ValueError("command must be a nonempty argv list of strings without NUL")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a finite positive number")
    if (isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0):
        raise ValueError("max_output_bytes must be a positive integer")
    if not isinstance(input_text, str):
        raise ValueError("input_text must be a string")
    if sys.platform != "linux":
        raise ProcessError("run_guarded requires Linux subreaping")

    started = time.monotonic()
    deadline = started + timeout
    label = repr(command[0][:200])
    if cancel is not None and cancel.is_set():
        raise ProcessError(f"process {label} cancelled before launch; stderr: (no stderr)")
    try:
        workdir = Path(cwd).resolve()
        artifacts = Path(artifact_dir).resolve()
        repo = next((p for p in (workdir, *workdir.parents)
                     if (p / ".git").exists()), workdir)
        if artifacts.is_relative_to(repo) or repo.is_relative_to(artifacts):
            raise ValueError("artifact_dir must be outside the candidate repository")
        artifacts.mkdir(mode=0o700, parents=True, exist_ok=True)
        if next(artifacts.iterdir(), None) is not None:
            raise ValueError("artifact_dir must be fresh (new or empty)")
        with (
            tempfile.TemporaryFile() as source,
            (artifacts / "stdout.bin").open("x+b") as stdout,
            (artifacts / "stderr.bin").open("x+b") as stderr,
            (artifacts / "process.json").open("x+b") as status,
        ):
            source.write(input_text.encode("utf-8"))
            source.seek(0)
            if cancel is not None and cancel.is_set():
                raise ProcessError(f"process {label} cancelled before launch; stderr: (no stderr)")
            if time.monotonic() >= deadline:
                raise ProcessError(f"process {label} timed out before launch; stderr: (no stderr)")
            fds = [f.fileno() for f in (source, stdout, stderr, status)]
            argv = [
                sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--guardian",
                str(os.getpid()), str(deadline), str(max_output_bytes),
                *(str(fd) for fd in fds), str(workdir), *command,
            ]
            worker = subprocess.Popen(
                argv, cwd=artifacts, pass_fds=tuple(fds), start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            failure = None
            hard_deadline = deadline + 4.5
            try:
                while worker.poll() is None:
                    if cancel is not None and cancel.is_set():
                        failure = "cancelled"
                        break
                    remaining = hard_deadline - time.monotonic()
                    if remaining <= 0:
                        failure = "timed out; guardian cleanup deadline exceeded"
                        break
                    time.sleep(min(0.01, remaining))
            finally:
                _stop_guardian(worker, min(hard_deadline, time.monotonic() + 4.5))
            if cancel is not None and cancel.is_set():
                failure = failure or "cancelled"
            if failure:
                raise ProcessError(f"process {label} {failure}; stderr tail: {_tail(stderr.fileno())}")
            if worker.returncode != 0:
                raise ProcessError(
                    f"process {label} guardian exited {worker.returncode}; "
                    f"stderr tail: {_tail(stderr.fileno())}")
            status.seek(0)
            try:
                record = json.loads(status.read(65536))
                if not isinstance(record, dict) or set(record) != {"returncode", "error"}:
                    raise ValueError("invalid guardian result")
                if record["error"] is not None:
                    raise ValueError(str(record["error"]))
                if type(record["returncode"]) is not int:
                    raise ValueError("missing command exit status")
                out, err = _capture(stdout.fileno(), stderr.fileno(), max_output_bytes)
            except (ValueError, UnicodeError) as exc:
                raise ProcessError(
                    f"process {label} {exc}; stderr tail: {_tail(stderr.fileno())}") from exc
            return ProcessResult(record["returncode"], out, err, time.monotonic() - started)
    except OSError as exc:
        raise ProcessError(f"process {label} I/O or launch failure: {exc}") from exc


if __name__ == "__main__":
    if len(sys.argv) < 11 or sys.argv[1] != "--guardian":
        raise SystemExit("private guardian entry point")
    raise SystemExit(_guardian(
        int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]),
        *(int(fd) for fd in sys.argv[5:9]), sys.argv[9], sys.argv[10:],
    ))
