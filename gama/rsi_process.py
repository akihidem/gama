"""Bounded subprocess execution for the fixed RSI controller (stdlib only)."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time


class ProcessError(RuntimeError):
    """A process could not be started or exceeded its execution limits."""


@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def _stderr_tail(stream, limit: int = 4096) -> str:
    size = os.fstat(stream.fileno()).st_size
    stream.seek(max(0, size - limit))
    tail = stream.read(limit).decode("utf-8", errors="replace").strip()
    return ("[last bytes] " if size > limit else "") + (tail or "(no stderr)")


def _stop_process(process: subprocess.Popen) -> bool:
    """Stop the entire POSIX process group, including surviving descendants."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return False
    return True


def run_process(
    command: list[str],
    *,
    cwd: Path,
    input_text: str = "",
    timeout: float = 300,
    cancel: threading.Event | None = None,
    max_output_bytes: int = 1048576,
    env: dict | None = None,
) -> ProcessResult:
    """Execute argv without a shell and capture complete UTF-8 output.

    Ordinary nonzero exits are returned for the caller to interpret. Launch errors,
    cancellation, timeout, invalid UTF-8, and output overflow raise ``ProcessError``.
    ``max_output_bytes`` bounds stdout and stderr *combined*. Temporary files also
    prevent an unread stdin pipe from blocking timeout/cancellation handling.

    ``env`` has subprocess semantics: None inherits the environment; a supplied
    mapping replaces it. This helper never changes HOME or any environment entry.
    Remaining descendants in the child's POSIX process group are killed on return.
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

    label = repr(command[0][:200])
    started = time.monotonic()
    if cancel is not None and cancel.is_set():
        raise ProcessError(f"process {label} cancelled before launch; stderr: (no stderr)")

    try:
        with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, \
                tempfile.TemporaryFile() as stderr:
            stdin.write(input_text.encode("utf-8"))
            stdin.seek(0)
            if cancel is not None and cancel.is_set():
                raise ProcessError(
                    f"process {label} cancelled before launch; stderr: (no stderr)")
            if time.monotonic() - started >= timeout:
                raise ProcessError(
                    f"process {label} timed out before launch; stderr: (no stderr)")
            try:
                process = subprocess.Popen(
                    command, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                    env=env, shell=False, start_new_session=(os.name == "posix"),
                )
            except (OSError, ValueError) as exc:
                raise ProcessError(
                    f"could not start process {label}: {exc}; stderr: {_stderr_tail(stderr)}"
                ) from exc

            failure = None
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        failure = "cancelled"
                        break
                    size = (os.fstat(stdout.fileno()).st_size
                            + os.fstat(stderr.fileno()).st_size)
                    if size > max_output_bytes:
                        failure = f"exceeded combined output limit of {max_output_bytes} bytes"
                        break
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        failure = f"timed out after {timeout:g}s"
                        break
                    if process.poll() is not None:
                        break
                    delay = min(0.02, remaining)
                    if cancel is None:
                        time.sleep(delay)
                    else:
                        cancel.wait(delay)
            finally:
                stopped = _stop_process(process)

            if not stopped:
                failure = (failure or "termination failed") + "; did not exit after kill"
            if failure:
                raise ProcessError(
                    f"process {label} {failure}; stderr tail: {_stderr_tail(stderr)}")

            # Recheck after termination: a short-lived process can finish between
            # polling its output and polling its exit code.
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(max_output_bytes + 1)
            err = stderr.read(max_output_bytes + 1)
            if len(out) + len(err) > max_output_bytes:
                raise ProcessError(
                    f"process {label} exceeded combined output limit of "
                    f"{max_output_bytes} bytes; stderr tail: {_stderr_tail(stderr)}")
            try:
                out_text = out.decode("utf-8")
                err_text = err.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProcessError(
                    f"process {label} returned invalid UTF-8; "
                    f"stderr tail: {_stderr_tail(stderr)}") from exc
            return ProcessResult(
                returncode=process.returncode,
                stdout=out_text,
                stderr=err_text,
                elapsed_s=time.monotonic() - started,
            )
    except OSError as exc:
        raise ProcessError(f"process {label} I/O failure: {exc}") from exc
