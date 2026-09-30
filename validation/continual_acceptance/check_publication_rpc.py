"""Two offline RPC backpressure checks; run from candidate cwd with pinned Python.

Instantiate the real _Runner without entering its driver context, then supply a
real Popen whose child keeps stdin open but never reads it. No response or product
method is mocked. The unchanged core rsi_guard contains each probe and its child,
including a blocked Python write. No Git, mission, model, or network is invoked.
"""
from __future__ import annotations

import array
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import unittest

PY = "/home/akhd/work/gama-rsi/.venv/bin/python"
ROOT = Path.cwd().resolve()
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

CANCEL_LIMIT = 2.0
DEADLINE_DURATION = 0.25
DEADLINE_LIMIT = DEADLINE_DURATION + 10.0 + 1.0  # Existing overhead + scheduling slack.
GUARD_LIMITS = {"cancel": 5.0, "deadline": 13.0}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_identity(pid):
    try:
        # comm may contain spaces or ')'; the remaining fields start at state.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return [pid, fields[19]]
    except FileNotFoundError:
        return None


def probe(case, root):
    publisher = importlib.import_module("gama.continual_publish")
    ready = root / "child-ready"
    process = subprocess.Popen(
        [PY, "-I", "-B", str(SCRIPT), "--nonreading-child", str(ready)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True, close_fds=True,
    )
    observer = None
    done = threading.Event()
    try:
        startup_deadline = time.monotonic() + 2.0
        while not ready.exists():
            if process.poll() is not None or time.monotonic() >= startup_deadline:
                raise RuntimeError("non-reading child did not become ready")
            time.sleep(0.005)
        fd = process.stdin.fileno()
        capacity = fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)
        payload = "x" * max(65536, capacity * 4)
        cancellation = threading.Event()
        duration = 30.0 if case == "cancel" else DEADLINE_DURATION
        runner = publisher._Runner(ROOT, root, duration, cancellation)
        runner.logs.mkdir()
        runner.process = process
        started = time.monotonic()
        source = Path(publisher.__file__).resolve()
        write_json(root / "started.json", {
            "case": case, "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "worker": process_identity(os.getpid()),
            "child": process_identity(process.pid),
            "pipe_capacity": capacity, "payload_bytes": len(payload),
            "duration": duration, "started": started,
        })

        def observe_backpressure():
            while not done.wait(0.005):
                pending = array.array("i", [0])
                fcntl.ioctl(fd, termios.FIONREAD, pending, True)
                writable = select.select([], [fd], [], 0)[1]
                if pending[0] > 0 and not writable:
                    observation = {"pending_bytes": pending[0],
                                   "observed": time.monotonic()}
                    if case == "cancel":
                        observation["cancel_at"] = time.monotonic()
                        cancellation.set()
                    write_json(root / "backpressure.json", observation)
                    return

        observer = threading.Thread(target=observe_backpressure, daemon=True)
        observer.start()
        result = {}
        try:
            runner.run([PY, "-I", "-B", "-c", "pass"],
                       timeout=duration, input_text=payload)
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc),
                          expected_error=isinstance(exc, (publisher.PublicationError,
                                                          TimeoutError)))
        else:
            result.update(error_type=None, error="run returned success", expected_error=False)
        returned = time.monotonic()
        result.update(returned=returned, elapsed=returned - started,
                      child_alive_at_return=process.poll() is None,
                      cancel_set=cancellation.is_set())
        write_json(root / "returned.json", result)
    finally:
        done.set()
        if observer is not None:
            observer.join(timeout=1.0)
        # Close the read end by terminating/reaping our child before closing a
        # potentially buffered writer. A stuck probe instead uses rsi_guard drain.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2.0)
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.stdout.close()


class PublicationRPC(unittest.TestCase):
    def check_case(self, case):
        from gama.rsi_guard import run_guarded
        from gama.rsi_process import ProcessError

        with tempfile.TemporaryDirectory(prefix="gama-publication-rpc-") as directory:
            root = Path(directory)
            artifact = root / "guardian"
            failure, outcome = None, None
            started = time.monotonic()
            try:
                outcome = run_guarded(
                    [PY, "-I", "-B", str(SCRIPT), "--probe", case, str(root)],
                    cwd=ROOT, timeout=GUARD_LIMITS[case], artifact_dir=artifact,
                )
            except ProcessError as exc:
                failure = str(exc)
            evidence = {"case": case, "wall_seconds": time.monotonic() - started,
                        "guard_error": failure}
            for name in ("started", "backpressure", "returned"):
                path = root / (name + ".json")
                evidence[name] = json.loads(path.read_text()) if path.exists() else None
            status = artifact / "process.json"
            evidence["guard_status"] = json.loads(status.read_text()) if status.stat().st_size else None
            stderr = artifact / "stderr.bin"
            evidence["probe_stderr"] = stderr.read_text(errors="replace")[-2000:]
            before = evidence["started"]
            evidence["descendants_gone"] = bool(before) and all(
                process_identity(identity[0]) != identity
                for identity in (before["worker"], before["child"])
            )
            print("RPC_OBSERVATION " + json.dumps(evidence, sort_keys=True), flush=True)

            self.assertIsNotNone(before, "probe failed before entering _Runner.run")
            self.assertEqual(Path(before["source"]), ROOT / "gama/continual_publish.py")
            self.assertGreater(before["payload_bytes"], before["pipe_capacity"])
            self.assertIsNotNone(evidence["backpressure"],
                                 "must observe real input pipe backpressure")
            self.assertTrue(evidence["descendants_gone"], "external guard left a probe descendant")
            self.assertIsNone(failure,
                              f"{case}: _Runner.run hung until external guard timeout: {failure}")
            self.assertEqual(outcome.returncode, 0, outcome.stderr)
            result = evidence["returned"]
            self.assertIsNotNone(result, "_Runner.run did not return to its caller")
            self.assertTrue(result["child_alive_at_return"],
                            "child exit must not be what unblocks input transmission")
            self.assertTrue(result["expected_error"], result)
            message = result["error"].lower()
            if case == "cancel":
                cancelled = evidence["backpressure"]["cancel_at"]
                self.assertTrue(result["cancel_set"])
                self.assertGreaterEqual(result["returned"], cancelled)
                self.assertLessEqual(result["returned"] - cancelled, CANCEL_LIMIT,
                                     "cancellation must interrupt an in-flight request write")
                self.assertTrue("cancel" in message or "stop" in message, result)
            else:
                self.assertFalse(result["cancel_set"])
                self.assertGreaterEqual(result["elapsed"], DEADLINE_DURATION)
                self.assertLessEqual(result["elapsed"], DEADLINE_LIMIT,
                                     "deadline must include input transmission")
                self.assertTrue(any(word in message for word in ("deadline", "timeout", "timed out")),
                                result)

    def test_cancellation_interrupts_backpressured_request(self):
        self.check_case("cancel")

    def test_deadline_includes_backpressured_request_transmission(self):
        self.check_case("deadline")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--nonreading-child"]:
        Path(sys.argv[2]).touch()
        while True:
            signal.pause()
    elif sys.argv[1:2] == ["--probe"]:
        probe(sys.argv[2], Path(sys.argv[3]))
    else:
        unittest.main(verbosity=2)
