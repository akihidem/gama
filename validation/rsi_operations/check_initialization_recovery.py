"""Offline CLI regression. Run from a product checkout; --stub-bridge is component-only."""
import ctypes, fcntl, json, os, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "validation/rsi_operations"))
from ops_common import Env, PY, wait_for

STUB = "--stub-bridge" in sys.argv
if STUB:
    sys.argv.remove("--stub-bridge")


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def unlocked(path):
    if not path.exists():
        return False
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def cleanup_owned():
    # This standalone test process subreaps only descendants of its own fixtures.
    end = time.monotonic() + 2
    while time.monotonic() < end:
        for pid in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        time.sleep(.01)
    raise AssertionError("fixture descendants did not drain")


class InitializationRecovery(unittest.TestCase):
    def build(self, ceilings=(.25, .25), gate="wait"):
        tmp = tempfile.TemporaryDirectory(prefix="gama-init-recovery-")
        self.addCleanup(tmp.cleanup)
        self.addCleanup(cleanup_owned)
        self.e = e = Env(Path(tmp.name), stub_bridge=STUB, ceilings=ceilings)
        self.marker, self.log = e.tmp / "gate.marker", e.tmp / "gate.jsonl"
        self.marker.write_text(gate)
        check = e.tmp / "seed_check.py"
        check.write_text(f"""import fcntl,json,os,sys,time
from pathlib import Path
marker=Path({str(self.marker)!r})
def note(event):
 fd=os.open({str(self.log)!r},os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 calls=Path({str(e.calls)!r})
 row={{"event":event,"calls":len(calls.read_text().splitlines()) if calls.exists() else 0}}
 os.write(fd,(json.dumps(row)+"\\n").encode());os.close(fd)
with open({str(e.tmp / 'seed.lock')!r},"a+b") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 note("start")
 if marker.exists() and marker.read_text()=="fail": sys.exit(23)
 while marker.exists(): time.sleep(.02)
 note("passed")
""")
        config = json.loads(e.rsi.read_text())
        config["checks"] = [[PY, "-B", str(check)]]
        config["evaluation_files"].append(str(check))
        config["evaluation_timeout"] = 4
        e.rsi.write_text(json.dumps(config))
        self.frozen = {p: p.read_bytes() for p in
                       (check, e.rsi, e.bridge, e.mission, e.tmp / "score.py")}
        self.checkout = self.git("status", "--porcelain"), self.git("rev-parse", "HEAD")
        self.baseline = None

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.e.repo, timeout=5)

    def core(self):
        return json.loads((self.e.state / "rsi/state.json").read_text())

    def checkpoint(self):
        core = self.core()
        self.assertEqual((core["phase"], core["reserved_proposals"], core["archive"]),
                         ("initializing", 0, []))
        self.assertNotIn("champion", core)
        self.baseline = tuple(core[k] for k in ("run_id", "base", "contract_hash"))
        self.assertEqual(self.e.events(), [])

    def observe(self, phase, ownership):
        with self.subTest(stage="status", expected_phase=phase):
            p, data = self.e.run("status", timeout=5)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual((data["phase"], data["ownership"]), (phase, ownership), data)
            self.assertEqual((data["cycle_reservations"], data["counts"]["proposals"]), (0, 0))
            self.assertIsNone(data["champion"])

    def drain(self):
        e = self.e
        self.assertTrue(wait_for(lambda: all(unlocked(p) for p in
                        (e.tmp / "seed.lock", e.state / "owner.lock", e.state / "rsi/run.lock")),
                        timeout=5), "previous owner/check/core still holds a lock")

    def recover(self, budget=0):
        self.marker.unlink()
        self.assertEqual({p: p.read_bytes() for p in self.frozen}, self.frozen)
        p, data = self.e.run("resume", timeout=20)
        with self.subTest(stage="resume"):
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        if p.returncode:
            return  # Other test methods still exercise their independent failure paths.
        core = self.core()
        self.assertEqual(tuple(core[k] for k in ("run_id", "base", "contract_hash")), self.baseline)
        self.assertEqual((self.e.reserved(), data["cycle_reservations"]), (budget, budget))
        self.assertLessEqual(budget, 4)
        events = rows(self.e.state / "rsi/events.jsonl")
        tags = [row["event"] for row in events]
        self.assertEqual(tags.count("seed"), 1)
        if "round_start" in tags:
            self.assertLess(tags.index("seed"), tags.index("round_start"))
        self.assertEqual(next(row for row in rows(self.log) if row["event"] == "passed")["calls"], 0)
        self.assertEqual(data["counts"]["cycles"], 1)
        if budget == 0:
            self.assertEqual((data["phase"], core["phase"]), ("saturated", "finalized"))
            self.assertEqual(core["champion"], "seed")
            self.assertEqual(self.e.events(), [])
            self.assertEqual(self.e.evaluations.read_text().splitlines().count("sealed"), 1)
            measured = self.e.evaluations.read_bytes()
            for action in ("run", "resume"):
                p, data = self.e.run(action, timeout=10)
                self.assertEqual((p.returncode, data["phase"]), (0, "saturated"))
                self.assertEqual((self.e.reserved(), self.e.events()), (0, []))
                self.assertEqual(self.e.evaluations.read_bytes(), measured)
        else:
            self.assertEqual((data["phase"], core["phase"]), ("waiting", "ready"))
            self.assertEqual(tags.count("round_start"), 2)
            self.assertEqual(sum(row["role"] == "builder" for row in self.e.events()), 4)
            self.assertNotIn("sealed", self.e.evaluations.read_text().splitlines())
        self.assertEqual({p: p.read_bytes() for p in self.frozen}, self.frozen)
        self.assertEqual((self.git("status", "--porcelain"), self.git("rev-parse", "HEAD")), self.checkout)

    def interrupt(self, sig, budget=0):
        self.build(ceilings=(1., 1.) if budget else (.25, .25))
        p = subprocess.Popen(self.e.cmd("run"), cwd=self.e.repo, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertTrue(wait_for(lambda: bool(rows(self.log)) or p.poll() is not None, timeout=5))
        self.assertIsNone(p.poll(), "mission exited before the seed check could be interrupted")
        self.checkpoint()
        self.observe("active", "live")
        p.send_signal(sig)
        p.wait(timeout=5)
        self.drain()
        self.assertEqual(self.core()["phase"], "initializing")
        self.assertEqual(self.e.reserved(), 0)
        self.recover(budget)

    def test_sigkill_seed_saturation(self):
        self.interrupt(signal.SIGKILL)

    def test_sigterm_seed_saturation(self):
        self.interrupt(signal.SIGTERM)

    def test_sigkill_seed_then_bounded_search(self):
        self.interrupt(signal.SIGKILL, budget=4)

    def test_transient_check_failure_requires_explicit_resume(self):
        self.build(gate="fail")
        p, data = self.e.run("run", timeout=12)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.checkpoint()  # A failed check is not a ready seed, regardless of unchanged counters.
        self.drain()
        self.observe("blocked", "none")
        before = rows(self.log)
        p, data = self.e.run("run", timeout=5)
        with self.subTest(stage="blocked tick"):
            self.assertEqual(p.returncode, 4, p.stdout + p.stderr)
            self.assertEqual(rows(self.log), before)
            self.assertEqual(self.e.events(), [])
        self.recover()


if __name__ == "__main__":
    assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
    unittest.main(verbosity=2)
