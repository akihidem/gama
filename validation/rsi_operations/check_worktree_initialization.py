"""Real interrupted Git checkout recovery. Run from the integration/product checkout."""
import ctypes, fcntl, json, os, shlex, signal, subprocess, sys, tempfile, time, unittest
from contextlib import contextmanager
from pathlib import Path

ROOT, PY = Path.cwd().resolve(), sys.executable
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/rsi_operations"))
from ops_common import alive, wait_for
from gama.rsi_workspace import Workspaces, WorkspaceError


def owned_children():
    return [int(p) for p in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()]


def cleanup_owned():
    # Only this standalone test's descendants are adopted by its private subreaper.
    end = time.monotonic() + 2
    while time.monotonic() < end:
        for pid in owned_children():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        time.sleep(.01)
    raise AssertionError("fixture descendants did not drain")


def files(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


class WorktreeInitialization(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="gama-worktree-init-")
        self.addCleanup(tmp.cleanup)
        self.addCleanup(cleanup_owned)
        self.tmp = Path(tmp.name)
        self.repo, self.root = self.tmp / "repo", self.tmp / "worktrees"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        (self.repo / ".gitattributes").write_text("payload.txt filter=hold\n")
        (self.repo / "payload.txt").write_text("original source\n")
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture")
        self.ws = Workspaces(self.repo, self.root)
        self.foreign = self.root / "unreceipted"
        self.git("worktree", "add", "--detach", str(self.foreign), "HEAD")
        self.git("worktree", "lock", "--reason", "user keep", str(self.foreign))
        (self.root / "notes").mkdir()
        (self.root / "notes/keep.txt").write_text("unrelated directory\n")
        (self.repo / "payload.txt").write_text("caller edit\n")
        (self.repo / "personal.txt").write_text("caller untracked file\n")
        self.hold, self.ready = self.tmp / "hold", self.tmp / "filter.ready"
        smudge = self.tmp / "smudge.py"
        smudge.write_text(f"""import fcntl,os,sys,time
from pathlib import Path
data=sys.stdin.buffer.read()
with open({str(self.tmp / 'filter.lock')!r},"a+b") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if Path({str(self.hold)!r}).exists():
  Path({str(self.ready)!r}).write_text(str(os.getpid()))
  while Path({str(self.hold)!r}).exists(): time.sleep(.01)
sys.stdout.buffer.write(data)
""")
        self.git("config", "filter.hold.smudge", shlex.join([PY, "-B", str(smudge)]))
        self.git("config", "filter.hold.clean", "cat")
        self.git("config", "filter.hold.required", "true")
        self.caller_before = self.caller()
        self.foreign_before = files(self.foreign)

    def git(self, *args, cwd=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
        return subprocess.check_output(
            ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgSign=false", "-c", f"core.hooksPath={os.devnull}", *args],
            cwd=cwd or self.repo, env=env, timeout=5, stderr=subprocess.PIPE)

    def caller(self):
        return (self.git("symbolic-ref", "HEAD"), self.git("rev-parse", "HEAD"),
                self.git("status", "--porcelain"), (self.repo / "payload.txt").read_bytes(),
                (self.repo / "personal.txt").read_bytes())

    def records(self):
        records, row = {}, {}
        for field in self.git("worktree", "list", "--porcelain", "-z").split(b"\0"):
            if not field:
                if row:
                    records[Path(row["worktree"])] = row
                row = {}
            else:
                key, _, value = os.fsdecode(field).partition(" ")
                row[key] = value
        return records

    @contextmanager
    def ownership(self):
        with (self.tmp / "run.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def admin(self, path):
        return Path(self.git("rev-parse", "--absolute-git-dir", cwd=path).decode().strip())

    def unchanged(self):
        self.assertEqual(self.caller(), self.caller_before)
        self.assertEqual(files(self.foreign), self.foreign_before)
        self.assertEqual(self.records()[self.foreign]["locked"], "user keep")
        self.assertEqual((self.root / "notes/keep.txt").read_text(), "unrelated directory\n")

    def interrupted(self):
        self.hold.touch()
        path = self.root / "partial"
        prefix = f"import sys;sys.path.insert(0,{str(ROOT)!r});from pathlib import Path\n"
        create = prefix + f"""import fcntl
from gama.rsi_workspace import Workspaces
with open({str(self.tmp / 'run.lock')!r},"a+b") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 Workspaces({str(self.repo)!r},{str(self.root)!r}).create("partial","HEAD")
 Path({str(self.tmp / 'returned')!r}).touch()
"""
        owner = prefix + f"""from gama.rsi_guard import run_guarded
r=run_guarded({[PY, '-B', '-c', create]!r},cwd=Path({str(self.repo)!r}),
 timeout=15,artifact_dir=Path({str(self.tmp / 'guard')!r}))
raise SystemExit(r.returncode)
"""
        p = subprocess.Popen([PY, "-B", "-c", owner], cwd=ROOT, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertTrue(wait_for(lambda: self.ready.exists() or p.poll() is not None, timeout=5))
        self.assertIsNone(p.poll(), "guarded creator exited before checkout paused")
        self.assertTrue(self.ready.exists())
        self.assertFalse((self.tmp / "returned").exists(), "create() already completed")
        before = self.records()[path]
        self.assertIn("detached", before)
        p.kill()
        self.assertEqual(p.wait(timeout=2), -signal.SIGKILL)
        self.assertTrue(wait_for(lambda: not any(alive(pid) for pid in owned_children()), timeout=5))
        with self.ownership(), (self.tmp / "filter.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        after = self.records()[path]
        print(json.dumps({"case": self.id().split(".")[-1], "paused_lock": before.get("locked"),
                          "after_drain_lock": after.get("locked"), "create_returned": False}), flush=True)
        self.hold.unlink()
        return path

    def test_interrupted_checkout_recovers(self):
        path = self.interrupted()
        with self.ownership():
            try:
                recovered = Workspaces(self.repo, self.root).recover()
            except WorkspaceError as exc:
                self.fail(f"all creator descendants drained, but recovery refused: {exc}")
            self.assertIn(path, recovered)
            self.assertFalse(path.exists())
            self.assertNotIn(path, self.records())
            self.assertEqual(Workspaces(self.repo, self.root).recover(), [])
        self.unchanged()

    def completed_lock(self, reason):
        path = self.ws.create("completed", "HEAD")
        self.assertEqual((path / "payload.txt").read_text(), "original source\n")
        self.assertEqual(self.git("status", "--porcelain", cwd=path), b"")
        self.git("worktree", "lock", "--reason", reason, str(path))
        before = files(self.root), files(self.admin(path)), self.records()
        with self.ownership():
            for operation in ("recover", "remove"):
                with self.subTest(operation=operation, reason=reason):
                    ws = Workspaces(self.repo, self.root)
                    with self.assertRaises(WorkspaceError):
                        ws.recover() if operation == "recover" else ws.remove(path)
                    self.assertEqual((files(self.root), files(self.admin(path)), self.records()), before)
        self.unchanged()

    def test_completed_user_lock(self):
        self.completed_lock("maintenance by user")

    def test_completed_user_lock_named_initializing(self):
        self.completed_lock("initializing")

    def test_invalid_receipt_cannot_unlock_partial_checkout(self):
        path = self.interrupted()
        receipts = [p for p in self.root.iterdir() if p.is_file() and
                    json.loads(p.read_text()).get("name") == path.name]
        self.assertTrue(receipts, "native creation did not leave an ownership receipt")
        receipts[0].write_bytes(b'{"broken":')
        before = files(self.root), files(self.admin(path)), self.records()
        with self.ownership(), self.assertRaises(WorkspaceError):
            Workspaces(self.repo, self.root).recover()
        self.assertEqual((files(self.root), files(self.admin(path)), self.records()), before)
        self.unchanged()


if __name__ == "__main__":
    assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
    unittest.main(verbosity=2)
