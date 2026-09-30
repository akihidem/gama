"""External lifecycle gates: real Git, original API calls, no product source edits."""
import ctypes, json, os, signal, subprocess, sys, time, unittest, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_worktree_initialization as fixture


class CreationLifecycle(fixture.WorktreeInitialization):
    # Inherited tests retain the original smudge, manual-lock and bad-receipt checks.
    def pause_creation(self, point):
        hold, ready = self.tmp / "call.hold", self.tmp / "call.ready"
        returned, path = self.tmp / "create.returned", self.root / "partial"
        hold.touch()
        prefix = f"import sys;sys.path.insert(0,{str(fixture.ROOT)!r});from pathlib import Path\n"
        creator = prefix + f"""import fcntl,time
from gama.rsi_workspace import Workspaces
original=Workspaces._git
def pause():
 with open({str(self.tmp / 'pause.lock')!r},"a+b") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  Path({str(ready)!r}).touch()
  while Path({str(hold)!r}).exists(): time.sleep(.01)
def observed(cwd,*args,**kwargs):
 add=args[:2]==("worktree","add")
 if add and {point!r}=="before-add": pause()
 result=original(cwd,*args,**kwargs)
 if add and {point!r}=="after-add": pause()
 return result
Workspaces._git=staticmethod(observed)
with open({str(self.tmp / 'run.lock')!r},"a+b") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 Workspaces({str(self.repo)!r},{str(self.root)!r}).create("partial","HEAD")
 Path({str(returned)!r}).touch()
"""
        owner = prefix + f"""from gama.rsi_guard import run_guarded
r=run_guarded({[fixture.PY, '-B', '-c', creator]!r},cwd=Path({str(self.repo)!r}),
 timeout=15,artifact_dir=Path({str(self.tmp / 'guard')!r}))
raise SystemExit(r.returncode)
"""
        p = subprocess.Popen([fixture.PY, "-B", "-c", owner], cwd=fixture.ROOT,
                             start_new_session=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        self.assertTrue(fixture.wait_for(lambda: ready.exists() or p.poll() is not None, timeout=5))
        self.assertIsNone(p.poll(), "creator ended before its lifecycle gate")
        self.assertTrue(ready.exists())
        self.assertFalse(returned.exists(), "create() must still be in progress")
        record = self.records().get(path)
        if point == "after-add":
            self.assertIsNotNone(record)
            self.assertTrue((self.admin(path) / "index").is_file(), "checkout did not finish")
            self.assertEqual((path / "payload.txt").read_text(), "original source\n")
        else:
            self.assertIsNone(record)
        print(json.dumps({"gate": point, "registered": record is not None,
                          "lock": (record or {}).get("locked"), "create_returned": False}), flush=True)
        p.kill()
        self.assertEqual(p.wait(timeout=2), -signal.SIGKILL)
        self.assertTrue(fixture.wait_for(
            lambda: not any(fixture.alive(pid) for pid in fixture.owned_children()), timeout=5))
        with self.ownership(), (self.tmp / "pause.lock").open("rb") as lock:
            fixture.fcntl.flock(lock, fixture.fcntl.LOCK_EX | fixture.fcntl.LOCK_NB)
        hold.unlink()
        with self.ownership():
            self.assertIn(path, fixture.Workspaces(self.repo, self.root).recover())
            self.assertFalse(path.exists())
            self.assertNotIn(path, self.records())
            self.assertEqual(fixture.Workspaces(self.repo, self.root).recover(), [])
        self.unchanged()

    def test_before_git_add_is_recoverable(self):
        self.pause_creation("before-add")

    def test_after_git_add_before_create_returns_is_recoverable(self):
        self.pause_creation("after-add")

    def test_real_git_nonce_lock_spans_checkout(self):
        """Verify Git's primitive directly; do not fabricate a product receipt."""
        path = self.root / "native-token"
        token = "gama-rsi-creating:" + uuid.uuid4().hex
        self.hold.touch()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.git, "worktree", "add", "--detach", "--lock",
                                 "--reason", token, str(path), "HEAD")
            try:
                self.assertTrue(fixture.wait_for(lambda: self.ready.exists(), timeout=3))
                admin = self.admin(path)
                during = {"lock": self.records()[path].get("locked"),
                          "index": (admin / "index").exists(),
                          "index_lock": (admin / "index.lock").is_file()}
                self.assertEqual(during, {"lock": token, "index": False, "index_lock": True})
                self.assertEqual((admin / "locked").read_bytes(), (token + "\n").encode())
            finally:
                self.hold.unlink(missing_ok=True)
            future.result(timeout=5)
        after = {"lock": self.records()[path].get("locked"),
                 "index": (admin / "index").is_file(), "index_lock": (admin / "index.lock").exists()}
        self.assertEqual(after, {"lock": token, "index": True, "index_lock": False})
        self.assertEqual((admin / "locked").read_bytes(), (token + "\n").encode())
        print(json.dumps({"native_nonce": {"during_smudge": during, "after_add": after}}), flush=True)
        self.git("worktree", "unlock", str(path))
        self.assertNotIn("locked", self.records()[path])
        self.git("worktree", "remove", "--force", str(path))
        self.unchanged()


if __name__ == "__main__":
    assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
    unittest.main(verbosity=2)
