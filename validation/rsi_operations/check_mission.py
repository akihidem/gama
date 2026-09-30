"""Frozen offline mission acceptance."""
import argparse, fcntl, json, os, signal, subprocess, tempfile
from contextlib import contextmanager
from pathlib import Path

from ops_common import Env, alive, wait_for

COMPONENT = False


@contextmanager
def fixture(**kw):
  with tempfile.TemporaryDirectory() as tmp:
    e = Env(Path(tmp), stub_bridge=COMPONENT, **kw)
    try:
      yield e
    finally:
      for pid in {r["pid"] for r in e.events()}:
        try:
          if alive(pid) and Path(f"/proc/{pid}/cwd").resolve().is_relative_to(e.tmp):
            os.kill(pid, signal.SIGKILL)
        except OSError:
          pass

def call(e, action, code=0):
  p, data = e.run(action)
  assert p.returncode == code and isinstance(data, dict), (
    action, p.returncode, p.stdout[-1000:], p.stderr[-1000:])
  return data

def idle(e, action, code=0):
  before = e.reserved(), e.events()
  s = call(e, action, code)
  assert (e.reserved(), e.events()) == before
  return s

def git(e, *args):
  return subprocess.check_output(["git", *args], cwd=e.repo)

def snap(e):
  return [git(e, *a.split()) for a in ("symbolic-ref HEAD", "rev-parse HEAD",
          "status --porcelain --untracked-files=all")] + [
          (e.repo / n).read_bytes() for n in ("value.py", "caller.txt")]

def drained(e):
  if any(alive(r["pid"]) for r in e.events()):
    return False
  with (e.state / "rsi/run.lock").open("rb") as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return False
    fcntl.flock(lock, fcntl.LOCK_UN)
  return True

@contextmanager
def running(e):
  p = subprocess.Popen(e.cmd("run"), cwd=e.repo, start_new_session=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  try:
    assert wait_for(lambda: e.reserved() >= 2 or p.poll() is not None, timeout=60)
    assert e.reserved() == 2, p.poll()
    yield p
  finally:
    p.kill()
    p.wait(timeout=10)

def progress():
  with fixture() as e:
    (e.repo / "value.py").write_text("VALUE = 1\n# caller edit\n")
    (e.repo / "caller.txt").write_bytes(b"caller data\n")
    before = snap(e)
    s = call(e, "run")
    assert (s["phase"], s["last_outcome"]) == ("waiting", "promoted"), s
    assert e.reserved() == s["cycle_reservations"] == s["counts"]["proposals"] == 4
    assert s["reserved_baseline"] == 0 and not s["cycle_open"]
    assert int(git(e, "show", s["ref"] + ":value.py").split(b"=")[1]) > 1
    assert b"value.py" in Path(s["patch"]).read_bytes()
    roles = [r["role"] for r in e.events()]
    assert roles.count("builder") == roles.count("reviewer") == 4, roles
    for _ in range(2):
      assert not idle(e, "resume")["cycle_open"]
    s = call(e, "run")
    assert s["reserved_baseline"] == 4 < e.reserved() <= 8
    assert s["cycle_reservations"] == e.reserved()-4 <= 4
    assert len(e.events()) > len(roles)
    assert snap(e) == before

def saturated():
  with fixture(ceilings=(.25, .25)) as e:
    assert call(e, "run")["phase"] == "saturated"
    assert json.loads((e.state / "rsi/state.json").read_text())["phase"] == "finalized"
    measured = e.evaluations.read_text()
    assert measured.splitlines().count("sealed") == 1
    assert e.reserved() == 0 and not e.events()
    for action in ("run", "resume", "run"):
      assert idle(e, action)["phase"] == "saturated"
      assert e.evaluations.read_text() == measured

def stop_and_overlap():
  with fixture(mode="hang") as e:
    with running(e) as p:
      s = call(e, "status")
      assert (s["phase"], s["ownership"]) == ("active", "live"), s
      call(e, "run", 3)
      assert e.reserved() == 2
      call(e, "stop")
      p.wait(timeout=10)
    assert wait_for(lambda: drained(e), timeout=8)
    for _ in range(2):
      assert idle(e, "run", 4)["phase"] == "stopped"
    e.mode_file.write_text("increment")
    call(e, "resume")
    assert e.reserved() <= 4

def blocked():
  for mode in ("error", "reject"):
    with fixture(mode=mode) as e:
      p, s = e.run("run")
      assert p.returncode in (0, 2) and isinstance(s, dict), p.stderr[-1000:]
      assert s["phase"] == "blocked", s
      assert 0 < e.reserved() <= 4
      core = json.loads((e.state / "rsi/state.json").read_text())
      assert core["champion"] == "seed" and len(core["archive"]) == 1
      assert e.events()
      for _ in range(2):
        assert idle(e, "run", 4)["phase"] == "blocked"

def interrupted():
  for sig in (signal.SIGKILL, signal.SIGTERM):
    with fixture(mode="hang") as e:
      with running(e) as p:
        p.send_signal(sig)
        p.wait(timeout=10)
      assert wait_for(lambda: drained(e), timeout=8), sig
      s = call(e, "status")
      assert s["ownership"] in ("stale", "none"), s
      assert s["cycle_open"] and e.reserved() == 2, s
      e.mode_file.write_text("increment")
      s = call(e, "resume")
      assert e.reserved() == s["cycle_reservations"] == 4, (sig, s)
      assert s["reserved_baseline"] == 0 and not s["cycle_open"]
      idle(e, "resume")

def frozen():
  with fixture(ceilings=(.25, .25)) as e:
    assert call(e, "run")["phase"] == "saturated"
    measured = e.evaluations.read_bytes()
    for path, key, value in ((e.rsi, "goal", "changed"),
                            (e.bridge, "timeout", 19),
                            (e.mission, "search_ceiling", .5),
                            (e.mission, "confirm_ceiling", .5)):
      old = path.read_bytes()
      data = json.loads(old)
      data[key] = value
      try:
        path.write_text(json.dumps(data))
        for action in ("run", "resume"):
          idle(e, action, 5)
          assert e.evaluations.read_bytes() == measured
      finally:
        path.write_bytes(old)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--component", action="store_true")
  COMPONENT = parser.parse_args().component
  for check in (progress, saturated, stop_and_overlap, blocked, interrupted, frozen):
    check()
    print(check.__name__ + " passed")
