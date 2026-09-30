import os, signal, subprocess, sys, tempfile, threading, time
from pathlib import Path
from ops_common import ROOT, PY, alive, wait_for

ME = str(Path(__file__).resolve())
PY = str(PY)
now = time.monotonic
OWNED = ("leaf", "leader", "guard")
TEXT = "日本語\n" * 20000
OUT = "os.write(1,b'x'*32);os.write(2,b'y'*32)"
CASES = {
  "normal": ("sys.stdout.write(sys.stdin.read())", (0, TEXT, "")),
  "nonzero": ("os.write(1,b'o');os.write(2,b'e');sys.exit(7)", (7, "o", "e")),
  "utf8": ("os.write(1,b'\\xff')", None),
  "utf8err": ("os.write(2,b'\\xff')", None),
  "limit": (OUT, (0, "x"*32, "y"*32)),
  "overflow": (OUT, None),
}

def stat(pid):
  return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()

def mark(path, pid):
  tmp = path.with_suffix(".tmp")
  tmp.write_text(f"{pid} {stat(pid)[19]}")
  tmp.replace(path)

def live(path):
  try:
    pid, born = path.read_text().split()
    return alive(int(pid)) and stat(pid)[19] == born
  except (OSError, ValueError): return False

def gone(d):
  return not any(live(d / n) for n in OWNED)

def spawn(d, owner):
  mark(d / "leader", os.getpid())
  pid = os.fork()
  if pid == 0:
    os.setsid()
    ppid = os.getpid()
    if os.fork(): os._exit(0)
    while os.getppid() == ppid: time.sleep(.001)
    guard = node = os.getppid()
    for _ in range(32):
      if node in (1, owner): break
      node = int(stat(node)[1])
    assert guard != owner and node == owner
    mark(d / "guard", guard)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    mark(d / "leaf", os.getpid())
    time.sleep(60)
    os._exit(0)
  os.waitpid(pid, 0)
  time.sleep(60)

def caller(d, m):
  sys.path.insert(0, str(ROOT))
  from gama.rsi_guard import run_guarded
  from gama.rsi_process import ProcessError, ProcessResult
  cmd = lambda code: [PY, "-B", "-c", "import os,sys,time;" + code]
  mark(d / "sentinel", subprocess.Popen(cmd("time.sleep(60)")).pid)
  (d / "cwd").mkdir()
  argv = [PY, "-B", ME, "spawn", str(d), str(os.getpid())]
  if m in CASES:
    argv = cmd("assert os.path.basename(os.getcwd())=='cwd';" + CASES[m][0])
  ev, when = threading.Event(), []
  if m == "cancel":
    def trigger():
      wait_for(lambda: live(d / "leaf"), timeout=3)
      when.append(now())
      ev.set()
    threading.Thread(target=trigger, daemon=True).start()
  t0 = now()
  got = None
  try:
    r = run_guarded(argv, cwd=d / "cwd", artifact_dir=d / "artifacts",
            input_text=TEXT, timeout=1 if m == "timeout" else 12,
            max_output_bytes={"limit": 64, "overflow": 63}.get(m, 1048576),
            cancel=ev)
    assert isinstance(r, ProcessResult) and 0 <= r.elapsed_s <= 17
    got = (r.returncode, r.stdout, r.stderr)
  except ProcessError: pass
  assert now() <= t0 + 17
  assert m != "death" and got == CASES.get(m, ("", None))[1], m
  if m not in CASES:
    end = (t0 + 1 if m == "timeout" else when[0]) + 5
    assert (d / "leaf").exists()
    wait_for(lambda: gone(d), timeout=max(.001, end - now()))
    assert gone(d) and now() <= end

def check(m):
  with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    p = subprocess.Popen([PY, "-B", ME, "call", tmp, m], start_new_session=True)
    try:
      if m == "death":
        wait_for(lambda: live(d / "leaf"), timeout=5)
        assert live(d / "leaf")
        t0 = now()
        p.kill()
        assert p.wait(timeout=1) == -signal.SIGKILL
        wait_for(lambda: gone(d), timeout=5)
        assert gone(d) and now() <= t0 + 5
      else:
        assert p.wait(timeout=20) == 0, m
      assert live(d / "sentinel")
    finally:
      if p.poll() is None: p.kill()
      p.wait(timeout=2)
      for name in (*OWNED, "sentinel"):
        path = d / name
        if live(path):
          try:
            os.kill(int(path.read_text().split()[0]), signal.SIGKILL)
          except ProcessLookupError: pass

if __name__ == "__main__":
  args = sys.argv[1:]
  if not args:
    for m in (*CASES, "timeout", "cancel", "death"):
      check(m)
    print("guard OK")
  elif args[0] == "spawn": spawn(Path(args[1]), int(args[2]))
  else: caller(Path(args[1]), args[2])
