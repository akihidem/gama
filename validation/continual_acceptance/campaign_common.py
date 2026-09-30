"""Offline campaign fixtures. All Git remotes, adapters and processes are local."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

ROOT = Path.cwd().resolve()
PY = sys.executable
BRANCH = "feature/offline-campaign"
OLD = {
    "gama/fixture_total.py": "def total(values):\n    return sum(values) + 1\n",
    "gama/fixture_text.py": "def canonical(text):\n    return text.strip()\n",
}
NEW = {
    "gama/fixture_total.py": "def total(values):\n    return sum(values)\n",
    "gama/fixture_text.py": "def canonical(text):\n    return text.strip().casefold()\n",
}


def goal(ident, path, module, function, examples):
    tests = {}
    for role, cases in zip(("search", "confirm", "sealed"), examples):
        source = f"import unittest\nfrom gama.{module} import {function}\n\n"
        source += "class Behavior(unittest.TestCase):\n"
        for i, (argument, expected) in enumerate(cases):
            source += (f"    def test_case_{i}(self):\n"
                       f"        self.assertEqual({function}({argument!r}), {expected!r})\n")
        source += "\nif __name__ == '__main__':\n    unittest.main()\n"
        tests[role] = source
    return {"id": ident, "title": ident, "goal": f"Make {function} satisfy its fixed examples.",
            "allowed_paths": [path], "tests": tests}


GOALS = [
    goal("total-zero", "gama/fixture_total.py", "fixture_total", "total", (
        [([], 0), ([2, -2], 0)], [([2, 3], 5), ([-7], -7)],
        [([100, -10, 3], 93), ([0, 0], 0)])),
    goal("canonical-case", "gama/fixture_text.py", "fixture_text", "canonical", (
        [(" A ", "a"), ("B", "b")], [(" MIXED ", "mixed"), ("", "")],
        [("\tStraße\n", "strasse"), ("  AbC  ", "abc")])),
]

FAKE = r'''
import difflib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class LiveBackend:
 def __init__(self,config): self.config=config
 def identity(self,role):
  b=role=="builder"
  return {"provider":"codex" if b else "bedrock","model":"astra" if b else "opus",
   "family":"openai" if b else "anthropic",
   "resolved_model":"bedrock-astra" if b else "global.anthropic.claude-opus-5",
   "simulated":True}
 def complete(self,role,prompt,*,cwd,output_dir,cancel=None):
  out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
  (out/"prompt.txt").write_text(prompt)
  (out/"meta.json").write_text(json.dumps({"usage":{"fixture_tokens":1},"simulated":True}))
  stat=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
  row={"role":role,"pid":os.getpid(),"start":stat[19],
       "discovery":"DISCOVERY_" in prompt,"output_dir":str(out)}
  fd=os.open(ROOT/"calls.jsonl",os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
  os.write(fd,(json.dumps(row)+"\n").encode());os.close(fd)
  mode=(ROOT/"mode").read_text().strip()
  if role=="builder":
   if mode=="hang":
    time.sleep(40)
   if mode=="error": raise RuntimeError("offline provider failure")
   if "\nDISCOVERY_JSON\n" in prompt: return "{}"
   req=json.loads(prompt.split("\nREQUEST_JSON\n",1)[1])
   fixes=json.loads((ROOT/"fixes.json").read_text())
   result=[]
   for name,old in req["files"].items():
    if name in fixes:
     result.extend(difflib.unified_diff(old.splitlines(True),fixes[name].splitlines(True),
                   fromfile="a/"+name,tofile="b/"+name))
   return "".join(result)
  return json.dumps({"verdict":"PASS","reason":"independent offline fixture"})
'''


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def wait_for(predicate, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.03)
    return bool(predicate())


class Env:
    def __init__(self, tmp, *, empty=False):
        self.tmp = Path(tmp)
        self.repo, self.state = self.tmp / "repo", self.tmp / "state"
        self.repo.mkdir()
        shutil.copytree(ROOT / "gama", self.repo / "gama",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path, source in OLD.items():
            (self.repo / path).write_text(source)
        # A pre-existing mandatory check, independent of the new goal scores.
        (self.repo / "mandatory.py").write_text(
            "import ast\nfrom pathlib import Path\n"
            "for p in Path('gama').glob('fixture_*.py'): ast.parse(p.read_text())\n")
        self.git_env = dict(os.environ, GIT_AUTHOR_NAME="offline fixture",
                            GIT_AUTHOR_EMAIL="fixture@example.invalid",
                            GIT_COMMITTER_NAME="offline fixture",
                            GIT_COMMITTER_EMAIL="fixture@example.invalid",
                            GIT_TERMINAL_PROMPT="0", PYTHONDONTWRITEBYTECODE="1")
        self.git("init", "-q", "-b", BRANCH)
        self.git("add", "-A")
        self.git("commit", "-qm", "offline baseline")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.remote = self.tmp / "remote.git"
        self.git("init", "-q", "--bare", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "-u", "origin", BRANCH)
        adapter = self.tmp / "adapter"
        (adapter / "astra_loop").mkdir(parents=True)
        (adapter / "astra_loop/__init__.py").write_text("")
        (adapter / "astra_loop/backends.py").write_text(FAKE)
        (adapter / "fixes.json").write_text(json.dumps(NEW))
        self.mode = adapter / "mode"
        self.mode.write_text("fix")
        self.calls = adapter / "calls.jsonl"
        self.bridge = self.tmp / "bridge.json"
        self.bridge.write_text(json.dumps({
            "astra_loop_root": str(adapter), "artifact_root": str(self.tmp / "artifacts"),
            "backend": {"timeout_seconds": 12, "bedrock_max_tokens": 512,
                        "max_context_bytes": 110000}, "timeout": 18}))
        descriptors = []
        for item in GOALS:
            path = self.tmp / (item["id"] + ".json")
            path.write_text(json.dumps(item))
            descriptors.append(str(path))
        self.config = self.tmp / "campaign.json"
        self.config.write_text(json.dumps({
            "campaign_id": "offline-campaign", "repo": str(self.repo),
            "state_dir": str(self.state), "branch": BRANCH, "remote": "origin",
            "bridge_config": str(self.bridge), "goals": [] if empty else descriptors,
            "checks": [[PY, "-B", "mandatory.py"]], "evaluation_timeout": 15,
            "max_goal_cycles": 2, "workers": 2, "max_proposals_per_tick": 4,
            "schedule_hours": [9, 21], "timezone": "Asia/Tokyo"}))
        self.children = []

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo,
                                       env=self.git_env, text=True, stderr=subprocess.PIPE,
                                       timeout=15)

    def remote_head(self):
        return self.git("ls-remote", "origin", "refs/heads/" + BRANCH).split()[0]

    def events(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines() if line]

    def builders(self):
        return [r for r in self.events() if r["role"] == "builder"]

    def drained(self):
        return all(process_identity(r["pid"]) != r["start"] for r in self.events())

    def start(self, action, *, now=None):
        args = [action, "--config", str(self.config)]
        if now is None:
            command = [PY, "-B", "-m", "gama.continual", *args]
        else:
            # Private unit-test seam only; no production CLI/env clock override.
            harness = (
                "import sys\nfrom datetime import datetime\n"
                "from unittest.mock import patch\nfrom gama import continual\n"
                "instant=datetime.fromisoformat(sys.argv[1])\n"
                "with patch.object(continual, '_now', return_value=instant):\n"
                "    raise SystemExit(continual.main(sys.argv[2:]))\n"
            )
            command = [PY, "-B", "-c", harness, now, *args]
        p = subprocess.Popen(command, cwd=self.repo,
                             env=self.git_env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        self.children.append(p)
        return p

    def run(self, action, timeout=90, *, now=None):
        p = self.start(action, now=now)
        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate(timeout=8)
            raise AssertionError(f"{action} exceeded bounded deadline")
        try:
            value = json.loads(stdout)
        except ValueError:
            value = None
        return p.returncode, value, stdout[-2500:], stderr[-2500:]

    def cleanup(self):
        for p in self.children:
            if p.poll() is None:
                p.kill()
            p.communicate(timeout=8)
        wait_for(self.drained, timeout=3)
        for row in self.events():
            if process_identity(row["pid"]) == row["start"]:
                try:
                    if Path(f"/proc/{row['pid']}/cwd").resolve().is_relative_to(self.tmp):
                        os.kill(row["pid"], signal.SIGKILL)
                except OSError:
                    pass


@contextmanager
def fixture(**kwargs):
    with tempfile.TemporaryDirectory(prefix="gama-campaign-acceptance-") as tmp:
        env = Env(tmp, **kwargs)
        try:
            yield env
        finally:
            env.cleanup()
