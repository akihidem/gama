"""Offline fixtures for immutable Astra Loop acceptance; no real inference."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path.cwd().resolve()
PY = sys.executable
BASE = "d4497d36d17182ad4c8e832c3ae6814b3aaea486"
sys.path.insert(0, str(ROOT))


def alive(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def wait_for(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.02)
    return bool(predicate())


FAKE = r'''
import difflib,json,os,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class LiveBackend:
 def __init__(self,config): self.config=config
 def identity(self,role):
  builder=role=="builder"
  family="openai" if builder else "anthropic"
  if (ROOT/"mode").read_text().strip()=="bad_identity": family="openai"
  return {"provider":"codex" if builder else "bedrock","model":"astra" if builder else "opus","family":family,"resolved_model":"bedrock-astra" if builder else "global.anthropic.claude-opus-5","simulated":True}
 def complete(self,role,prompt,*,cwd,output_dir,cancel=None):
  out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
  (out/"prompt.txt").write_text(prompt)
  (out/"meta.json").write_text(json.dumps({"usage":{"fixture_tokens":1},"simulated":True}))
  fd=os.open(ROOT/"calls.jsonl",os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
  os.write(fd,(json.dumps({"role":role,"pid":os.getpid(),"output_dir":str(out)})+"\n").encode());os.close(fd)
  mode=(ROOT/"mode").read_text().strip()
  if role=="builder":
   if mode=="error": raise RuntimeError("fixture provider failure")
   if mode=="noise": print("fixture diagnostic",flush=True)
   if mode=="detached":
    code="import os,time,sys; from pathlib import Path; p=os.fork();\nif p: os._exit(0)\nos.setsid();p=os.fork();\nif p: os._exit(0)\nPath(sys.argv[1]).write_text(str(os.getpid()));time.sleep(90)\n"
    subprocess.Popen([sys.executable,"-B","-c",code,str(ROOT/"leaf.pid")],start_new_session=True)
   if mode in ("hang","detached"):
    (ROOT/"worker.pid").write_text(str(os.getpid()))
    time.sleep(90)
   if mode=="invalid_patch": return "not a unified diff"
   req=json.loads(prompt.split("\nREQUEST_JSON\n",1)[1])
   name,old=next(iter(req["files"].items()))
   value=int(old.split("=")[1].strip())
   new="VALUE = %d\n"%(4 if mode=="perfect" else value+1)
   return "".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile="a/"+name,tofile="b/"+name))
  if mode=="invalid_review": return "not JSON"
  if mode=="extra_review": return '{"verdict":"PASS","reason":"ok","unexpected":true}'
  if mode=="invalid_constant": return '{"verdict":"PASS","reason":NaN}'
  return json.dumps({"verdict":"FAIL" if mode=="reject" else "PASS","reason":"offline fixture"})
'''

STUB_BRIDGE = r'''
import argparse,json,sys,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());sys.path.insert(0,cfg["astra_loop_root"])
from astra_loop.backends import LiveBackend
b=LiveBackend(cfg["backend"]);req=json.load(sys.stdin)
out=Path(cfg["artifact_root"])/("proposal-"+uuid.uuid4().hex);out.mkdir(parents=True)
diff=b.complete("builder","Fixture\nREQUEST_JSON\n"+json.dumps(req),cwd=Path.cwd(),output_dir=out/"builder")
v=json.loads(b.complete("reviewer","Fixture\nREVIEW_JSON\n"+json.dumps({"request":req,"patch":diff}),cwd=Path.cwd(),output_dir=out/"reviewer"))
if v["verdict"]!="PASS": raise SystemExit(1)
sys.stdout.write(diff)
'''


class Env:
    def __init__(self, tmp, *, stub_bridge=False, mode="increment",
                 ceilings=(1., 1.), rounds=2):
        self.tmp = Path(tmp)
        self.repo, self.state = self.tmp / "repo", self.tmp / "state"
        shutil.copytree(ROOT, self.repo, ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", "*.pyc"))
        if stub_bridge:
            (self.repo / "gama/rsi_bridge.py").write_text(STUB_BRIDGE)
        (self.repo / "value.py").write_text("VALUE = 1\n")
        git_env = dict(os.environ, GIT_AUTHOR_NAME="fixture",
                       GIT_AUTHOR_EMAIL="fixture@example.invalid",
                       GIT_COMMITTER_NAME="fixture",
                       GIT_COMMITTER_EMAIL="fixture@example.invalid")
        for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                     ["commit", "-qm", "fixture"]):
            subprocess.run(["git", *args], cwd=self.repo, env=git_env,
                           check=True, capture_output=True)
        adapter = self.tmp / "adapter"
        (adapter / "astra_loop").mkdir(parents=True)
        (adapter / "astra_loop/__init__.py").write_text("")
        (adapter / "astra_loop/backends.py").write_text(FAKE)
        self.calls, self.mode_file = adapter / "calls.jsonl", adapter / "mode"
        self.leaf_pid = adapter / "leaf.pid"
        self.worker_pid = adapter / "worker.pid"
        self.mode_file.write_text(mode)
        self.artifacts = self.tmp / "artifacts"
        self.evaluations = self.tmp / "evaluations.jsonl"
        scorer = self.tmp / "score.py"
        scorer.write_text(
            "import ast,json,os,sys\nfrom pathlib import Path\n"
            "value=ast.literal_eval(ast.parse(Path('value.py').read_text()).body[0].value)\n"
            f"fd=os.open({str(self.evaluations)!r},os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\n"
            "os.write(fd,(sys.argv[1]+'\\n').encode());os.close(fd)\n"
            "print(json.dumps({'score':min(1.,max(0.,value/4))}))\n")
        self.rsi, self.bridge, self.mission = (
            self.tmp / name for name in ("rsi.json", "bridge.json", "mission.json"))
        rsi = {
            "goal": "Increase VALUE generally; fixed offline acceptance.",
            "allowed_paths": ["value.py"],
            "agents": [{"name": "will-be-bound", "command": [PY, "-c", "raise SystemExit(99)"]}],
            "checks": [[PY, "-B", "-c", "import ast;ast.parse(open('value.py').read())"]],
            "search_command": [PY, "-B", str(scorer), "search"],
            "confirm_command": [PY, "-B", str(scorer), "confirm"],
            "sealed_command": [PY, "-B", str(scorer), "sealed"],
            "evaluation_files": [str(scorer)], "workers": 2, "batch_size": 2,
            "timeout": 30, "evaluation_timeout": 30,
            "search_repeats": 1, "confirm_repeats": 1, "min_gain": 0., "seed": 0}
        self.rsi.write_text(json.dumps(rsi))
        self.bridge.write_text(json.dumps({
            "astra_loop_root": str(adapter), "artifact_root": str(self.artifacts),
            "backend": {"timeout_seconds": 10, "bedrock_max_tokens": 512,
                        "max_context_bytes": 110000},
            "timeout": 20}))
        self.mission.write_text(json.dumps({
            "mission_id": "offline-fixture", "repo": str(self.repo),
            "state_dir": str(self.state), "rsi_config": str(self.rsi),
            "bridge_config": str(self.bridge), "search_ceiling": ceilings[0],
            "confirm_ceiling": ceilings[1], "rounds_per_cycle": rounds}))

    def cmd(self, action):
        return [PY, "-B", "-m", "gama.rsi_mission", action,
                "--mission", str(self.mission)]

    def run(self, action, timeout=90):
        return self._run(self.cmd(action), timeout=timeout)

    def _run(self, argv, *, timeout, input_text=None):
        p = subprocess.run(argv, cwd=self.repo, input=input_text,
                           capture_output=True, text=True, timeout=timeout)
        try:
            data = json.loads(p.stdout)
        except ValueError:
            data = None
        return p, data

    def reserved(self):
        path = self.state / "rsi/state.json"
        return json.loads(path.read_text())["reserved_proposals"] if path.exists() else 0

    def events(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines() if line]

    def request(self):
        return {"goal": "Increase VALUE; keep the fixed evaluator unchanged.",
                "parent": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=self.repo, text=True).strip(),
                "allowed_paths": ["value.py"],
                "files": {"value.py": (self.repo / "value.py").read_text()},
                "feedback": {}, "papers": []}

    def bridge_cmd(self):
        return [PY, "-I", "-B", str(self.repo / "gama/rsi_bridge.py"),
                "--config", str(self.bridge)]

    def bridge_run(self, request=None, timeout=30):
        return self._run(self.bridge_cmd(), timeout=timeout,
                         input_text=json.dumps(self.request() if request is None else request))
