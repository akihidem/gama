"""Explicit offline LiveBackend replacement; installed only in temporary fixtures."""
import json
from pathlib import Path
from campaign_common import GOALS

FAKE = r'''
import copy,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.write(1,b"fake import diagnostic\n")
class LiveBackend:
 def __init__(self,config): self.config=config
 def identity(self,role):
  b=role=="builder"
  return {"provider":"codex" if b else "bedrock","model":"astra" if b else "opus",
   "family":"openai" if b else ("openai" if (ROOT/"mode").read_text()=="identity" else "anthropic"),
   "resolved_model":"bedrock-astra" if b else "global.anthropic.claude-opus-5","simulated":True}
 def complete(self,role,prompt,*,cwd,output_dir,cancel=None):
  out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
  (out/"prompt.txt").open("x").write(prompt)
  (out/"meta.json").open("x").write(json.dumps({"usage":{"fixture_tokens":1},"simulated":True}))
  stat=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
  row={"role":role,"pid":os.getpid(),"start":stat[19],"output_dir":str(out),"cwd":str(cwd)}
  fd=os.open(ROOT/"calls.jsonl",os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
  os.write(fd,(json.dumps(row)+"\n").encode());os.close(fd)
  os.write(1,b"fake provider diagnostic\n")
  mode=(ROOT/"mode").read_text().strip()
  if role=="reviewer":
   assert "\nDISCOVERY_REVIEW_JSON\n" in prompt
   json.loads(prompt.split("\nDISCOVERY_REVIEW_JSON\n",1)[1])
   if mode=="review_json": return "PASS"
   if mode=="review_extra": return '{"verdict":"PASS","reason":"ok","extra":1}'
   return json.dumps({"verdict":"FAIL" if mode=="reject" else "PASS","reason":"offline review"})
  req=json.loads(prompt.split("\nDISCOVERY_JSON\n",1)[1])
  assert {"files","seen","schema"} <= set(req)
  assert sum(len(v.encode()) for v in req["files"].values())<=60000
  # A serialized implementation fails this rendezvous before producing a goal.
  end=time.monotonic()+3
  while len([json.loads(x) for x in (ROOT/"calls.jsonl").read_text().splitlines()
             if json.loads(x)["role"]=="builder"])<2:
   if time.monotonic()>end: raise RuntimeError("scouts were not parallel")
   time.sleep(.02)
  if mode=="hang":
   if os.fork()==0:
    os.setsid()
    if os.fork(): os._exit(0)
    s=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
    (ROOT/("leaf-"+str(os.getpid())+".json")).write_text(json.dumps({"pid":os.getpid(),"start":s[19]}))
    time.sleep(60);os._exit(0)
   time.sleep(60)
  if mode=="invalid": return "not JSON"
  if mode=="oversize": return "x"*1048577
  goals=json.loads((ROOT/"discovery-goals.json").read_text())
  matches=[g for g in goals if set(g["allowed_paths"])<=set(req["files"])]
  if not matches: raise RuntimeError("fixture source omitted from scout context")
  g=copy.deepcopy(matches[0])
  if mode=="semantic_duplicate": g.update(id="renamed-"+g["id"],title="Different label")
  if mode=="extra": g["unexpected"]=True
  if mode=="path": g["allowed_paths"]=["gama/rsi.py"]
  if mode=="no_tests": g["tests"]["search"]="import unittest\n"
  if mode=="broken": g["tests"]["search"]="class Broken(\n"
  if mode=="test_error": g["tests"]["search"]=g["tests"]["search"].replace("self.assertEqual(","raise RuntimeError(")
  if mode=="satisfied":
   g["tests"]["search"]="import unittest\nclass Baseline(unittest.TestCase):\n def test_ok(self):\n  self.assertEqual(1,1)\n"
  text=json.dumps(g)
  if mode=="duplicate_keys": text='{"id":"duplicate",'+text[1:]
  if mode=="nan": text=text[:-1]+',"extra":NaN}'
  return text
'''


def install(e):
    adapter = e.mode.parent
    (adapter / "astra_loop/backends.py").write_text(FAKE)
    goals = json.loads(json.dumps(GOALS))
    for goal in goals:
        marker = e.tmp / "sealed-was-executed"
        goal["tests"]["sealed"] = (
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
            + goal["tests"]["sealed"])
    (adapter / "discovery-goals.json").write_text(json.dumps(goals))
    # Keep controller/import behavior intact while limiting eligible context to
    # the two small fixture modules. No source in ROOT is changed.
    for p in (e.repo / "gama").glob("*.py"):
        if p.stem.startswith(("rsi", "continual", "fixture_")) or p.name in ("__init__.py", "__main__.py", "cli.py"):
            continue
        p.write_text("#" + "x" * 60001 + "\n" + p.read_text())
    e.git("add", "gama")
    e.git("commit", "-qm", "offline discovery source inventory")
    e.base = e.git("rev-parse", "HEAD").strip()


def snapshot(repo):
    return {str(p.relative_to(repo)): p.read_bytes() for p in repo.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(repo).parts}
