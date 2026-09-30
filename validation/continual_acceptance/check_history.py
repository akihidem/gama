"""Independent offline discovery-history regression; never edits active sources."""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, tempfile, threading
from pathlib import Path

FAKE = r'''
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class LiveBackend:
 def __init__(self,config): pass
 def identity(self,role):
  b=role=="builder"
  return {"provider":"codex" if b else "bedrock","model":"astra" if b else "opus",
   "family":"openai" if b else "anthropic",
   "resolved_model":"bedrock-astra" if b else "global.anthropic.claude-opus-5"}
 def complete(self,role,prompt,*,cwd,output_dir,cancel=None):
  out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
  (out/"prompt.txt").write_text(prompt)
  (out/"meta.json").write_text('{"usage":{"fixture_tokens":1}}')
  fd=os.open(ROOT/"calls",os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
  os.write(fd,(role+"\n").encode());os.close(fd)
  if role=="reviewer": return '{"verdict":"PASS","reason":"offline history probe"}'
  req=json.loads(prompt.split("\nDISCOVERY_JSON\n",1)[1])
  path=next(iter(req["files"]));name=Path(path).stem
  source=("import unittest\nfrom gama."+name+" import solve\n"
   "class Cases(unittest.TestCase):\n"
   " def test_zero(self): self.assertEqual(solve(0),0)\n"
   " def test_two(self): self.assertEqual(solve(2),2)\n")
  return json.dumps({"id":name+"-improvement","title":"General identity",
   "goal":"Return nonnegative input unchanged.","allowed_paths":[path],
   "tests":{s:source for s in ("search","confirm","sealed")}})
'''


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="offline history",
               GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="offline history",
               GIT_COMMITTER_EMAIL="fixture@example.invalid",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.check_output(["git", *args], cwd=repo, env=env,
                                   stderr=subprocess.PIPE, timeout=10).decode()


def descriptor(index=0, padded=False):
    source = ("import unittest\nfrom gama.old import solve\n"
              "class Historical(unittest.TestCase):\n"
              f" def test_value(self): self.assertEqual(solve({index}),{index})\n")
    if padded:
        source += "#" + "x" * 5600 + "\n"
    assert len(source.encode()) <= 6144
    return {"id": f"historical-{index}", "title": "Earlier behavior", "goal": "Preserve identity.",
            "allowed_paths": ["gama/old.py"],
            "tests": {split: source for split in ("search", "confirm", "sealed")}}


def run_case(module, root, name):
    root.mkdir()
    repo = root / "repo"
    (repo / "gama").mkdir(parents=True)
    (repo / "gama/__init__.py").write_text("")
    (repo / "gama/old.py").write_text("def solve(x):\n return x\n")
    (repo / "gama/fresh.py").write_text("def solve(x):\n return min(x,1)\n")
    git(repo, "init", "-q", "-b", "feature/history")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "formerly valid history")
    old = descriptor()
    # Establish validity before simulating later accepted source evolution.
    module._sibling("continual_tasks").validate_goal(old, repo)
    if name == "grown":
        with (repo / "gama/old.py").open("a") as stream:
            stream.write("#" + "x" * 60001 + "\n")
    elif name == "gone":
        (repo / "gama/old.py").unlink()
    if name in ("grown", "gone"):
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "later source evolution")
    seen = [descriptor(i, padded=True) for i in range(1000)] if name == "large_history" else [old]
    adapter = root / "adapter"
    (adapter / "astra_loop").mkdir(parents=True)
    (adapter / "astra_loop/__init__.py").write_text("")
    (adapter / "astra_loop/backends.py").write_text(FAKE)
    bridge = root / "bridge.json"
    bridge.write_text(json.dumps({"astra_loop_root": str(adapter),
        "artifact_root": str(root / "model-evidence"), "timeout": 20,
        "backend": {"timeout_seconds": 10, "max_context_bytes": 110000}}))
    before = {str(p.relative_to(repo)): p.read_bytes() for p in (repo / "gama").rglob("*") if p.is_file()}
    head = git(repo, "rev-parse", "HEAD")
    result, error = None, None
    try:
        result = module.discover({"repo": str(repo), "branch": "feature/history",
            "bridge_config": str(bridge), "evaluation_timeout": 5},
            directory=root / "evidence", seen=seen, cursor=0, cancel=threading.Event())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    after = {str(p.relative_to(repo)): p.read_bytes() for p in (repo / "gama").rglob("*") if p.is_file()}
    calls = (adapter / "calls").read_text().splitlines() if (adapter / "calls").exists() else []
    preserved = before == after and head == git(repo, "rev-parse", "HEAD") and not git(repo, "status", "--porcelain")
    fresh = result is not None and any("gama/fresh.py" in g["allowed_paths"] for g in result["goals"])
    return {"case": name, "passed": bool(fresh and preserved), "error": error,
            "fresh_goal": fresh, "repo_preserved": bool(preserved),
            "builder_calls": calls.count("builder"), "seen_count": len(seen),
            "seen_json_bytes": len(json.dumps(seen, separators=(",", ":")).encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd() / "gama/continual_discover.py")
    parser.add_argument("--support", type=Path, default=Path.cwd() / "gama")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.directory or Path(tempfile.mkdtemp(prefix="discovery-history-"))
    if args.directory:
        root.mkdir(parents=True)
    root = root.resolve()
    controller = root / "controller"
    shutil.copytree(args.support, controller, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = args.source.read_bytes()
    target = controller / "continual_discover.py"
    target.write_bytes(source)  # Test-only copy; no diff normalization or active edits.
    spec = importlib.util.spec_from_file_location("history_discovery_under_test", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [run_case(module, root / name, name)
            for name in ("control", "grown", "gone", "large_history")]
    report = {"source": str(args.source.resolve()), "source_sha256": hashlib.sha256(source).hexdigest(),
              "directory": str(root), "passed": sum(r["passed"] for r in rows), "total": len(rows),
              "source_unchanged": source == args.source.read_bytes(), "cases": rows}
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
