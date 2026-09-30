"""Publication fixtures: real RSI, real Git, explicitly simulated model calls.

Acceptance uses the public freeze_goal API, never a fabricated core result.
The standalone --self-check exercises the same core/bridge fixture before the
new tasks/publisher modules exist; its legacy freezer is NOT an acceptance
fallback. Run with the pinned Python, from the implementation checkout.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path.cwd().resolve()
PY = "/home/akhd/work/gama-rsi/.venv/bin/python"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
BRANCH = "acceptance/publication"
SOURCE = "gama/publish_fixture.py"
OTHER = "gama/publish_other.py"

FAKE_BACKEND = r'''
import difflib, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
class LiveBackend:
    def __init__(self, config): self.config = config
    def identity(self, role):
        builder = role == "builder"
        return {"provider": "offline-fixture", "model": "astra" if builder else "opus",
                "family": "openai" if builder else "anthropic",
                "resolved_model": "offline-astra" if builder else "offline-claude",
                "simulated": True}
    def complete(self, role, prompt, *, cwd, output_dir, cancel=None):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps({"simulated": True,
                                                 "usage": {"fixture_calls": 1}}))
        fd = os.open(ROOT / "calls.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.write(fd, (json.dumps({"role": role, "cwd": str(cwd)}) + "\n").encode())
        os.close(fd)
        if role != "builder":
            return '{"verdict":"PASS","reason":"deterministic offline patch"}'
        request = json.loads(prompt.split("\nREQUEST_JSON\n", 1)[1])
        name = "gama/publish_fixture.py"; old = request["files"][name]
        mode = (ROOT / "mode").read_text()
        new = (old + "# behavior unchanged\n" if mode == "unimproved"
               else old.replace("min(value, 1)", "value"))
        changes = [(name, old, new)]
        if mode == "extra":
            name = "gama/publish_other.py"; old = request["files"][name]
            changes.append((name, old, old.replace("True", "False")))
        return "".join("".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            fromfile="a/" + name, tofile="b/" + name)) for name, old, new in changes)
'''


def git(repo, *args, data=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_AUTHOR_NAME="publication acceptance",
               GIT_AUTHOR_EMAIL="offline@example.invalid",
               GIT_COMMITTER_NAME="publication acceptance",
               GIT_COMMITTER_EMAIL="offline@example.invalid")
    result = subprocess.run(["git", *args], cwd=repo, env=env, input=data,
                            capture_output=True, timeout=20)
    if result.returncode:
        raise AssertionError(f"fixture git {args!r}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=3) + "\n", encoding="utf-8")


def strings(value):
    """Read values without depending on names/shape of private journal stages."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from strings(key)
            yield from strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from strings(child)


def test_source(split):
    values = {"search": [2, 3], "confirm": [4, 7], "sealed": [11, 19]}[split]
    text = f'''# Frozen {split}: exact Unicode bytes — 保持
import json
import os
from pathlib import Path
import unittest
from gama import publish_fixture as candidate

class RetainedBehavior(unittest.TestCase):
    def test_general_values_and_negative_floor(self):
        self.assertNotEqual(os.environ.get("GAMA_PUBLICATION_ACCEPTANCE_FAIL"), "1")
        self.assertTrue(Path(candidate.__file__).resolve().is_relative_to(Path.cwd().resolve()))
        self.assertEqual(candidate.solve(-5), 0)
        self.assertEqual(candidate.solve(0), 0)
        for value in {values!r}:
            self.assertEqual(candidate.solve(value), value)
        marker = os.environ.get("GAMA_PUBLICATION_ACCEPTANCE_PROBE")
        if marker:
            receipt_path = Path.cwd().parent / ("." + Path.cwd().name + ".applied.json")
            receipt = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else None
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.write(fd, (json.dumps({{"split": {split!r}, "cwd": str(Path.cwd()),
                                      "receipt": receipt}}) + "\\n").encode())
            os.close(fd)

if __name__ == "__main__":
    unittest.main()
'''
    return text.replace("\n", "\r\n") if split == "confirm" else text


class Fixture:
    def __init__(self, root, *, freezer, finalized=True, mode="improved"):
        self.root = Path(root)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        shutil.copytree(ROOT / "gama", self.repo / "gama",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (self.repo / SOURCE).write_text(
            "def solve(value):\n    return max(0, min(value, 1))\n")
        (self.repo / OTHER).write_text("UNCHANGED = True\n")
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "publication acceptance")
        git(self.repo, "config", "user.email", "offline@example.invalid")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "offline fixture baseline")
        self.base = self.head()
        git(self.repo, "checkout", "-qb", BRANCH)
        self.remote = self.root / "remote.git"
        git(self.root, "init", "--bare", "-q", str(self.remote))
        git(self.remote, "config", "receive.denyNonFastForwards", "true")
        git(self.repo, "remote", "add", "origin", str(self.remote))
        git(self.repo, "push", "-q", "origin", "main", BRANCH)

        adapter = self.root / "adapter"
        (adapter / "astra_loop").mkdir(parents=True)
        (adapter / "astra_loop/__init__.py").write_text("")
        (adapter / "astra_loop/backends.py").write_text(FAKE_BACKEND)
        (adapter / "mode").write_text(mode)
        self.calls = adapter / "calls.jsonl"
        self.probe = self.root / "regressions-executed.jsonl"
        self.git_log = self.root / "publication-git.jsonl"
        self.lose_push_reply = self.root / "lose-successful-push-reply"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        real_git = shutil.which("git")
        if not real_git:
            raise AssertionError("Git is required for acceptance")
        wrapper = self.bin / "git"
        wrapper.write_text(
            f"#!{PY}\nimport json,os,subprocess,sys\nfrom pathlib import Path\n"
            f"fd=os.open({str(self.git_log)!r},os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\n"
            "os.write(fd,(json.dumps(sys.argv[1:])+'\\n').encode());os.close(fd)\n"
            f"lost=Path({str(self.lose_push_reply)!r})\n"
            "if 'push' in sys.argv[1:] and lost.exists():\n"
            f" result=subprocess.run([{real_git!r},*sys.argv[1:]])\n"
            " if result.returncode==0:\n"
            "  lost.unlink();sys.exit(71)\n"
            " sys.exit(result.returncode)\n"
            f"os.execv({real_git!r},[{real_git!r},*sys.argv[1:]])\n")
        wrapper.chmod(0o755)
        self.bridge = self.root / "bridge.json"
        write_json(self.bridge, {
            "astra_loop_root": str(adapter), "artifact_root": str(self.root / "model-evidence"),
            "backend": {"timeout_seconds": 10, "bedrock_max_tokens": 512,
                        "max_context_bytes": 110000}, "timeout": 20})
        self.goal = {
            "id": "preserve-clamp-floor", "title": "Remove an arbitrary positive cap",
            "goal": "Return nonnegative integers unchanged and negative integers as zero.",
            "allowed_paths": [SOURCE, OTHER] if mode == "extra" else [SOURCE],
            "tests": {split: test_source(split) for split in ("search", "confirm", "sealed")}}
        self.descriptor = self.root / "portable-goal.json"
        write_json(self.descriptor, self.goal)
        self.config = {
            "campaign_id": "offline-publication", "repo": str(self.repo),
            "state_dir": str(self.root / "campaign"), "branch": BRANCH, "remote": "origin",
            "bridge_config": str(self.bridge), "goals": [str(self.descriptor)],
            "checks": [[PY, "-I", "-B", "-c",
                        "import sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()));"
                        "from gama.publish_fixture import solve;"
                        "assert solve(-5)==0 and solve(0)==0 and solve(1)==1"]],
            "evaluation_timeout": 15, "max_goal_cycles": 2, "workers": 2,
            "max_proposals_per_tick": 4, "schedule_hours": [9, 21], "timezone": "Asia/Tokyo"}
        self.frozen = self.root / "frozen-goal"
        self.mission_path = Path(freezer(self.config, copy.deepcopy(self.goal),
                                        self.frozen, []))
        self.mission = json.loads(self.mission_path.read_bytes())
        self.state = Path(self.mission["state_dir"])
        if finalized:
            self.command([PY, "-B", "-m", "gama.rsi_mission", "run",
                          "--mission", str(self.mission_path)])
        else:
            self.core(finalize=False)
        self.result_path = self.state / "rsi/result.json"
        self.state_path = self.state / "rsi/state.json"
        self.result = json.loads(self.result_path.read_bytes())
        if finalized and self.result["phase"] != "finalized":
            self.core(finalize=True)
            self.command([PY, "-B", "-m", "gama.rsi_mission", "resume",
                          "--mission", str(self.mission_path)])
            self.result = json.loads(self.result_path.read_bytes())
        self.source = self.result["champion"]["commit"]
        self.journal = {"base_commit": self.base}
        self.snapshots = []
        self.cancel = threading.Event()
        self.initial_calls = self.calls.read_bytes() if self.calls.exists() else b""
        self.frozen_descriptors = [
            p.read_bytes() for p in self.frozen.rglob("*.json")
            if self._is_descriptor(p)]
        if not self.frozen_descriptors:
            raise AssertionError("freeze_goal did not retain the portable descriptor")
        for script in self.goal["tests"].values():
            if not any(p.read_bytes() == script.encode("utf-8")
                       for p in self.frozen.rglob("*.py")):
                raise AssertionError("freeze_goal did not retain exact test bytes")

    def _is_descriptor(self, path):
        try:
            return json.loads(path.read_bytes()) == self.goal
        except (ValueError, OSError):
            return False

    def command(self, argv):
        p = subprocess.run(argv, cwd=self.repo, text=True, capture_output=True, timeout=100)
        if p.returncode:
            raise AssertionError(f"fixture command failed: {argv!r}\n{p.stdout}\n{p.stderr}")
        return p

    def core(self, *, finalize):
        code = ("from pathlib import Path;from gama.rsi_runtime import load_inputs;"
                "from gama.rsi import run_rsi;"
                f"i=load_inputs(Path({str(self.mission_path)!r}));"
                "m=i['mission'];run_rsi(i['rsi_config'],repo=m['repo'],"
                "state_dir=Path(m['state_dir'])/'rsi',rounds=1,"
                f"resume={finalize!r},finalize={finalize!r})")
        self.command([PY, "-B", "-c", code])

    def head(self):
        return git(self.repo, "rev-parse", "HEAD").decode().strip()

    def remote_head(self):
        return git(self.repo, "ls-remote", "origin", "refs/heads/" + BRANCH).decode().split()[0]

    def save(self, journal):
        snapshot = json.loads(json.dumps(journal, allow_nan=False))
        self.snapshots.append(snapshot)
        path = self.root / "durable-journal.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(snapshot, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def reload(self):
        self.journal = json.loads((self.root / "durable-journal.json").read_bytes())

    def publish(self, publisher, *, save=None, goal=None):
        from unittest.mock import patch
        with patch.dict(os.environ, {
                "GAMA_PUBLICATION_ACCEPTANCE_PROBE": str(self.probe),
                "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", "")}):
            return publisher(self.config, goal=self.goal if goal is None else goal,
                             mission_path=self.mission_path, journal=self.journal,
                             save=self.save if save is None else save, cancel=self.cancel)

    def snapshot_checkout(self):
        return (self.head(), git(self.repo, "symbolic-ref", "HEAD"),
                git(self.repo, "status", "--porcelain=v1", "--untracked-files=all"),
                git(self.repo, "diff", "--binary"), git(self.repo, "diff", "--cached", "--binary"),
                (self.repo / SOURCE).read_bytes(), (self.repo / OTHER).read_bytes())

    def core_bytes(self):
        return {str(p.relative_to(self.state / "rsi")): p.read_bytes()
                for p in (self.state / "rsi").rglob("*") if p.is_file()}

    def git_commands(self):
        return ([json.loads(line) for line in self.git_log.read_text().splitlines()]
                if self.git_log.exists() else [])


def legacy_freezer(config, goal, directory, prior_goals):
    """Only the standalone fixture smoke check uses this existing-core freezer."""
    directory.mkdir()
    write_json(directory / "goal.json", goal)
    scorer = directory / "score.py"
    scorer.write_text(
        "import contextlib,io,json,runpy,sys,unittest\nfrom pathlib import Path\n"
        "sys.path.insert(0,str(Path.cwd()))\n"
        "with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):\n"
        " scope=runpy.run_path(sys.argv[1],run_name='frozen_goal')\n"
        " suite=unittest.TestSuite()\n"
        " for v in scope.values():\n"
        "  if isinstance(v,type) and issubclass(v,unittest.TestCase):\n"
        "   suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(v))\n"
        " result=unittest.TextTestRunner(stream=io.StringIO()).run(suite)\n"
        "if not result.testsRun: raise RuntimeError('empty suite')\n"
        "print(json.dumps({'score':(result.testsRun-len(result.failures)-len(result.errors))"
        "/result.testsRun,'tests_run':result.testsRun,"
        "'failures':len(result.failures),'errors':len(result.errors)}))\n")
    for split, text in goal["tests"].items():
        (directory / (split + ".py")).write_bytes(text.encode("utf-8"))
    rsi = {
        "goal": goal["goal"], "allowed_paths": goal["allowed_paths"],
        "agents": [{"name": "bound-by-runtime", "command": [PY, "-c", "raise SystemExit(99)"]}],
        "checks": config["checks"], "workers": 2, "batch_size": 2,
        "timeout": 30, "evaluation_timeout": 15, "search_repeats": 1,
        "confirm_repeats": 3, "min_gain": 0,
        "evaluation_files": [str(p) for p in directory.iterdir() if p.is_file()]}
    for split in goal["tests"]:
        rsi[split + "_command"] = [PY, "-I", "-B", str(scorer), str(directory / (split + ".py"))]
    write_json(directory / "rsi.json", rsi)
    mission = directory / "mission.json"
    write_json(mission, {
        "mission_id": goal["id"], "repo": config["repo"],
        "state_dir": str(directory / "state"), "rsi_config": str(directory / "rsi.json"),
        "bridge_config": config["bridge_config"], "search_ceiling": 1,
        "confirm_ceiling": 1, "rounds_per_cycle": 1, "batch_size": 2,
        "max_reservations_per_cycle": 4})
    return mission


if __name__ == "__main__":
    import tempfile
    if sys.argv[1:] != ["--self-check"]:
        raise SystemExit("use --self-check; normal acceptance imports Fixture")
    with tempfile.TemporaryDirectory(prefix="gama-publish-fixture-") as root:
        fixture = Fixture(root, freezer=legacy_freezer)
        assert fixture.result["phase"] == "finalized", fixture.result
        assert fixture.result["sealed_verdict"] == "improved", fixture.result
        assert fixture.result["champion"]["confirm"]["samples"] == [1.0] * 3
        assert fixture.result["sealed"]["champion"]["samples"] == [1.0] * 3
        assert fixture.head() == fixture.remote_head() == fixture.base
        assert fixture.source != fixture.base
        print(json.dumps({"fixture": "PASS", "real_core": True, "simulated_models": True,
                          "reserved": fixture.result["reserved_proposals"],
                          "sealed_verdict": fixture.result["sealed_verdict"]}))
