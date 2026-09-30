"""Frozen offline tasks acceptance. Run from the implementation checkout."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path.cwd().resolve()
PY = "/home/akhd/work/gama-rsi/.venv/bin/python"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
SPLITS = ("search", "confirm", "sealed")
TARGET = "gama/_task_acceptance.py"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def source(assertion="self.assertEqual(candidate.VALUE, 2)"):
    return f'''# Exact UTF-8 bytes — 保持
from pathlib import Path
import unittest
from gama import _task_acceptance as candidate
print("captured import noise")
class Behavior(unittest.TestCase):
    def test_binding(self):
        print("captured test noise")
        self.assertTrue(Path(candidate.__file__).resolve().is_relative_to(Path.cwd().resolve()))
        self.assertGreaterEqual(candidate.VALUE, 1)
    def test_behavior(self):
        {assertion}
if __name__ == "__main__":
    raise RuntimeError("a scorer must load tests, not run the main guard")
'''


class SeedReady(BaseException):
    pass


class TasksAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = importlib.import_module("gama.continual_tasks")
        cls.cli = str(Path(cls.tasks.__file__).resolve())

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="gama-tasks-acceptance-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        shutil.copytree(ROOT / "gama", self.repo / "gama",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.candidate(1)
        for name in ("rsi_fixture.py", "continual_fixture.py", "cli.py"):
            (self.repo / "gama" / name).touch()
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_fixed.py").write_text("# fixed\n")
        (self.repo / "gama/nested").mkdir()
        (self.repo / "gama/nested/source.py").write_text("VALUE=1\n")
        (self.repo / "gama/settings.json").write_text("{}\n")
        (self.repo / "gama/link.py").symlink_to("_task_acceptance.py")
        self.git("init", "-q", "-b", "acceptance/tasks")
        self.git("add", ".")
        self.git("commit", "-qm", "offline candidate baseline")
        remote = self.root / "remote.git"
        self.git("init", "--bare", "-q", str(remote))
        self.git("remote", "add", "origin", str(remote))
        self.git("push", "-q", "origin", "acceptance/tasks")
        self.goal = dict(id="finite-task", title="Candidate behavior",
                         goal="Return the corrected value while preserving valid output.",
                         allowed_paths=[TARGET],
                         tests={s: source() for s in SPLITS})
        # Preserve line endings as well as non-ASCII source bytes.
        self.goal["tests"]["confirm"] = source().replace("\n", "\r\n")

    def git(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_AUTHOR_NAME="offline", GIT_COMMITTER_NAME="offline",
                   GIT_AUTHOR_EMAIL="offline@example.invalid",
                   GIT_COMMITTER_EMAIL="offline@example.invalid")
        return subprocess.check_output(["git", *args], cwd=self.repo, env=env,
                                       stderr=subprocess.PIPE, timeout=20)

    def candidate(self, value):
        (self.repo / TARGET).write_text(f"VALUE = {value}\n")

    def command(self, args):
        return subprocess.run(args, cwd=self.repo, text=True, capture_output=True,
                              timeout=30)

    def score(self, text):
        script = self.root / "external-test.py"
        script.write_bytes(text.encode("utf-8"))
        return self.command([PY, "-I", "-B", self.cli, "score", "--test", str(script)])

    def measured(self, proc, score, failures=0, errors=0):
        self.assertEqual(proc.returncode, 0, proc.stderr)
        def invalid(value):
            raise AssertionError(f"nonfinite JSON: {value}")
        data = json.loads(proc.stdout, parse_constant=invalid)
        self.assertIn(type(data["score"]), (int, float))
        self.assertTrue(math.isfinite(data["score"]))
        self.assertEqual(data["score"], score)
        self.assertEqual((data["tests_run"], data["failures"], data["errors"]),
                         (2, failures, errors))

    def test_schema_and_strict_tracked_paths(self):
        self.assertEqual(self.tasks.validate_goal(copy.deepcopy(self.goal), self.repo),
                         self.goal)
        unexecuted = copy.deepcopy(self.goal)
        unexecuted["tests"]["sealed"] = "raise RuntimeError('compile only')\n" + source()
        self.tasks.validate_goal(unexecuted, self.repo)
        invalid = []
        for key, value in (("id", "../bad"), ("unknown", 1),
                           ("title", ""), ("allowed_paths", [])):
            row = copy.deepcopy(self.goal); row[key] = value; invalid.append(row)
        for text in ("def broken(:", "#" + "界" * 2100):
            row = copy.deepcopy(self.goal); row["tests"]["sealed"] = text
            invalid.append(row)
        row = copy.deepcopy(self.goal); del row["tests"]["confirm"]; invalid.append(row)
        row = copy.deepcopy(self.goal); row["tests"]["extra"] = source(); invalid.append(row)
        (self.repo / "gama/untracked.py").write_text("VALUE=1\n")
        for name in ("tests/test_fixed.py", "gama/untracked.py", "gama/missing.py",
                     TARGET + "/x", "gama/*.py", "gama/../gama/_task_acceptance.py",
                     str(self.repo / TARGET), "gama/link.py", "gama/__init__.py",
                     "gama/__main__.py", "gama/cli.py", "gama/rsi_fixture.py",
                     "gama/continual_fixture.py", "gama/nested/source.py",
                     "gama/settings.json"):
            row = copy.deepcopy(self.goal); row["allowed_paths"] = [name]
            invalid.append(row)
        for row in invalid:
            with self.subTest(row=str(row)[:180]):
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    self.tasks.validate_goal(row, self.repo)

    def test_real_candidate_scoring_and_empty_suite_refusal(self):
        before = self.git("status", "--porcelain")
        self.measured(self.score(source()), 0.5, failures=1)
        self.assertEqual(self.git("status", "--porcelain"), before)
        self.candidate(2)
        self.measured(self.score(source()), 1.0)
        self.measured(self.score(source("raise ValueError('test error')")), 0.5, errors=1)
        for text in ("import unittest\n", "raise ImportError('broken suite')\n"):
            proc = self.score(text)
            self.assertNotEqual(proc.returncode, 0, proc.stdout)

    def test_frozen_inputs_core_hashes_and_retained_regressions(self):
        descriptor = self.root / "goal.json"; write_json(descriptor, self.goal)
        prior = copy.deepcopy(self.goal); prior["id"] = "prior-goal"
        prior["tests"] = {s: source(f"self.assertNotEqual(candidate.VALUE, {i})")
                          for i, s in enumerate(SPLITS, 2)}
        prior_path = self.root / "prior.json"; write_json(prior_path, prior)
        older = copy.deepcopy(prior); older["id"] = "older-goal"
        older["tests"] = {s: source(f"self.assertNotEqual(candidate.VALUE, {i})")
                          for i, s in enumerate(SPLITS, 5)}
        older_path = self.root / "older.json"; write_json(older_path, older)
        prior_paths = [str(older_path), str(prior_path)]
        adapter = self.root / "adapter"; (adapter / "astra_loop").mkdir(parents=True)
        (adapter / "astra_loop/__init__.py").write_text("")
        (adapter / "astra_loop/backends.py").write_text(
            "raise RuntimeError('tasks acceptance must never invoke providers')\n")
        bridge = self.root / "bridge.json"
        write_json(bridge, dict(astra_loop_root=str(adapter),
                               artifact_root=str(self.root / "old-artifacts"),
                               backend={}, timeout=20))
        check = [PY, "-I", "-B", "-c", "assert 1 + 1 == 2"]
        cfg = dict(campaign_id="offline-tasks", repo=str(self.repo),
                   state_dir=str(self.root / "campaign"), branch="acceptance/tasks",
                   remote="origin", bridge_config=str(bridge), goals=[str(descriptor)],
                   checks=[check], evaluation_timeout=15, max_goal_cycles=2, workers=2,
                   max_proposals_per_tick=4, schedule_hours=[9, 21], timezone="Asia/Tokyo")
        frozen = self.root / "frozen"
        mission_path = Path(self.tasks.freeze_goal(cfg, self.goal, frozen, prior_paths))
        mission = json.loads(mission_path.read_bytes())
        rsi_path, bridge_path = (Path(mission[k]) for k in ("rsi_config", "bridge_config"))
        rsi = json.loads(rsi_path.read_bytes())
        for key, value in dict(rounds_per_cycle=1, max_reservations_per_cycle=4,
                               batch_size=2, search_ceiling=1, confirm_ceiling=1).items():
            self.assertEqual(mission[key], value)
        for key, value in dict(workers=2, batch_size=2, search_repeats=1,
                               confirm_repeats=3, min_gain=0).items():
            self.assertEqual(rsi[key], value)
        self.assertNotIn("sealed_repeats", rsi)
        self.assertIn(check, rsi["checks"])
        self.assertEqual(rsi["allowed_paths"], [TARGET])
        copied_bridge = json.loads(bridge_path.read_bytes())
        self.assertNotEqual(bridge_path, bridge)
        artifacts = Path(copied_bridge["artifact_root"])
        self.assertTrue(artifacts.is_absolute())
        self.assertNotEqual(artifacts, self.root / "old-artifacts")
        self.assertFalse(artifacts.is_relative_to(self.repo))
        regression = [c for c in rsi["checks"] if "regressions" in c and "--goals" in c]
        self.assertEqual(len(regression), 1)
        regression = regression[0]
        manifest = Path(regression[regression.index("--goals") + 1])
        required = {mission_path, rsi_path, bridge_path, manifest, Path(self.cli)}
        for split in SPLITS:
            cmd = rsi[split + "_command"]
            self.assertIn("-I", cmd); self.assertIn("-B", cmd)
            test = Path(cmd[cmd.index("--test") + 1])
            self.assertTrue(test.is_absolute())
            self.assertEqual(test.read_bytes(), self.goal["tests"][split].encode("utf-8"))
            required.add(test)
        inputs = {Path(p) for p in rsi["evaluation_files"]}
        self.assertTrue(required <= inputs, required - inputs)
        self.assertTrue(all(p.is_absolute() and p.is_file() for p in inputs))
        retained, found = [], set()
        descriptors = {x["id"]: x for x in (self.goal, prior, older)}
        for path in inputs:
            if path.suffix == ".json":
                obj = json.loads(path.read_bytes())
                if isinstance(obj, dict) and obj.get("id") in descriptors:
                    self.assertEqual(obj, descriptors[obj["id"]])
                    found.add(obj["id"])
                    if obj["id"] != self.goal["id"]:
                        retained.append(path)
        self.assertEqual(found, set(descriptors), "all descriptors must be frozen inputs")
        snapshot = {p: p.read_bytes() for p in frozen.rglob("*") if p.is_file()}
        changed = copy.deepcopy(self.goal); changed["goal"] += " CHANGED"
        for row in (self.goal, changed):
            try:
                self.tasks.freeze_goal(cfg, row, frozen, prior_paths)
            except (ValueError, RuntimeError, FileExistsError):
                pass
            self.assertEqual({p: p.read_bytes() for p in frozen.rglob("*") if p.is_file()},
                             snapshot)
        self.assertEqual(self.command(regression).returncode, 0)
        for value in range(2, 8):
            self.candidate(value)
            self.assertNotEqual(self.command(regression).returncode, 0,
                                f"retained split for value {value} was not enforced")
        self.candidate(1)
        from gama.rsi import RSIError, run_rsi
        from gama.rsi_runtime import load_inputs
        effective = load_inputs(mission_path)["rsi_config"]
        state = self.root / "core-proof"
        def stop_at_seed(event):
            if event["event"] == "seed":
                raise SeedReady()
        with self.assertRaises(SeedReady):
            run_rsi(effective, repo=self.repo, state_dir=state, on_event=stop_at_seed)
        core = json.loads((state / "state.json").read_bytes())
        self.assertEqual(core["reserved_proposals"], 0)
        hashes = core["contract"]["evaluation_files"]
        for path in inputs:
            self.assertEqual(hashes[str(path)], hashlib.sha256(path.read_bytes()).hexdigest())
        test.chmod(test.stat().st_mode | 0o200)
        test.write_bytes(test.read_bytes() + b"\n# changed after seed\n")
        with self.assertRaises(RSIError):
            run_rsi(effective, repo=self.repo, state_dir=state, resume=True)
        for path in retained:
            path.chmod(path.stat().st_mode | 0o200)
            path.write_bytes(path.read_bytes() + b"\n")
        self.assertNotEqual(self.command(regression).returncode, 0,
                            "changed frozen descriptor bytes must be rejected")


if __name__ == "__main__":
    unittest.main()
