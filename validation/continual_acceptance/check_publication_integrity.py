"""Independent publication regressions; only disposable fixtures are mutated.

Run with the pinned Python from the candidate checkout. This imports the existing
immutable publication Fixture and exercises real core/bridge/Git paths. The only
model adapter is that fixture's explicitly simulated adapter.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path.cwd().resolve()
HELPERS = ROOT / "validation/continual_acceptance"
if not (HELPERS / "publish_common.py").is_file():
    HELPERS = Path(__file__).resolve().parents[1] / "acceptance"
sys.path.insert(0, str(HELPERS))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from publish_common import BRANCH, PY, SOURCE, Fixture, git, write_json


def digest(value):
    # Canonical digest of the unchanged RSI core's recorded contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


PRIOR_TEST = '''import json, os, unittest
from pathlib import Path
from gama.publish_fixture import solve

class PriorFloorRegression(unittest.TestCase):
    def test_floor_is_retained(self):
        marker = os.environ.get("GAMA_PUBLICATION_INTEGRITY_PRIOR_LOG")
        if marker:
            fd = os.open(marker, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            os.write(fd, (json.dumps({"cwd": str(Path.cwd())}) + "\\n").encode())
            os.close(fd)
        self.assertNotEqual(os.environ.get("GAMA_PUBLICATION_INTEGRITY_PRIOR_FAIL"), "1")
        self.assertEqual(solve(-7), 0)
        self.assertEqual(solve(0), 0)
        self.assertEqual(solve(1), 1)

if __name__ == "__main__":
    unittest.main()
'''


SWITCH_SHIM = r'''
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
effective = Path.cwd()
command = None
tail = []
i = 0
while i < len(args):
    token = args[i]
    if token == "-C":
        effective = (effective / args[i + 1]).resolve(); i += 2
    elif token.startswith("-C") and token != "-C":
        effective = (effective / token[2:]).resolve(); i += 1
    elif token == "-c":
        i += 2
    elif token == "--work-tree":
        effective = Path(args[i + 1]).resolve(); i += 2
    elif token.startswith("--work-tree="):
        effective = Path(token.split("=", 1)[1]).resolve(); i += 1
    elif token == "--git-dir":
        i += 2
    elif token.startswith("-"):
        i += 1
    else:
        command, tail = token, args[i + 1:]
        break
mutation = command in ("merge", "read-tree")
if command == "update-ref":
    # Ignore archive/receipt pins. --stdin also covers a real ref transaction
    # without consuming or altering its protocol/input stream.
    mutation = ("--stdin" in tail or "HEAD" in tail
                or any(arg.startswith("refs/heads/") for arg in tail))
if effective.resolve() == Path(CALLER) and mutation and not Path(MARKER).exists():
    try:
        stream = open(MARKER, "x")
    except FileExistsError:
        stream = None
    if stream is not None:
        with stream:
            switched = subprocess.run(
                [REAL_GIT, "-c", "core.hooksPath=/dev/null", "-C", CALLER,
                 "checkout", "--quiet", "main"],
                capture_output=True, text=True, timeout=10)
            json.dump({"intercepted_command": command, "argv": args,
                       "switch_returncode": switched.returncode,
                       "switch_stderr": switched.stderr[-2000:]}, stream)
            stream.flush(); os.fsync(stream.fileno())
os.execv(DELEGATE, [DELEGATE, *args])
'''


class PublicationIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.publish = staticmethod(importlib.import_module("gama.continual_publish").publish)
        cls.freeze = staticmethod(importlib.import_module("gama.continual_tasks").freeze_goal)
        cls.source_hash = hashlib.sha256(
            (ROOT / "gama/continual_publish.py").read_bytes()).hexdigest()

    def fixture(self):
        tmp = tempfile.TemporaryDirectory(prefix="gama-publication-integrity-")
        self.addCleanup(tmp.cleanup)

        def with_prior(config, goal, directory, prior_goals):
            prior = {
                "id": "prior-negative-floor", "title": "Keep negative inputs floored",
                "goal": "Retain zero as the floor while improving positive inputs.",
                "allowed_paths": [SOURCE],
                "tests": {split: "# " + split + "\n" + PRIOR_TEST
                          for split in ("search", "confirm", "sealed")},
            }
            path = Path(directory).parent / "previous-accepted-goal.json"
            write_json(path, prior)
            return self.freeze(config, goal, directory, [*prior_goals, str(path)])

        f = Fixture(tmp.name, freezer=with_prior)
        state = json.loads(f.state_path.read_bytes())
        self.assertEqual(f.result["phase"], "finalized")
        self.assertEqual(f.result["sealed_verdict"], "improved")
        self.assertNotEqual(f.source, f.base)
        self.assertEqual(f.result["champion"]["confirm"]["samples"], [1.0] * 3)
        self.assertEqual(f.result["sealed"]["champion"]["samples"], [1.0] * 3)
        self.assertEqual(state["contract_hash"], digest(state["contract"]))
        checks = state["contract"]["config"]["checks"]
        retained = [row for row in checks if "regressions" in row and "--goals" in row]
        self.assertEqual(len(retained), 1, "fixture needs a real cumulative regression command")
        f.retained_command = retained[0]
        f.prior_log = f.root / "prior-regression-executions.jsonl"
        self.assertIn(f.retained_command,
                      json.loads(Path(f.mission["rsi_config"]).read_bytes())["checks"])
        return f

    def observe(self, f, **extra):
        value = {
            "test": self._testMethodName, "publisher_sha256": self.source_hash,
            "base": f.base, "source": f.source, "local_head": f.head(),
            "main": git(f.repo, "rev-parse", "refs/heads/main").decode().strip(),
            "branch": git(f.repo, "symbolic-ref", "--short", "HEAD").decode().strip(),
            "remote_feature": f.remote_head(),
            "prior_regression_runs": (len(f.prior_log.read_text().splitlines())
                                      if f.prior_log.exists() else 0),
            **extra,
        }
        print("PUBLICATION_INTEGRITY_OBSERVATION " + json.dumps(value, sort_keys=True), flush=True)
        return value

    def invoke(self, f, *, prior_must_fail=False):
        with patch.dict(os.environ, {
                "GAMA_PUBLICATION_INTEGRITY_PRIOR_LOG": str(f.prior_log),
                "GAMA_PUBLICATION_INTEGRITY_PRIOR_FAIL": "1" if prior_must_fail else "0"}):
            try:
                return f.publish(self.publish), None
            except (AssertionError, ImportError, AttributeError, KeyError, TypeError):
                raise
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"

    def assert_no_inference(self, f):
        self.assertEqual(f.calls.read_bytes(), f.initial_calls)

    def assert_published(self, f, result, error):
        self.assertIsNone(error, error)
        self.assertIsInstance(result, dict)
        self.assertNotIn(f.head(), (f.base, f.source))
        self.assertEqual(f.remote_head(), f.head())
        self.assertEqual(git(f.repo, "rev-parse", "main").decode().strip(), f.base)
        self.assertGreaterEqual(len(f.prior_log.read_text().splitlines()), 3,
                                "untampered publication must actually run retained history")
        self.assert_no_inference(f)

    def test_untampered_frozen_controls_publish_and_execute_prior_regressions(self):
        f = self.fixture()
        result, error = self.invoke(f)
        self.observe(f, error=error, returned=result)
        self.assert_published(f, result, error)

    def test_operator_branch_switch_immediately_before_adoption_never_advances_main(self):
        f = self.fixture()
        initial_source = (f.repo / SOURCE).read_bytes()
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        delegate = f.bin / "git-delegate"
        (f.bin / "git").rename(delegate)
        marker = f.root / "branch-switch-attempt.json"
        settings = {"CALLER": str(f.repo), "MARKER": str(marker),
                    "REAL_GIT": real_git, "DELEGATE": str(delegate)}
        wrapper = f.bin / "git"
        wrapper.write_text("#!" + PY + "\n" +
                           "".join(f"{key} = {value!r}\n" for key, value in settings.items()) +
                           SWITCH_SHIM)
        wrapper.chmod(0o755)
        result, error = self.invoke(f)
        self.assertTrue(marker.is_file(),
                        "must intercept real caller merge/read-tree/update-ref, not a private helper")
        attempt = json.loads(marker.read_bytes())
        observation = self.observe(f, error=error, returned=result, injection=attempt)
        self.assertEqual(observation["main"], f.base,
                         "publication advanced main after the injected operator checkout")
        self.assertEqual(git(f.repo, "ls-remote", "origin", "refs/heads/main")
                         .decode().split()[0], f.base)
        if attempt["switch_returncode"] == 0:
            self.assertEqual(observation["branch"], "main",
                             "publication must preserve a successful operator branch switch")
            self.assertEqual((f.repo / SOURCE).read_bytes(), initial_source,
                             "publication must not install feature source into the operator's main")
            self.assertFalse(git(f.repo, "status", "--porcelain=v1", "--untracked-files=all"))
        else:
            # A real transaction may hold a Git lock and safely block the switch.
            self.assertTrue(attempt["switch_stderr"])
            self.assertEqual(observation["branch"], BRANCH)
        self.assert_no_inference(f)

    def corrupt_controls(self, *, recompute):
        f = self.fixture()
        state = json.loads(f.state_path.read_bytes())
        original_contract_hash = state["contract_hash"]
        frozen_bytes = {name: Path(name).read_bytes()
                        for name in state["contract"]["evaluation_files"]}
        commands = state["contract"]["config"]["checks"]
        commands.remove(f.retained_command)
        self.assertTrue(commands, "ordinary campaign checks remain mandatory")
        if recompute:
            state["contract_hash"] = digest(state["contract"])
        else:
            self.assertEqual(state["contract_hash"], original_contract_hash)
            self.assertNotEqual(state["contract_hash"], digest(state["contract"]))
        f.state_path.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False))
        tampered_state = f.state_path.read_bytes()
        before = f.snapshot_checkout()
        result, error = self.invoke(f, prior_must_fail=True)
        self.observe(f, error=error, returned=result, recomputed_contract_hash=recompute,
                     original_contract_hash=original_contract_hash,
                     supplied_contract_hash=state["contract_hash"],
                     removed_command=f.retained_command)
        for name, content in frozen_bytes.items():
            self.assertEqual(Path(name).read_bytes(), content, "frozen inputs were altered: " + name)
        self.assertEqual(f.state_path.read_bytes(), tampered_state,
                         "publisher must not repair/overwrite core evidence")
        self.assertEqual(f.snapshot_checkout(), before,
                         "modified effective controls must be refused before caller adoption")
        self.assertEqual(f.remote_head(), f.base,
                         "modified effective controls must never be pushed")
        self.assertTrue(error or (isinstance(result, dict) and
                                  (result.get("error") or result.get("reason") or result.get("message"))),
                        "modified frozen controls need an explicit refusal")
        self.assert_no_inference(f)

    def test_removed_cumulative_check_with_original_contract_hash_is_refused(self):
        self.corrupt_controls(recompute=False)

    def test_rehashed_modified_contract_still_binds_original_frozen_rsi_inputs(self):
        self.corrupt_controls(recompute=True)

    def test_success_reclaims_owned_release_worktree_and_preserves_operator_worktree(self):
        f = self.fixture()
        operator = f.root / "operator-worktree"
        git(f.repo, "worktree", "add", "--detach", str(operator), f.base)
        note = operator / "operator-notes.txt"
        note.write_text("keep this independent checkout and its untracked data\n")
        result, error = self.invoke(f)
        worktrees = [Path(row.removeprefix("worktree "))
                     for row in git(f.repo, "worktree", "list", "--porcelain").decode().splitlines()
                     if row.startswith("worktree ")]
        self.observe(f, error=error, returned=result,
                     registered_worktrees=[str(path) for path in worktrees])
        self.assert_published(f, result, error)
        self.assertEqual(note.read_text(), "keep this independent checkout and its untracked data\n")
        self.assertEqual(set(worktrees), {f.repo, operator},
                         "completed publication leaked an owned release checkout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
