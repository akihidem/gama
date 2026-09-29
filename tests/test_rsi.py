"""End-to-end source evolution using real worktrees and offline patch processes."""
from __future__ import annotations

import copy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from gama.rsi import RSIConfig, RSIError, run_rsi, select_parents


SOURCE = """LIMIT = 1
BROKEN_CONFIRM = False
MUTATE = False

def solve(value):
    return max(0, min(value, LIMIT))
"""
CHECK = """import program
assert program.solve(0) == 0
assert program.solve(1) == 1
assert program.solve(-3) == 0
"""
SCORER = """import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("candidate", pathlib.Path.cwd() / "program.py")
program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)
split = sys.argv[1]
if split == "confirm" and program.BROKEN_CONFIRM:
    raise RuntimeError("confirmation measurement failed")
values = {"search": range(8), "confirm": range(0, 16, 2), "sealed": range(0, 24, 3)}[split]
score = sum(program.solve(v) == v for v in values) / len(values)
if program.MUTATE:
    with open("program.py", "a") as f:
        f.write("# evaluation rewrote source\\n")
if len(sys.argv) > 2:
    with open(sys.argv[2], "a") as f:
        f.write(split + "\\n")
print(json.dumps({"score": score}))
"""
GENERATOR = """import difflib, json, pathlib, re, sys, time
request = json.load(sys.stdin)
mode, amount, label, delay, markers = sys.argv[1:]
if markers != "-":
    marker_dir = pathlib.Path(markers)
    (marker_dir / ("started-" + label)).touch()
    deadline = time.monotonic() + 5
    while len(list(marker_dir.glob("started-*"))) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError("second generator did not run concurrently")
        time.sleep(.01)
time.sleep(float(delay))
name = "program.py"
old = request["files"][name]
limit = int(re.search(r"^LIMIT = (\\d+)", old)[1])
target = int(amount) + (limit if mode == "add" else 0)
new = re.sub(r"^LIMIT = \\d+", "LIMIT = " + str(target), old)
if mode == "failcheck":
    new = new.replace("return max(0, min(value, LIMIT))", "return -1")
elif mode == "badpath":
    name = "test_check.py"
    old = pathlib.Path(name).read_text()
    new = "# disabled checks\\n"
elif mode == "dirty":
    pathlib.Path(name).write_text(new)
elif mode == "confirmfail":
    new = new.replace("BROKEN_CONFIRM = False", "BROKEN_CONFIRM = True")
elif mode == "mutate":
    new = new.replace("MUTATE = False", "MUTATE = True")
elif mode == "delete":
    new = ""
if markers != "-":
    (pathlib.Path(markers) / ("finished-" + label)).write_text(str(time.monotonic()))
sys.stdout.write("".join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), fromfile="a/" + name,
    tofile="/dev/null" if mode == "delete" else "b/" + name)))
"""


class SourceRun(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gama-rsi-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "program.py").write_text(SOURCE)
        (self.repo / "test_check.py").write_text(CHECK)
        self.scorer = self.root / "score.py"
        self.scorer.write_text(SCORER)
        self.generator = self.root / "agent.py"
        self.generator.write_text(GENERATOR)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=RSI test", "-c", "user.email=test@localhost",
                 "commit", "-qm", "seed")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.state_dir = self.root / "run"
        self.config = {
            "goal": "Return each nonnegative input unchanged, retaining the negative-input floor.",
            "allowed_paths": ["program.py"],
            "agents": [self.agent("small", 2), self.agent("large", 4)],
            "checks": [[sys.executable, "-B", "test_check.py"]],
            "search_command": [sys.executable, "-B", str(self.scorer), "search"],
            "confirm_command": [sys.executable, "-B", str(self.scorer), "confirm"],
            "sealed_command": [sys.executable, "-B", str(self.scorer), "sealed"],
            "evaluation_files": [str(self.scorer)],
            "workers": 2, "batch_size": 2, "confirm_repeats": 2,
            "timeout": 10, "evaluation_timeout": 5, "seed": 6,
        }

    def git(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, env=env, text=True, stderr=subprocess.PIPE,
        )

    def agent(self, name, amount, *, mode="set", delay=0, markers="-"):
        return {"name": name, "command": [
            sys.executable, "-B", str(self.generator),
            mode, str(amount), name, str(delay), str(markers),
        ]}

    def run_source(self, **kwargs):
        return run_rsi(self.config, repo=self.repo, state_dir=self.state_dir, **kwargs)

    def state(self):
        return json.loads((self.state_dir / "state.json").read_text())

    def events(self):
        return [json.loads(row) for row in (self.state_dir / "events.jsonl").read_text().splitlines()]

    def assert_no_worktrees(self):
        lines = self.git("worktree", "list", "--porcelain").splitlines()
        self.assertEqual([x for x in lines if x.startswith("worktree ")],
                         ["worktree " + str(self.repo)])

    def test_parallel_source_improvement_keeps_nonwinner_and_callers_dirty_checkout(self):
        markers = self.root / "markers"
        markers.mkdir()
        self.config["agents"] = [
            self.agent("small", 2, delay=.15, markers=markers),
            self.agent("large", 4, markers=markers),
        ]
        (self.repo / "program.py").write_text(SOURCE + "# user's staged work\n")
        self.git("add", "program.py")
        (self.repo / "notes.txt").write_text("user's untracked work")
        before = self.git("status", "--porcelain")
        staged = self.git("diff", "--cached")
        result = self.run_source()
        archive = self.state()["archive"]
        self.assertEqual(result["champion"]["id"], "r0000-s001")
        self.assertEqual([e["id"] for e in archive], ["seed", "r0000-s000", "r0000-s001"])
        self.assertEqual(archive[1]["parent"], "seed")
        self.assertEqual(result["archive_size"], 3)
        self.assertGreater(result["champion"]["confirm"]["score"], archive[0]["confirm"]["score"])
        self.assertIn("+LIMIT = 4", Path(result["patch"]).read_text())
        self.assertEqual(self.git("rev-parse", result["ref"]).strip(), result["champion"]["commit"])
        self.assertEqual(self.git("rev-parse", result["ref"] + "^").strip(), self.base)
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.base)
        self.assertEqual(self.git("status", "--porcelain"), before)
        self.assertEqual(self.git("diff", "--cached"), staged)
        self.assertEqual((self.repo / "notes.txt").read_text(), "user's untracked work")
        self.assertLess(float((markers / "finished-large").read_text()),
                        float((markers / "finished-small").read_text()))
        self.assertEqual([e["id"] for e in self.events() if e["event"] == "candidate"],
                         ["r0000-s000", "r0000-s001"])
        self.assert_no_worktrees()

    def test_duplicate_source_is_scored_once_before_confirmation(self):
        log = self.root / "score-calls"
        self.config["search_command"].append(str(log))
        self.config["agents"] = [self.agent("a", 2), self.agent("b", 2)]
        result = self.run_source()
        self.assertEqual(result["archive_size"], 2)
        candidates = [e for e in self.events() if e["event"] == "candidate"]
        self.assertEqual([e["status"] for e in candidates], ["viable", "duplicate"])
        self.assertEqual(log.read_text().splitlines(), ["search", "search"])
        self.assert_no_worktrees()

    def test_bad_patch_failed_checks_dirty_generator_and_rewritten_evaluation_are_rejected(self):
        self.config["agents"] = [
            self.agent("unauthorized", 4, mode="badpath"),
            self.agent("broken", 4, mode="failcheck"),
            self.agent("direct-write", 4, mode="dirty"),
            self.agent("rewrites-source", 4, mode="mutate"),
        ]
        self.config["batch_size"] = 4
        result = self.run_source()
        self.assertEqual(result["archive_size"], 1)
        self.assertEqual(result["champion"]["commit"], self.base)
        self.assertEqual(Path(result["patch"]).read_text(), "")
        candidates = [e for e in self.events() if e["event"] == "candidate"]
        self.assertEqual([e["status"] for e in candidates], ["rejected"] * 4)
        self.assertEqual([e["stage"] for e in candidates],
                         ["proposal", "evaluation", "proposal", "evaluation"])
        self.assert_no_worktrees()

    def test_failed_confirmation_keeps_viable_source_but_does_not_promote(self):
        self.config["agents"] = [self.agent("confirmation-broken", 4, mode="confirmfail")]
        self.config["batch_size"] = 1
        result = self.run_source()
        self.assertEqual(result["archive_size"], 2)
        self.assertEqual(result["champion"]["commit"], self.base)
        self.assertEqual(result["rounds_completed"], 1)
        self.assertIn("confirmation measurement failed", self.state()["archive"][1]["confirm_error"])
        self.assertTrue(any(e["event"] == "confirmation_failed" for e in self.events()))
        self.assert_no_worktrees()

    def test_resume_evolves_archived_source_and_advances_champion_ref(self):
        self.config["agents"] = [self.agent("a", 2, mode="add"), self.agent("b", 3, mode="add")]
        first = self.run_source()
        first_ref = first["ref"]
        second = self.run_source(resume=True, rounds=2)
        state = self.state()
        self.assertEqual(second["rounds_completed"], 3)
        self.assertEqual(second["reserved_proposals"], 6)
        self.assertNotEqual(second["ref"], first_ref)
        self.assertEqual(self.git("rev-parse", first_ref).strip(), first["champion"]["commit"])
        by_id = {e["id"]: e for e in state["archive"]}
        later = [e for e in state["archive"] if e["round"] > 0]
        self.assertTrue(any(e["parent"] != "seed" for e in later))
        for entry in later:
            self.assertEqual(self.git("rev-parse", entry["commit"] + "^").strip(),
                             by_id[entry["parent"]]["commit"])
        self.assert_no_worktrees()

    def test_interrupted_uncommitted_round_retries_without_ref_collision_or_lost_budget(self):
        def interrupt(event):
            if event["event"] == "confirmation":
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_source(on_event=interrupt)
        checkpoint = self.state()
        self.assertEqual(checkpoint["next_round"], 0)
        self.assertEqual(checkpoint["reserved_proposals"], 2)
        self.assertIsNotNone(checkpoint["pending"])
        self.assert_no_worktrees()
        result = self.run_source(resume=True)
        self.assertEqual(result["rounds_completed"], 1)
        self.assertEqual(result["reserved_proposals"], 4)
        self.assertEqual(result["archive_size"], 3)
        self.assertIsNone(self.state()["pending"])
        self.assertEqual(len(list((self.state_dir / "attempts").iterdir())), 2)
        self.assert_no_worktrees()

    def test_completed_round_survives_interrupt_and_contract_changes_are_rejected(self):
        def interrupt(event):
            if event["event"] == "round_complete":
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_source(on_event=interrupt)
        self.assertEqual(self.state()["next_round"], 1)
        original = copy.deepcopy(self.config)
        self.config["goal"] += " changed goal"
        with self.assertRaisesRegex(RSIError, "changed"):
            self.run_source(resume=True)
        self.config = original
        with patch("gama.rsi.controller_fingerprint", return_value="different-controller"):
            with self.assertRaisesRegex(RSIError, "changed"):
                self.run_source(resume=True)
        self.scorer.write_text(SCORER + "# evaluator changed\n")
        with self.assertRaisesRegex(RSIError, "changed"):
            self.run_source(resume=True)
        self.scorer.write_text(SCORER)
        result = self.run_source(resume=True)
        self.assertEqual(result["rounds_completed"], 2)
        self.assert_no_worktrees()

    def test_holdout_opens_only_on_finalization_and_permanently_closes_search(self):
        log = self.root / "sealed-calls"
        self.config["sealed_command"].append(str(log))
        first = self.run_source()
        self.assertEqual(first["sealed_verdict"], "not_opened")
        self.assertFalse(log.exists())
        result = self.run_source(resume=True, finalize=True)
        self.assertEqual(result["phase"], "finalized")
        self.assertEqual(result["sealed_verdict"], "improved")
        self.assertEqual(len(log.read_text().splitlines()), 4)
        again = self.run_source(resume=True, finalize=True)
        self.assertEqual(again, result)
        self.assertEqual(len(log.read_text().splitlines()), 4)
        with self.assertRaisesRegex(RSIError, "cannot search"):
            self.run_source(resume=True)
        self.assert_no_worktrees()

    def test_failed_holdout_cannot_resume_search(self):
        self.config["sealed_command"] = [sys.executable, "-c", "raise RuntimeError('holdout down')"]
        self.run_source()
        with self.assertRaisesRegex(RuntimeError, "holdout down"):
            self.run_source(resume=True, finalize=True)
        self.assertEqual(self.state()["phase"], "finalizing")
        with self.assertRaisesRegex(RSIError, "cannot search"):
            self.run_source(resume=True)
        self.assert_no_worktrees()

    def test_run_lock_and_existing_state_prevent_duplicate_coordinators(self):
        import fcntl
        self.state_dir.mkdir()
        with (self.state_dir / "run.lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RSIError, "another RSI coordinator"):
                self.run_source()
            self.assertFalse((self.state_dir / "state.json").exists())
        self.run_source()
        with self.assertRaisesRegex(RSIError, "already exists"):
            self.run_source()

    def test_seed_measurement_failure_is_not_a_zero_score(self):
        self.config["search_command"] = [sys.executable, "-c", "print('not JSON')"]
        with self.assertRaisesRegex(RuntimeError, "JSON"):
            self.run_source()
        state = self.state()
        self.assertEqual(state["phase"], "initializing")
        self.assertEqual(state["archive"], [])
        self.assertEqual(state["reserved_proposals"], 0)
        self.assert_no_worktrees()

    def test_clean_commit_swap_during_seed_checks_is_not_scored_as_the_seed(self):
        (self.repo / "program.py").write_text(SOURCE.replace("LIMIT = 1", "LIMIT = 4"))
        self.git("add", "program.py")
        self.git("-c", "user.name=RSI test", "-c", "user.email=test@localhost",
                 "commit", "-qm", "different source")
        alternate = self.git("rev-parse", "HEAD").strip()
        self.git("checkout", "--detach", self.base)
        self.config["checks"].insert(0, ["git", "checkout", "--detach", alternate])
        with self.assertRaisesRegex(RuntimeError, "commit"):
            self.run_source()
        self.assertEqual(self.state()["archive"], [])
        self.assert_no_worktrees()

    def test_evaluator_drift_aborts_before_dispatch_without_archiving_mixed_scores(self):
        def change_evaluator(event):
            if event["event"] == "round_start":
                self.scorer.write_text(SCORER + "# changed after seed measurement\n")
        with self.assertRaisesRegex(RSIError, "evaluator input changed"):
            self.run_source(on_event=change_evaluator)
        self.assertEqual(len(self.state()["archive"]), 1)
        self.assertEqual(self.state()["next_round"], 0)
        self.assert_no_worktrees()

    def test_resume_preserves_truncated_ledger_fragment_and_restores_jsonl(self):
        self.run_source()
        fragment = b'{"event": "interrupted", "unicode": "\xe3'
        with (self.state_dir / "events.jsonl").open("ab") as fh:
            fh.write(fragment)
        self.run_source(resume=True)
        recovered = [e for e in self.events() if e["event"] == "resumed"]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(Path(recovered[0]["recovered_event_fragment"]).read_bytes(), fragment)
        self.assertEqual(self.state()["next_round"], 2)
        self.assert_no_worktrees()

    def test_new_run_does_not_overwrite_unrelated_existing_artifacts(self):
        self.state_dir.mkdir()
        artifact = self.state_dir / "result.json"
        artifact.write_text('{"user": "original result"}')
        with self.assertRaisesRegex(RSIError, "empty state directory"):
            self.run_source()
        self.assertEqual(artifact.read_text(), '{"user": "original result"}')
        self.assertFalse((self.state_dir / "state.json").exists())

    def test_configured_evaluator_inputs_cannot_be_mutation_targets(self):
        (self.repo / "floor.py").write_text(CHECK)
        (self.repo / "judge.py").write_text(SCORER)
        (self.repo / "data.json").write_text('{"fixed": true}')
        self.git("add", ".")
        self.git("-c", "user.name=RSI test", "-c", "user.email=test@localhost",
                 "commit", "-qm", "fixed evaluator inputs")
        original = copy.deepcopy(self.config)
        overlaps = [
            ("floor.py", {"checks": [[sys.executable, "./floor.py"]]}),
            ("judge.py", {"search_command": [sys.executable, "judge.py", "search"]}),
            ("judge.py", {"search_command": [sys.executable, "-m", "judge", "search"]}),
            ("judge.py", {"search_command": [sys.executable, "-mjudge", "search"]}),
            ("judge.py", {"search_command": [sys.executable, "-IBmjudge", "search"]}),
            ("judge.py", {"search_command": [sys.executable, "-IBm", "judge", "search"]}),
            ("judge.py", {"confirm_command": [
                sys.executable, str(self.repo / "judge.py"), "confirm"]}),
            ("data.json", {"evaluation_files": [str(self.repo / "data.json")]}),
        ]
        for index, (source, change) in enumerate(overlaps):
            with self.subTest(source=source, change=change):
                self.state_dir = self.root / f"overlap-{index}"
                self.config = {**original, "allowed_paths": ["program.py", source], **change}
                with self.assertRaisesRegex(ValueError, "evaluator input cannot be mutated"):
                    self.run_source()
                self.assertFalse((self.state_dir / "state.json").exists())

    def test_deleted_editable_source_is_rejected_before_archive_admission(self):
        self.config["agents"] = [self.agent("deletes-source", 4, mode="delete")]
        self.config["batch_size"] = 1
        self.config["checks"] = [[sys.executable, "-c", "pass"]]
        score = [sys.executable, "-c", "print('{\"score\": 0.5}')"]
        self.config["search_command"] = [*score, "search"]
        self.config["confirm_command"] = [*score, "confirm"]
        result = self.run_source(rounds=2)
        self.assertEqual(result["archive_size"], 1)
        self.assertEqual(result["rounds_completed"], 2)
        candidates = [e for e in self.events() if e["event"] == "candidate"]
        self.assertEqual([e["status"] for e in candidates], ["rejected", "rejected"])
        for entry in candidates:
            self.assertIn("tracked regular file", entry["error"])
            self.assertTrue((Path(entry["artifact_dir"]) / "result.json").exists())
        self.assert_no_worktrees()

    def test_unreadable_parent_request_is_recorded_without_aborting_its_sibling(self):
        from gama.rsi_workspace import Workspaces, WorkspaceError
        read_sources = Workspaces.read_sources

        def unreadable(workspace, path, *args, **kwargs):
            if Path(path).name.startswith("r0000-s000-"):
                raise WorkspaceError("source became unreadable")
            return read_sources(workspace, path, *args, **kwargs)

        with patch.object(Workspaces, "read_sources", unreadable):
            result = self.run_source()
        self.assertEqual(result["archive_size"], 2)
        failures = [e for e in self.events() if e["event"] == "candidate"
                    and e["status"] == "rejected"]
        self.assertEqual(len(failures), 1)
        self.assertTrue((Path(failures[0]["artifact_dir"]) / "result.json").exists())
        self.assert_no_worktrees()

    def test_interrupted_export_preserves_the_previous_results_immutable_patch(self):
        import gama.rsi as rsi
        self.config["agents"] = [self.agent("a", 2, mode="add"), self.agent("b", 3, mode="add")]
        first = self.run_source()
        published = (self.state_dir / "result.json").read_bytes()
        old_patch = Path(first["patch"]).read_bytes()
        atomic_json = rsi._atomic_json

        def fail_summary(path, value):
            if path.name == "result.json" and path.parent == self.state_dir:
                raise KeyboardInterrupt()
            return atomic_json(path, value)

        with patch.object(rsi, "_atomic_json", fail_summary):
            with self.assertRaises(KeyboardInterrupt):
                self.run_source(resume=True, rounds=2)
        self.assertEqual((self.state_dir / "result.json").read_bytes(), published)
        self.assertEqual(Path(first["patch"]).read_bytes(), old_patch)
        self.assertEqual(self.state()["next_round"], 3)
        self.assert_no_worktrees()

    def test_abrupt_coordinator_exit_recovers_owned_worktrees_under_the_run_lock(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        script = (
            "import json, os, sys\n"
            "from gama.rsi import run_rsi\n"
            "from gama.rsi_workspace import Workspaces\n"
            "def crash(*args, **kwargs):\n"
            "    os._exit(71)\n"
            "Workspaces.read_sources = crash\n"
            "run_rsi(json.load(open(sys.argv[1])), repo=sys.argv[2], state_dir=sys.argv[3])\n"
        )
        died = subprocess.run(
            [sys.executable, "-B", "-c", script, str(config_path),
             str(self.repo), str(self.state_dir)],
            cwd=Path(__file__).resolve().parents[1], timeout=20, capture_output=True, text=True,
        )
        self.assertEqual(died.returncode, 71, died.stderr)
        self.assertEqual(self.state()["phase"], "initializing")
        trees = [line.removeprefix("worktree ") for line in
                 self.git("worktree", "list", "--porcelain").splitlines()
                 if line.startswith("worktree ")]
        orphan = next(path for path in trees if path != str(self.repo))
        result = self.run_source(resume=True)
        self.assertEqual(result["rounds_completed"], 1)
        self.assertFalse(Path(orphan).exists())
        recovery = next(e for e in self.events() if e["event"] == "resumed")
        self.assertEqual(recovery["recovered_worktrees"], [orphan])
        self.assert_no_worktrees()

    def test_reexport_preserves_crlf_source_patch_bytes(self):
        (self.repo / "program.py").write_bytes(SOURCE.replace("\n", "\r\n").encode())
        self.git("-c", "core.autocrlf=false", "add", "program.py")
        self.git("-c", "user.name=RSI test", "-c", "user.email=test@localhost",
                 "commit", "-qm", "CRLF source")
        self.base = self.git("rev-parse", "HEAD").strip()
        first = self.run_source()
        exported = Path(first["patch"]).read_bytes()
        self.assertIn(b"\r\n", exported)
        again = self.run_source(resume=True)
        self.assertEqual(again["patch"], first["patch"])
        self.assertEqual(Path(again["patch"]).read_bytes(), exported)
        self.assertEqual(Path(again["latest_patch"]).read_bytes(), exported)
        self.assert_no_worktrees()

    def test_shared_git_worktree_mutations_do_not_overlap_between_workers(self):
        from gama.rsi_workspace import Workspaces
        counter_lock = threading.Lock()
        running = 0
        peak = 0

        def track(operation):
            def measured(*args, **kwargs):
                nonlocal running, peak
                with counter_lock:
                    running += 1
                    peak = max(peak, running)
                try:
                    # Expose overlapping registry scans without relying on Git's
                    # filesystem race occurring on this particular machine.
                    time.sleep(.02)
                    return operation(*args, **kwargs)
                finally:
                    with counter_lock:
                        running -= 1
            return measured

        with patch.object(Workspaces, "create", track(Workspaces.create)), \
                patch.object(Workspaces, "remove", track(Workspaces.remove)):
            result = self.run_source()
        self.assertEqual(result["archive_size"], 3)
        self.assertEqual(peak, 1)
        self.assert_no_worktrees()

    def test_cli_runs_and_reports_reviewable_patch(self):
        from gama.cli import main
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["rsi", "--config", str(config_path), "--repo", str(self.repo),
                         "--state-dir", str(self.state_dir)])
        self.assertEqual(code, 0, err.getvalue())
        result = json.loads(out.getvalue())
        self.assertTrue(Path(result["patch"]).is_file())
        self.assertIn("promoted", err.getvalue())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["rsi", "--config", str(config_path) + ".missing",
                                   "--state-dir", str(self.root / "missing")]), 2)


class ConfigurationAndParents(unittest.TestCase):
    def configuration(self):
        return {
            "goal": "Improve code", "allowed_paths": ["gama/backends.py"],
            "agents": [{"name": "local", "backend": {"backend": "echo"}}],
            "checks": [["python", "-m", "unittest"]],
            "search_command": ["score", "search"], "confirm_command": ["score", "confirm"],
        }

    def test_invalid_configurations_rejected_before_work(self):
        invalid = [
            {"allowed_paths": [["not a string"]]}, {"allowed_paths": ["."]},
            {"allowed_paths": ["tests/test_x.py"]}, {"allowed_paths": ["gama/rsi.py"]},
            {"allowed_paths": ["test_check.py"]}, {"allowed_paths": ["nested/tests/check.py"]},
            {"allowed_paths": ["../source.py"]}, {"allowed_paths": ["/source.py"]},
            {"allowed_paths": ["a.py", "a.py"]}, {"allowed_paths": ["*.py"]},
            {"allowed_paths": ["a/\n.py"]}, {"allowed_paths": [".Git/config"]},
            {"workers": 0}, {"workers": True}, {"batch_size": 1.5},
            {"timeout": float("nan")}, {"evaluation_timeout": -1},
            {"confirm_repeats": 0}, {"min_gain": float("inf")}, {"seed": -1},
            {"checks": []}, {"search_command": "python score.py"},
            {"confirm_command": ["score", "search"]}, {"sealed_command": ["score", "confirm"]},
            {"agents": [{"name": 8, "command": ["echo", "patch"]}]},
            {"agents": [{"name": "a", "backend": {"backend": "echo"}, "command": ["echo"]}]},
            {"evaluation_files": ["relative-input.json"]}, {"typo": 1},
            {"papers": [{"title": "DGM", "url": "https://arxiv.org/abs/2505.22954"}]},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    RSIConfig.from_dict({**self.configuration(), **changes})
        config = RSIConfig.from_dict(self.configuration())
        self.assertEqual(RSIConfig.from_dict(asdict(config)), config)

    def test_dgm_weights_use_functioning_direct_children_and_stable_sampling(self):
        archive = [
            {"id": "seed", "parent": None, "search": {"score": .5}},
            {"id": "child", "parent": "seed", "search": {"score": .5}},
            {"id": "grandchild", "parent": "child", "search": {"score": .5}},
            {"id": "perfect", "parent": "seed", "search": {"score": 1}},
        ]
        sample = select_parents(archive, 12000, seed=42, round_number=3)
        counts = {key: sum(e["id"] == key for e in sample) for key in
                  ("seed", "child", "grandchild", "perfect")}
        # weights are 1/6, 1/4, 1/2; a non-winning functioning child still has offspring.
        self.assertEqual(counts["perfect"], 0)
        self.assertTrue(2.8 < counts["grandchild"] / counts["seed"] < 3.2, counts)
        self.assertTrue(1.85 < counts["grandchild"] / counts["child"] < 2.15, counts)
        self.assertEqual(sample, select_parents(list(reversed(archive)), 12000,
                                               seed=42, round_number=3))
        self.assertEqual(select_parents([archive[-1]], 2, seed=0, round_number=0), [])
