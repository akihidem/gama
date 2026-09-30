"""Frozen publication behavior; run from candidate cwd with pinned Python.

Public dependencies: continual_tasks.freeze_goal and continual_publish.publish.
No private publication journal fields are assumed beyond base_commit. Expected
refusals must be explicit, preserve caller bytes/index/refs, and never push.
"""
from __future__ import annotations

from collections import Counter
import copy
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

try:
    from .publish_common import BRANCH, OTHER, SOURCE, Fixture, git, strings
except ImportError:
    from publish_common import BRANCH, OTHER, SOURCE, Fixture, git, strings


class SimulatedCrash(BaseException):
    """Do not let a production `except Exception` turn a crash into progress."""


class PublicationAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.publisher = staticmethod(importlib.import_module("gama.continual_publish").publish)
        cls.freezer = staticmethod(importlib.import_module("gama.continual_tasks").freeze_goal)

    def fixture(self, **kwargs):
        directory = tempfile.TemporaryDirectory(prefix="gama-publish-acceptance-")
        self.addCleanup(directory.cleanup)
        return Fixture(directory.name, freezer=self.freezer, **kwargs)

    def assert_real_improved(self, f):
        self.assertEqual(f.result["phase"], "finalized")
        self.assertEqual(f.result["sealed_verdict"], "improved")
        self.assertNotEqual(f.source, f.base)
        self.assertEqual(f.result["champion"]["confirm"]["samples"], [1.0] * 3)
        self.assertEqual(f.result["sealed"]["champion"]["samples"], [1.0] * 3)
        state = json.loads(f.state_path.read_bytes())
        self.assertEqual(state["phase"], "finalized")
        self.assertEqual(state["sealed_verdict"], "improved")
        self.assertEqual(state["base"], f.base)
        outcomes = [json.loads(p.read_bytes())
                    for p in (f.state / "rsi").rglob("result.json")]
        self.assertTrue(any(row.get("commit") == f.source and row.get("checks")
                            and row.get("status") == "viable" for row in outcomes),
                        "a real winning candidate with mandatory check evidence is required")

    def assert_no_inference(self, f):
        self.assertEqual(f.calls.read_bytes() if f.calls.exists() else b"", f.initial_calls)
        for argv in f.git_commands():
            self.assertNotIn("reset", argv, "publication may never reset a checkout")
            if "push" in argv:
                tail = argv[argv.index("push") + 1:]
                self.assertFalse(any(a.startswith(("--force", "+"))
                                     or (a.startswith("-") and not a.startswith("--")
                                         and "f" in a[1:]) for a in tail),
                                 f"force push is forbidden even when it would succeed: {argv!r}")

    def assert_core_preserved(self, f, before):
        for name, contents in before.items():
            self.assertEqual((f.state / "rsi" / name).read_bytes(), contents,
                             f"publication rewrote core acceptance: {name}")

    def assert_refusal(self, f, *, goal=None):
        before, remote = f.snapshot_checkout(), f.remote_head()
        core = f.core_bytes()
        try:
            result = f.publish(self.publisher, goal=goal)
        except (AssertionError, ImportError, AttributeError, KeyError, TypeError):
            raise
        except Exception as exc:
            self.assertTrue(str(exc).strip(), "refusal needs a concrete reason")
        else:
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get("error") or result.get("reason") or result.get("message"),
                            f"unsafe publication returned without a refusal: {result!r}")
        self.assertEqual(f.snapshot_checkout(), before)
        self.assertEqual(f.remote_head(), remote)
        self.assert_core_preserved(f, core)
        self.assert_no_inference(f)

    def assert_release(self, f, evidence):
        self.assertIsInstance(evidence, dict)
        json.dumps(evidence, allow_nan=False)
        selected = f.head()
        self.assertNotIn(selected, (f.base, f.source), "release must add exact regression artifacts")
        self.assertEqual(f.remote_head(), selected)
        self.assertEqual(git(f.repo, "symbolic-ref", "--short", "HEAD").decode().strip(), BRANCH)
        self.assertEqual(git(f.repo, "rev-parse", "refs/heads/main").decode().strip(), f.base)
        self.assertEqual(git(f.repo, "ls-remote", "origin", "refs/heads/main").decode().split()[0],
                         f.base)
        self.assertFalse(git(f.repo, "status", "--porcelain=v1", "--untracked-files=all"))
        git(f.repo, "merge-base", "--is-ancestor", f.base, selected)
        self.assertEqual(git(f.repo, "rev-list", "--parents", "-n", "1", selected).decode().split(),
                         [selected, f.source], "regression commit must directly preserve source")
        prefix = "validation/continual/" + f.goal["id"] + "/"
        changes = git(f.repo, "diff", "--name-only", "-z", f.source, selected).decode().split("\0")
        self.assertEqual(len([p for p in changes if p]), 4)
        self.assertTrue(all(p.startswith(prefix) for p in changes if p))
        rows = git(f.repo, "ls-tree", "-rz", selected, "--", prefix).split(b"\0")
        payloads = []
        for row in filter(None, rows):
            metadata, name = row.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            self.assertIn(mode, (b"100644", b"100755"))
            self.assertEqual(kind, b"blob")
            payloads.append(git(f.repo, "cat-file", "blob", oid.decode()))
        self.assertEqual(len(payloads), 4)
        scripts = [value.encode("utf-8") for value in f.goal["tests"].values()]
        remaining = Counter(payloads) - Counter(scripts)
        self.assertEqual(sum(remaining.values()), 1)
        self.assertTrue(all(Counter(payloads)[s] == 1 for s in scripts))
        descriptor = next(iter(remaining))
        self.assertIn(descriptor, f.frozen_descriptors)
        self.assertEqual(json.loads(descriptor), f.goal)
        self.assertIn(selected, set(strings(evidence)), "return verified selected commit evidence")
        self.assertTrue(any(selected in set(strings(s)) for s in f.snapshots),
                        "selected release SHA must be durably journaled")
        refs = git(f.repo, "for-each-ref", "--points-at", selected,
                   "--format=%(refname)").decode().splitlines()
        self.assertTrue(any(not ref.startswith(("refs/heads/", "refs/remotes/")) for ref in refs),
                        "release needs a durable owned ref in addition to caller branch")
        probes = [json.loads(line) for line in f.probe.read_text().splitlines()]
        self.assertEqual({row["split"] for row in probes}, {"search", "confirm", "sealed"})
        self.assertTrue(all(Path(row["cwd"]) != f.repo for row in probes),
                        "retained regressions must be validated before caller adoption")
        self.assertTrue(any(isinstance(row.get("receipt"), dict)
                            and row["receipt"].get("version") == 1
                            and row["receipt"].get("parent") == f.source
                            and row["receipt"].get("repo") == str(f.repo / ".git")
                            for row in probes),
                        "release validation must use an owned Workspaces receipt on source SHA")
        self.assert_no_inference(f)
        commands = f.git_commands()
        for index, argv in enumerate(commands):
            if "push" in argv:
                self.assertIn(selected + ":refs/heads/" + BRANCH, argv,
                              "push must explicitly name the journaled SHA and configured branch")
                self.assertTrue(any("ls-remote" in later for later in commands[index + 1:]),
                                "a successful push must be followed by remote SHA verification")
        return selected

    def test_real_sealed_winner_exact_regressions_fastforward_verified_push(self):
        f = self.fixture()
        self.assert_real_improved(f)
        core = f.core_bytes()
        evidence = f.publish(self.publisher)
        self.assert_release(f, evidence)
        self.assert_core_preserved(f, core)

    def test_successful_retry_reuses_selected_release_and_budget(self):
        f = self.fixture()
        first = f.publish(self.publisher)
        selected = self.assert_release(f, first)
        refs = git(f.repo, "show-ref")
        f.reload()
        second = f.publish(self.publisher)
        self.assertEqual(self.assert_release(f, second), selected)
        self.assertEqual(git(f.repo, "show-ref"), refs)

    def test_crashes_at_each_durable_checkpoint_reconcile_without_second_release(self):
        # First discover checkpoint count/shape through the documented callback.
        probe = self.fixture()
        probe.publish(self.publisher)
        self.assertGreaterEqual(len(probe.snapshots), 1)
        for index, when in ((i, when) for i in range(len(probe.snapshots))
                            for when in ("before", "after")):
            with self.subTest(checkpoint=index, crash=when):
                f = self.fixture()
                f.save(f.journal)
                seen = 0
                crashed = False
                def save(journal):
                    nonlocal seen, crashed
                    current = seen
                    seen += 1
                    if current == index and when == "before":
                        crashed = True
                        raise SimulatedCrash()
                    f.save(journal)
                    if current == index:
                        crashed = True
                        raise SimulatedCrash()
                with self.assertRaises(SimulatedCrash):
                    f.publish(self.publisher, save=save)
                self.assertTrue(crashed)
                saved = copy.deepcopy(f.snapshots[-1])
                selected = []
                for value in strings(saved):
                    if len(value) == 40 and all(c in "0123456789abcdef" for c in value):
                        if value not in (f.base, f.source):
                            try:
                                paths = git(f.repo, "ls-tree", "-r", "--name-only", value).decode()
                            except AssertionError:
                                continue
                            if "validation/continual/" in paths:
                                selected.append(value)
                f.reload()
                evidence = f.publish(self.publisher)
                actual = self.assert_release(f, evidence)
                self.assertTrue(all(value == actual for value in selected),
                                "a journaled release may never be replaced on retry")

    def test_stop_after_selection_blocks_adoption_and_push_then_resume_same_sha(self):
        f = self.fixture()
        selected = []
        def save(journal):
            f.save(journal)
            if f.head() != f.base:
                self.fail("adopted before durably selecting release SHA")
            for value in strings(journal):
                if len(value) == 40 and value not in (f.base, f.source):
                    try:
                        names = git(f.repo, "ls-tree", "-r", "--name-only", value)
                    except AssertionError:
                        continue
                    if b"validation/continual/" in names:
                        selected.append(value)
                        f.cancel.set()
        try:
            f.publish(self.publisher, save=save)
        except (AssertionError, ImportError, AttributeError, KeyError, TypeError):
            raise
        except Exception:
            pass
        self.assertTrue(selected, "release selection must be saved before adoption")
        self.assertEqual(f.head(), f.base)
        self.assertEqual(f.remote_head(), f.base)
        f.cancel.clear()
        f.reload()
        self.assertEqual(self.assert_release(f, f.publish(self.publisher)), selected[0])

    def test_cancel_already_set_preserves_checkout_and_remote(self):
        f = self.fixture()
        f.cancel.set()
        self.assert_refusal(f)

    def test_cancel_after_clean_adoption_prevents_push_and_can_resume(self):
        f = self.fixture()
        class StopAfterAdoption(threading.Event):
            def is_set(self):
                if (not super().is_set() and f.head() != f.base
                        and not git(f.repo, "status", "--porcelain=v1")):
                    self.set()
                return super().is_set()
        f.cancel = StopAfterAdoption()
        try:
            f.publish(self.publisher)
        except (AssertionError, ImportError, AttributeError, KeyError, TypeError):
            raise
        except Exception:
            pass
        self.assertTrue(f.cancel.is_set(), "must reach clean local adoption")
        selected = f.head()
        self.assertNotEqual(selected, f.base)
        self.assertEqual(f.remote_head(), f.base, "STOP must be checked again before push")
        f.cancel = threading.Event()
        f.reload()
        self.assertEqual(self.assert_release(f, f.publish(self.publisher)), selected)

    def test_retained_regression_failure_prevents_adoption_and_push(self):
        f = self.fixture()
        self.assert_real_improved(f)
        with patch.dict("os.environ", {"GAMA_PUBLICATION_ACCEPTANCE_FAIL": "1"}):
            self.assert_refusal(f)

    def test_search_perfect_but_not_finalized_is_refused(self):
        f = self.fixture(finalized=False)
        self.assertEqual(f.result["phase"], "ready")
        self.assertEqual(f.result["champion"]["search"]["score"], 1)
        self.assertEqual(f.result["sealed_verdict"], "not_opened")
        self.assert_refusal(f)

    def test_real_finalized_not_improved_is_refused(self):
        f = self.fixture(mode="unimproved")
        self.assertEqual(f.result["phase"], "finalized")
        self.assertEqual(f.result["sealed_verdict"], "not_separable")
        self.assert_refusal(f)

    def test_missing_serial_confirmation_is_refused(self):
        f = self.fixture()
        for path in (f.result_path, f.state_path):
            value = json.loads(path.read_bytes())
            if path == f.result_path:
                value["champion"].pop("confirm", None)
            else:
                for row in value["archive"]:
                    if row["id"] == value["champion"]:
                        row.pop("confirm", None)
            path.write_text(json.dumps(value))
        self.assert_refusal(f)

    def test_missing_sealed_measurement_is_refused(self):
        f = self.fixture()
        result = json.loads(f.result_path.read_bytes())
        result["sealed"] = None
        f.result_path.write_text(json.dumps(result))
        state = json.loads(f.state_path.read_bytes())
        state.pop("sealed_champion")
        f.state_path.write_text(json.dumps(state))
        self.assert_refusal(f)

    def test_missing_mandatory_check_evidence_is_refused(self):
        f = self.fixture()
        changed = False
        for path in (f.state / "rsi").rglob("result.json"):
            row = json.loads(path.read_bytes())
            if row.get("commit") == f.source and "checks" in row:
                row.pop("checks")
                path.write_text(json.dumps(row))
                changed = True
        self.assertTrue(changed)
        # The core also records candidate checks in its append-only event log.
        # Remove the same evidence there, so recovery cannot legitimately find it.
        for path in f.state.rglob("events.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            for row in rows:
                if row.get("commit") == f.source:
                    row.pop("checks", None)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assert_refusal(f)

    def test_candidate_delta_outside_goal_paths_is_refused(self):
        f = self.fixture(mode="extra")
        self.assertEqual(set(git(f.repo, "diff", "--name-only", f.base, f.source)
                             .decode().splitlines()), {SOURCE, OTHER})
        goal = copy.deepcopy(f.goal)
        goal["allowed_paths"] = [SOURCE]
        self.assert_refusal(f, goal=goal)

    def test_goal_traversal_and_protected_source_are_refused(self):
        for paths in (["../escape.py"], ["/tmp/escape.py"], ["gama/continual_publish.py"],
                      ["gama/rsi.py"], ["gama/*.py"]):
            with self.subTest(paths=paths):
                f = self.fixture()
                goal = copy.deepcopy(f.goal)
                goal["allowed_paths"] = paths
                self.assert_refusal(f, goal=goal)

    def test_goal_id_cannot_escape_regression_directory(self):
        f = self.fixture()
        goal = copy.deepcopy(f.goal)
        goal["id"] = "../../operator-files"
        self.assert_refusal(f, goal=goal)
        self.assertFalse((f.repo / "operator-files").exists())

    def test_dirty_tracked_staged_and_untracked_work_is_preserved(self):
        for kind in ("unstaged", "staged", "untracked"):
            with self.subTest(kind=kind):
                f = self.fixture()
                path = f.repo / ("operator-notes.txt" if kind == "untracked" else SOURCE)
                path.write_bytes(path.read_bytes() + b"# keep operator edits\n"
                                 if path.exists() else b"private operator notes\n")
                if kind == "staged":
                    git(f.repo, "add", SOURCE)
                before = path.read_bytes()
                self.assert_refusal(f)
                self.assertEqual(path.read_bytes(), before)

    def test_unexpected_local_head_and_nonowned_branch_are_refused(self):
        for wrong_branch in (False, True):
            with self.subTest(wrong_branch=wrong_branch):
                f = self.fixture()
                if wrong_branch:
                    git(f.repo, "checkout", "-q", "main")
                else:
                    git(f.repo, "commit", "--allow-empty", "-qm", "operator advancement")
                self.assert_refusal(f)

    def test_campaign_base_must_equal_core_base(self):
        f = self.fixture()
        f.journal["base_commit"] = f.source
        self.assert_refusal(f)

    def test_remote_divergence_preserved_and_no_force_push(self):
        f = self.fixture()
        tree = git(f.repo, "rev-parse", f.base + "^{tree}").decode().strip()
        other = git(f.repo, "commit-tree", tree, "-p", f.base,
                    data=b"independent remote advancement\n").decode().strip()
        git(f.repo, "push", "-q", "origin", other + ":refs/heads/" + BRANCH)
        self.assert_refusal(f)
        self.assertEqual(f.remote_head(), other)

    def test_completed_journal_does_not_hide_later_remote_advancement(self):
        f = self.fixture()
        selected = self.assert_release(f, f.publish(self.publisher))
        tree = git(f.repo, "rev-parse", selected + "^{tree}").decode().strip()
        other = git(f.repo, "commit-tree", tree, "-p", selected,
                    data=b"operator advanced the published remote\n").decode().strip()
        git(f.repo, "push", "-q", "origin", other + ":refs/heads/" + BRANCH)
        f.reload()
        self.assert_refusal(f)
        self.assertEqual(f.head(), selected)
        self.assertEqual(f.remote_head(), other)

    def test_remote_push_lost_acknowledgment_reconciles_same_selected_commit(self):
        f = self.fixture()
        f.lose_push_reply.touch()
        try:
            f.publish(self.publisher)
        except (AssertionError, ImportError, AttributeError, KeyError, TypeError):
            raise
        except Exception:
            pass
        self.assertFalse(f.lose_push_reply.exists(), "must exercise a real successful Git push")
        selected = f.head()
        self.assertNotEqual(selected, f.base)
        self.assertEqual(f.remote_head(), selected)
        f.reload()
        self.assertEqual(self.assert_release(f, f.publish(self.publisher)), selected)

    def test_push_failure_after_adoption_then_retry_preserves_selected_commit(self):
        f = self.fixture()
        hook = f.remote / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\ncat >/dev/null\nexit 1\n")
        hook.chmod(0o755)
        try:
            f.publish(self.publisher)
        except (AssertionError, ImportError, AttributeError, KeyError, TypeError):
            raise
        except Exception:
            pass
        self.assertNotEqual(f.head(), f.base, "fixture must reach local adoption before failed push")
        selected = f.head()
        self.assertEqual(f.remote_head(), f.base)
        self.assertTrue(any(selected in set(strings(row)) for row in f.snapshots))
        hook.unlink()
        f.reload()
        self.assertEqual(self.assert_release(f, f.publish(self.publisher)), selected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
