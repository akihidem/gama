"""Exercise the RSI adapter against real temporary repositories, without models."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

from gama import rsi_workspace
from gama.rsi_workspace import WorkspaceError, Workspaces


PATCH = """diff --git a/main.py b/main.py
--- a/main.py
+++ b/main.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 42
"""
OTHER_PATCH = """diff --git a/other.py b/other.py
--- a/other.py
+++ b/other.py
@@ -1 +1 @@
-OTHER = 2
+OTHER = 99
"""


def git(path, *args, data=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, LC_ALL="C")
    result = subprocess.run(
        ["git", "--no-pager", "-c", f"core.hooksPath={os.devnull}",
         "-c", "user.name=Fixture", "-c", "user.email=fixture@localhost",
         "-c", "commit.gpgSign=false", *args],
        cwd=path, env=env, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=False, timeout=15,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


class TestWorkspaces(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gama-workspace-tests-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.repo = self.directory / "caller checkout"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "--template=")
        (self.repo / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.repo / "other.py").write_text("OTHER = 2\n", encoding="utf-8")
        (self.repo / "pkg").mkdir()
        (self.repo / "pkg/with space.py").write_text("VALUE = 3\n", encoding="utf-8")
        (self.repo / ".gitignore").write_text("__pycache__/\n*.cache\n", encoding="utf-8")
        self.base = self.save_base()
        self.root = self.directory / "state/worktrees"
        self.workspaces = Workspaces(self.repo, self.root)
        self.patch_number = 0

    def save_base(self):
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "fixture")
        return git(self.repo, "rev-parse", "HEAD").decode().strip()

    def candidate(self, name="candidate"):
        return self.workspaces.create(name, self.base)

    def generated_patch(self, changes):
        """Let Git generate quoting/binary/add/delete headers used as test inputs."""
        self.patch_number += 1
        path = self.candidate(f"patch-input-{self.patch_number}")
        try:
            for name, content in changes.items():
                target = path / name
                if content is None:
                    target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content if isinstance(content, bytes) else content.encode())
            git(path, "add", "--all")
            return git(
                path, "diff", "--cached", "--binary", "--no-renames", self.base, "--"
            ).decode("utf-8")
        finally:
            self.workspaces.remove(path)

    def assert_pristine(self, path):
        self.assertEqual(git(path, "status", "--porcelain", "--untracked-files=all"), b"")
        self.assertEqual((path / "main.py").read_text(), "VALUE = 1\n")
        self.assertEqual(git(path, "rev-parse", "HEAD").decode().strip(), self.base)

    def test_apply_commit_keep_cleanup_preserve_caller_edits_and_index(self):
        branch = git(self.repo, "symbolic-ref", "HEAD")
        (self.repo / "main.py").write_text("staged caller edit\n")
        git(self.repo, "add", "main.py")
        (self.repo / "main.py").write_text("unstaged caller edit\n")
        (self.repo / "notes.txt").write_text("untracked caller file\n")
        index_before = (self.repo / ".git/index").read_bytes()
        config_before = (self.repo / ".git/config").read_bytes()

        candidate = self.candidate()
        self.assertEqual(self.workspaces.repo, self.repo)
        self.assertEqual(self.workspaces.root, self.root)
        self.assertEqual(self.workspaces.head(), self.base)
        self.assertEqual(self.workspaces.read_sources(candidate, ["main.py"]),
                         {"main.py": "VALUE = 1\n"})
        self.assertEqual(self.workspaces.apply_patch(candidate, PATCH, ["main.py"]), ["main.py"])
        commit = self.workspaces.commit(candidate, self.base, "Improve value")
        self.assertEqual(git(candidate, "rev-parse", "HEAD").decode().strip(), commit)
        self.assertEqual(git(candidate, "rev-list", "--parents", "-n", "1", commit).split(),
                         [commit.encode(), self.base.encode()])
        self.assertEqual(git(candidate, "show", f"{commit}:main.py"), b"VALUE = 42\n")
        self.assertEqual(git(candidate, "show", f"{commit}:other.py"), b"OTHER = 2\n")
        self.assertEqual(git(candidate, "show", "-s", "--format=%an <%ae>", commit).strip(),
                         b"Gama RSI <gama-rsi@localhost>")
        self.workspaces.assert_clean(candidate)
        ref = self.workspaces.keep(commit, "run-1/candidate-0")
        self.assertEqual(ref, "refs/gama-rsi/run-1/candidate-0")
        self.assertEqual(self.workspaces.keep(commit, "run-1/candidate-0"), ref)
        self.workspaces.remove(candidate)
        self.assertFalse(candidate.exists())
        self.assertEqual(self.workspaces.head(), self.base)
        self.assertEqual(git(self.repo, "symbolic-ref", "HEAD"), branch)
        self.assertEqual((self.repo / ".git/index").read_bytes(), index_before)
        self.assertEqual((self.repo / ".git/config").read_bytes(), config_before)
        self.assertEqual((self.repo / "main.py").read_text(), "unstaged caller edit\n")
        self.assertEqual((self.repo / "notes.txt").read_text(), "untracked caller file\n")
        git(self.repo, "reflog", "expire", "--expire=now", "--all")
        git(self.repo, "gc", "--prune=now")
        self.assertEqual(git(self.repo, "rev-parse", ref).decode().strip(), commit)
        self.assertEqual(git(self.repo, "show", f"{ref}:main.py"), b"VALUE = 42\n")

    def test_exported_diff_round_trip_and_tree_dedup(self):
        first = self.candidate("first")
        self.workspaces.apply_patch(first, PATCH, ["main.py"])
        commit = self.workspaces.commit(first, self.base, "first commit")
        exported = self.workspaces.diff(self.base, commit)
        second = self.candidate("second")
        self.workspaces.apply_patch(second, exported, ["main.py"])
        another = self.workspaces.commit(second, self.base, "different metadata")
        self.assertNotEqual(commit, another)
        self.assertEqual(self.workspaces.tree(commit), self.workspaces.tree(another))
        self.assertNotEqual(self.workspaces.tree(commit), self.workspaces.tree(self.base))
        self.assertEqual(self.workspaces.diff(commit, another), "")

    def test_commit_receipt_survives_restart_and_retry(self):
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        restarted = Workspaces(self.repo, self.root)
        commit = restarted.commit(candidate, self.base, "resumable")
        restarted_again = Workspaces(self.repo, self.root)
        self.assertEqual(restarted_again.commit(candidate, self.base, "resumable"), commit)
        self.assertEqual(git(candidate, "rev-parse", "HEAD").decode().strip(), commit)
        with self.assertRaises(WorkspaceError):
            restarted_again.commit(candidate, self.base, "different retry")

    def test_creation_receipt_records_ownership_but_does_not_approve_a_commit(self):
        candidate = self.candidate()
        receipt = json.loads((self.root / ".candidate.applied.json").read_text())
        self.assertEqual(receipt, {
            "version": 1, "name": "candidate", "repo": str(self.repo / ".git"),
            "parent": self.base, "tree": self.workspaces.tree(self.base),
            "paths": [], "commit": self.base, "stage": "created",
        })
        (candidate / "main.py").write_text("VALUE = 42\n")
        git(candidate, "add", "main.py")
        with self.assertRaisesRegex(WorkspaceError, "approved patch"):
            self.workspaces.commit(candidate, self.base, "No approval")
        self.assertEqual(git(candidate, "rev-parse", "HEAD").decode().strip(), self.base)

    def test_recover_created_approved_and_committed_orphans_preserves_unrelated_paths(self):
        created = self.candidate("a-created")
        approved = self.candidate("b-approved")
        self.workspaces.apply_patch(approved, PATCH, ["main.py"])
        committed = self.candidate("c-committed")
        self.workspaces.apply_patch(committed, PATCH, ["main.py"])
        commit = self.workspaces.commit(committed, self.base, "Retained candidate")
        ref = self.workspaces.keep(commit, "recovery/retained")
        (committed / "test-output.log").write_text("untracked output")
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        (unrelated / "marker").write_text("retain")
        manual = self.root / "manual-worktree"
        git(self.repo, "worktree", "add", "--detach", str(manual), self.base)
        external = self.directory / "external-worktree"
        git(self.repo, "worktree", "add", "--detach", str(external), self.base)
        (self.repo / "main.py").write_text("user's staged edit")
        git(self.repo, "add", "main.py")
        (self.repo / "main.py").write_text("user's unstaged edit")
        index = (self.repo / ".git/index").read_bytes()
        config = (self.repo / ".git/config").read_bytes()
        branch = git(self.repo, "symbolic-ref", "HEAD")

        restarted = Workspaces(self.repo, self.root)
        self.assertEqual(restarted.recover(), [created, approved, committed])
        self.assertEqual(restarted.recover(), [])
        for path in (created, approved, committed):
            self.assertFalse(path.exists())
            self.assertFalse((self.root / f".{path.name}.applied.json").exists())
        self.assertEqual((unrelated / "marker").read_text(), "retain")
        self.assertTrue((manual / "main.py").is_file())
        self.assertTrue((external / "main.py").is_file())
        self.assertEqual((self.repo / "main.py").read_text(), "user's unstaged edit")
        self.assertEqual((self.repo / ".git/index").read_bytes(), index)
        self.assertEqual((self.repo / ".git/config").read_bytes(), config)
        self.assertEqual(git(self.repo, "symbolic-ref", "HEAD"), branch)
        self.assertEqual(restarted.head(), self.base)
        self.assertEqual(git(self.repo, "rev-parse", ref).decode().strip(), commit)

    def test_recover_empty_unregistered_reservation_and_missing_path_receipt(self):
        missing = self.candidate("missing")
        reservation = self.candidate("reservation")
        # Git removes the checkout and registration, leaving the adapter's receipt.
        # Recreate only the empty reservation to model a crash before worktree add.
        for path in (missing, reservation):
            git(self.repo, "worktree", "remove", "--force", str(path))
        reservation.mkdir()
        restarted = Workspaces(self.repo, self.root)
        self.assertEqual(restarted.recover(), [missing, reservation])
        self.assertEqual(list(self.root.glob(".*.applied.json")), [])
        self.assertFalse(reservation.exists())
        self.assertEqual(restarted.recover(), [])
        recreated = restarted.create("reservation", self.base)
        self.assertEqual(recreated, reservation)

    def test_recover_registered_worktree_whose_directory_is_missing(self):
        candidate = self.candidate()
        shutil.rmtree(candidate)
        self.assertIn(str(candidate).encode(), git(self.repo, "worktree", "list", "--porcelain"))
        self.assertEqual(Workspaces(self.repo, self.root).recover(), [candidate])
        self.assertNotIn(str(candidate).encode(), git(self.repo, "worktree", "list", "--porcelain"))
        self.assertFalse((self.root / ".candidate.applied.json").exists())

    def test_recover_refuses_nonempty_unregistered_directory_before_any_cleanup(self):
        valid = self.candidate("a-valid")
        unsafe = self.candidate("z-unregistered")
        git(self.repo, "worktree", "remove", "--force", str(unsafe))
        unsafe.mkdir()
        (unsafe / "user-file").write_text("must survive")
        with self.assertRaisesRegex(WorkspaceError, "unregistered nonempty"):
            Workspaces(self.repo, self.root).recover()
        self.assertTrue(valid.is_dir())
        self.assertTrue((self.root / ".a-valid.applied.json").is_file())
        self.assertEqual((unsafe / "user-file").read_text(), "must survive")
        self.assertTrue((self.root / ".z-unregistered.applied.json").is_file())

    def test_recover_refuses_wrong_repository_and_tampered_receipts(self):
        valid = self.candidate("a-valid")
        candidate = self.candidate("z-candidate")
        receipt_path = self.root / ".z-candidate.applied.json"
        original = json.loads(receipt_path.read_text())
        changes = (
            {"repo": str(self.directory / "another-repo/.git")},
            {"name": "a-valid"},
            {"name": "../outside"},
            {"version": True},
            {"parent": "HEAD"},
            {"tree": "invalid-object"},
            {"stage": "approved", "paths": []},
            {"path": str(self.repo)},
        )
        restarted = Workspaces(self.repo, self.root)
        for change in changes:
            receipt_path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(WorkspaceError):
                restarted.recover()
            self.assertTrue(valid.is_dir())
            self.assertTrue(candidate.is_dir())
            self.assertTrue((self.repo / "main.py").is_file())
        for contents in ("[]", '{"version": 1,' + json.dumps(original)[1:], "x" * 128_001):
            receipt_path.write_text(contents)
            with self.subTest(size=len(contents)), self.assertRaises(WorkspaceError):
                restarted.recover()
            self.assertTrue(valid.is_dir())
            self.assertTrue(candidate.is_dir())
        receipt_path.write_text(json.dumps(original))
        self.assertEqual(restarted.recover(), [valid, candidate])

    def test_recover_refuses_invalid_receipt_name_and_symlink(self):
        candidate = self.candidate()
        receipt = self.root / ".candidate.applied.json"
        invalid = self.root / ".bad..name.applied.json"
        invalid.write_bytes(receipt.read_bytes())
        with self.assertRaises(WorkspaceError):
            Workspaces(self.repo, self.root).recover()
        self.assertTrue(candidate.is_dir())
        invalid.unlink()
        outside = self.directory / "outside-receipt.json"
        outside.write_bytes(receipt.read_bytes())
        receipt.unlink()
        receipt.symlink_to(outside)
        with self.assertRaises(WorkspaceError):
            Workspaces(self.repo, self.root).recover()
        self.assertTrue(candidate.is_dir())
        self.assertTrue(outside.is_file())
        self.assertTrue(receipt.is_symlink())

    def test_parallel_worktrees_have_independent_indexes_and_commits(self):
        def generate(slot):
            path = self.candidate(f"parallel-{slot}")
            patch = PATCH.replace("VALUE = 42", f"VALUE = {slot + 10}")
            self.workspaces.apply_patch(path, patch, ["main.py"])
            commit = self.workspaces.commit(path, self.base, f"slot {slot}")
            self.workspaces.keep(commit, f"parallel/{slot}")
            self.workspaces.assert_clean(path)
            return git(path, "show", f"{commit}:main.py")

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(generate, range(3)))
        self.assertEqual(results, [b"VALUE = 10\n", b"VALUE = 11\n", b"VALUE = 12\n"])
        self.assertEqual(self.workspaces.head(), self.base)

    def test_rejects_invalid_repository_and_internal_roots(self):
        invalid = self.directory / "not-a-repo"
        invalid.mkdir()
        for repo, root in (
            (invalid, self.directory / "unused"),
            (self.repo, self.repo),
            (self.repo, self.repo / "state/worktrees"),
            (self.repo, self.repo / ".git/worktrees"),
            (self.repo, self.directory),
        ):
            with self.subTest(repo=repo, root=root), self.assertRaises(WorkspaceError):
                Workspaces(repo, root)
        self.assertFalse((self.directory / "unused").exists())
        self.assertFalse((self.repo / "state").exists())

    def test_repo_subdirectory_is_normalized_to_checkout_root(self):
        adapter = Workspaces(self.repo / "pkg", self.directory / "another-root")
        self.assertEqual(adapter.repo, self.repo)
        self.assertEqual(adapter.head(), self.base)

    def test_rejects_symlinked_repo_root_and_traversal(self):
        alias = self.directory / "repo-alias"
        alias.symlink_to(self.repo, target_is_directory=True)
        root_alias = self.directory / "root-alias"
        root_alias.symlink_to(self.root, target_is_directory=True)
        for repo, root in (
            (alias, self.directory / "unused"),
            (self.repo, root_alias),
            (self.repo, self.directory / "../escape"),
        ):
            with self.subTest(root=root), self.assertRaises(WorkspaceError):
                Workspaces(repo, root)

    def test_validates_names_and_commitish_arguments(self):
        for name in ("", "../escape", "a/b", "-option", ".git", "a..b", "name.lock", "a\nb"):
            with self.subTest(name=name), self.assertRaises(WorkspaceError):
                self.workspaces.create(name, self.base)
        blob = git(self.repo, "rev-parse", "HEAD:main.py").decode().strip()
        for parent in ("missing-ref", "--help", blob, "", "HEAD\n"):
            with self.subTest(parent=parent), self.assertRaises(WorkspaceError):
                self.workspaces.create("unused", parent)
        git(self.repo, "tag", "-a", "base-tag", "-m", "tag", self.base)
        tagged = self.workspaces.create("tagged", "base-tag")
        self.assertEqual(git(tagged, "rev-parse", "HEAD").decode().strip(), self.base)
        with self.assertRaises(WorkspaceError):
            self.workspaces.tree(blob)

    def test_create_never_reuses_existing_paths_or_registered_name(self):
        existing = self.root / "occupied"
        existing.mkdir()
        marker = existing / "user.txt"
        marker.write_text("retain")
        with self.assertRaises(WorkspaceError):
            self.workspaces.create("occupied", self.base)
        self.assertEqual(marker.read_text(), "retain")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(WorkspaceError):
            self.workspaces.create("empty", self.base)
        candidate = self.candidate()
        with self.assertRaises(WorkspaceError):
            self.candidate()
        self.assert_pristine(candidate)

    def test_read_sources_has_total_byte_budget_and_exact_paths(self):
        candidate = self.candidate()
        self.assertEqual(self.workspaces.read_sources(candidate, ["main.py", "other.py"], 20),
                         {"main.py": "VALUE = 1\n", "other.py": "OTHER = 2\n"})
        with self.assertRaises(WorkspaceError):
            self.workspaces.read_sources(candidate, ["main.py", "other.py"], 19)
        for allowed in ([], ["*.py"], ["pkg"], ["missing.py"], ["../main.py"],
                        ["/main.py"], ["./main.py"], ["pkg//with space.py"],
                        [".git/config"], ["pkg/.GIT/config"], ["main.py\x00"]):
            with self.subTest(allowed=allowed), self.assertRaises(WorkspaceError):
                self.workspaces.read_sources(candidate, allowed)
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(WorkspaceError):
                self.workspaces.read_sources(candidate, ["main.py"], limit)

    def test_read_rejects_symlinks_binary_and_special_files(self):
        (self.repo / "link.py").symlink_to("main.py")
        (self.repo / "binary.py").write_bytes(b"\0binary")
        (self.repo / "invalid.py").write_bytes(b"\xffinvalid")
        self.base = self.save_base()
        candidate = self.candidate()
        for name in ("link.py", "binary.py", "invalid.py"):
            with self.subTest(name=name), self.assertRaises(WorkspaceError):
                self.workspaces.read_sources(candidate, [name])
        (candidate / "main.py").unlink()
        if hasattr(os, "mkfifo"):
            os.mkfifo(candidate / "main.py")
            with self.assertRaises(WorkspaceError):
                self.workspaces.read_sources(candidate, ["main.py"])

    def test_read_rejects_symlinked_source_ancestor(self):
        candidate = self.candidate()
        outside = self.directory / "outside"
        outside.mkdir()
        (outside / "with space.py").write_text("outside source")
        (candidate / "pkg/with space.py").unlink()
        (candidate / "pkg").rmdir()
        (candidate / "pkg").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(WorkspaceError):
            self.workspaces.read_sources(candidate, ["pkg/with space.py"])
        self.assertEqual((outside / "with space.py").read_text(), "outside source")

    def test_rejects_builder_edits_including_ignored_files(self):
        actions = (
            lambda path: (path / "other.py").write_text("builder edit"),
            lambda path: (path / "untracked.py").write_text("builder edit"),
            lambda path: (path / "builder.cache").write_text("ignored edit"),
            lambda path: git(path, "update-index", "--assume-unchanged", "main.py"),
            lambda path: git(path, "update-index", "--skip-worktree", "main.py"),
        )
        for number, action in enumerate(actions):
            candidate = self.candidate(f"dirty-{number}")
            action(candidate)
            with self.subTest(number=number), self.assertRaises(WorkspaceError):
                self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
            self.assertEqual((candidate / "main.py").read_text(), "VALUE = 1\n")

    def test_rejects_already_staged_builder_change(self):
        candidate = self.candidate()
        (candidate / "other.py").write_text("builder edit")
        git(candidate, "add", "other.py")
        with self.assertRaises(WorkspaceError):
            self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        self.assertEqual(git(candidate, "show", ":other.py"), b"builder edit")

    def test_disallowed_multifile_patch_is_rejected_without_partial_application(self):
        candidate = self.candidate()
        with self.assertRaises(WorkspaceError):
            self.workspaces.apply_patch(candidate, PATCH + OTHER_PATCH, ["main.py"])
        self.assert_pristine(candidate)
        self.assertEqual((candidate / "other.py").read_text(), "OTHER = 2\n")

    def test_git_parses_quoted_and_space_containing_paths(self):
        quoted = 'pkg/quo"ted.py'
        patch = self.generated_patch({
            "pkg/with space.py": "VALUE = 4\n",
            quoted: "QUOTED = True\n",
        })
        candidate = self.candidate()
        self.assertEqual(
            self.workspaces.apply_patch(candidate, patch, [quoted, "pkg/with space.py"]),
            sorted([quoted, "pkg/with space.py"]),
        )
        commit = self.workspaces.commit(candidate, self.base, "Quoted files")
        self.assertEqual(git(candidate, "show", f"{commit}:{quoted}"), b"QUOTED = True\n")
        self.assertEqual(self.workspaces.read_sources(candidate, [quoted]),
                         {quoted: "QUOTED = True\n"})

    def test_add_delete_and_executable_regular_sources(self):
        patch = self.generated_patch({"main.py": None, "new.py": "NEW = 7\n"})
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, patch, ["main.py", "new.py"])
        commit = self.workspaces.commit(candidate, self.base, "Move implementation")
        self.assertFalse((candidate / "main.py").exists())
        self.assertEqual(git(candidate, "show", f"{commit}:new.py"), b"NEW = 7\n")
        executable = self.candidate("executable")
        mode_patch = PATCH.replace(
            "--- a/main.py", "old mode 100644\nnew mode 100755\n--- a/main.py"
        )
        self.workspaces.apply_patch(executable, mode_patch, ["main.py"])
        executable_commit = self.workspaces.commit(executable, self.base, "Executable")
        self.assertTrue(git(executable, "ls-tree", executable_commit, "main.py").startswith(b"100755"))

    def test_patch_cannot_traverse_or_modify_git_metadata(self):
        candidate = self.candidate()
        outside = self.directory / "outside.py"
        outside.write_text("outside must survive")
        for name in ("../outside.py", ".git/config", "dir/.git/config", "/tmp/outside.py"):
            patch = PATCH.replace("a/main.py", f"a/{name}").replace("b/main.py", f"b/{name}")
            with self.subTest(name=name), self.assertRaises(WorkspaceError):
                self.workspaces.apply_patch(candidate, patch, ["main.py"])
            self.assert_pristine(candidate)
        self.assertEqual(outside.read_text(), "outside must survive")

    def test_binary_symlink_and_submodule_patches_are_rejected(self):
        binary = self.generated_patch({"main.py": b"binary\0data\n"})
        symlink = """diff --git a/link.py b/link.py
new file mode 120000
--- /dev/null
+++ b/link.py
@@ -0,0 +1 @@
+main.py
"""
        submodule = f"""diff --git a/module.py b/module.py
new file mode 160000
--- /dev/null
+++ b/module.py
@@ -0,0 +1 @@
+Subproject commit {self.base}
"""
        for number, patch in enumerate((binary, symlink, submodule)):
            candidate = self.candidate(f"rejected-{number}")
            with self.subTest(number=number), self.assertRaises(WorkspaceError):
                self.workspaces.apply_patch(candidate, patch, ["main.py", "link.py", "module.py"])
            self.assert_pristine(candidate)

    def test_rename_cannot_smuggle_a_disallowed_source(self):
        candidate = self.candidate()
        patch = """diff --git a/other.py b/renamed.py
similarity index 100%
rename from other.py
rename to renamed.py
"""
        with self.assertRaises(WorkspaceError):
            self.workspaces.apply_patch(candidate, patch, ["renamed.py"])
        self.assert_pristine(candidate)
        self.assertTrue((candidate / "other.py").is_file())

    def test_rejects_empty_invalid_oversized_and_mode_only_patches(self):
        candidate = self.candidate()
        mode_only = """diff --git a/main.py b/main.py
old mode 100644
new mode 100755
"""
        for patch in ("", " \n", "not a diff", PATCH + "\0", "x" * 1_000_001, mode_only):
            with self.subTest(size=len(patch)), self.assertRaises(WorkspaceError):
                self.workspaces.apply_patch(candidate, patch, ["main.py"])
            self.assert_pristine(candidate)

    def test_commit_requires_an_approved_nonempty_patch_and_correct_parent(self):
        candidate = self.candidate()
        (candidate / "main.py").write_text("unapproved change")
        git(candidate, "add", "main.py")
        with self.assertRaises(WorkspaceError):
            self.workspaces.commit(candidate, self.base, "Not approved")
        git(candidate, "reset", "--hard", self.base)
        self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        (self.repo / "other.py").write_text("a different parent")
        wrong_parent = self.save_base()
        with self.assertRaises(WorkspaceError):
            self.workspaces.commit(candidate, wrong_parent, "Wrong parent")
        for message in ("", " \n", "bad\0message"):
            with self.subTest(message=message), self.assertRaises(WorkspaceError):
                self.workspaces.commit(candidate, self.base, message)

    def test_commit_rejects_changed_source_or_index_after_approval(self):
        for number, staged in enumerate((False, True)):
            candidate = self.candidate(f"tampered-{number}")
            self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
            (candidate / "other.py").write_text("unapproved later change")
            if staged:
                git(candidate, "add", "other.py")
            with self.subTest(staged=staged), self.assertRaises(WorkspaceError):
                self.workspaces.commit(candidate, self.base, "Reject tampering")
            self.assertEqual(git(candidate, "rev-parse", "HEAD").decode().strip(), self.base)

    def test_commit_rejects_source_replaced_with_symlink(self):
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        outside = self.directory / "outside.py"
        outside.write_text("VALUE = 42\n")
        (candidate / "main.py").unlink()
        (candidate / "main.py").symlink_to(outside)
        with self.assertRaises(WorkspaceError):
            self.workspaces.commit(candidate, self.base, "Reject symlink")
        self.assertEqual(outside.read_text(), "VALUE = 42\n")

    def test_commit_rejects_recreated_deleted_file(self):
        patch = self.generated_patch({"main.py": None})
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, patch, ["main.py"])
        (candidate / "main.py").write_text("different evaluated code")
        with self.assertRaises(WorkspaceError):
            self.workspaces.commit(candidate, self.base, "Reject recreated source")

    def test_assert_clean_allows_build_outputs_but_rejects_tracked_edits(self):
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        commit = self.workspaces.commit(candidate, self.base, "Evaluate")
        (candidate / "result.log").write_text("test output")
        (candidate / "compiler.cache").write_text("ignored output")
        (candidate / "__pycache__").mkdir()
        (candidate / "__pycache__/main.pyc").write_bytes(b"\0compiled")
        self.workspaces.assert_clean(candidate)
        self.assertEqual(self.workspaces.commit(candidate, self.base, "Evaluate"), commit)
        (candidate / "main.py").write_text("rewritten during evaluation")
        with self.assertRaises(WorkspaceError):
            self.workspaces.assert_clean(candidate)
        git(candidate, "add", "main.py")
        with self.assertRaises(WorkspaceError):
            self.workspaces.assert_clean(candidate)
        git(candidate, "reset", "--hard", self.base)
        with self.assertRaises(WorkspaceError):
            self.workspaces.assert_clean(candidate)

    def test_assert_clean_checks_expected_head_without_an_apply_receipt(self):
        alternative = self.candidate("alternative")
        self.workspaces.apply_patch(alternative, PATCH, ["main.py"])
        other_commit = self.workspaces.commit(alternative, self.base, "Alternative source")
        fresh = self.candidate("fresh-evaluation")
        self.workspaces.assert_clean(fresh, self.base)
        git(fresh, "checkout", "--detach", other_commit)
        self.assertEqual(git(fresh, "status", "--porcelain"), b"")
        with self.assertRaisesRegex(WorkspaceError, "candidate commit"):
            self.workspaces.assert_clean(fresh, expected_commit=self.base)
        # The creation receipt also protects callers that omit expected_commit.
        with self.assertRaisesRegex(WorkspaceError, "candidate commit"):
            self.workspaces.assert_clean(fresh)

    def test_keep_validates_names_and_never_overwrites_another_commit(self):
        self.workspaces.keep(self.base, "retained")
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        commit = self.workspaces.commit(candidate, self.base, "Another commit")
        with self.assertRaises(WorkspaceError):
            self.workspaces.keep(commit, "retained")
        self.assertEqual(git(self.repo, "rev-parse", "refs/gama-rsi/retained").decode().strip(),
                         self.base)
        for name in ("", "../outside", "/root", "part//name", "name.lock", "name@{1}", "-option"):
            with self.subTest(name=name), self.assertRaises(WorkspaceError):
                self.workspaces.keep(commit, name)

    def test_keep_does_not_accept_a_symbolic_ref_as_an_immutable_pin(self):
        branch = git(self.repo, "symbolic-ref", "HEAD").decode().strip()
        ref = "refs/gama-rsi/symbolic"
        git(self.repo, "symbolic-ref", ref, branch)
        with self.assertRaises(WorkspaceError):
            self.workspaces.keep(self.base, "symbolic")
        self.assertEqual(git(self.repo, "symbolic-ref", ref).decode().strip(), branch)

    def test_remove_only_registered_detached_worktrees_in_owned_root(self):
        unmanaged = self.root / "ordinary"
        unmanaged.mkdir()
        (unmanaged / "marker").write_text("keep")
        external = self.directory / "external-worktree"
        git(self.repo, "worktree", "add", "--detach", str(external), self.base)
        branch_tree = self.root / "branch-tree"
        git(self.repo, "worktree", "add", "-b", "other-branch", str(branch_tree), self.base)
        for path in (self.repo, unmanaged, external, branch_tree, self.root / "../external"):
            with self.subTest(path=path), self.assertRaises(WorkspaceError):
                self.workspaces.remove(path)
        self.assertEqual((unmanaged / "marker").read_text(), "keep")
        self.assertTrue((external / "main.py").exists())
        candidate = self.candidate()
        (candidate / "untracked-output").write_text("disposable")
        self.workspaces.remove(candidate)
        self.assertFalse(candidate.exists())
        with self.assertRaises(WorkspaceError):
            self.workspaces.remove(candidate)

    def test_foreign_repo_worktree_in_root_cannot_be_read_or_removed(self):
        foreign = self.directory / "foreign"
        foreign.mkdir()
        git(foreign, "init", "-q", "--template=")
        (foreign / "file.py").write_text("foreign")
        git(foreign, "add", "file.py")
        git(foreign, "commit", "-qm", "foreign")
        path = self.root / "foreign-child"
        git(foreign, "worktree", "add", "--detach", str(path), "HEAD")
        with self.assertRaises(WorkspaceError):
            self.workspaces.read_sources(path, ["file.py"])
        with self.assertRaises(WorkspaceError):
            self.workspaces.remove(path)
        self.assertEqual((path / "file.py").read_text(), "foreign")

    def test_no_hooks_external_diff_or_config_mutations(self):
        marker = self.directory / "hook-ran"
        hooks = self.directory / "custom hooks"
        hooks.mkdir()
        script = "#!/bin/sh\nprintf x >> " + shlex.quote(str(marker)) + "\n"
        for name in ("post-checkout", "pre-commit", "prepare-commit-msg", "commit-msg",
                     "post-commit", "reference-transaction", "fsmonitor"):
            hook = hooks / name
            hook.write_text(script)
            hook.chmod(0o755)
        git(self.repo, "config", "core.hooksPath", str(hooks))
        git(self.repo, "config", "core.fsmonitor", str(hooks / "fsmonitor"))
        git(self.repo, "config", "diff.external", str(hooks / "fsmonitor"))
        config = (self.repo / ".git/config").read_bytes()
        candidate = self.candidate()
        self.workspaces.apply_patch(candidate, PATCH, ["main.py"])
        commit = self.workspaces.commit(candidate, self.base, "No hooks")
        self.workspaces.keep(commit, "no-hooks")
        self.assertIn("+VALUE = 42", self.workspaces.diff(self.base, commit))
        self.workspaces.assert_clean(candidate)
        self.workspaces.remove(candidate)
        self.assertFalse(marker.exists())
        self.assertEqual((self.repo / ".git/config").read_bytes(), config)

    @unittest.skipUnless(os.name == "posix", "Process-group timeout test uses POSIX")
    def test_git_timeout_kills_slow_checkout_filter(self):
        (self.repo / ".gitattributes").write_text("*.py filter=slow\n")
        self.base = self.save_base()
        git(self.repo, "config", "filter.slow.smudge", "sleep 5; cat")
        git(self.repo, "config", "filter.slow.required", "true")
        previous_timeout = rsi_workspace.GIT_TIMEOUT
        try:
            rsi_workspace.GIT_TIMEOUT = 0.15
            start = time.monotonic()
            with self.assertRaisesRegex(WorkspaceError, "timed out"):
                self.candidate("slow")
            self.assertLess(time.monotonic() - start, 3)
        finally:
            rsi_workspace.GIT_TIMEOUT = previous_timeout


if __name__ == "__main__":
    unittest.main()
