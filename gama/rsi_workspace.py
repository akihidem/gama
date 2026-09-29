"""Git storage for source-code RSI candidates; never checks out the caller's branch.

Candidate worktrees are detached, patches are checked in a temporary index, and
only the approved index tree can be committed. Small receipts beside the worktrees
allow another Workspaces instance to finish a commit or recover abandoned trees
after a coordinator restart.
This is checkout isolation, not a sandbox for executing candidate code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile


GIT_TIMEOUT = 30
_MAX_PATCH_BYTES = 1_000_000
_MAX_BLOB_BYTES = 8_000_000
_MAX_RECEIPT_BYTES = 128_000
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_FILE_MODES = {"100644", "100755"}
_GIT_OPTIONS = (
    "--no-pager", "--no-replace-objects", "--literal-pathspecs",
    "-c", f"core.hooksPath={os.devnull}",
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
    "-c", "core.sparseCheckout=false",
    "-c", "core.fileMode=true",
    "-c", "core.trustctime=true",
    "-c", "core.checkStat=default",
    "-c", "core.autocrlf=false",
    "-c", "commit.gpgSign=false",
    "-c", "gc.auto=0",
    "-c", "maintenance.auto=false",
    "-c", "user.name=Gama RSI",
    "-c", "user.email=gama-rsi@localhost",
    "-c", "user.useConfigOnly=true",
)


class WorkspaceError(RuntimeError):
    """An invalid workspace, rejected source change, or failed Git operation."""


def _absolute(value: str | Path) -> Path:
    try:
        path = Path(value).expanduser()
        if ".." in path.parts:
            raise WorkspaceError("Path traversal is not allowed")
        path = Path(os.path.abspath(path))
        for part in (*reversed(path.parents), path):
            try:
                mode = part.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise WorkspaceError(f"Symlinks are not allowed: {part}")
        return path
    except (OSError, ValueError, TypeError) as exc:
        raise WorkspaceError(f"Invalid filesystem path: {exc}") from exc


def _name(value: str, *, ref: bool = False) -> str:
    parts = value.split("/") if isinstance(value, str) and ref else [value]
    if not parts or any(
        not isinstance(part, str) or not _NAME.fullmatch(part)
        or ".." in part or part.endswith(".") or part.lower().endswith(".lock")
        for part in parts
    ):
        raise WorkspaceError("Names must use safe alphanumeric, dot, dash or underscore components")
    return value


def _source_name(value: str) -> str:
    if not isinstance(value, str) or not value or any(
        ord(char) < 32 or ord(char) == 127 or char in "\\:*?[]" for char in value
    ):
        raise WorkspaceError("Allowed sources must be exact relative file paths, without globs")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.lower().rstrip(" .") == ".git" for part in parts):
        raise WorkspaceError(f"Unsafe source path: {value!r}")
    return value


def _allowed(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not values:
        raise WorkspaceError("At least one exact allowed source path is required")
    return list(dict.fromkeys(_source_name(value) for value in values))


def _regular_bytes(path: Path, limit: int) -> bytes:
    """Bounded reads, including for a FIFO or symlink substituted for a source."""
    _absolute(path)
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise WorkspaceError(f"Not a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise WorkspaceError(f"Not a regular file: {path}")
            if info.st_size > limit:
                raise WorkspaceError(f"Source byte limit exceeded: {path}")
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise WorkspaceError(f"Source byte limit exceeded: {path}")
        return data
    except OSError as exc:
        raise WorkspaceError(f"Cannot read regular file {path}: {exc}") from exc


def _text(data: bytes, label: str) -> str:
    try:
        if b"\0" in data:
            raise ValueError("NUL byte")
        return data.decode("utf-8")
    except (UnicodeError, ValueError) as exc:
        raise WorkspaceError(f"Binary or non-UTF-8 source is not allowed: {label}") from exc


class Workspaces:
    """Manage detached worktrees directly below an owned ``root`` directory."""

    def __init__(self, repo: str | Path, root: str | Path):
        candidate = _absolute(repo)
        self.repo = _absolute(os.fsdecode(self._git(candidate, "rev-parse", "--show-toplevel")).strip())
        self._common = self._common_dir(self.repo)
        self.root = _absolute(root)
        if (self.root == self.repo or self.root in self.repo.parents
                or self.repo in self.root.parents):
            raise WorkspaceError("The workspace root must be outside the caller checkout")
        if self.root == self._common or self._common in self.root.parents:
            raise WorkspaceError("The workspace root must not be inside Git metadata")
        if any(path == self.root or path in self.root.parents for path in self._registered()):
            raise WorkspaceError("The workspace root must be outside registered worktrees")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(f"Cannot create workspace root: {exc}") from exc
        if not self.root.is_dir():
            raise WorkspaceError("Workspace root is not a directory")

    @staticmethod
    def _git(cwd: Path, *args: str, data: bytes | None = None,
             index: Path | None = None) -> bytes:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update({
            "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_LAZY_FETCH": "1", "LC_ALL": "C",
        })
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        try:
            with subprocess.Popen(
                ["git", *_GIT_OPTIONS, *args], cwd=cwd, env=env, shell=False,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            ) as process:
                try:
                    output, error = process.communicate(data, timeout=GIT_TIMEOUT)
                except subprocess.TimeoutExpired as exc:
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.communicate(timeout=5)
                    raise WorkspaceError(f"git {args[0]} timed out after {GIT_TIMEOUT}s") from exc
                if process.returncode:
                    detail = error.decode("utf-8", errors="replace").strip()[-2000:]
                    raise WorkspaceError(f"git {args[0]} failed: {detail}")
                return output
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkspaceError(f"Cannot run git {args[0]}: {exc}") from exc

    def _common_dir(self, path: Path) -> Path:
        raw = self._git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
        return _absolute(os.fsdecode(raw).strip())

    def _resolve(self, value: str, path: Path | None = None) -> str:
        if not isinstance(value, str) or not value or len(value) > 1024 or any(
            ord(char) < 32 for char in value
        ):
            raise WorkspaceError("A commit SHA or ref is required")
        result = self._git(
            path or self.repo, "rev-parse", "--verify", "--end-of-options", value + "^{commit}"
        ).decode("ascii").strip()
        if not _OID.fullmatch(result):
            raise WorkspaceError("Git did not resolve a commit object")
        return result

    def head(self) -> str:
        return self._resolve("HEAD")

    def tree(self, commit: str) -> str:
        """Return the immutable source tree ID, independent of commit metadata."""
        oid = self._resolve(commit)
        return self._git(self.repo, "rev-parse", "--verify", oid + "^{tree}").decode("ascii").strip()

    def _registered(self) -> dict[Path, dict[str, str]]:
        result: dict[Path, dict[str, str]] = {}
        current: dict[str, str] = {}
        for record in self._git(self.repo, "worktree", "list", "--porcelain", "-z").split(b"\0"):
            if not record:
                if "worktree" in current:
                    result[Path(current["worktree"])] = current
                current = {}
            else:
                key, _, value = os.fsdecode(record).partition(" ")
                current[key] = value
        return result

    def _managed(self, value: str | Path, *, missing: bool = False) -> Path:
        path = _absolute(value)
        _absolute(self.root)
        if path.parent != self.root or path == self.repo:
            raise WorkspaceError("Candidate must be directly inside the owned workspace root")
        _name(path.name)
        record = self._registered().get(path)
        if record is None or "detached" not in record:
            raise WorkspaceError("Not a registered detached worktree for this repository")
        if missing and not path.exists():
            return path
        if not path.is_dir():
            raise WorkspaceError("Candidate worktree directory is missing")
        _regular_bytes(path / ".git", _MAX_RECEIPT_BYTES)
        if self._common_dir(path) != self._common:
            raise WorkspaceError("Candidate belongs to a different Git repository")
        top = _absolute(os.fsdecode(self._git(path, "rev-parse", "--show-toplevel")).strip())
        if top != path:
            raise WorkspaceError("Candidate is not the registered worktree root")
        return path

    def create(self, name: str, parent: str) -> Path:
        name = _name(name)
        _absolute(self.root)
        base = self._resolve(parent)
        base_tree = self.tree(base)
        path = self.root / name
        receipt = self._receipt_path(path)
        if path in self._registered() or os.path.lexists(receipt):
            raise WorkspaceError(f"Workspace name already exists: {name}")
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            raise WorkspaceError(f"Workspace path already exists or cannot be created: {path}") from exc
        try:
            # Publish ownership before Git can register a worktree. A killed
            # coordinator leaves either an empty reservation or a recoverable tree.
            self._save_receipt(path, {
                "version": 1, "name": name, "repo": str(self._common),
                "parent": base, "tree": base_tree, "paths": [],
                "commit": base, "stage": "created",
            })
            self._git(self.repo, "worktree", "add", "--detach", "--", str(path), base)
            return self._managed(path)
        except WorkspaceError:
            # Only remove our empty reservation. Git handles its own failed checkout;
            # never recursively delete a path that might now contain somebody's files.
            try:
                path.rmdir()
            except OSError:
                pass
            raise

    def read_sources(self, path: str | Path, allowed_paths: list[str],
                     max_bytes: int = 120000) -> dict[str, str]:
        """Read existing tracked UTF-8 files; ``max_bytes`` is a total byte budget."""
        worktree = self._managed(path)
        allowed = _allowed(allowed_paths)
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise WorkspaceError("max_bytes must be a positive integer")
        entries = self._git(worktree, "ls-files", "--stage", "-z", "--", *allowed).split(b"\0")
        modes = {}
        for entry in entries:
            if entry:
                metadata, name = entry.split(b"\t", 1)
                mode, _, stage = metadata.split()
                if stage != b"0":
                    raise WorkspaceError("Unmerged sources are not allowed")
                modes[os.fsdecode(name)] = mode.decode("ascii")
        result = {}
        remaining = max_bytes
        for name in allowed:
            if modes.get(name) not in _FILE_MODES:
                raise WorkspaceError(f"Source is not a tracked regular file: {name}")
            data = _regular_bytes(worktree / name, remaining)
            result[name] = _text(data, name)
            remaining -= len(data)
        return result

    def _check_index_flags(self, path: Path) -> None:
        entries = self._git(path, "ls-files", "-v", "-z").split(b"\0")
        if any(entry and not entry.startswith(b"H ") for entry in entries):
            raise WorkspaceError("Unmerged, assume-unchanged or skip-worktree entries are not allowed")

    def _clean(self, path: Path) -> None:
        self._check_index_flags(path)
        if self._git(
            path, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored"
        ):
            raise WorkspaceError("Builder already changed worktree files; expected a clean candidate")

    def assert_clean(self, path: str | Path, expected_commit: str | None = None) -> None:
        """Reject tracked changes and a changed HEAD, allowing untracked outputs.

        Passing the recorded commit also checks the coordinator's source identity.
        Creation and approved-patch receipts independently retain their HEAD checks.
        """
        worktree = self._managed(path)
        if expected_commit is not None and self._resolve("HEAD", worktree) != self._resolve(expected_commit):
            raise WorkspaceError("Evaluation changed the candidate commit")
        self._check_index_flags(worktree)
        if self._git(
            worktree, "status", "--porcelain=v1", "-z", "--untracked-files=no",
            "--ignore-submodules=none",
        ):
            raise WorkspaceError("Evaluation changed tracked worktree files")
        receipt_path = self._receipt_path(worktree)
        if os.path.lexists(receipt_path):
            try:
                receipt = json.loads(_regular_bytes(receipt_path, _MAX_RECEIPT_BYTES))
                expected = receipt.get("commit")
            except (ValueError, TypeError, AttributeError) as exc:
                raise WorkspaceError("Invalid approved patch receipt") from exc
            if expected is not None and self._resolve("HEAD", worktree) != expected:
                raise WorkspaceError("Evaluation changed the candidate commit")

    def _changes(self, path: Path, parent: str, *, index: Path | None = None) -> list[tuple]:
        raw = self._git(
            path, "diff", "--cached", "--raw", "-z", "--no-abbrev", "--no-renames",
            "--no-ext-diff", "--no-textconv", parent, "--", index=index,
        ).split(b"\0")
        result = []
        for offset in range(0, len(raw) - 1, 2):
            if offset + 1 >= len(raw) or not raw[offset].startswith(b":"):
                raise WorkspaceError("Invalid Git diff records")
            fields = raw[offset][1:].decode("ascii").split()
            if len(fields) != 5:
                raise WorkspaceError("Invalid Git diff metadata")
            old_mode, new_mode, old_oid, new_oid, status_code = fields
            name = _source_name(os.fsdecode(raw[offset + 1]))
            if status_code not in {"A", "D", "M"} or any(
                mode not in _FILE_MODES | {"000000"} for mode in (old_mode, new_mode)
            ):
                raise WorkspaceError(f"Only regular-file source changes are allowed: {name}")
            result.append((name, old_mode, new_mode, old_oid, new_oid))
        return result

    def _blob(self, path: Path, oid: str) -> bytes:
        size = int(self._git(path, "cat-file", "-s", oid))
        if size > _MAX_BLOB_BYTES:
            raise WorkspaceError("Source blob exceeds the byte limit")
        return self._git(path, "cat-file", "blob", oid)

    def _validate_diff(self, path: Path, parent: str, allowed: list[str],
                       index: Path) -> list[tuple]:
        changes = self._changes(path, parent, index=index)
        if not changes:
            raise WorkspaceError("A nonempty source diff is required")
        for name, old_mode, new_mode, old_oid, new_oid in changes:
            if name not in allowed:
                raise WorkspaceError(f"Patch changes a file outside allowed_paths: {name}")
            _absolute(path / name)
            if old_mode != "000000":
                _regular_bytes(path / name, _MAX_BLOB_BYTES)
                _text(self._blob(path, old_oid), name)
            elif os.path.lexists(path / name):
                raise WorkspaceError(f"New source path already exists: {name}")
            if new_mode != "000000":
                _text(self._blob(path, new_oid), name)
        numstat = self._git(
            path, "diff", "--cached", "--numstat", "-z", "--no-renames", "--no-ext-diff",
            "--no-textconv", parent, "--", index=index,
        )
        changed_lines = 0
        for record in numstat.split(b"\0"):
            if record:
                added, removed, _ = record.split(b"\t", 2)
                if not added.isdigit() or not removed.isdigit():
                    raise WorkspaceError("Binary patches are not allowed")
                changed_lines += int(added) + int(removed)
        if not changed_lines:
            raise WorkspaceError("A nonempty source diff is required; mode-only patches are rejected")
        return changes

    def _receipt_path(self, path: Path) -> Path:
        return self.root / f".{path.name}.applied.json"

    def _save_receipt(self, path: Path, receipt: dict) -> None:
        target = self._receipt_path(path)
        _absolute(target)
        if os.path.lexists(target) and not stat.S_ISREG(target.lstat().st_mode):
            raise WorkspaceError("Invalid workspace receipt")
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".rsi-receipt-", dir=self.root)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            raise WorkspaceError(f"Cannot save approved patch receipt: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _recovery_receipt(self, path: Path) -> dict:
        """Validate ownership metadata before it can authorize cleanup."""
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate receipt field")
                result[key] = value
            return result

        try:
            receipt = json.loads(
                _regular_bytes(self._receipt_path(path), _MAX_RECEIPT_BYTES),
                object_pairs_hook=unique_fields,
            )
            required = {"version", "name", "repo", "parent", "tree", "paths", "stage"}
            if (not isinstance(receipt, dict) or not required <= receipt.keys()
                    or receipt.keys() - required - {"commit", "message"}
                    or type(receipt["version"]) is not int or receipt["version"] != 1
                    or receipt["name"] != path.name
                    or receipt["repo"] != str(self._common)
                    or receipt["stage"] not in {"created", "approved"}):
                raise ValueError("receipt ownership or schema does not match")
            for key in ("parent", "tree"):
                if not isinstance(receipt[key], str) or not _OID.fullmatch(receipt[key]):
                    raise ValueError(f"invalid {key} object ID")
            if self._resolve(receipt["parent"]) != receipt["parent"]:
                raise ValueError("invalid parent commit")
            if receipt["stage"] == "created":
                if (receipt["paths"] != [] or receipt.get("commit") != receipt["parent"]
                        or "message" in receipt or self.tree(receipt["parent"]) != receipt["tree"]):
                    raise ValueError("invalid creation receipt")
            else:
                paths = _allowed(receipt["paths"])
                if paths != receipt["paths"] or len(paths) != len(set(paths)):
                    raise ValueError("invalid approved source paths")
                if self._git(self.repo, "cat-file", "-t", receipt["tree"]).strip() != b"tree":
                    raise ValueError("invalid approved tree")
                if "commit" in receipt:
                    commit = receipt["commit"]
                    if (not isinstance(commit, str) or not _OID.fullmatch(commit)
                            or not isinstance(receipt.get("message"), str)
                            or not receipt["message"].strip() or "\0" in receipt["message"]
                            or self.tree(commit) != receipt["tree"]):
                        raise ValueError("invalid approved commit")
                    parents = self._git(self.repo, "rev-list", "--parents", "-n", "1", commit).split()
                    if parents != [commit.encode("ascii"), receipt["parent"].encode("ascii")]:
                        raise ValueError("approved commit has a different parent")
                elif "message" in receipt:
                    raise ValueError("receipt message has no commit")
            return receipt
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise WorkspaceError(f"Invalid recovery receipt for {path.name}: {exc}") from exc

    def recover(self) -> list[Path]:
        """Clean receipted orphans; call only under the coordinator's run lock.

        Registered detached worktrees are removed through ``remove``. Empty
        unregistered reservations and receipts for missing paths can be cleared.
        Unregistered nonempty directories and worktrees without receipts are kept.
        All receipts are preflighted before any cleanup; repeated recovery is safe.
        """
        _absolute(self.root)
        registered = self._registered()
        plan = []
        try:
            for receipt_path in sorted(self.root.glob(".*.applied.json")):
                name = _name(receipt_path.name[1:-len(".applied.json")])
                path = _absolute(self.root / name)
                self._recovery_receipt(path)
                if path in registered:
                    self._managed(path, missing=True)
                    if "locked" in registered[path]:
                        raise WorkspaceError(f"Cannot recover a locked worktree: {path}")
                elif path.exists():
                    if not path.is_dir() or next(path.iterdir(), None) is not None:
                        raise WorkspaceError(f"Refusing unregistered nonempty reservation: {path}")
                plan.append((path, path in registered))
            recovered = []
            for path, is_registered in plan:
                if is_registered:
                    self.remove(path)
                else:
                    # rmdir is intentionally non-recursive and fails if anything
                    # was placed in the reservation after preflight.
                    _absolute(path)
                    if path.exists():
                        path.rmdir()
                    self._receipt_path(path).unlink()
                recovered.append(path)
            return recovered
        except OSError as exc:
            raise WorkspaceError(f"Cannot recover owned worktrees: {exc}") from exc

    def apply_patch(self, path: str | Path, patch: str, allowed_paths: list[str]) -> list[str]:
        """Validate and stage a text patch atomically with respect to rejection."""
        worktree = self._managed(path)
        allowed = _allowed(allowed_paths)
        self._clean(worktree)
        if not isinstance(patch, str) or not patch.strip():
            raise WorkspaceError("A nonempty unified diff is required")
        try:
            data = patch.encode("utf-8")
        except UnicodeError as exc:
            raise WorkspaceError("Patch must be UTF-8 text") from exc
        if b"\0" in data or len(data) > _MAX_PATCH_BYTES:
            raise WorkspaceError("Patch is binary or exceeds the byte limit")
        parent = self._resolve("HEAD", worktree)
        with tempfile.TemporaryDirectory(prefix=".rsi-index-", dir=self.root) as temporary:
            index = Path(temporary) / "index"
            self._git(worktree, "read-tree", parent, index=index)
            self._git(
                worktree, "apply", "--cached", "--whitespace=nowarn", "-", data=data, index=index
            )
            changes = self._validate_diff(worktree, parent, allowed, index)
            tree = self._git(worktree, "write-tree", index=index).decode("ascii").strip()
        self._clean(worktree)
        self._git(worktree, "apply", "--index", "--whitespace=nowarn", "-", data=data)
        actual_tree = self._git(worktree, "write-tree").decode("ascii").strip()
        if actual_tree != tree:
            raise WorkspaceError("Applied index differs from the validated patch")
        names = sorted(change[0] for change in changes)
        self._save_receipt(worktree, {
            "version": 1, "name": worktree.name, "repo": str(self._common), "parent": parent,
            "tree": tree, "paths": names, "stage": "approved",
        })
        return names

    def commit(self, path: str | Path, parent: str, message: str) -> str:
        """Commit only an approved patch; advance this worktree's detached HEAD."""
        worktree = self._managed(path)
        base = self._resolve(parent)
        if not isinstance(message, str) or not message.strip() or "\0" in message:
            raise WorkspaceError("A nonempty commit message is required")
        if len(message.encode("utf-8")) > _MAX_RECEIPT_BYTES // 2:
            raise WorkspaceError("Commit message exceeds the byte limit")
        try:
            receipt = json.loads(_regular_bytes(self._receipt_path(worktree), _MAX_RECEIPT_BYTES))
            if (receipt["version"] != 1 or receipt["repo"] != str(self._common)
                    or receipt["parent"] != base or not _OID.fullmatch(receipt["tree"])):
                raise WorkspaceError("Approved patch does not match this repository and parent")
            if receipt.get("stage") == "created":
                raise WorkspaceError("A nonempty approved patch is required before commit")
            allowed = _allowed(receipt["paths"])
        except (ValueError, KeyError, TypeError) as exc:
            raise WorkspaceError("Missing or invalid approved patch receipt") from exc
        current = self._resolve("HEAD", worktree)
        if current not in {base, receipt.get("commit")}:
            raise WorkspaceError("Worktree HEAD changed after patch approval")
        self._check_index_flags(worktree)
        tree = self._git(worktree, "write-tree").decode("ascii").strip()
        if tree != receipt["tree"]:
            raise WorkspaceError("Staged files changed after patch approval")
        if self._git(
            worktree, "diff", "--raw", "-z", "--no-ext-diff", "--no-textconv", "--no-renames", "--"
        ):
            raise WorkspaceError("Worktree files changed after patch approval")
        changes = self._changes(worktree, base)
        if not changes or sorted(change[0] for change in changes) != sorted(allowed):
            raise WorkspaceError("A nonempty approved source diff is required")
        for name, _, mode, _, oid in changes:
            if mode == "000000":
                if os.path.lexists(worktree / name):
                    raise WorkspaceError(f"Deleted source was recreated: {name}")
            else:
                content = _regular_bytes(worktree / name, _MAX_BLOB_BYTES)
                algorithm = hashlib.sha1 if len(oid) == 40 else hashlib.sha256
                actual = algorithm(b"blob " + str(len(content)).encode("ascii") + b"\0" + content)
                if actual.hexdigest() != oid:
                    raise WorkspaceError(f"Source differs from the approved Git blob: {name}")
        commit = receipt.get("commit")
        if commit is None:
            commit = self._git(
                worktree, "commit-tree", tree, "-p", base, data=message.encode("utf-8")
            ).decode("ascii").strip()
            receipt.update(commit=commit, message=message)
            self._save_receipt(worktree, receipt)
        elif receipt.get("message") != message or self._resolve(commit) != commit:
            raise WorkspaceError("Commit retry does not match the approved commit")
        if current == base:
            self._git(worktree, "update-ref", "--no-deref", "HEAD", commit, base)
        return commit

    def keep(self, commit: str, name: str) -> str:
        """Pin a commit under refs/gama-rsi; existing different refs are preserved."""
        ref = "refs/gama-rsi/" + _name(name, ref=True)
        oid = self._resolve(commit)
        self._git(self.repo, "check-ref-format", ref)
        # An all-zero expected old value makes creation atomic and refuses overwrites.
        try:
            self._git(self.repo, "update-ref", "--no-deref", ref, oid, "0" * len(oid))
        except WorkspaceError:
            # Resolving a symbolic ref to the same commit is insufficient: its
            # target could subsequently move, dropping the archive's GC root.
            existing = self._git(
                self.repo, "for-each-ref", "--format=%(refname)%00%(objectname)%00%(symref)", ref
            ).splitlines()
            if existing != [(ref + "\0" + oid + "\0").encode("ascii")]:
                raise WorkspaceError(f"Archive ref already exists or cannot be created: {ref}")
        return ref

    def diff(self, base: str, head: str) -> str:
        before, after = self._resolve(base), self._resolve(head)
        output = self._git(
            self.repo, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
            "--no-color", "--src-prefix=a/", "--dst-prefix=b/", before, after, "--",
        )
        return _text(output, "commit diff")

    def remove(self, path: str | Path) -> None:
        """Remove only a registered detached worktree directly inside this root."""
        worktree = self._managed(path, missing=True)
        receipt = self._receipt_path(worktree)
        _absolute(receipt)
        if os.path.lexists(receipt) and not stat.S_ISREG(receipt.lstat().st_mode):
            raise WorkspaceError("Invalid workspace receipt")
        self._git(self.repo, "worktree", "remove", "--force", "--", str(worktree))
        try:
            receipt.unlink(missing_ok=True)
        except OSError as exc:
            raise WorkspaceError(f"Cannot remove workspace receipt: {exc}") from exc
