"""Persistent, bounded evolution of source code under a fixed external evaluator.

The archive and parent weights follow DGM (2505.22954, Appendix C.2). Parallel
batches use an immutable archive snapshot; the coordinator integrates results in
slot order. Generation may use the selected parent's gama backend implementation,
but it cannot replace this controller or its acceptance rules during a run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from .rsi_agent import propose_patch
from .rsi_evaluate import Evaluation, evaluate, promotion, run_checks
from .rsi_workspace import Workspaces

SCHEMA_VERSION = 1
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_CONTROL_FILES = {
    "gama/rsi.py", "gama/rsi_agent.py", "gama/rsi_process.py",
    "gama/rsi_workspace.py", "gama/rsi_evaluate.py", "gama/rsi_cli.py",
}


class RSIError(RuntimeError):
    """A run cannot proceed with the recorded inputs or evaluation contract."""


def _positive(value, name: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0 \
            or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def _argv(value, name: str) -> None:
    if not isinstance(value, list) or not value \
            or any(not isinstance(x, str) or not x or "\0" in x for x in value):
        raise ValueError(f"{name} must be a nonempty JSON array of command arguments")


@dataclass(frozen=True)
class RSIConfig:
    goal: str
    allowed_paths: list[str]
    agents: list[dict]
    checks: list[list[str]]
    search_command: list[str]
    confirm_command: list[str]
    sealed_command: Optional[list[str]] = None
    workers: int = 2
    batch_size: int = 2
    timeout: float = 600
    evaluation_timeout: float = 180
    search_repeats: int = 1
    confirm_repeats: int = 3
    min_gain: float = 0.0
    seed: int = 0
    papers: list[dict] = field(default_factory=list)
    evaluation_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict) -> "RSIConfig":
        if not isinstance(value, dict):
            raise ValueError("RSI config must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown RSI config keys: {', '.join(sorted(unknown))}")
        try:
            config = cls(**copy.deepcopy(value))
        except TypeError as exc:
            raise ValueError(f"incomplete RSI config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must describe the intended source improvement")
        if not isinstance(self.allowed_paths, list) or not self.allowed_paths:
            raise ValueError("allowed_paths must list exact source files")
        for name in self.allowed_paths:
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("allowed_paths must contain relative file names")
            p = PurePosixPath(name)
            if p.is_absolute() or not p.parts or ".." in p.parts or str(p) != name \
                    or any(ord(c) < 32 or c in "\\:*?[]" for c in name) \
                    or any(x.lower().startswith(".git") for x in p.parts):
                raise ValueError(f"unsafe source path: {name!r}")
            if any(part in ("tests", "test") for part in p.parts) \
                    or p.name.startswith("test_") or p.name.endswith("_test.py") \
                    or p.name == "conftest.py" or name in _CONTROL_FILES:
                raise ValueError(f"the fixed tests/controller cannot be mutated: {name}")
        if len(set(self.allowed_paths)) != len(self.allowed_paths):
            raise ValueError("allowed_paths contains duplicates")
        if not isinstance(self.agents, list) or not self.agents:
            raise ValueError("agents must contain at least one patch generator")
        names = set()
        for agent in self.agents:
            if not isinstance(agent, dict) or not isinstance(agent.get("name"), str) \
                    or not _NAME.fullmatch(agent["name"]):
                raise ValueError("each agent needs a short unique name")
            name = agent["name"]
            if name in names:
                raise ValueError(f"duplicate agent name: {name}")
            names.add(name)
            if set(agent) - {"name", "backend", "command"}:
                raise ValueError(f"unknown fields in agent {name!r}")
            if ("backend" in agent) == ("command" in agent):
                raise ValueError(f"agent {name!r} needs exactly one of backend or command")
            if "command" in agent:
                _argv(agent["command"], f"agent {name} command")
            elif not isinstance(agent["backend"], dict) \
                    or not isinstance(agent["backend"].get("backend"), str) \
                    or not agent["backend"]["backend"].strip():
                raise ValueError(f"agent {name!r} needs a gama backend spec")
        if not isinstance(self.checks, list) or not self.checks:
            raise ValueError("checks must include at least one fixed test command")
        for command in self.checks:
            _argv(command, "check")
        _argv(self.search_command, "search_command")
        _argv(self.confirm_command, "confirm_command")
        if self.search_command == self.confirm_command:
            raise ValueError("search_command and confirm_command must select separate evaluations")
        if self.sealed_command is not None:
            _argv(self.sealed_command, "sealed_command")
            if self.sealed_command in (self.search_command, self.confirm_command):
                raise ValueError("sealed_command must select a separate evaluation")
        for name in ("workers", "batch_size", "search_repeats", "confirm_repeats"):
            _positive(getattr(self, name), name, integer=True)
        for name in ("timeout", "evaluation_timeout"):
            _positive(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(self.min_gain, bool) or not isinstance(self.min_gain, (int, float)) \
                or not math.isfinite(self.min_gain) or not 0 <= self.min_gain <= 1:
            raise ValueError("min_gain must be finite and between 0 and 1")
        if not isinstance(self.papers, list):
            raise ValueError("papers must be a list of title/url/notes objects")
        for paper in self.papers:
            if not isinstance(paper, dict) or set(paper) - {"title", "url", "notes"} \
                    or any(not isinstance(paper.get(k), str) or not paper[k].strip()
                           for k in ("title", "url", "notes")):
                raise ValueError("each paper needs title, url, and source-grounded notes")
            if not paper["url"].startswith(("https://", "http://")):
                raise ValueError("paper url must be an HTTP(S) source")
        if not isinstance(self.evaluation_files, list) or any(
            not isinstance(p, str) or "\0" in p or not Path(p).is_absolute()
            for p in self.evaluation_files
        ):
            raise ValueError("evaluation_files must list absolute paths to fixed evaluator inputs")
        if len(set(self.evaluation_files)) != len(self.evaluation_files):
            raise ValueError("evaluation_files contains duplicates")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def controller_fingerprint() -> str:
    """Identify the outer code whose tests and promotion decisions a run used."""
    root = Path(__file__).parent
    return _digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.glob("rsi*.py"))})


def _file_digests(paths: list[str]) -> dict[str, str]:
    """Fingerprint declared external evaluators/data without loading them into RAM."""
    result = {}
    for name in paths:
        path = Path(name)
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise RSIError(f"evaluation input is not a regular file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(block)
            result[name] = digest.hexdigest()
        except OSError as exc:
            raise RSIError(f"cannot fingerprint evaluation input {path}: {exc}") from exc
    return result


def _validate_evaluation_scope(config: RSIConfig, repo: Path) -> None:
    """Keep configured command inputs and declared dependencies outside mutation."""
    fixed = list(config.evaluation_files)
    commands = [*config.checks, config.search_command, config.confirm_command]
    if config.sealed_command:
        commands.append(config.sealed_command)
    for command in commands:
        inline = False
        module = False
        for token in command:
            if inline:
                inline = False
                continue
            if token == "-c":
                inline = True
                continue
            # Python accepts both -m pkg and -Impkg / -Im pkg. Leaving the
            # attached spelling opaque would make a fixed module editable.
            attached_module = re.fullmatch(r"-[bBdEhiIOPqRsSuvVx]*m(.*)", token)
            if attached_module and not attached_module[1]:
                module = True
                continue
            if attached_module:
                name = attached_module[1].replace(".", "/")
                fixed.extend((name + ".py", name + "/__main__.py"))
            elif module:
                fixed.extend((token.replace(".", "/") + ".py",
                              token.replace(".", "/") + "/__main__.py"))
                module = False
            elif token.startswith("-"):
                if "=" in token:
                    fixed.append(token.split("=", 1)[1])
            else:
                fixed.append(token.split("::", 1)[0])
    mutable = {repo / name: name for name in config.allowed_paths}
    for token in fixed:
        if not token:
            continue
        path = Path(os.path.abspath(repo / token))
        # Include symlink aliases of declared inputs without requiring argv strings
        # (which also contain scalar options) to name existing files.
        try:
            aliases = {path, path.resolve()}
        except OSError:
            aliases = {path}
        for candidate, name in mutable.items():
            if candidate in aliases:
                raise ValueError(f"fixed test/evaluator input cannot be mutated: {name}")


def select_parents(archive: list[dict], count: int, *, seed: int, round_number: int) -> list[dict]:
    """DGM C.2: sigmoid(10(score-.5))/(1+functioning direct children).

    Scores are bounded [0,1]. A score-one parent is ineligible in the paper's
    rule. Sampling is with replacement, over a stable ordering and batch seed.
    """
    _positive(count, "parent count", integer=True)
    children: dict[str, int] = {}
    for entry in archive:
        parent = entry.get("parent")
        if parent:
            children[parent] = children.get(parent, 0) + 1
    eligible, weights = [], []
    for entry in sorted(archive, key=lambda e: e["id"]):
        score = entry["search"]["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) \
                or not math.isfinite(score) or not 0 <= score <= 1:
            raise RSIError("archive contains an invalid search score")
        if score == 1:
            continue
        eligible.append(entry)
        weights.append(1 / (1 + math.exp(-10 * (score - .5)))
                       / (1 + children.get(entry["id"], 0)))
    if not eligible:
        return []
    rng = random.Random(f"gama-rsi:{seed}:{round_number}")
    return rng.choices(eligible, weights=weights, k=count)


def _evaluation(value: dict) -> Evaluation:
    return Evaluation(score=value["score"], samples=tuple(value["samples"]),
                      results=tuple(value["results"]))


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _repair_events(directory: Path) -> Optional[str]:
    """Preserve an interrupted final write before appending new JSONL records."""
    ledger = directory / "events.jsonl"
    if not ledger.exists() or ledger.stat().st_size == 0:
        return None
    with ledger.open("r+b") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return None
        position = fh.tell()
        boundary = 0
        while position:
            start = max(0, position - 65536)
            fh.seek(start)
            newline = fh.read(position - start).rfind(b"\n")
            if newline >= 0:
                boundary = start + newline + 1
                break
            position = start
        fragment = directory / f"events.interrupted-{uuid.uuid4().hex}.bin"
        fh.seek(boundary)
        with fragment.open("xb") as saved:
            shutil.copyfileobj(fh, saved)
        fh.truncate(boundary)
    return str(fragment)


@contextmanager
def _run_lock(directory: Path):
    # A kernel lock is released after SIGKILL too; a stale pid file is not a lock.
    try:
        import fcntl
    except ImportError as exc:
        raise RSIError("RSI run locking requires Linux/WSL or macOS") from exc
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RSIError(f"another RSI coordinator owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class _Session:
    def __init__(self, config: RSIConfig, repo: Path, directory: Path,
                 on_event: Optional[Callable[[dict], None]]):
        self.config = config
        self.repo, self.directory = repo, directory
        self.workspace = Workspaces(repo, directory / "worktrees")
        self.on_event = on_event
        self.cancel = threading.Event()
        self.active: dict[Path, str] = {}
        self.active_lock = threading.Lock()
        self.state: dict = {}

    def emit(self, event: str, **fields) -> None:
        row = {"event": event, **fields}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if self.on_event:
            self.on_event(row)

    def save(self) -> None:
        _atomic_json(self.directory / "state.json", self.state)

    def tree(self, commit: str, label: str) -> Path:
        # Git's worktree commands scan shared registry paths. Concurrent remove
        # calls can delete a sibling's metadata while another command scans it.
        # Serialize these short mutations, leaving generation/scoring parallel.
        with self.active_lock:
            path = self.workspace.create(f"{label}-{uuid.uuid4().hex[:12]}", commit)
            self.active[path] = commit
        return path

    def remove(self, path: Path) -> None:
        with self.active_lock:
            self.workspace.remove(path)
            self.active.pop(path, None)

    def close(self) -> None:
        self.cancel.set()
        errors = []
        for path in list(self.active):
            try:
                self.remove(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        if errors:
            self.emit("cleanup_failed", paths=errors)

    def fixed_inputs(self) -> None:
        if controller_fingerprint() != self.state["contract"]["controller"]:
            raise RSIError("the RSI controller source changed during this run")
        if _file_digests(self.config.evaluation_files) != self.state["contract"]["evaluation_files"]:
            raise RSIError("a declared evaluator input changed during this run")

    def parallel(self, function: Callable, items: list) -> list:
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(self.config.workers, len(items))) as pool:
            futures = [pool.submit(function, item) for item in items]
            try:
                # Task completion order cannot determine archive admission or a winner.
                return [future.result() for future in futures]
            except BaseException:
                self.cancel.set()
                for future in futures:
                    future.cancel()
                raise

    def check(self, path: Path) -> list[dict]:
        self.fixed_inputs()
        result = run_checks(self.config.checks, cwd=path,
                            timeout=self.config.evaluation_timeout, cancel=self.cancel)
        self.workspace.assert_clean(path, expected_commit=self.active[path])
        self.fixed_inputs()
        return result

    def score(self, path: Path, command: list[str], repeats: int) -> Evaluation:
        self.fixed_inputs()
        result = evaluate(command, cwd=path, timeout=self.config.evaluation_timeout,
                          repeats=repeats, cancel=self.cancel)
        self.workspace.assert_clean(path, expected_commit=self.active[path])
        self.fixed_inputs()
        return result

    def score_commit(self, commit: str, command: list[str], repeats: int,
                     label: str) -> Evaluation:
        path = self.tree(commit, label)
        try:
            self.check(path)
            return self.score(path, command, repeats)
        finally:
            self.remove(path)

    def initialize(self) -> None:
        base = self.state["base"]
        path = self.tree(base, "seed")
        try:
            self.workspace.read_sources(path, self.config.allowed_paths)
            self.check(path)
            search = self.score(path, self.config.search_command, self.config.search_repeats)
            confirm = self.score(path, self.config.confirm_command, self.config.confirm_repeats)
        finally:
            self.remove(path)
        entry = {"id": "seed", "commit": base, "tree": self.workspace.tree(base),
                 "parent": None, "agent": None, "round": -1,
                 "search": asdict(search), "confirm": asdict(confirm)}
        entry["ref"] = self.workspace.keep(base, f"{self.state['run_id']}/seed")
        self.state.update(archive=[entry], champion="seed", phase="ready", next_round=0)
        self.save()
        self.emit("seed", commit=base, search=search.score, confirm=confirm.score)

    def prepare(self, job: dict) -> dict:
        parent, agent = job["parent_entry"], job["agent_spec"]
        outcome = {k: job[k] for k in ("id", "round", "parent", "agent", "attempt", "artifact_dir")}
        artifacts = Path(job["artifact_dir"])
        artifacts.mkdir(parents=True, exist_ok=True)
        path = None
        keep = False
        try:
            path = self.tree(parent["commit"], job["id"])
            request = {
                "goal": self.config.goal, "parent": parent["commit"],
                "allowed_paths": list(self.config.allowed_paths),
                "files": self.workspace.read_sources(path, self.config.allowed_paths),
                "feedback": {"search": parent["search"],
                             "confirm": ({k: parent["confirm"][k] for k in ("score", "samples")}
                                         if parent.get("confirm") else None),
                             "recent_failures": self.state.get("recent_failures", [])},
                "papers": copy.deepcopy(self.config.papers),
            }
            _atomic_json(artifacts / "request.json", request)
            proposal = propose_patch(agent, request=request, cwd=path,
                                     timeout=self.config.timeout, cancel=self.cancel)
            (artifacts / "response.txt").write_text(proposal.output, encoding="utf-8")
            (artifacts / "proposal.diff").write_text(proposal.patch, encoding="utf-8")
            changed = self.workspace.apply_patch(path, proposal.patch, self.config.allowed_paths)
            # Every archived parent must still provide the declared editable files.
            self.workspace.read_sources(path, self.config.allowed_paths)
            commit = self.workspace.commit(
                path, parent["commit"], f"RSI {job['id']}: {self.config.goal[:160]}")
            with self.active_lock:
                self.active[path] = commit
            outcome.update(commit=commit, tree=self.workspace.tree(commit), changed=changed,
                           usage=proposal.usage, worktree=str(path), status="prepared")
            keep = True
        except Exception as exc:
            outcome.update(status="rejected", stage="proposal", error=f"{type(exc).__name__}: {exc}")
        finally:
            if path is not None and not keep:
                self.remove(path)
        return outcome

    def assess(self, outcome: dict) -> dict:
        outcome = dict(outcome)
        path = Path(outcome.pop("worktree"))
        try:
            outcome["checks"] = self.check(path)
            search = self.score(path, self.config.search_command, self.config.search_repeats)
            outcome.update(status="viable", search=asdict(search))
        except Exception as exc:
            outcome.update(status="rejected", stage="evaluation",
                           error=f"{type(exc).__name__}: {exc}")
        finally:
            self.remove(path)
        return outcome

    def round(self) -> bool:
        self.fixed_inputs()
        cfg = self.config
        gen = self.state["next_round"]
        parents = select_parents(self.state["archive"], cfg.batch_size,
                                 seed=cfg.seed, round_number=gen)
        if not parents:
            self.emit("stop", reason="all archived parents have search score 1", round=gen)
            return False
        attempt = uuid.uuid4().hex[:12]
        jobs = []
        for slot, parent in enumerate(parents):
            agent = cfg.agents[(gen * cfg.batch_size + slot) % len(cfg.agents)]
            ident = f"r{gen:04d}-s{slot:03d}"
            jobs.append({"id": ident, "round": gen, "parent": parent["id"],
                         "agent": agent["name"], "parent_entry": copy.deepcopy(parent),
                         "agent_spec": copy.deepcopy(agent),
                         "attempt": attempt,
                         "artifact_dir": str(self.directory / "attempts" / attempt / ident)})
        # Reserve the whole batch before dispatch. Failed/interrupted attempts remain
        # counted even if a later invocation retries this unfinished round.
        self.state["reserved_proposals"] += len(jobs)
        self.state["pending"] = {"round": gen, "attempt": attempt,
                                 "jobs": [{k: j[k] for k in ("id", "parent", "agent")} for j in jobs]}
        self.save()
        self.emit("round_start", **self.state["pending"], workers=cfg.workers)
        self.fixed_inputs()
        prepared = self.parallel(self.prepare, jobs)
        self.fixed_inputs()

        # Deduplicate BEFORE evaluation, including equal source produced concurrently.
        # Slot order picks the representative, independently of which LLM finished first.
        seen = {e["tree"] for e in self.state["archive"]}
        evaluate_these = []
        outcomes: dict[str, dict] = {}
        for result in prepared:
            if result["status"] != "prepared":
                outcomes[result["id"]] = result
            elif result["tree"] in seen:
                self.remove(Path(result.pop("worktree")))
                result.update(status="duplicate", stage="proposal")
                outcomes[result["id"]] = result
            else:
                seen.add(result["tree"])
                evaluate_these.append(result)
        for result in self.parallel(self.assess, evaluate_these):
            outcomes[result["id"]] = result
        self.fixed_inputs()

        # Stage archive changes in memory. On interruption, the prior round's checkpoint
        # remains the decision state, and this attempt's artifacts stay available.
        archive = copy.deepcopy(self.state["archive"])
        viable = []
        for job in jobs:
            result = outcomes[job["id"]]
            _atomic_json(Path(job["artifact_dir"]) / "result.json", result)
            self.emit("candidate", **result)
            if result["status"] == "viable":
                entry = {k: result[k] for k in
                         ("id", "round", "parent", "agent", "attempt", "commit", "tree", "search")}
                entry["ref"] = self.workspace.keep(
                    entry["commit"], f"{self.state['run_id']}/{entry['id']}-{entry['commit']}")
                archive.append(entry)
                viable.append(entry)
        current = next(e for e in archive if e["id"] == self.state["champion"])
        champion = current
        if viable:
            challenger = min(viable, key=lambda e: (-e["search"]["score"], e["id"]))
            if challenger["search"]["score"] >= current["search"]["score"]:
                # Confirmation is serialized: independent of shared evaluator/GPU load
                # caused by the candidate-search batch, and at most one challenge per round.
                before = self.score_commit(current["commit"], cfg.confirm_command,
                                           cfg.confirm_repeats, "incumbent-confirm")
                current["confirm"] = asdict(before)
                try:
                    after = self.score_commit(challenger["commit"], cfg.confirm_command,
                                              cfg.confirm_repeats, "candidate-confirm")
                except Exception as exc:
                    challenger["confirm_error"] = f"{type(exc).__name__}: {exc}"
                    self.emit("confirmation_failed", round=gen, candidate=challenger["id"],
                              error=challenger["confirm_error"])
                else:
                    accepted, reason = promotion(after, before, cfg.min_gain)
                    challenger["confirm"] = asdict(after)
                    self.emit("confirmation", round=gen, candidate=challenger["id"],
                              incumbent=current["id"], accepted=accepted, reason=reason,
                              before=asdict(before), after=asdict(after))
                    if accepted:
                        champion = challenger
            else:
                self.emit("confirmation_skipped", round=gen, candidate=challenger["id"],
                          reason="best new search score is below the incumbent")
        failures = [{"candidate": r["id"], "stage": r.get("stage"),
                     "error": r.get("error", "")[:6000]}
                    for r in outcomes.values() if r["status"] == "rejected"]
        self.fixed_inputs()
        self.state.update(archive=archive, champion=champion["id"], next_round=gen + 1,
                          pending=None, recent_failures=failures[-4:])
        self.save()
        self.emit("round_complete", round=gen, champion=champion["id"],
                  archived=len(viable), archive_size=len(archive))
        return True

    def finalize(self) -> None:
        if self.config.sealed_command is None:
            raise RSIError("finalization needs a separate sealed_command")
        # Once holdout is opened, this run can never search again—even if scoring
        # crashes halfway through. A resume may finish finalization only.
        self.state["phase"] = "finalizing"
        self.save()
        current = next(e for e in self.state["archive"] if e["id"] == self.state["champion"])
        if "sealed_base" not in self.state:
            score = self.score_commit(self.state["base"], self.config.sealed_command,
                                      self.config.confirm_repeats, "base-sealed")
            self.state["sealed_base"] = asdict(score)
            self.save()
        if "sealed_champion" not in self.state:
            score = (_evaluation(self.state["sealed_base"])
                     if current["commit"] == self.state["base"] else
                     self.score_commit(current["commit"], self.config.sealed_command,
                                       self.config.confirm_repeats, "champion-sealed"))
            self.state["sealed_champion"] = asdict(score)
            self.save()
        before, after = (_evaluation(self.state[k]) for k in ("sealed_base", "sealed_champion"))
        better, reason = promotion(after, before, self.config.min_gain)
        worse = max(after.samples) < min(before.samples) - self.config.min_gain
        verdict = "improved" if better else "regressed" if worse else "not_separable"
        self.state.update(phase="finalized", sealed_verdict=verdict)
        self.save()
        self.emit("finalized", verdict=verdict, reason=reason,
                  base=asdict(before), champion=asdict(after))

    def result(self) -> dict:
        self.fixed_inputs()
        current = next(e for e in self.state["archive"] if e["id"] == self.state["champion"])
        reference = self.workspace.keep(current["commit"], current["ref"].removeprefix("refs/gama-rsi/"))
        patch_dir = self.directory / "patches"
        patch_dir.mkdir(exist_ok=True)
        patch = patch_dir / f"{current['commit']}.patch"
        text = self.workspace.diff(self.state["base"], current["commit"])
        if patch.exists():
            if patch.read_bytes() != text.encode("utf-8"):
                raise RSIError(f"an exported immutable patch was changed: {patch}")
        else:
            _atomic_text(patch, text)
        # The summary points to the immutable artifact. Even an interrupted export
        # cannot change the patch referenced by the previously published result.
        latest = self.directory / "champion.patch"
        _atomic_text(latest, text)
        summary = {"phase": self.state["phase"], "base": self.state["base"],
                   "champion": copy.deepcopy(current), "ref": reference,
                   "archive_size": len(self.state["archive"]),
                   "rounds_completed": self.state["next_round"],
                   "reserved_proposals": self.state["reserved_proposals"],
                   "sealed_verdict": self.state.get("sealed_verdict", "not_opened"),
                   "sealed": ({"base": copy.deepcopy(self.state["sealed_base"]),
                               "champion": copy.deepcopy(self.state["sealed_champion"])}
                              if self.state["phase"] == "finalized" else None),
                   "patch": str(patch), "latest_patch": str(latest),
                   "state_dir": str(self.directory)}
        # asdict(Evaluation) retains tuples until checkpoint serialization. Return
        # the same JSON-shaped values before and after a process restart.
        summary = json.loads(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        _atomic_json(self.directory / "result.json", summary)
        return summary


def run_rsi(config: RSIConfig | dict, *, repo, state_dir, rounds: int = 1,
            resume: bool = False, finalize: bool = False,
            on_event: Optional[Callable[[dict], None]] = None) -> dict:
    """Run bounded source-evolution batches, or finalize an existing run's holdout.

    ``rounds`` is additional work per invocation, not a changed frozen budget.
    ``resume`` can extend a ready run. ``finalize`` opens the sealed evaluator and
    permanently closes that run to search. The caller's checkout is never replaced.
    """
    config = RSIConfig.from_dict(asdict(config) if isinstance(config, RSIConfig) else config)
    _positive(rounds, "rounds", integer=True)
    repo, directory = Path(repo).resolve(), Path(state_dir).resolve()
    if directory == repo or repo in directory.parents:
        raise ValueError("state_dir must be outside the repository checkout")
    if finalize and config.sealed_command is None:
        raise ValueError("finalize requires a sealed_command")
    contract = {"schema": SCHEMA_VERSION, "config": asdict(config),
                "repo": str(repo), "controller": controller_fingerprint(),
                "python": [sys.version_info.major, sys.version_info.minor],
                "evaluation_files": _file_digests(config.evaluation_files)}
    with _run_lock(directory):
        checkpoint = directory / "state.json"
        if not resume:
            if checkpoint.exists():
                raise RSIError(f"checkpoint already exists in {directory}; use --resume")
            if any(p.name != "run.lock" for p in directory.iterdir()):
                raise RSIError("a new run needs an empty state directory")
            if finalize:
                raise RSIError("initialize and run search before --resume --finalize")
        session = _Session(config, repo, directory, on_event)
        try:
            _validate_evaluation_scope(config, session.workspace.repo)
            if resume:
                if not checkpoint.is_file():
                    raise RSIError(f"no checkpoint in {directory}")
                session.state = json.loads(checkpoint.read_text(encoding="utf-8"))
                if session.state.get("contract_hash") != _digest(contract):
                    raise RSIError("cannot resume with changed config, repository, controller "
                                   "or Python version; use the frozen inputs or a new run")
                phase = session.state.get("phase")
                if phase in ("finalizing", "finalized") and not finalize:
                    raise RSIError("the sealed evaluator has been opened; this run cannot search "
                                   "again (use --finalize to finish/report it)")
                recovered = [str(path) for path in session.workspace.recover()]
                fragment = _repair_events(directory)
                session.emit("resumed", next_round=session.state["next_round"],
                             interrupted_attempt=session.state.get("pending"),
                             recovered_event_fragment=fragment, recovered_worktrees=recovered)
            else:
                session.state = {
                    "contract_hash": _digest(contract), "contract": contract,
                    "run_id": uuid.uuid4().hex, "base": session.workspace.head(),
                    "phase": "initializing", "next_round": 0, "reserved_proposals": 0,
                    "archive": [], "pending": None,
                }
                session.workspace.keep(session.state["base"], f"{session.state['run_id']}/seed")
                session.save()
            if session.state["phase"] == "initializing":
                session.initialize()
            if finalize:
                if session.state["phase"] != "finalized":
                    session.finalize()
            else:
                for _ in range(rounds):
                    if not session.round():
                        break
            return session.result()
        finally:
            session.close()
