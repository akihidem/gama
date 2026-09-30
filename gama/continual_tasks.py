"""Portable continual goals and frozen unittest evaluators.

Limits: 64-byte IDs, 512-byte titles, 8192-byte goal text, 16 target files,
60000 total UTF-8 source bytes and 6144 UTF-8 bytes per test. Descriptor JSON
is limited to 128 KiB; control files and manifests to 16 MiB.

The CLI deliberately imports only stdlib code until candidate cwd is first on
sys.path. Timeouts and process-tree cancellation belong to the invoking guard.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
import unittest

SPLITS = ("search", "confirm", "sealed")
MAX_TEST_BYTES = 6144
MAX_SOURCE_BYTES = 60000
MAX_DESCRIPTOR_BYTES = 128 * 1024
MAX_CONTROL_BYTES = 16 * 1024 * 1024
GOAL_KEYS = {"id", "title", "goal", "allowed_paths", "tests"}


def _shape(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label}: expected exactly {sorted(keys)}")


def _text(value, label, limit):
    if type(value) is not str or not value.strip() or "\0" in value:
        raise ValueError(f"{label}: expected nonempty text without NUL")
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{label}: exceeds {limit} UTF-8 bytes")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label}: expected a positive finite number")
    return value


def _constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        _constant(value)
    return result


def _loads(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant, parse_float=_float)


def _dump(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise ValueError(f"expected an absolute path without symlinks: {path}")
    return path


def _read(path, limit=MAX_CONTROL_BYTES):
    path = _path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"not a bounded regular file: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds {limit} bytes: {path}")
    return data


def _disjoint(left, right):
    if left.is_relative_to(right) or right.is_relative_to(left):
        raise ValueError(f"paths must be disjoint: {left}, {right}")


def _git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    proc = subprocess.run(["git", *args], cwd=repo, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if proc.returncode:
        raise RuntimeError("git validation failed: " +
                           proc.stderr.decode("utf-8", "replace")[:1000])
    return proc.stdout


def _index(repo, paths):
    output = _git(repo, "--literal-pathspecs", "ls-files", "--stage", "-z", "--", *paths)
    entries = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        header, name = record.decode("utf-8").split("\t", 1)
        mode, _, stage = header.split()
        if stage != "0" or name in entries or mode not in ("100644", "100755"):
            raise ValueError(f"not an unambiguous tracked regular file: {name}")
        entries[name] = mode
    return entries


def _goal(raw):
    _shape(raw, GOAL_KEYS, "goal")
    ident = _text(raw["id"], "id", 64)
    if re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", ident) is None:
        raise ValueError("id must be a lowercase slug")
    _text(raw["title"], "title", 512)
    _text(raw["goal"], "goal", 8192)
    paths = raw["allowed_paths"]
    if type(paths) is not list or not 1 <= len(paths) <= 16:
        raise ValueError("allowed_paths must contain 1 to 16 exact source paths")
    for name in paths:
        _text(name, "allowed path", 255)
        parts = name.split("/")
        if (len(parts) != 2 or parts[0] != "gama" or
                not parts[1].endswith(".py") or parts[1][:-3] in ("", ".", "..") or
                any(c in name for c in "\\*?[]") or any(ord(c) < 32 for c in name)):
            raise ValueError(f"not an exact gama Python module path: {name}")
        if parts[1].startswith(("rsi", "continual")) or parts[1] in (
                "__init__.py", "__main__.py", "cli.py"):
            raise ValueError(f"protected source path: {name}")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate allowed path")
    _shape(raw["tests"], SPLITS, "tests")
    for split in SPLITS:
        source = _text(raw["tests"][split], f"{split} test", MAX_TEST_BYTES)
        try:
            compile(source, f"<{ident}/{split}>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError(f"invalid {split} test source: {exc}") from exc
    if len(_dump(raw)) > MAX_DESCRIPTOR_BYTES:
        raise ValueError("descriptor exceeds 128 KiB")
    # Copy containers without normalizing newlines or Unicode in test source.
    return {**raw, "allowed_paths": list(paths), "tests": dict(raw["tests"])}


def validate_goal(raw: dict, repo: Path) -> dict:
    goal = _goal(raw)
    repo = _path(repo)
    if Path(_git(repo, "rev-parse", "--show-toplevel").decode("utf-8").rstrip("\n")) != repo:
        raise ValueError("repo must be the repository root")
    if set(_index(repo, goal["allowed_paths"])) != set(goal["allowed_paths"]):
        raise ValueError("every allowed path must already be tracked")
    total = 0
    for name in goal["allowed_paths"]:
        data = _read(repo / name, MAX_SOURCE_BYTES)
        data.decode("utf-8")
        total += len(data)
    if total > MAX_SOURCE_BYTES:
        raise ValueError("combined target source exceeds 60000 UTF-8 bytes")
    return goal


def _configuration(raw):
    required = {"campaign_id", "repo", "state_dir", "branch", "remote", "bridge_config",
                "goals", "evaluation_timeout", "max_goal_cycles"}
    optional = {"checks", "bootstrap_history", "workers", "max_proposals_per_tick",
                "schedule_hours", "timezone"}
    if type(raw) is not dict or not required <= set(raw) or set(raw) - required - optional:
        raise ValueError("invalid continual configuration fields")
    cfg = _loads(_dump(raw))
    _text(cfg["campaign_id"], "campaign_id", 64)
    if re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", cfg["campaign_id"]) is None:
        raise ValueError("campaign_id must be a lowercase slug")
    for key, value in (("workers", 2), ("max_proposals_per_tick", 4),
                       ("schedule_hours", [9, 21]), ("timezone", "Asia/Tokyo")):
        cfg.setdefault(key, value)
        if type(cfg[key]) is not type(value) or cfg[key] != value:
            raise ValueError(f"{key} is fixed at {value!r}")
    if any(type(hour) is not int for hour in cfg["schedule_hours"]):
        raise ValueError("schedule hours must be integers")
    _number(cfg["evaluation_timeout"], "evaluation_timeout")
    if type(cfg["max_goal_cycles"]) is not int or cfg["max_goal_cycles"] <= 0:
        raise ValueError("max_goal_cycles must be a positive integer")
    for key in ("repo", "state_dir", "bridge_config"):
        cfg[key] = str(_path(cfg[key]))
    _disjoint(Path(cfg["repo"]), Path(cfg["state_dir"]))
    _text(cfg["branch"], "branch", 256)
    _text(cfg["remote"], "remote", 128)
    if cfg["branch"] in ("main", "master"):
        raise ValueError("a feature branch is required")
    if _git(Path(cfg["repo"]), "symbolic-ref", "--short", "HEAD").decode().strip() != cfg["branch"]:
        raise ValueError("configured feature branch is not checked out")
    if type(cfg["goals"]) is not list:
        raise ValueError("goals must be a list of absolute descriptor paths")
    cfg["goals"] = [str(_path(p)) for p in cfg["goals"]]
    if len(set(cfg["goals"])) != len(cfg["goals"]):
        raise ValueError("duplicate initial descriptor path")
    if "bootstrap_history" in cfg and type(cfg["bootstrap_history"]) is not list:
        raise ValueError("bootstrap_history must be a list")
    cfg.setdefault("checks", [[sys.executable, "-B", "-m", "unittest", "discover",
                              "-s", "tests", "-t", ".", "-q"]])
    if type(cfg["checks"]) is not list or not cfg["checks"]:
        raise ValueError("at least one mandatory check is required")
    for command in cfg["checks"]:
        if type(command) is not list or not command:
            raise ValueError("checks must be nonempty argv arrays")
        for arg in command:
            _text(arg, "check argument", 16384)
    return cfg


def _check_inputs(repo, checks):
    # Freeze helper/data files too: argv alone misses unittest discovery imports.
    paths = {repo / name for name in _index(repo, [
        "tests", "validation", "pyproject.toml", "setup.cfg", "tox.ini",
        "pytest.ini", "conftest.py"])}
    for command in checks:
        for index, arg in enumerate(command):
            if arg.startswith("-") or (index == 0 and not arg.endswith((".py", ".sh"))):
                continue
            path = Path(arg)
            path = path if path.is_absolute() else repo / path
            try:
                if path.is_file():
                    paths.add(_path(path))
            except OSError:
                continue
        if "-m" in command:
            index = command.index("-m") + 1
            if index < len(command) and re.fullmatch(r"\w+(?:\.\w+)*", command[index]):
                module = repo.joinpath(*command[index].split("."))
                for path in (module.with_suffix(".py"), module / "__init__.py",
                             module / "__main__.py"):
                    if path.is_file():
                        paths.add(_path(path))
    return paths


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze_goal(config: dict, goal: dict, directory: Path, prior_goals: list) -> Path:
    cfg = _configuration(config)
    repo, directory = Path(cfg["repo"]), _path(directory)
    _disjoint(repo, directory)
    goal = validate_goal(goal, repo)
    if type(prior_goals) is not list:
        raise ValueError("prior_goals must be a list of absolute descriptor paths")
    # Lazy sibling import keeps the standalone evaluator independent of gama.
    from .rsi_runtime import load_bridge_config

    bridge_source = Path(cfg["bridge_config"])
    external = {bridge_source: _read(bridge_source)}
    original_bridge = _loads(external[bridge_source])
    validated_bridge = load_bridge_config(bridge_source)
    if _read(bridge_source) != external[bridge_source]:
        raise RuntimeError("bridge changed while freezing")
    artifacts = _path(original_bridge["artifact_root"])
    for path in (repo, Path(cfg["state_dir"]), directory):
        _disjoint(artifacts, path)
    key = _sha(str(directory).encode("utf-8"))[:24]
    bridge = {**original_bridge, "artifact_root": str(_path(
        artifacts / "continual" / cfg["campaign_id"] / key))}
    bridge_timeout = _number(validated_bridge["timeout"], "bridge timeout")
    planned = {}
    descriptor_bytes = _dump(goal)
    for name in cfg["goals"]:
        path = Path(name)
        data = _read(path, MAX_DESCRIPTOR_BYTES)
        external[path] = data
        initial = _goal(_loads(data))
        if initial == goal:
            descriptor_bytes = data
    planned[directory / "goal.json"] = descriptor_bytes
    for split in SPLITS:
        planned[directory / f"{split}.py"] = goal["tests"][split].encode("utf-8")
    retained, ids = [], {goal["id"]}
    for name in prior_goals:
        path = _path(name)
        data = _read(path, MAX_DESCRIPTOR_BYTES)
        # Accepted history survives later changes to source size or layout.
        prior = _goal(_loads(data))
        if prior["id"] in ids:
            raise ValueError(f"duplicate goal in regression history: {prior['id']}")
        ids.add(prior["id"])
        external[path] = data
        target = directory / "regressions" / prior["id"]
        descriptor = target / "goal.json"
        planned[descriptor] = data
        entry = {"descriptor": str(descriptor), "sha256": _sha(data), "tests": {}}
        for split in SPLITS:
            script = target / f"{split}.py"
            body = prior["tests"][split].encode("utf-8")
            planned[script] = body
            entry["tests"][split] = {"path": str(script), "sha256": _sha(body)}
        retained.append(entry)
    cli = Path(__file__).resolve()
    for parent in {cli.parent, repo / "gama"}:
        for pattern in ("continual*.py", "rsi*.py", "__init__.py", "__main__.py", "cli.py"):
            for path in parent.glob(pattern):
                external[_path(path)] = _read(path)
    for path in _check_inputs(repo, cfg["checks"]):
        external[_path(path)] = _read(path)
    if any(repo / name in external for name in goal["allowed_paths"]):
        raise ValueError("a source mutation target is also an evaluator")
    manifest = directory / "regressions.json"
    planned[manifest] = _dump({"schema_version": 1, "goals": retained})
    planned[directory / "campaign.json"] = _dump(cfg)
    bridge_path, rsi_path = directory / "bridge.json", directory / "rsi.json"
    mission_path, inventory = directory / "mission.json", directory / "inputs.json"
    planned[bridge_path] = _dump(bridge)
    python = sys.executable  # Resolving the venv symlink can select a different Python.
    if not Path(python).is_absolute():
        raise ValueError("the interpreter must have an absolute executable spelling")
    prefix = [python, "-I", "-B", str(cli)]
    checks = list(cfg["checks"])
    if retained:
        checks.append(prefix + ["regressions", "--goals", str(manifest)])
    inputs = set(planned) | set(external) | {rsi_path, mission_path, inventory, cli}
    rsi = {
        "goal": goal["goal"], "allowed_paths": goal["allowed_paths"],
        # load_inputs replaces this with the two fixed bridge agents.
        "agents": [], "checks": checks, "workers": 2, "batch_size": 2,
        "timeout": max(600, bridge_timeout + 5),
        "evaluation_timeout": cfg["evaluation_timeout"],
        "search_repeats": 1, "confirm_repeats": 3, "min_gain": 0,
        "evaluation_files": [str(path) for path in sorted(inputs)],
    }
    for split in SPLITS:
        rsi[split + "_command"] = prefix + ["score", "--test", str(directory / f"{split}.py")]
    planned[rsi_path] = _dump(rsi)
    planned[mission_path] = _dump({
        "mission_id": "continual-" + key, "repo": str(repo),
        "state_dir": str(directory / "state"), "rsi_config": str(rsi_path),
        "bridge_config": str(bridge_path), "rounds_per_cycle": 1,
        "batch_size": 2, "max_reservations_per_cycle": 4,
        "search_ceiling": 1, "confirm_ceiling": 1,
    })
    if set(planned) & set(external):
        raise ValueError("frozen output overlaps an input file")
    # The inventory hashes everything except itself; the core hashes it too.
    planned[inventory] = _dump({
        "schema_version": 1, "python": python,
        "files": {str(path): _sha(data) for path, data in sorted({**external, **planned}.items())},
    })
    if any(len(data) > MAX_CONTROL_BYTES for data in planned.values()):
        raise ValueError("frozen control files exceed 16 MiB")
    if directory.exists():
        for path, data in planned.items():
            if _read(path) != data:
                raise RuntimeError(f"existing frozen input differs: {path}")
        return mission_path
    # An interrupted creation is evidence, never a directory to overwrite.
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    folders = {directory.parent}
    for path, data in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        folders.update(p for p in path.parents if p.is_relative_to(directory))
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    for parent in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        _fsync_dir(parent)
    return mission_path


@contextlib.contextmanager
def _quiet():
    # Redirect inherited fds as well as Python streams, including subprocess noise.
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield sink
    finally:
        for fd, backup in zip((1, 2), saved):
            os.dup2(backup, fd)
            os.close(backup)


class _Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = 0

    def addSuccess(self, test):
        self.passed += 1
        super().addSuccess(test)


def _suite(body, filename):
    repo = Path.cwd().resolve()
    if not _path(repo / "gama" / "__init__.py").is_file():
        raise ValueError("candidate cwd must contain a regular gama package")
    source = body.decode("utf-8")
    code = compile(source, str(filename), "exec", dont_inherit=True)
    name = "_continual_test_" + _sha(str(filename).encode("utf-8"))
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(filename), ""
    previous = {k: v for k, v in sys.modules.items() if k == "gama" or k.startswith("gama.")}
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    old_module = sys.modules.get(name)
    try:
        for key in previous:
            del sys.modules[key]
        sys.path.insert(0, str(repo))
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        sys.modules[name] = module
        with _quiet() as sink:
            # A non-main module name intentionally bypasses unittest.main guards.
            exec(code, module.__dict__)
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            if loader.errors or suite.countTestCases() == 0:
                raise ValueError("suite failed to load or contains zero tests")
            result = unittest.TextTestRunner(stream=sink, verbosity=0,
                                             resultclass=_Result).run(suite)
        if result.testsRun <= 0 or not 0 <= result.passed <= result.testsRun:
            raise ValueError("suite did not execute a positive, consistent test count")
        for key, loaded in list(sys.modules.items()):
            if key == "gama" or key.startswith("gama."):
                origin = getattr(loaded, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(repo / "gama"):
                    raise ValueError(f"import escaped the candidate package: {key}")
        # Skips and expected failures are not passing behavioral measurements.
        return result.passed, result.testsRun, len(result.failures), len(result.errors)
    finally:
        os.chdir(repo)
        sys.path, sys.dont_write_bytecode = old_path, old_bytecode
        for key in list(sys.modules):
            if key == "gama" or key.startswith("gama."):
                del sys.modules[key]
        sys.modules.update(previous)
        if old_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module


def _verified(path, digest, limit):
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid frozen SHA-256")
    path = _path(path)
    body = _read(path, limit)
    if _sha(body) != digest:
        raise ValueError(f"changed frozen regression bytes: {path}")
    return path, body


def _regressions(path):
    manifest = _loads(_read(_path(path)))
    _shape(manifest, {"schema_version", "goals"}, "regression manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unknown regression manifest version")
    if type(manifest["goals"]) is not list or not manifest["goals"]:
        raise ValueError("regression manifest must contain at least one goal")
    suites, ids, paths = [], set(), set()
    for entry in manifest["goals"]:
        _shape(entry, {"descriptor", "sha256", "tests"}, "regression entry")
        descriptor, data = _verified(entry["descriptor"], entry["sha256"], MAX_DESCRIPTOR_BYTES)
        goal = _goal(_loads(data))
        if goal["id"] in ids or descriptor in paths:
            raise ValueError("duplicate retained regression")
        ids.add(goal["id"])
        paths.add(descriptor)
        _shape(entry["tests"], SPLITS, "regression tests")
        for split in SPLITS:
            item = entry["tests"][split]
            _shape(item, {"path", "sha256"}, "regression script")
            script, body = _verified(item["path"], item["sha256"], MAX_TEST_BYTES)
            if script in paths or body != goal["tests"][split].encode("utf-8"):
                raise ValueError("regression script is duplicated or differs from its descriptor")
            paths.add(script)
            suites.append((body, script))
    # Verify all bytes before executing any retained suite, including sealed.
    total = (0, 0, 0, 0)
    for body, script in suites:
        total = tuple(a + b for a, b in zip(total, _suite(body, script)))
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("score").add_argument("--test", required=True)
    sub.add_parser("regressions").add_argument("--goals", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            path = _path(args.test)
            counts = _suite(_read(path, MAX_TEST_BYTES), path)
        else:
            counts = _regressions(args.goals)
        passed, tests_run, failures, errors = counts
        result = {"score": passed / tests_run, "tests_run": tests_run,
                  "failures": failures, "errors": errors}
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
        return 0 if args.command == "score" or passed == tests_run else 1
    except (Exception, SystemExit) as exc:
        print(f"continual evaluator: {type(exc).__name__}: {str(exc)[:1000]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
