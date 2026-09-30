"""Verified publication of sealed winners; never invokes proposal/model work.

The caller owns the campaign lock and proposal ledger. Its save callback must
atomically fsync each journal snapshot before returning.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import re
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import uuid
from collections import Counter


class PublicationError(RuntimeError):
    """Publication was refused without discarding operator or remote work."""


def _need(condition, message):
    if not condition:
        raise PublicationError(message)


def _stop(cancel):
    if cancel is not None and cancel.is_set():
        raise PublicationError("publication stopped; resume retains the selected release")


def _pairs(items):
    result = {}
    for key, value in items:
        _need(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(text):
    def invalid(value):
        raise PublicationError("nonfinite JSON value: " + value)
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid,
                          parse_float=lambda number: float(number) if math.isfinite(float(number)) else invalid(number))
    except (ValueError, UnicodeError) as exc:
        raise PublicationError("invalid publication JSON: " + str(exc)) from exc


def _dump(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PublicationError("nonportable publication data: " + str(exc)) from exc


def _path(value):
    _need(isinstance(value, (str, Path)), "expected an absolute filesystem path")
    path = Path(value)
    _need(path.is_absolute() and path.resolve() == path,
          "path must be absolute and free of symlinks: " + str(path))
    return path


def _disjoint(left, right):
    _need(left != right and left not in right.parents and right not in left.parents,
          "publication state and repository must be disjoint")


def _bytes(path, limit=8 * 1024 * 1024):
    path = _path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            _need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
                  "not a regular evidence file: " + str(path))
            data = stream.read(limit + 1)
        _need(len(data) <= limit, "oversize evidence file: " + str(path))
        return data
    except OSError as exc:
        raise PublicationError("cannot read evidence " + str(path) + ": " + str(exc)) from exc


def _object(path):
    value = _json(_bytes(path))
    _need(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _oid(value):
    _need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", value),
          "invalid Git object ID")
    return value


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _argv(value):
    _need(isinstance(value, list) and value and value[0]
          and all(isinstance(arg, str) and "\0" not in arg for arg in value),
          "expected a nonempty command argv array")
    return value


def _sibling(name):
    package = "_gama_continual_publication"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path(__file__).resolve().parent)]
        sys.modules[package] = module
    return importlib.import_module(package + "." + name)


def _git_environment():
    # Inherited Git plumbing variables must not redirect a checked cwd/index.
    blocked = {"GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
               "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
               "GIT_NAMESPACE", "GIT_REPLACE_REF_BASE", "GIT_SHALLOW_FILE",
               "GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"}
    for key in list(os.environ):
        if key in blocked or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            os.environ.pop(key, None)
    os.environ["GIT_NO_REPLACE_OBJECTS"] = "1"
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"


class _Runner:
    def __init__(self, repo, root, timeout, cancel):
        self.repo, self.root, self.timeout, self.cancel = repo, root, timeout, cancel
        self.logs = _path(root / "commands")

    def __enter__(self):
        lease = os.open(self.root / "drain.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            deadline = time.monotonic() + 8
            while True:
                _stop(self.cancel)
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    _need(time.monotonic() < deadline, "previous publication guardians have not drained")
                    time.sleep(0.02)
            marker = self.root / "drain.json"
            if marker.exists():
                _need(_object(marker).get("drained") is True, "previous publication containment is unproven")
            else:
                _need(not self.logs.exists() or not any(self.logs.iterdir()),
                      "prior commands have no drainage lease; preserve their evidence")
            self.logs.mkdir(parents=True, exist_ok=True)
            # The inherited lease outlives SIGKILL; keep the venv executable spelling.
            _put(marker, {"drained": False})
            with (self.root / ("driver-" + uuid.uuid4().hex + ".log")).open("xb") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                     "_guard", str(self.root), str(lease)],
                    cwd=self.repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, pass_fds=(lease,), start_new_session=True)
            return self
        finally:
            os.close(lease)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=8)
            if exc_type is None:
                _need(self.process.returncode == 0, "publication guardian driver failed")
        except subprocess.TimeoutExpired:
            if exc_type is None:
                raise PublicationError("publication guardians are still draining; retain the journal")
        finally:
            self.process.stdout.close()

    def run(self, command, *, cwd=None, timeout=None, accepted=(0,), input_text=""):
        _stop(self.cancel)
        command = _argv(command)
        artifact = self.logs / uuid.uuid4().hex
        duration = timeout or self.timeout
        request = {"command": command, "cwd": str(cwd or self.repo), "timeout": duration,
                   "artifact_dir": str(artifact), "input_text": input_text}
        deadline = time.monotonic() + duration + 10
        pending = memoryview((_dump(request) + "\n").encode("utf-8"))
        fd = self.process.stdin.fileno()
        blocking = os.get_blocking(fd)
        try:
            # Raw writes leave no buffered data for __exit__ to flush after STOP.
            os.set_blocking(fd, False)
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_WRITE)
                while pending:
                    _stop(self.cancel)
                    remaining = deadline - time.monotonic()
                    _need(remaining > 0, "publication guardian response deadline exceeded")
                    if not selector.select(min(0.05, remaining)):
                        continue
                    _stop(self.cancel)
                    _need(time.monotonic() < deadline, "publication guardian response deadline exceeded")
                    try:
                        sent = os.write(fd, pending[:65536])
                    except BlockingIOError:
                        continue
                    _need(sent > 0, "publication guardian request write made no progress")
                    pending = pending[sent:]
        finally:
            os.set_blocking(fd, blocking)
        data = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in data:
                _stop(self.cancel)
                _need(time.monotonic() < deadline, "publication guardian response deadline exceeded")
                if not selector.select(0.05):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                _need(chunk, "publication guardian exited without a drainage receipt")
                data.extend(chunk)
                _need(len(data) <= 16 * 1024 * 1024, "oversize publication guardian response")
        result = _json(bytes(data))
        _need(isinstance(result, dict), "invalid publication guardian response")
        _need(not result.get("error"), "publication command failed: " + str(result.get("error")))
        _need(result["returncode"] in accepted,
              "publication command failed (%s): %s; %s" %
              (result["returncode"], command[0], result["stderr"][-2000:]))
        return {"command": command, "returncode": result["returncode"],
                "stdout": result["stdout"], "stderr": result["stderr"],
                "artifact_dir": str(artifact)}

    def git(self, *args, cwd=None, accepted=(0,)):
        command = ["git", "--no-replace-objects", "-c", "core.fsmonitor=false",
                   "-c", "submodule.recurse=false", *args]
        return self.run(command, cwd=cwd, timeout=60, accepted=accepted)["stdout"]

    def worker(self, request):
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_workspace"]
        value = _json(self.run(command, timeout=120, input_text=_dump(request))["stdout"])
        _need(isinstance(value, dict), "invalid publication worker receipt")
        return value


@contextlib.contextmanager
def _lock(root):
    fd = os.open(root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublicationError("another publisher owns this mission") from exc
        yield
    finally:
        os.close(fd)


def _checkout(runner, branch, allowed, *, clean=True):
    _need(runner.git("symbolic-ref", "--quiet", "HEAD").strip() == "refs/heads/" + branch,
          "caller is not on the configured feature branch")
    head = _oid(runner.git("rev-parse", "--verify", "HEAD").strip())
    _need(head in allowed, "unexpected caller HEAD: " + head)
    if clean:
        _need(not runner.git("status", "--porcelain=v1", "--untracked-files=all"),
              "caller has tracked, staged, or untracked changes; preserve them before publication")
    return head


def _transport(runner, remote):
    fetch = runner.git("remote", "get-url", "--all", remote).splitlines()
    push = runner.git("remote", "get-url", "--push", "--all", remote).splitlines()
    _need(len(fetch) == len(push) == 1 and fetch == push and fetch[0],
          "remote must have one identical fetch/push destination for SHA verification")
    return _hash(fetch[0].encode("utf-8"))


def _remote_head(runner, remote, branch):
    ref = "refs/heads/" + branch
    text = runner.git("ls-remote", "--exit-code", "--refs", remote, ref, accepted=(0, 2))
    if not text.strip():
        return None
    rows = text.strip().splitlines()
    _need(len(rows) == 1, "ambiguous remote branch evidence")
    fields = rows[0].split()
    _need(len(fields) == 2 and fields[1] == ref, "unexpected ls-remote response")
    return _oid(fields[0])


def _tree(runner, commit, paths):
    entries = {}
    for row in runner.git("ls-tree", "-rz", commit, "--", *paths).split("\0"):
        if not row:
            continue
        metadata, separator, name = row.partition("\t")
        fields = metadata.split()
        _need(separator and len(fields) == 3 and name not in entries, "invalid Git tree listing")
        entries[name] = tuple(fields)
    return entries


def _measurement(value, repeats, command, label):
    _need(isinstance(value, dict), "missing " + label + " measurement")
    samples, results = value.get("samples"), value.get("results")
    _need(isinstance(samples, list) and len(samples) == repeats
          and all(_number(item) and 0 <= item <= 1 for item in samples),
          "invalid " + label + " samples")
    _need(_number(value.get("score")) and min(samples) <= value["score"] <= max(samples),
          "invalid " + label + " score")
    _need(isinstance(results, list) and len(results) == repeats,
          "missing " + label + " evaluator receipts")
    for sample, result in zip(samples, results):
        _need(isinstance(result, dict) and result, "empty " + label + " evaluator receipt")
        if "returncode" in result:
            _need(type(result["returncode"]) is int and result["returncode"] == 0,
                  label + " evaluator failed")
        if "command" in result:
            _need(result["command"] == command, label + " evaluator command changed")
        if "score" in result:
            _need(_number(result["score"]) and result["score"] == sample,
                  label + " receipt and sample disagree")
    return samples


def _checks(receipts, commands):
    _need(isinstance(receipts, list) and receipts, "missing mandatory check evidence")
    actual = []
    for receipt in receipts:
        _need(isinstance(receipt, dict) and type(receipt.get("returncode")) is int
              and receipt["returncode"] == 0, "mandatory check did not succeed")
        actual.append(tuple(_argv(receipt.get("command"))))
    _need(not (Counter(map(tuple, commands)) - Counter(actual)),
          "mandatory command is absent from accepted check receipts")


def _unchanged(files):
    for name, digest in files.items():
        _need(_hash(_bytes(Path(name))) == digest, "frozen publication evidence changed: " + name)


def _accepted(config, goal, mission_path, base, runner):
    mission = _object(mission_path)
    mission_state = _path(mission.get("state_dir"))
    _disjoint(runner.repo, mission_state)
    core = _path(mission_state / "rsi")
    state_path, result_path = core / "state.json", core / "result.json"
    state, result = _object(state_path), _object(result_path)
    _need(state.get("phase") == result.get("phase") == "finalized",
          "source has not been finalized by the RSI core")
    _need(state.get("sealed_verdict") == result.get("sealed_verdict") == "improved",
          "core sealed verdict is not improved")
    _need(state.get("base") == result.get("base") == base,
          "core candidate base differs from campaign base_commit")
    _need(_path(result.get("state_dir")) == core, "core result names a different state directory")
    contract = state.get("contract")
    _need(isinstance(contract, dict) and type(contract.get("schema")) is int and contract["schema"] == 1,
          "missing frozen core contract")
    _need(state.get("contract_hash") == _hash(_dump(contract).encode("utf-8")),
          "recorded core contract hash does not authenticate its controls")
    _need(_path(contract.get("repo")) == runner.repo, "core contract belongs to another repository")
    _need(contract.get("python") == [sys.version_info.major, sys.version_info.minor],
          "publication interpreter differs from the accepted core interpreter")
    controls = contract.get("config")
    _need(isinstance(controls, dict), "missing frozen RSI controls")
    commands = controls.get("checks")
    _need(isinstance(commands, list) and commands, "no frozen mandatory commands")
    commands = [_argv(command) for command in commands]
    if "checks" in config:
        required = config["checks"]
        _need(isinstance(required, list) and required, "campaign checks must be nonempty")
        required = [_argv(command) for command in required]
        _need(not (Counter(map(tuple, required)) - Counter(map(tuple, commands))),
              "campaign mandatory checks are absent from the frozen core")
    else:
        _need(any(["-m", "unittest", "discover"] == command[i:i + 3]
                  for command in commands for i in range(len(command) - 2)),
              "default full unittest suite is absent from the frozen core")
    repeats = controls.get("confirm_repeats")
    _need(type(repeats) is int and repeats == 3, "three confirmation/sealed repeats are required")
    _need(type(controls.get("search_repeats")) is int and controls["search_repeats"] == 1, "one frozen search repeat is required")
    gain = controls.get("min_gain")
    _need(_number(gain) and gain == 0, "unexpected frozen improvement threshold")
    archive = state.get("archive")
    _need(isinstance(archive, list), "missing core archive")
    winners = [row for row in archive if isinstance(row, dict) and row.get("id") == state.get("champion")]
    _need(len(winners) == 1 and isinstance(result.get("champion"), dict),
          "missing or ambiguous finalized champion")
    winner = winners[0]
    _need(result["champion"] == winner, "core state and result disagree on the champion")
    source = _oid(winner.get("commit"))
    _need(source != base, "sealed winner has no source improvement")
    _need(_oid(winner.get("tree")) == runner.git("rev-parse", source + "^{tree}").strip(),
          "accepted candidate tree does not match its commit")
    runner.git("merge-base", "--is-ancestor", base, source)
    ref = result.get("ref")
    _need(isinstance(ref, str) and ref.startswith("refs/gama-rsi/"), "missing retained source ref")
    _need(runner.git("rev-parse", "--verify", "--end-of-options", ref + "^{commit}").strip() == source,
          "core source ref no longer retains the champion")
    validated = runner.worker({"action": "validate", "repo": str(runner.repo), "goal": goal})
    _need(validated.get("goal") == goal, "portable goal validation changed its meaning")
    allowed = goal.get("allowed_paths")
    _need(isinstance(allowed, list) and allowed, "goal has no source mutation paths")
    permissions = _argv(controls.get("allowed_paths"))
    _need(len(permissions) == len(allowed) and set(permissions) == set(allowed), "goal differs from frozen source permissions")
    changed = runner.git("diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                         "--name-only", "-z", base, source, "--").split("\0")
    changed = [name for name in changed if name]
    _need(changed and set(changed) <= set(allowed), "candidate changed source outside goal.allowed_paths")
    before, after = _tree(runner, base, allowed), _tree(runner, source, allowed)
    _need(set(before) == set(after) == set(allowed), "candidate added or deleted a source target")
    for name in allowed:
        _need(before[name][:2] == after[name][:2]
              and before[name][0] in ("100644", "100755") and before[name][1] == "blob",
              "source target is not a preserved regular file: " + name)
    files = contract.get("evaluation_files")
    _need(isinstance(files, dict) and files, "missing frozen evaluator hashes")
    files = dict(files)
    payloads = {}
    for name, digest in files.items():
        _need(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
              "invalid frozen evaluator digest")
        payloads[name] = _bytes(_path(name))
        _need(_hash(payloads[name]) == digest, "frozen evaluator changed: " + name)
    scorer = str(Path(__file__).resolve().with_name("continual_tasks.py"))
    for name in (str(mission_path), str(Path(__file__).resolve()), scorer):
        _need(name in files, "publication control was not frozen by the core: " + name)
    rsi_path = str(_path(mission.get("rsi_config")))
    _need(rsi_path in payloads, "original RSI controls were not frozen by the core")
    frozen = _json(payloads[rsi_path])
    _need(isinstance(frozen, dict), "invalid original frozen RSI controls")
    defaults = {"sealed_command": None, "workers": 2, "batch_size": 2, "timeout": 600,
                "evaluation_timeout": 180, "search_repeats": 1, "confirm_repeats": 3,
                "min_gain": 0, "seed": 0, "papers": [], "evaluation_files": []}
    # Apply runtime normalization, then authenticate the inputs the loader rereads.
    try:
        normalized = _sibling("rsi_runtime").load_inputs(mission_path)["rsi_config"]
    except ValueError as exc:
        raise PublicationError("invalid frozen mission inputs: " + str(exc)) from exc
    _unchanged(files)
    expected = {**defaults, **normalized}
    agents = controls.get("agents")
    _need(isinstance(agents, list) and len(agents) == 2
          and all(isinstance(agent, dict) for agent in agents),
          "invalid frozen bridge agents")
    command = _argv(agents[0].get("command"))
    trusted_bridge = expected["agents"][0]["command"][3]
    # Another core checkout may supply the bridge only if its frozen bytes match.
    _need(len(command) == 6 and Path(command[3]).name == "rsi_bridge.py"
          and trusted_bridge in payloads and command[3] in payloads
          and payloads[command[3]] == payloads[trusted_bridge],
          "frozen agent bridge differs from the trusted controller")
    for agent in expected["agents"]:
        agent["command"][3] = command[3]
    _need(controls == expected,
          "core execution controls differ from the original frozen RSI input")
    _need(_json(payloads[str(mission_path)]) == mission, "original frozen mission binding changed")
    originals = expected["evaluation_files"]
    _need(isinstance(originals, list)
          and all(isinstance(name, str) and name in files for name in originals),
          "original evaluator inputs are absent from the core contract")
    descriptors = []
    for name, data in payloads.items():
        if Path(name).suffix == ".json":
            value = _json(data)
            if value == goal:
                descriptors.append((name, data))
    _need(descriptors, "no exact portable goal descriptor in the frozen evaluators")
    descriptor_path, descriptor = sorted(descriptors)[0]
    goal_id = goal.get("id")
    _need(isinstance(goal_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", goal_id),
          "unsafe retained regression goal ID")
    prefix = "validation/continual/" + goal_id + "/"
    blobs = {prefix + "goal.json": descriptor.decode("utf-8")}
    role_commands = {}
    for role in ("search", "confirm", "sealed"):
        command = _argv(controls.get(role + "_command"))
        _need(command.count("--test") == 1 and scorer in command and "score" in command,
              "frozen " + role + " does not use the portable goal evaluator")
        index = command.index("--test")
        _need(index + 1 < len(command), "missing frozen " + role + " script")
        name = command[index + 1]
        script = goal["tests"][role]
        _need(name in payloads and payloads[name] == script.encode("utf-8"),
              "frozen " + role + " script differs from the portable goal")
        blobs[prefix + role + ".py"] = script
        role_commands[role] = command
    _measurement(winner.get("search"), 1, role_commands["search"], "search")
    _measurement(winner.get("confirm"), repeats, role_commands["confirm"], "serial confirmation")
    sealed = result.get("sealed")
    _need(isinstance(sealed, dict), "missing sealed measurements")
    _need(sealed.get("base") == state.get("sealed_base")
          and sealed.get("champion") == state.get("sealed_champion"),
          "state and result disagree on sealed measurements")
    before_scores = _measurement(sealed.get("base"), repeats, role_commands["sealed"], "sealed baseline")
    after_scores = _measurement(sealed.get("champion"), repeats, role_commands["sealed"], "sealed champion")
    _need(min(after_scores) > max(before_scores) + gain, "sealed samples do not prove improvement")
    outcomes = list((core / "attempts").rglob("result.json"))
    _need(len(outcomes) <= 4096, "oversize candidate evidence inventory")
    matches = []
    for path in sorted(outcomes):
        outcome = _object(path)
        if outcome.get("commit") == source and outcome.get("id") == winner.get("id") \
                and outcome.get("status") == "viable":
            _need(outcome.get("tree") == winner["tree"], "candidate receipt tree mismatch")
            _checks(outcome.get("checks"), commands)
            matches.append(path)
    _need(matches, "no viable source outcome with mandatory check receipts")
    for path in [mission_path, state_path, result_path, *matches]:
        files[str(path)] = _hash(_bytes(path))
    return {"source": source, "files": files, "commands": commands, "blobs": blobs,
            "prefix": prefix, "descriptor": descriptor_path}


def _verify_release(runner, source, selected, ref, blobs):
    _oid(selected)
    parents = runner.git("rev-list", "--parents", "-n", "1", selected).split()
    _need(parents == [selected, source], "release is not a direct regression-only child of source")
    names = runner.git("diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                       "--name-only", "-z", source, selected, "--").split("\0")
    _need(set(filter(None, names)) == set(blobs), "release changed more than the exact regressions")
    entries = _tree(runner, selected, list(blobs))
    _need(set(entries) == set(blobs), "release regression artifacts are incomplete")
    _need(not _tree(runner, source, list(blobs)), "regression artifacts would overwrite existing source")
    for name, text in blobs.items():
        mode, kind, oid = entries[name]
        _need(mode == "100644" and kind == "blob", "release artifact is not a regular UTF-8 file")
        # Hash Git's framing so even CRLF and absent final newlines are compared as bytes.
        data = text.encode("utf-8")
        digest = hashlib.sha1 if len(_oid(oid)) == 40 else hashlib.sha256
        actual = digest(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        _need(actual == oid, "release regression bytes changed: " + name)
    _need(runner.git("rev-parse", "--verify", "--end-of-options", ref + "^{commit}").strip() == selected,
          "owned release ref is missing or changed")


def _score_command(path):
    return [sys.executable, "-I", "-B", str(Path(__file__).resolve().with_name("continual_tasks.py")),
            "score", "--test", str(path)]


def _score_ok(value):
    _need(isinstance(value, dict) and _number(value.get("score")) and value["score"] == 1,
          "retained regression did not pass")
    _need(type(value.get("tests_run")) is int and value["tests_run"] > 0
          and type(value.get("failures")) is int and value["failures"] == 0
          and type(value.get("errors")) is int and value["errors"] == 0,
          "retained regression has missing tests, failures, or errors")


def _brief(receipt):
    return {key: receipt[key] for key in ("command", "returncode", "artifact_dir")}


def _proof(proof, accepted, workspace_root):
    _need(isinstance(proof, dict), "missing release validation receipts")
    path = _path(proof.get("worktree"))
    _need(path.parent == workspace_root, "release validation used an unowned worktree")
    _checks(proof.get("checks"), accepted["commands"])
    regressions = proof.get("regressions")
    _need(isinstance(regressions, dict) and set(regressions) == {"search", "confirm", "sealed"},
          "incomplete retained regression validation")
    for role, receipt in regressions.items():
        _need(isinstance(receipt, dict) and type(receipt.get("returncode")) is int
              and receipt["returncode"] == 0, "retained regression evaluator failed")
        _need(receipt.get("command") == _score_command(path / accepted["prefix"] / (role + ".py")),
              "release validation used a different regression script")
        _score_ok(receipt.get("measurement"))


def _build_release(runner, accepted, root, name):
    source, blobs = accepted["source"], accepted["blobs"]
    ref = "refs/gama-rsi/" + name
    existing = runner.git("rev-parse", "--verify", "--quiet", "--end-of-options",
                          ref + "^{commit}", accepted=(0, 1)).strip()
    if existing:
        _verify_release(runner, source, existing, ref, blobs)
    _need(not _tree(runner, source, [accepted["prefix"]]), "goal regression directory already exists")
    request = {"action": "create", "repo": str(runner.repo), "root": str(root),
               "source": source, "files": blobs}
    created = runner.worker(request)
    worktree = _path(created.get("worktree"))
    _need(worktree.parent == root, "workspace worker returned an unowned path")
    proof = {"worktree": str(worktree), "checks": [], "regressions": {}}
    for command in accepted["commands"]:
        proof["checks"].append(_brief(runner.run(command, cwd=worktree)))
    for role in ("search", "confirm", "sealed"):
        command = _score_command(worktree / accepted["prefix"] / (role + ".py"))
        result = runner.run(command, cwd=worktree)
        measurement = _json(result["stdout"])
        _score_ok(measurement)
        proof["regressions"][role] = {**_brief(result), "measurement": measurement}
    runner.git("diff", "--no-ext-diff", "--no-textconv", "--exit-code", "--", cwd=worktree)
    _need(not runner.git("ls-files", "--others", "--exclude-standard", "-z", cwd=worktree),
          "release checks left untracked work")
    if existing:
        tree = runner.git("write-tree", cwd=worktree).strip()
        _need(tree == runner.git("rev-parse", existing + "^{tree}").strip(),
              "retained release differs from the newly validated tree")
        selected = existing
    else:
        committed = runner.worker({"action": "commit", "repo": str(runner.repo), "root": str(root),
                                   "source": source, "worktree": str(worktree), "name": name,
                                   "message": "Retain continual regressions for " + accepted["prefix"]})
        selected = _oid(committed.get("commit"))
        _need(committed.get("ref") == ref, "workspace retained an unexpected release ref")
    _proof(proof, accepted, root)
    _unchanged(accepted["files"])
    _verify_release(runner, source, selected, ref, blobs)
    return selected, ref, proof


def _checkpoint(journal, record, save):
    snapshot = _json(_dump(journal))
    snapshot["publication"] = _json(_dump(record))
    save(snapshot)
    journal.clear()
    journal.update(snapshot)


def _publish(config, goal, mission_path, journal, save, cancel):
    _stop(cancel)
    _need(isinstance(config, dict) and isinstance(goal, dict) and isinstance(journal, dict),
          "publication requires config, goal, and journal objects")
    _need(callable(save), "publication requires a durable save callback")
    repo, state_dir, mission_path = _path(config.get("repo")), _path(config.get("state_dir")), _path(mission_path)
    _need(repo.is_dir(), "repository directory is missing")
    _disjoint(repo, state_dir)
    _disjoint(repo, mission_path.parent)
    branch, remote = config.get("branch"), config.get("remote", "origin")
    _need(isinstance(branch, str) and branch not in ("main", "master", "HEAD")
          and not branch.startswith(("refs/", "-")), "publication requires an owned feature branch")
    _need(isinstance(remote, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", remote),
          "publication requires a named Git remote")
    timeout = config.get("evaluation_timeout", 180)
    _need(_number(timeout) and timeout > 0, "evaluation_timeout must be positive and finite")
    base = _oid(journal.get("base_commit"))
    old = journal.get("publication")
    _need(old is None or isinstance(old, dict), "corrupt publication journal")
    if old is not None:
        _need(type(old.get("version")) is int and old["version"] == 1 and old.get("stage") in ("prepared", "selected", "complete"),
              "unrecognized publication journal stage")
    selected = old.get("release_commit") if old else None
    if old and old["stage"] != "prepared":
        _oid(selected)
    root = _path(state_dir / "publications" / _hash(str(mission_path).encode("utf-8")))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    runner = _Runner(repo, root, timeout, cancel)
    with _lock(root), runner:
        _need(runner.git("rev-parse", "--show-toplevel").strip() == str(repo),
              "configured repo is not its Git worktree root")
        runner.git("check-ref-format", "refs/heads/" + branch)
        allowed_heads = {selected} if old and old["stage"] == "complete" else {base}
        if selected:
            allowed_heads.add(selected)
        _checkout(runner, branch, allowed_heads,
                  clean=not (old and old["stage"] == "selected"))
        accepted = _accepted(config, goal, mission_path, base, runner)
        context = {"version": 1, "base_commit": base, "source_commit": accepted["source"],
                   "repo": str(repo), "branch": branch, "remote": remote,
                   "binding": _hash(_dump({"files": accepted["files"], "goal": goal}).encode("utf-8")),
                   "transport": _transport(runner, remote)}
        remote_head = _remote_head(runner, remote, branch)
        if old is None:
            _need(remote_head in (None, base), "remote advanced or diverged from campaign base")
            record = {**context, "stage": "prepared", "remote_before": remote_head}
            _checkpoint(journal, record, save)
        else:
            _need(all(old.get(key) == value for key, value in context.items()),
                  "publication inputs changed; preserve the journal and repair the mismatch")
            record = dict(old)
            _need("remote_before" in record and record["remote_before"] in (None, base),
                  "invalid recorded remote starting point")
            _need(remote_head == record["remote_before"] or (selected is not None and remote_head == selected),
                  "remote advanced, diverged, or was removed since publication began")
            if record["stage"] == "complete":
                _need(remote_head == selected, "completed publication remote no longer matches")
        name = "continual-" + context["binding"][:32]
        workspace_root = _path(root / ("workspaces-" + context["binding"][:32]))
        if record["stage"] == "prepared":
            selected, ref, proof = _build_release(runner, accepted, workspace_root, name)
            record.update(stage="selected", release_commit=selected, release_ref=ref, validation=proof)
            # This is the write-ahead decision; no caller ref has moved yet.
            _checkpoint(journal, record, save)
        else:
            ref = record.get("release_ref")
            _need(ref == "refs/gama-rsi/" + name, "unexpected journaled release ref")
            _proof(record.get("validation"), accepted, workspace_root)
            _verify_release(runner, accepted["source"], selected, ref, accepted["blobs"])
        _unchanged(accepted["files"])
        _stop(cancel)
        remote_head = _remote_head(runner, remote, branch)
        _need(remote_head in (record["remote_before"], selected), "remote changed before adoption")
        if record["stage"] != "complete":
            _stop(cancel)
            adopted = runner.worker({"action": "adopt", "repo": str(repo), "root": str(root),
                                     "branch": branch, "base": base, "selected": selected})
            _need(adopted.get("head") == selected, "caller did not adopt the selected release")
        _checkout(runner, branch, {selected})
        _stop(cancel)
        _unchanged(accepted["files"])
        _need(_transport(runner, remote) == context["transport"], "remote destination changed before push")
        remote_head = _remote_head(runner, remote, branch)
        _need(remote_head in (record["remote_before"], selected), "remote changed before push")
        if remote_head != selected:
            _stop(cancel)
            # No force/lease/default refspec: only this recorded object and this feature branch.
            runner.git("-c", "push.followTags=false", "-c", "remote." + remote + ".mirror=false",
                       "push", "--porcelain", "--no-verify", "--no-follow-tags",
                       "--recurse-submodules=no", remote, selected + ":refs/heads/" + branch)
        verified = _remote_head(runner, remote, branch)
        _need(verified == selected, "remote did not acknowledge the selected release SHA")
        _checkout(runner, branch, {selected})
        _unchanged(accepted["files"])
        if record["stage"] != "complete":
            record.update(stage="complete", remote_commit=verified)
            _checkpoint(journal, record, save)
        runner.worker({"action": "remove", "repo": str(repo), "root": str(workspace_root),
                       "source": accepted["source"],
                       "worktree": record["validation"]["worktree"]})
        return {"phase": "published", "commit": selected, "release_commit": selected,
                "source_commit": accepted["source"], "remote_commit": verified,
                "branch": branch, "remote": remote, "ref": ref,
                "regression_descriptor": str(repo / accepted["prefix"] / "goal.json"),
                "regression_files": [str(repo / name) for name in sorted(accepted["blobs"])]}


def publish(config: dict, *, goal: dict, mission_path: Path, journal: dict, save, cancel) -> dict:
    """Reconcile an accepted release; failures leave the selected SHA recoverable."""
    try:
        return _publish(config, goal, mission_path, journal, save, cancel)
    except (OSError, UnicodeError) as exc:
        raise PublicationError("publication could not complete: " + str(exc)) from exc


def _write(path, data):
    """Atomically persist publisher-owned state (also used while index.lock is held)."""
    path = _path(path)
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def _put(path, value):
    _write(path, (_dump(value) + "\n").encode("utf-8"))


def _drained():
    # As a subreaper, ECHILD proves that even orphaned/setsid descendants are gone.
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid == 0:
            time.sleep(0.02)
    return False


def _guard_worker():
    import ctypes

    _need(len(sys.argv) == 4 and sys.argv[1] == "_guard", "internal guardian arguments required")
    root, lease = _path(sys.argv[2]), int(sys.argv[3])
    held, named = os.fstat(lease), (root / "drain.lock").stat()
    _need((held.st_dev, held.st_ino) == (named.st_dev, named.st_ino), "guardian lease changed")
    _need(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0,
          "publication requires Linux child subreaping")
    _git_environment()
    marker, cancellation, requests = root / "drain.json", threading.Event(), queue.Queue()

    def receive():
        try:
            while True:
                raw = sys.stdin.buffer.readline(16 * 1024 * 1024 + 1)
                if not raw:
                    break
                _need(len(raw) <= 16 * 1024 * 1024 and raw.endswith(b"\n"),
                      "invalid guardian request framing")
                requests.put(_json(raw))
        finally:
            cancellation.set()
            requests.put(None)

    threading.Thread(target=receive, daemon=True).start()
    _put(marker, {"drained": True})
    try:
        while True:
            request = requests.get()
            if request is None or cancellation.is_set():
                break
            # A killed driver leaves this latch closed even if its lease FD disappears.
            _put(marker, {"drained": False})
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = _sibling("rsi_guard").run_guarded(
                        _argv(request["command"]), cwd=_path(request["cwd"]),
                        timeout=request["timeout"], artifact_dir=_path(request["artifact_dir"]),
                        input_text=request["input_text"], cancel=cancellation)
                response = {"returncode": result.returncode, "stdout": result.stdout,
                            "stderr": result.stderr}
            except Exception as exc:
                response = {"error": str(exc)}
            _need(_drained(), "publication guardian descendants remain; containment is unproven")
            _put(marker, {"drained": True})
            if not cancellation.is_set():
                print(_dump(response), flush=True)
    finally:
        os.close(lease)


def _addition_patch(files):
    chunks = []
    for name, text in sorted(files.items()):
        lines = text.split("\n")
        if lines[-1] == "":
            lines.pop()
        _need(lines, "empty retained regression artifact")
        chunks.extend(["diff --git a/" + name + " b/" + name + "\n",
                       "new file mode 100644\n", "--- /dev/null\n", "+++ b/" + name + "\n",
                       "@@ -0,0 +1," + str(len(lines)) + " @@\n"])
        chunks.extend("+" + line + "\n" for line in lines)
        if not text.endswith("\n"):
            chunks.append("\\ No newline at end of file\n")
    return "".join(chunks)


def _local_git(repo, *args, env=None):
    # Only called in a workspace worker already bounded by the publication guardian.
    result = subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.fsmonitor=false",
         "-c", "submodule.recurse=false", *args],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, encoding="utf-8", timeout=60)
    _need(result.returncode == 0, "publication Git transition failed: " + result.stderr[-2000:])
    return result.stdout


def _owns_index_lock(path, owner):
    if not isinstance(owner, dict) or not isinstance(owner.get("token"), str) or not path.exists():
        return False
    info = path.lstat()
    return (stat.S_ISREG(info.st_mode) and owner.get("identity") == [info.st_dev, info.st_ino]
            and _bytes(path, 256) == owner["token"].encode("ascii"))


def _adopt(repo, root, branch, base, selected):
    def interrupted(signum, frame):
        raise PublicationError("publication transition interrupted; resume the selected SHA")

    signal.signal(signal.SIGTERM, interrupted)
    identity = {"repo": str(repo), "branch": branch, "base": base, "selected": selected}
    ledger = root / "adoption.json"
    previous = _object(ledger) if ledger.exists() else {}
    _need(not previous or all(previous.get(key) == value for key, value in identity.items()),
          "caller transition belongs to different publication inputs")
    recovering = previous.get("phase") == "installing"
    index = _path(_local_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index").strip())
    lock = _path(Path(str(index) + ".lock"))
    if lock.exists():
        _need(_owns_index_lock(lock, previous.get("lock")), "caller has an unowned index lock")
        lock.unlink()
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    owner = {"identity": [info.st_dev, info.st_ino], "token": uuid.uuid4().hex}
    transaction, temporary = None, root / ("index-" + uuid.uuid4().hex)
    try:
        os.write(fd, owner["token"].encode("ascii"))
        os.fsync(fd)
        receipt = {**identity, "phase": previous.get("phase", "locked"), "lock": owner}
        _put(ledger, receipt)
        # index.lock stops checkouts before they can touch files; prepare pins HEAD and the ref.
        head = _oid(_local_git(repo, "rev-parse", "--verify", "HEAD").strip())
        _need(head in (base, selected), "unexpected caller HEAD during adoption")
        ref = "refs/heads/" + branch
        transaction = subprocess.Popen(
            ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
             "update-ref", "--stdin"],
            cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8")

        def step(command, expected):
            transaction.stdin.write(command + "\n")
            transaction.stdin.flush()
            _need(transaction.stdout.readline().strip() == expected,
                  "caller branch/ref transaction refused; preserve the checkout")

        step("start", "start: ok")
        transaction.stdin.write(("verify HEAD " + selected if head == selected else
                                 "update HEAD " + selected + " " + base) + "\n")
        step("prepare", "prepare: ok")
        # A prepared dereferencing HEAD operation locks HEAD and its referent.
        # Bind the branch under those locks before writing the index or worktree.
        _need(_local_git(repo, "symbolic-ref", "--quiet", "--no-recurse", "HEAD").strip() == ref,
              "caller is not on the configured feature branch")
        status = _local_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
        if head == base or status:
            _write(temporary, _bytes(index))
            environment = dict(os.environ, GIT_INDEX_FILE=str(temporary))
            selected_tree = _local_git(repo, "rev-parse", selected + "^{tree}").strip()
            if status:
                _need(recovering, "caller has operator changes; preserve them before publication")
                index_tree = _local_git(repo, "write-tree", env=environment).strip()
                base_tree = _local_git(repo, "rev-parse", base + "^{tree}").strip()
                _need(index_tree in (base_tree, selected_tree), "interrupted adoption index is ambiguous")
                # Only a complete selected tree is recoverable; partial writes/operator edits block.
                _local_git(repo, "read-tree", selected, env=environment)
                _local_git(repo, "update-index", "--refresh", env=environment)
            else:
                receipt["phase"] = "installing"
                _put(ledger, receipt)
                # Keep the real lock while read-tree uses a private index and refuses collisions.
                _local_git(repo, "read-tree", "-m", "-u", base, selected, env=environment)
            _need(_local_git(repo, "write-tree", env=environment).strip() == selected_tree,
                  "caller transition did not produce the selected index")
            _local_git(repo, "diff-files", "--quiet", "--no-ext-diff", "--no-textconv", "--", env=environment)
            _need(not _local_git(repo, "ls-files", "--others", "--exclude-standard", "-z", env=environment),
                  "caller has untracked operator work")
            # Install atomically without consuming index.lock; both locks still cover the transition.
            _write(index, _bytes(temporary))
        step("commit", "commit: ok")
        transaction.stdin.close()
        _need(transaction.wait(timeout=10) == 0, "caller ref transaction failed")
        receipt["phase"] = "complete"
        _put(ledger, receipt)
        return {"head": selected}
    finally:
        if transaction is not None and transaction.poll() is None:
            try:
                if transaction.stdin.closed:
                    transaction.wait(timeout=5)
                else:
                    transaction.communicate("abort\n", timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                transaction.terminate()
                try:
                    transaction.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        os.close(fd)
        if (transaction is None or transaction.poll() is not None) and _owns_index_lock(lock, owner):
            lock.unlink()
        temporary.unlink(missing_ok=True)


def _worker():
    _need(sys.argv[1:] == ["_workspace"], "internal publication worker arguments required")
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    _need(len(raw) <= 1024 * 1024, "oversize publication worker request")
    request = _json(raw)
    _need(isinstance(request, dict), "invalid publication worker request")
    _git_environment()
    # Workspace hooks are operator code, not part of the sealed source acceptance.
    os.environ.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="core.hooksPath",
                      GIT_CONFIG_VALUE_0="/dev/null", GIT_CONFIG_KEY_1="core.fsmonitor",
                      GIT_CONFIG_VALUE_1="false")
    repo = _path(request.get("repo"))
    with contextlib.redirect_stdout(sys.stderr):
        if request.get("action") == "validate":
            value = {"goal": _sibling("continual_tasks").validate_goal(request.get("goal"), repo)}
        elif request.get("action") == "adopt":
            root = _path(request.get("root"))
            _disjoint(repo, root)
            value = _adopt(repo, root, request.get("branch"),
                           _oid(request.get("base")), _oid(request.get("selected")))
        else:
            root, source = _path(request.get("root")), _oid(request.get("source"))
            _disjoint(repo, root)
            workspaces = _sibling("rsi_workspace").Workspaces(repo, root)
            if request.get("action") == "create":
                # Recover only this publication's root, with the core's nonce/receipt preflight.
                workspaces.recover()
                path = workspaces.create("release-" + uuid.uuid4().hex, source)
                files = request.get("files")
                _need(isinstance(files, dict) and len(files) == 4, "four regression artifacts are required")
                for name in files:
                    target = path / name
                    _need(target.resolve() == target and path in target.parents and not target.exists(),
                          "unsafe or existing regression destination: " + name)
                workspaces.apply_patch(path, _addition_patch(files), list(files))
                value = {"worktree": str(path)}
            elif request.get("action") == "commit":
                path = _path(request.get("worktree"))
                _need(path.parent == root, "commit worktree is outside the owned release root")
                commit = workspaces.commit(path, source, request.get("message"))
                value = {"commit": commit, "ref": workspaces.keep(commit, request.get("name"))}
            elif request.get("action") == "remove":
                path = _path(request.get("worktree"))
                _need(path.parent == root, "cleanup worktree is outside the owned release root")
                entry = "worktree " + str(path)
                listing = _local_git(repo, "worktree", "list", "--porcelain").splitlines()
                if path.exists() or entry in listing:
                    workspaces.remove(path)
                _need(not path.exists()
                      and entry not in _local_git(repo, "worktree", "list", "--porcelain").splitlines(),
                      "owned release worktree cleanup is incomplete")
                value = {"removed": str(path)}
            else:
                raise PublicationError("unknown publication workspace action")
    print(_dump(value))


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["_guard"]:
            _guard_worker()
        else:
            _worker()
    except Exception as exc:
        print("publication worker failed: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
