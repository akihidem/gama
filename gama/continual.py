"""Durable, bounded continual RSI coordination; controllers never propose source."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Isolated helpers load the frozen controller directory, not candidate __init__.
if not __package__:
    package = types.ModuleType("_gama_continual")
    package.__path__ = [str(Path(__file__).absolute().parent)]
    sys.modules[package.__name__] = package
    __package__ = package.__name__

from . import continual_discover, continual_publish, continual_tasks as tasks
from . import rsi_mission, rsi_runtime
from .rsi_guard import run_guarded

JST = ZoneInfo("Asia/Tokyo")
STATE_KEYS = set(("schema_version campaign_id config_path identity max_goal_cycles "
                  "initial_head expected_head phase error goals queue active_goal history "
                  "bootstrap cursor discoveries ledger inflight").split())
GOAL_KEYS = set(("id title descriptor digest work base prior mission mission_digest "
                 "mission_sha256 preparing needs_probe status outcome publication release").split())
KINDS = {"source", "probe", "finalize", "discovery", "publish"}


class Fault(Exception):
    def __init__(self, message, code=4):
        super().__init__(message)
        self.code = code


def _must(condition, message, code=4):
    if not condition:
        raise Fault(message, code)


def _short(value, limit=600):
    return str(value)[-limit:]


def _json(path):
    return tasks._loads(tasks._read(Path(path)))


def _abs(value):
    _must(isinstance(value, str) and Path(value).is_absolute(), "absolute path required", 5)
    path = Path(value)
    _must(path.resolve() == path, "symlink or noncanonical path: " + value, 5)
    return path


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(directory):
    directory = _abs(str(directory))
    if not directory.exists():
        _mkdir(directory.parent)
        directory.mkdir(exist_ok=True)
        _sync(directory.parent)
    _must(directory.is_dir(), "not a directory: " + str(directory))


def _put(path, value):
    _abs(str(path))
    _mkdir(path.parent)
    rsi_mission._put(path, value)


def _save(root, state):
    _put(root / "state.json", state)


def _now():
    return datetime.now(JST)


def _cadence(now):
    _must(isinstance(now, datetime) and now.utcoffset() is not None, "aware clock required")
    now = now.astimezone(JST)
    nine = now.replace(hour=9, minute=0, second=0, microsecond=0)
    evening = nine.replace(hour=21)
    if now < nine:
        return evening - timedelta(days=1), nine
    if now < evening:
        return nine, evening
    return evening, nine + timedelta(days=1)


def _slot_time(text):
    value = datetime.fromisoformat(text)
    _must(value.utcoffset() == timedelta(hours=9) and _cadence(value)[0].isoformat() == text,
          "invalid cadence ledger")
    return value


@contextlib.contextmanager
def _lock(path, nonblocking=False):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise Fault("campaign already owned", 3) from exc
        _sync(path.parent)
        yield
    finally:
        os.close(fd)


def _busy(root):
    try:
        fd = os.open(root / "owner.lock", os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)


def _stopped(root):
    return (root / "STOP").exists() or (root / "STOP").is_symlink()


def _control(root, stop):
    _mkdir(root)
    with _lock(root / "control.lock"):
        if stop:
            _put(root / "STOP", {"stop": True})
        elif _stopped(root):
            _abs(str(root / "STOP")).unlink()
            _sync(root)


def _owner_record(value):
    _must(isinstance(value, dict), "invalid owner receipt")
    if value:
        _must(set(value) == {"pid", "starttime", "boot_id"}
              and _integer(value["pid"], 1)
              and (_integer(value["starttime"]) or
                   isinstance(value["starttime"], str) and value["starttime"].isdigit())
              and isinstance(value["boot_id"], str) and value["boot_id"],
              "invalid owner receipt")
    return value


def _git(config, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(config["repo"]), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env)
    _must(result.returncode == 0 and len(result.stdout) <= 1048576,
          "Git " + args[0] + " refused: " + _short(result.stderr.decode("utf-8", "replace")))
    return result.stdout.decode("utf-8", "strict").strip()


def _checkout(config, expected=None):
    _must(_git(config, "symbolic-ref", "--short", "HEAD") == config["branch"], "feature branch changed")
    _must(not _git(config, "status", "--porcelain", "--untracked-files=all"), "checkout is dirty")
    head = _git(config, "rev-parse", "HEAD")
    _must(_sha(head) and (expected is None or head == expected), "unjournaled HEAD change")
    return head


def _fingerprint(path, body, config):
    files = [Path(config["bridge_config"]), *(Path(p) for p in config["goals"])]
    names = ("continual continual_tasks continual_discover continual_publish rsi rsi_cli "
             "rsi_mission rsi_runtime rsi_guard rsi_process rsi_bridge rsi_agent "
             "rsi_evaluate rsi_workspace").split()
    files += [Path(__file__).absolute().with_name(name + ".py") for name in names]
    venv = Path(sys.prefix) / "pyvenv.cfg"
    if venv.exists():
        files.append(venv)
    material = [(str(path), hashlib.sha256(body).hexdigest())]
    material += [(str(p), hashlib.sha256(tasks._read(p)).hexdigest()) for p in files]
    # Hash the target, but retain the venv executable's spelling and symlink.
    with open(sys.executable, "rb") as executable:
        _must(stat.S_ISREG(os.fstat(executable.fileno()).st_mode), "invalid interpreter", 5)
        digest = hashlib.file_digest(executable, "sha256").hexdigest()
    material.append((sys.executable, sys.prefix, sys.version, digest,
                     os.readlink(sys.executable) if Path(sys.executable).is_symlink() else None))
    return hashlib.sha256(tasks._dump(material)).hexdigest()


def _inputs(path, state=None):
    body = tasks._read(path)
    raw = tasks._loads(body)
    _must(isinstance(raw, dict), "configuration must be an object", 5)
    raw.setdefault("checks", [[sys.executable, "-B", "-m", "unittest", "discover",
                               "-s", "tests", "-t", ".", "-q"],
                              [sys.executable, "-B", "validation/json_extraction.py"]])
    try:
        config = tasks._configuration(raw)
        rsi_runtime.load_bridge_config(Path(config["bridge_config"]))
        identity = _fingerprint(path, body, config)
    except Exception as exc:
        raise Fault("invalid controls: " + _short(exc), 5) from exc
    _must(tasks._read(path) == body, "configuration changed while loading", 5)
    if state is not None:
        _must(state["identity"] == identity and state["config_path"] == str(path)
              and state["campaign_id"] == config["campaign_id"]
              and state["max_goal_cycles"] == config["max_goal_cycles"], "frozen controls changed", 5)
    return config, identity


def _goal_dir(root, key):
    return root / "goals" / key


def _job(root, flight):
    return root / "actions" / flight["id"]


def _descriptor(record):
    body = tasks._read(Path(record["descriptor"]))
    _must(hashlib.sha256(body).hexdigest() == record["digest"], "goal descriptor changed", 5)
    value = tasks._goal(tasks._loads(body))
    _must((value["id"], value["title"]) == (record["id"], record["title"]), "goal identity changed")
    return value


def _journal(root, key, record):
    path = _goal_dir(root, key) / "publication.json"
    value = _json(path)
    _must(isinstance(value, dict) and value.get("base_commit") == record["base"],
          "invalid publication journal: " + str(path))
    for field in ("commit", "release_commit", "source_commit", "remote_commit"):
        _must(value.get(field) is None or _sha(value[field]), "invalid publication commit")
    return value


def _load(root):
    if not root.exists():
        return None
    for path in root.glob("*.json"):
        _must(isinstance(_json(path), dict), "invalid persistent record: " + str(path))
    owner = _owner_record(_json(root / "owner.json")) if (root / "owner.json").exists() else {}
    if not (root / "state.json").exists():
        harmless = {"owner.lock", "owner.json", "control.lock", "STOP"}
        _must(not owner and all(p.name in harmless for p in root.iterdir()),
              "checkpoint missing with existing evidence; restore the checkpoint, never reset spending")
        return None
    s = _json(root / "state.json")
    _must(set(s) == STATE_KEYS and type(s["schema_version"]) is int and s["schema_version"] == 1,
          "unsupported campaign checkpoint")
    _must(_integer(s["max_goal_cycles"], 1) and _integer(s["cursor"])
          and isinstance(s["goals"], dict) and isinstance(s["ledger"], list)
          and s["phase"] in ("scheduled", "active", "blocked", "stopped")
          and (s["error"] is None or isinstance(s["error"], str))
          and isinstance(s["identity"], str) and re.fullmatch(r"[0-9a-f]{64}", s["identity"])
          and _sha(s["initial_head"]) and _sha(s["expected_head"]), "invalid campaign counters")
    _abs(s["config_path"])
    _must(isinstance(s["campaign_id"], str) and s["campaign_id"], "invalid campaign identity")
    totals, entries, previous = {}, {}, None
    for slot in s["ledger"]:
        _must(isinstance(slot, dict) and set(slot) == {"slot", "closed", "entries"}
              and type(slot["closed"]) is bool and isinstance(slot["entries"], list)
              and len(slot["entries"]) <= 2, "invalid reservation ledger")
        stamp = _slot_time(slot["slot"])
        _must(previous is None or stamp > previous, "nonmonotonic reservation ledger")
        previous = stamp
        for entry in slot["entries"]:
            _must(isinstance(entry, dict) and set(entry) == {"id", "kind", "goal"}
                  and isinstance(entry["id"], str) and re.fullmatch(r"[0-9a-f]{32}", entry["id"])
                  and entry["id"] not in entries and entry["kind"] in ("source", "discovery"),
                  "invalid reservation entry")
            entries[entry["id"]] = (slot["slot"], entry)
            if entry["kind"] == "source":
                _must(entry["goal"] in s["goals"], "reservation lost its goal")
                totals[entry["goal"]] = totals.get(entry["goal"], 0) + 1
            else:
                _must(entry["goal"] is None, "invalid discovery reservation")
    _must(isinstance(s["queue"], list) and isinstance(s["history"], list), "invalid goal queue")
    ordered = s["history"] + ([s["active_goal"]] if s["active_goal"] is not None else []) + s["queue"]
    _must(all(isinstance(k, str) for k in ordered) and len(set(ordered)) == len(ordered)
          and set(ordered) == set(s["goals"]), "inconsistent goal lifecycle")
    ids, prior, head = set(), [], s["initial_head"]
    for key in ordered:
        g = s["goals"][key]
        _must(re.fullmatch(r"[0-9a-f]{32}", key) and isinstance(g, dict) and set(g) == GOAL_KEYS,
              "invalid goal checkpoint")
        directory = _goal_dir(root, key)
        _must(g["descriptor"] == str(directory / "descriptor.json"), "unsafe descriptor path")
        _descriptor(g)
        _must(g["id"] not in ids and _integer(g["work"]) and g["work"] == totals.get(key, 0)
              and g["work"] <= s["max_goal_cycles"]
              and all(type(g[f]) is bool for f in ("preparing", "needs_probe", "publication"))
              and (g["status"] is None or isinstance(g["status"], dict)), "invalid goal accounting")
        ids.add(g["id"])
        if g["status"] is not None:
            status = g["status"]
            _must(set(status) <= {"phase", "cycle_open", "evidence", "digest", "last_outcome"}
                  and type(status.get("cycle_open")) is bool and isinstance(status.get("phase"), str),
                  "invalid mission status")
            evidence = _abs(status["evidence"])
            _must(evidence.parent.parent == root / "actions" and evidence.name == "result.json"
                  and re.fullmatch(r"[0-9a-f]{32}", evidence.parent.name), "unsafe status evidence")
            body = tasks._read(evidence)
            result = tasks._loads(body)
            _must(hashlib.sha256(body).hexdigest() == status["digest"] and result.get("ok") is True
                  and result["intent"]["goal"] == key and result["intent"]["id"] == evidence.parent.name
                  and result["intent"]["kind"] in ("source", "probe", "finalize")
                  and all(status[k] == result["value"][k] for k in ("phase", "cycle_open")),
                  "mission status evidence changed")
        if g["mission"] is not None:
            mission = _abs(g["mission"])
            _must(mission.is_relative_to(directory / "frozen") and not g["preparing"]
                  and hashlib.sha256(tasks._read(mission)).hexdigest() == g["mission_sha256"]
                  and isinstance(g["mission_digest"], str), "frozen mission changed", 5)
        else:
            _must(g["mission_digest"] is None and g["mission_sha256"] is None, "missing mission")
        if g["publication"] or (directory / "publication.json").exists():
            _must(g["mission"] is not None, "publication lost its original mission")
            _journal(root, key, g)
        if key in s["queue"]:
            _must(g["base"] is None and not g["prior"] and g["work"] == 0
                  and g["mission"] is None and g["outcome"] is None, "invalid queued goal")
            continue
        _must(g["base"] == head and g["prior"] == prior, "goal base or regressions changed")
        _must((key in s["history"]) == (g["outcome"] is not None), "invalid goal outcome")
        if g["outcome"] == "published":
            release = g["release"]
            _must(isinstance(release, dict) and _sha(release.get("release_commit"))
                  and release.get("remote_commit") == release["release_commit"]
                  and _sha(release.get("source_commit")), "incomplete release evidence")
            path = _abs(release["regression_descriptor"])
            _must(hashlib.sha256(tasks._read(path)).hexdigest() == release["descriptor_sha256"],
                  "retained regression descriptor changed", 5)
            prior = [*prior, str(path)]
            head = release["release_commit"]
        else:
            _must(g["release"] is None, "unaccepted release in history")
    _must(head == s["expected_head"], "expected HEAD does not match release history")
    _must(isinstance(s["bootstrap"], list) and isinstance(s["discoveries"], list), "invalid evidence index")
    for index, item in enumerate(s["bootstrap"]):
        path = root / "bootstrap" / (str(index) + ".json")
        _must(item["evidence"] == str(path)
              and hashlib.sha256(tasks._read(path)).hexdigest() == item["digest"], "history changed", 5)
    flight = s["inflight"]
    if flight is not None:
        _must(isinstance(flight, dict) and set(flight) == {"id", "kind", "goal", "action", "owner", "slot"}
              and isinstance(flight["id"], str) and re.fullmatch(r"[0-9a-f]{32}", flight["id"])
              and flight["kind"] in KINDS and s["ledger"], "invalid action intent")
        _owner_record(flight["owner"])
        _must(flight["owner"], "action missing owner identity")
        _abs(str(_job(root, flight)))
        if flight["kind"] in ("source", "discovery"):
            _must(entries.get(flight["id"]) == (flight["slot"],
                  {k: flight[k] for k in ("id", "kind", "goal")}), "unfunded action")
        if flight["kind"] == "discovery":
            _must(flight["goal"] is None and s["active_goal"] is None, "invalid discovery intent")
        else:
            _must(flight["goal"] == s["active_goal"] and flight["goal"] in s["goals"],
                  "action lost its active goal")
        expected = {"source": ("run", "resume"), "probe": ("status",), "finalize": ("resume",)}
        _must(flight["action"] in expected.get(flight["kind"], (None,)), "invalid delegated action")
    return s


def _receipt(root, flight):
    path = _job(root, flight) / "guard" / "process.json"
    deadline = time.monotonic() + 6
    changed = threading.Event()
    while True:
        try:
            value = _json(path)
        except Exception:
            value = None
        if (isinstance(value, dict) and set(value) == {"returncode", "error"}
                and (value["returncode"] is None or type(value["returncode"]) is int)
                and (value["error"] is None or isinstance(value["error"], str))):
            error = (value["error"] or "").lower()
            _must(not any(t in error for t in ("children did not exit after kill", "cleanup failed")),
                  "guard containment failed; inspect " + str(path))
            return value
        _must(time.monotonic() < deadline, "guard completion is unproven; inspect " + str(path))
        # Poll for the guardian's post-drain receipt; elapsed time never grants clearance.
        changed.wait(min(0.05, max(0, deadline - time.monotonic())))


@contextlib.contextmanager
def _owner(root, path):
    _mkdir(root)
    with _lock(root / "owner.lock", nonblocking=True):
        state = _load(root)
        config, identity = _inputs(path, state)
        if state and state["inflight"]:
            _receipt(root, state["inflight"])
        # Both preflight and this locked validation precede any owner.json replacement.
        _put(root / "owner.json", rsi_mission._identity(os.getpid()))
        try:
            yield config, identity, state
        finally:
            _put(root / "owner.json", {})


def _register(root, state, goal, repo):
    goal = tasks.validate_goal(goal, Path(repo))
    _must(all(g["id"] != goal["id"] for g in state["goals"].values()), "duplicate goal id")
    key = uuid.uuid4().hex
    path = _goal_dir(root, key) / "descriptor.json"
    _put(path, goal)
    state["goals"][key] = dict(id=goal["id"], title=goal["title"], descriptor=str(path),
        digest=hashlib.sha256(tasks._read(path)).hexdigest(), work=0, base=None, prior=[],
        mission=None, mission_digest=None, mission_sha256=None, preparing=False,
        needs_probe=False, status=None, outcome=None, publication=False, release=None)
    state["queue"].append(key)


def _initial(root, path, config, identity):
    head = _checkout(config)
    state = dict(schema_version=1, campaign_id=config["campaign_id"], config_path=str(path),
        identity=identity, max_goal_cycles=config["max_goal_cycles"], initial_head=head,
        expected_head=head, phase="scheduled", error=None, goals={}, queue=[], active_goal=None,
        history=[], bootstrap=[], cursor=0, discoveries=[], ledger=[], inflight=None)
    for goal_path in config["goals"]:
        _register(root, state, _json(Path(goal_path)), config["repo"])
    for index, item in enumerate(config.get("bootstrap_history", [])):
        evidence = root / "bootstrap" / (str(index) + ".json")
        _put(evidence, item)
        state["bootstrap"].append(dict(id=_short(item.get("id", "historical"), 120),
            title=_short(item.get("title", "Historical completed goal"), 240),
            outcome=_short(item.get("outcome", "historical"), 120), evidence=str(evidence),
            digest=hashlib.sha256(tasks._read(evidence)).hexdigest()))
    _save(root, state)
    return state


def _mission_inputs(record):
    loaded = rsi_runtime.load_inputs(Path(record["mission"]))
    mission = loaded["mission"]
    _must(all(type(mission.get(k)) is int and mission[k] == n for k, n in
              (("rounds_per_cycle", 1), ("batch_size", 2), ("max_reservations_per_cycle", 4)))
          and all(type(mission.get(k)) in (int, float) and mission[k] == 1
                  and math.isfinite(mission[k]) for k in ("search_ceiling", "confirm_ceiling")),
          "delegated mission exceeds its pair", 5)
    _must(record["mission_digest"] in (None, loaded["digest"]), "mission controls changed", 5)
    return loaded


def _finalization(record):
    if record["mission"] is None:
        return None, None
    _mission_inputs(record)
    root = _abs(_json(Path(record["mission"]))["state_dir"])
    directory = _abs(str(root / "rsi"))
    path = directory / "state.json"
    if not path.exists():
        return None, None
    core = _json(path)
    _must(isinstance(core, dict) and isinstance(core.get("phase"), str), "invalid core checkpoint")
    phase = core["phase"]
    if phase not in ("finalizing", "finalized"):
        return phase, None
    _must(core.get("base") == record["base"], "core base differs from goal base")
    archive = core.get("archive")
    _must(isinstance(archive, list) and all(isinstance(e, dict) for e in archive),
          "invalid core archive")
    champion = [e for e in archive if e.get("id") == core.get("champion")]
    _must(core.get("champion") is not None and len(champion) == 1
          and _sha(champion[0].get("commit")), "invalid core champion")
    if phase == "finalizing":
        return phase, None
    verdict = core.get("sealed_verdict")
    _must(verdict in ("improved", "regressed", "not_separable")
          and isinstance(core.get("sealed_base"), dict)
          and isinstance(core.get("sealed_champion"), dict), "invalid sealed core verdict")
    path = _abs(str(directory / "result.json"))
    if not path.exists():
        return phase, None
    result = _json(path)
    _must(isinstance(result, dict) and isinstance(result.get("phase"), str)
          and result.get("base") == record["base"] and result.get("state_dir") == str(directory),
          "core result base or directory differs")
    if result["phase"] != "finalized":
        # Only a ready report for the same champion is safe to replace.
        reported = result.get("champion")
        _must(result["phase"] == "ready" and isinstance(reported, dict)
              and all(reported.get(k) == champion[0][k] for k in ("id", "commit")),
              "stale core result disagrees with checkpoint")
        return phase, None
    _must(result.get("sealed_verdict") == verdict and result.get("champion") == champion[0]
          and result.get("sealed") == {"base": core["sealed_base"], "champion": core["sealed_champion"]},
          "finalized core result disagrees with checkpoint")
    return phase, verdict


def _recover_finalized_report(root, config, record, cancel):
    """Restore the report that the mission CLI skips for a finalized core."""
    loaded = _mission_inputs(record)
    mission_root = _abs(loaded["mission"]["state_dir"])
    directory = _abs(str(mission_root / "rsi"))

    def check_stop(_event=None):
        _must(not cancel.is_set() and not _stopped(root) and not _stopped(mission_root),
              "STOP requested")

    check_stop()
    with rsi_mission._owner(mission_root):
        # This probe releases run.lock; the exporter must acquire it itself.
        rsi_mission._wait_core(mission_root)
        check_stop()
        loaded = _mission_inputs(record)
        _must(Path(loaded["mission"]["repo"]) == Path(config["repo"]),
              "report recovery repository differs from campaign", 5)
        _checkout(config, record["base"])
        phase, verdict = _finalization(record)
        _must(phase == "finalized", "report recovery lacks a finalized core checkpoint")
        if verdict is None:
            checkpoint = tasks._read(directory / "state.json")
            from .rsi import run_rsi
            run_rsi(loaded["rsi_config"], repo=loaded["mission"]["repo"], state_dir=directory,
                    rounds=1, resume=True, finalize=True, on_event=check_stop)
            _must(tasks._read(directory / "state.json") == checkpoint,
                  "core checkpoint changed during report recovery")
            _must(_finalization(record)[1] is not None,
                  "core exporter did not restore an agreed finalized result")
        check_stop()
    # Release the mission owner before the CLI acquires it within the same action guard.
    check_stop()
    return subprocess.run([sys.executable, "-B", "-m", "gama.rsi_mission",
                           "resume", "--mission", record["mission"]],
                          cwd=config["repo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", errors="replace")


def _freeze(root, state, config, key):
    g = state["goals"][key]
    _must(not g["preparing"], "goal freeze was interrupted; inspect " + str(_goal_dir(root, key)))
    _checkout(config, state["expected_head"])
    g["preparing"] = True
    _save(root, state)
    mission = tasks.freeze_goal(config, _descriptor(g), _goal_dir(root, key) / "frozen", g["prior"])
    g["mission"] = str(_abs(str(mission)))
    g["mission_sha256"] = hashlib.sha256(tasks._read(mission)).hexdigest()
    g["mission_digest"] = _mission_inputs(g)["digest"]
    g["preparing"] = False
    _save(root, state)


def _dispatch(root, state, config, kind, cancel, action=None):
    if cancel.is_set() or _stopped(root):
        cancel.set()
        return
    _inputs(Path(state["config_path"]), state)
    key = state["active_goal"]
    g = state["goals"].get(key)
    if kind in ("source", "discovery"):
        _checkout(config, state["expected_head"])
    if kind in ("source", "probe"):
        _mission_inputs(g)
    mission_cli = kind in ("source", "probe", "finalize")
    if kind == "finalize":
        phase, _ = _finalization(g)
        _must(phase in ("finalizing", "finalized"),
              "finalization recovery lacks a terminal core checkpoint")
        mission_cli = phase == "finalizing"
    slot = state["ledger"][-1]
    flight = dict(id=uuid.uuid4().hex, kind=kind, goal=key, action=action,
                  owner=_json(root / "owner.json"), slot=slot["slot"])
    directory = _job(root, flight)
    _mkdir(directory)
    if kind in ("source", "discovery"):
        _must(not slot["closed"] and len(slot["entries"]) < 2, "cadence budget is closed")
        if kind == "source":
            _must(g["work"] < state["max_goal_cycles"], "goal work budget is closed")
            g["work"] += 1
        slot["entries"].append({k: flight[k] for k in ("id", "kind", "goal")})
    state["inflight"], state["phase"] = flight, "active"
    _save(root, state)  # Charge and startup intent are durable before any child starts.
    if _stopped(root) or cancel.is_set():
        state["inflight"] = None  # Definitely not launched; the conservative charge remains.
        _save(root, state)
        cancel.set()
        return
    bridge = rsi_runtime.load_bridge_config(Path(config["bridge_config"]))
    timeout = 12 * (bridge["timeout"] + config["evaluation_timeout"] *
                    (len(config["checks"]) + len(state["history"]) + 16)) + 60
    _must(math.isfinite(timeout), "derived deadline is not finite", 5)
    if mission_cli:
        command = [sys.executable, "-B", "-m", "gama.rsi_mission", action, "--mission", g["mission"]]
    else:
        command = [sys.executable, "-I", "-B", str(Path(__file__).absolute()),
                   "_worker", str(root), flight["id"]]
    try:
        result = run_guarded(command, cwd=Path(config["repo"]),
                             timeout=60 if kind == "probe" else timeout,
                             artifact_dir=directory / "guard", cancel=cancel)
        if mission_cli:
            _put(directory / "output.json", {"stdout": result.stdout, "stderr": result.stderr})
            value = tasks._loads(result.stdout.encode("utf-8"))
            _put(directory / "result.json", {"ok": True, "returncode": result.returncode,
                                            "value": value, "intent": flight})
        elif not (directory / "result.json").exists() and not cancel.is_set() and not _stopped(root):
            _put(directory / "output.json", {"stdout": result.stdout, "stderr": result.stderr})
            _put(directory / "result.json", {"ok": False, "code": 2, "intent": flight,
                                            "error": "worker exited without a result"})
    except Exception as exc:
        rsi_mission._write(directory / "error.txt", traceback.format_exc())
        if not cancel.is_set() and not _stopped(root) and not (directory / "result.json").exists():
            _put(directory / "result.json", {"ok": False, "code": getattr(exc, "code", 2), "intent": flight,
                                            "error": type(exc).__name__})
    _receipt(root, flight)


def _worker(root_text, token):
    root = _abs(root_text)
    state = _load(root)
    _must(state is not None and state["inflight"] is not None, "no authorized action")
    flight = state["inflight"]
    _must(flight["id"] == token and flight["kind"] in ("discovery", "publish", "finalize")
          and _busy(root) and _json(root / "owner.json") == flight["owner"]
          and rsi_mission._identity(flight["owner"]["pid"]) == flight["owner"]
          and not _stopped(root), "inactive action authorization")
    directory = _job(root, flight)
    # A private worker cannot replay a reservation, even while its owner is live.
    fd = os.open(directory / "started", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as marker:
        marker.write("started\n")
        marker.flush()
        os.fsync(marker.fileno())
    _sync(directory)
    try:
        config, _ = _inputs(Path(state["config_path"]), state)
        with rsi_mission._cancellation(root) as cancel:
            if flight["kind"] == "discovery":
                value = continual_discover.discover(config, directory=directory / "discovery",
                    seen=[_descriptor(g) for g in state["goals"].values()],
                    cursor=state["cursor"], cancel=cancel)
            elif flight["kind"] == "finalize":
                completed = _recover_finalized_report(root, config,
                    state["goals"][flight["goal"]], cancel)
                _put(directory / "output.json",
                     {"stdout": completed.stdout, "stderr": completed.stderr})
                value = tasks._loads(completed.stdout.encode("utf-8"))
            else:
                key = flight["goal"]
                g = state["goals"][key]
                value = continual_publish.publish(config, goal=_descriptor(g),
                    mission_path=Path(g["mission"]), journal=_journal(root, key, g),
                    save=lambda snapshot: _put(_goal_dir(root, key) / "publication.json", snapshot),
                    cancel=cancel)
        result = {"ok": True, "value": value, "intent": flight}
        if flight["kind"] == "finalize":
            result["returncode"] = completed.returncode
        _put(directory / "result.json", result)
    except Exception as exc:
        rsi_mission._write(directory / "error.txt", traceback.format_exc())
        _put(directory / "result.json", {"ok": False, "code": getattr(exc, "code", 2), "intent": flight,
                                       "error": type(exc).__name__})
    return 0


def _finish_goal(root, state, outcome):
    key = state["active_goal"]
    state["goals"][key]["outcome"] = outcome
    state["history"].append(key)
    state["active_goal"] = None
    _save(root, state)


def _complete(root, state, config):
    flight = state["inflight"]
    _receipt(root, flight)
    directory = _job(root, flight)
    result = _json(directory / "result.json") if (directory / "result.json").exists() else None
    _must(result is None or (isinstance(result, dict) and type(result.get("ok")) is bool
          and result.get("intent") == flight), "invalid action result: " + str(directory))
    kind, key = flight["kind"], flight["goal"]
    g = state["goals"].get(key)
    error = kind + " failed; evidence: " + str(directory)
    if result and not result["ok"] and result.get("code") == 5:
        raise Fault("action controls changed; evidence: " + str(directory), 5)
    if kind in ("source", "probe", "finalize"):
        g["needs_probe"] = True
    state["inflight"] = None
    if kind in ("source", "probe", "finalize"):
        if not result or not result["ok"]:
            state["error"] = error
            _save(root, state)
            _must(kind == "source", error)
            return 2 if result else 0
        value, code = result["value"], result["returncode"]
        _must(isinstance(value, dict) and isinstance(value.get("phase"), str)
              and type(value.get("cycle_open")) is bool and type(code) is int, "invalid mission status")
        g["status"] = {k: value[k] for k in ("phase", "cycle_open")}
        g["status"]["evidence"] = str(directory / "result.json")
        g["status"]["digest"] = hashlib.sha256(tasks._read(directory / "result.json")).hexdigest()
        if isinstance(value.get("last_outcome"), str):
            g["status"]["last_outcome"] = _short(value["last_outcome"], 160)
        g["needs_probe"] = False
        state["error"] = error if code else None
        _save(root, state)
        _must(code != 5, "mission frozen inputs changed; evidence: " + str(directory), 5)
        _must(code != 3, "mission ownership overlap; evidence: " + str(directory))
        text = str(value.get("error") or "").lower()
        integrity = ("fingerprint", "integrity", "corrupt", "dirty", "diverg", "unsafe",
                     "invalid checkpoint", "children did not exit after kill", "cleanup failed", "inputs changed")
        _must(not any(word in text for word in integrity), error)
        if kind == "finalize":
            _must(code == 0 and value["phase"] == "saturated", error)
        return 2 if kind == "source" and code else 0
    if kind == "publish":
        if not result:
            _save(root, state)  # Reconcile the same journal in a new enclosing guard.
            return 0
        _must(result["ok"], error)
        value = result["value"]
        _must(isinstance(value, dict) and value.get("phase") == "published"
              and _sha(value.get("release_commit")) and _sha(value.get("source_commit"))
              and value.get("remote_commit") == value["release_commit"]
              and value.get("branch") == config["branch"] and value.get("remote") == config["remote"]
              and value.get("ref") and value.get("regression_files"), "incomplete verified publication")
        journal = _journal(root, key, g)
        publication = journal.get("publication")
        _must(isinstance(publication, dict)
              and type(publication.get("version")) is int and publication["version"] == 1
              and publication.get("stage") == "complete"
              and publication.get("base_commit") == g["base"]
              and publication.get("release_commit") == value["release_commit"]
              and publication.get("source_commit") == value["source_commit"]
              and publication.get("remote_commit") == value["remote_commit"]
              and publication.get("branch") == config["branch"]
              and publication.get("remote") == config["remote"]
              and publication.get("release_ref") == value["ref"],
              "publication was not durably saved")
        regression = _abs(value["regression_descriptor"])
        _must(regression.is_relative_to(Path(config["repo"]))
              and tasks._goal(_json(regression)) == _descriptor(g), "published regression differs from goal")
        _checkout(config, value["release_commit"])
        fields = ("phase", "commit", "release_commit", "source_commit", "remote_commit",
                  "branch", "remote", "ref", "regression_descriptor")
        g["release"] = {field: value.get(field) for field in fields}
        g["release"]["descriptor_sha256"] = hashlib.sha256(tasks._read(regression)).hexdigest()
        state["expected_head"], state["error"] = value["release_commit"], None
        _finish_goal(root, state, "published")
        return 0
    slot = next(item for item in state["ledger"] if item["slot"] == flight["slot"])
    if result and result["ok"]:
        value = result["value"]
        _must(isinstance(value, dict) and _integer(value.get("cursor"))
              and isinstance(value.get("goals"), list) and isinstance(value.get("rejected"), list), "invalid discovery result")
        state["cursor"] = value["cursor"]
        for goal in value["goals"]:
            _register(root, state, goal, config["repo"])
        slot["closed"] = not value["goals"]
        rejected = [_short(item, 240) for item in value["rejected"][:16]]
        count, failure = len(value["goals"]), None
    else:
        state["cursor"] += 1
        slot["closed"], rejected, count, failure = True, [], 0, error
    state["discoveries"].append(dict(id=flight["id"], cursor=state["cursor"], count=count,
                                     rejected=rejected, error=failure, evidence=str(directory)))
    state["error"] = failure
    _save(root, state)
    return 2 if result and failure else 0


def _tick(root, state, config, cancel):
    stamp = _cadence(_now())[0]
    if not state["ledger"] or stamp > _slot_time(state["ledger"][-1]["slot"]):
        state["ledger"].append(dict(slot=stamp.isoformat(), closed=False, entries=[]))
        _save(root, state)
    slot = state["ledger"][-1]
    current, code = stamp == _slot_time(slot["slot"]), 0
    checked_finalization = set()
    while True:
        if cancel.is_set() or _stopped(root):
            _control(root, True)
            state["phase"] = "stopped"
            _save(root, state)
            return 4
        if state["inflight"]:
            code = max(code, _complete(root, state, config))
            continue
        key = state["active_goal"]
        g = state["goals"].get(key)
        if g:
            if g["needs_probe"]:
                _dispatch(root, state, config, "probe", cancel, "status")
                continue
            if g["publication"]:
                _dispatch(root, state, config, "publish", cancel)
                continue
            phase, verdict = _finalization(g)
            _must((g["status"] or {}).get("phase") != "saturated" or phase == "finalized",
                  "saturated mission lacks a finalized core checkpoint")
            if verdict is not None:
                if verdict != "improved":
                    _finish_goal(root, state, verdict)
                    continue
                path = _goal_dir(root, key) / "publication.json"
                if not path.exists():
                    _put(path, {"base_commit": g["base"]})
                _journal(root, key, g)
                g["publication"] = True
                _save(root, state)
                continue
            if phase in ("finalizing", "finalized"):
                _must(key not in checked_finalization,
                      "finalization recovery lacks an agreed finalized result")
                checked_finalization.add(key)
                # Terminal core checkpoints prove this resume cannot search.
                _dispatch(root, state, config, "finalize", cancel, "resume")
                continue
            if g["work"] >= state["max_goal_cycles"]:
                _finish_goal(root, state, "exhausted")
                continue
        if not current or slot["closed"] or len(slot["entries"]) == 2:
            state["phase"] = "scheduled"
            _save(root, state)
            return code
        if not g:
            _checkout(config, state["expected_head"])
            if not state["queue"]:
                _dispatch(root, state, config, "discovery", cancel)
                continue
            key = state["queue"].pop(0)
            state["active_goal"] = key
            g = state["goals"][key]
            g["base"] = state["expected_head"]
            g["prior"] = [state["goals"][k]["release"]["regression_descriptor"] for k in state["history"]
                          if state["goals"][k]["outcome"] == "published"]
            _save(root, state)
        if g["mission"] is None:
            _freeze(root, state, config, key)
        previous = g["status"] or {}
        action = "resume" if previous.get("cycle_open") or previous.get("phase") in (
            "blocked", "stopped", "active") else "run"
        _dispatch(root, state, config, "source", cancel, action)


def _summary(record):
    return dict(id=record["id"], title=record["title"], outcome=record["outcome"],
                work=record["work"], descriptor=record["descriptor"], mission_path=record["mission"],
                base_commit=record["base"], mission_status=record["status"], publication=record["release"])


def _view(root, state, error=None, phase=None):
    stamp, following = _cadence(_now())
    owner = "none"
    try:
        receipt = _owner_record(_json(root / "owner.json")) if (root / "owner.json").exists() else {}
        owner = "live" if _busy(root) else ("stale" if receipt else "none")
    except Exception:
        owner = "unknown"
    ledger = state["ledger"][-1] if state and state["ledger"] else None
    if ledger:
        following = max(following, _cadence(_slot_time(ledger["slot"]) + timedelta(seconds=1))[1])
    goals = state["goals"] if state else {}
    active = goals.get(state["active_goal"]) if state else None
    history = (state["bootstrap"] + [_summary(goals[k]) for k in state["history"]]) if state else []
    publication = next((goals[k]["release"] for k in reversed(state["history"])
                        if goals[k]["release"]), None) if state else None
    normal = state["phase"] if state else "scheduled"
    if normal == "active" and owner != "live":
        normal = "blocked"
        error = error or "Owner exited; run or resume will reconcile the guarded action."
    return dict(phase="stopped" if _stopped(root) else (phase or normal),
        active_goal=_summary(active) if active else None,
        queue=[{"id": goals[k]["id"], "title": goals[k]["title"]} for k in state["queue"]] if state else [],
        history=history, slot=ledger["slot"] if ledger else stamp.isoformat(),
        reserved=2 * len(ledger["entries"]) if ledger else 0, next_run=following.isoformat(),
        ownership=owner, error=error or (state["error"] if state else None),
        expected_head=state["expected_head"] if state else None, publication=publication,
        source_commit=publication["source_commit"] if publication else None,
        publication_journal=str(_goal_dir(root, state["active_goal"]) / "publication.json")
            if active and active["publication"] else None)


def _basic(path):
    raw = _json(path)
    _must(isinstance(raw, dict), "configuration must be an object", 5)
    root, repo = _abs(raw.get("state_dir")), _abs(raw.get("repo"))
    _must(not root.is_relative_to(repo) and not repo.is_relative_to(root), "state/repo overlap", 5)
    return root


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Fault(message, 5)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "_worker":
        return _worker(argv[1], argv[2])
    state, root, code, error, phase = None, None, 0, None, None
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument("action", choices=("run", "status", "stop", "resume"))
        parser.add_argument("--config", required=True)
        args = parser.parse_args(argv)
        path = _abs(args.config)
        root = _basic(path)
        if args.action == "stop":
            _control(root, True)
        try:
            state = _load(root)
        except Exception as exc:
            if args.action not in ("status", "stop"):
                raise
            error, phase = "checkpoint unavailable: " + _short(exc), "blocked"
        if args.action in ("run", "resume"):
            _must(not _busy(root), "campaign already owned", 3)
            _must(args.action == "resume" or not _stopped(root), "STOP is persistent", 4)
            _inputs(path, state)
            if state and state["inflight"]:
                _receipt(root, state["inflight"])
            with _owner(root, path) as (config, identity, latest):
                state = latest if latest is not None else _initial(root, path, config, identity)
                _must(args.action == "resume" or state["phase"] not in ("blocked", "stopped"),
                      state["error"] or "explicit resume required")
                if args.action == "resume":
                    _control(root, False)
                    state["error"] = None
                with rsi_mission._cancellation(root) as cancel:
                    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                    for sig in handlers:
                        signal.signal(sig, lambda *_: cancel.set())
                    try:
                        code = _tick(root, state, config, cancel)
                    except Exception as exc:
                        state["phase"] = "stopped" if _stopped(root) else "blocked"
                        state["error"] = _short(exc)
                        _save(root, state)
                        raise
                    finally:
                        for sig, handler in handlers.items():
                            signal.signal(sig, handler)
    except Exception as exc:
        code, error = getattr(exc, "code", 2), _short(exc)
        phase = "blocked" if code != 3 else None
    if root is None:
        value = dict(phase="blocked", active_goal=None, queue=[], history=[], slot=None, reserved=0,
                     next_run=_cadence(_now())[1].isoformat(), ownership="none", error=error,
                     expected_head=None, publication=None, source_commit=None)
    else:
        value = _view(root, state, error, phase)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
