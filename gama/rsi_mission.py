"""Persistent bounded missions over the unchanged source-RSI engine."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

sys.dont_write_bytecode = True
if not __package__:
    # Isolated private workers import only this installed source tree.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gama.rsi_guard import run_guarded
from gama.rsi_process import ProcessError
from gama.rsi_runtime import load_inputs


class _Changed(ValueError):
    pass


class _Overlap(RuntimeError):
    pass


class _Stopped(RuntimeError):
    pass


class _SeedReady(Exception):
    pass


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
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


def _put(path: Path, value) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _empty() -> dict:
    return {
        "schema_version": 1, "phase": "waiting", "cycle_open": False,
        "reserved_baseline": 0, "cycle_reservations": 0,
        "last_outcome": None, "champion": None, "ref": None, "patch": None, "error": None,
        "counts": {"cycles": 0, "proposals": 0}, "round_baseline": 0, "round_limit": 2,
        "cycle_champion": None, "cycle_healthy": False, "base_commit": None, "inflight": None,
    }


def _state(root: Path) -> dict:
    try:
        state = _read(root / "state.json")
    except FileNotFoundError:
        return _empty()
    if state.get("schema_version") != 1 or not state.get("digest"):
        raise ValueError("invalid mission checkpoint; refusing to reset its budget")
    return state


def _core(root: Path) -> dict | None:
    try:
        return _read(root / "rsi" / "state.json")
    except FileNotFoundError:
        return None


def _nat(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("invalid durable counter")
    return value


def _count(core: dict | None, key: str) -> int:
    return 0 if core is None else _nat(core[key])


def _entry(core: dict | None, identity=None) -> dict | None:
    if core is None:
        return None
    identity = core["champion"] if identity is None else identity
    archive = core["archive"]
    row = (archive.get(identity) if isinstance(archive, dict) else
           next((item for item in archive if item["id"] == identity), None))
    if not isinstance(row, dict) or not isinstance(row.get("commit"), str):
        raise ValueError("core champion is not a measured archive entry")
    return row


def _reconcile(state: dict, core: dict | None) -> None:
    total = _count(core, "reserved_proposals")
    spending = total - _nat(state["reserved_baseline"])
    completed = _count(core, "next_round") - _nat(state["round_baseline"])
    if (total < _nat(state["counts"]["proposals"]) or not 0 <= spending <= 4
            or spending % 2 or not 0 <= completed <= min(state["round_limit"], spending // 2)):
        raise ValueError("inconsistent core reservations or round progress; refusing dispatch")
    state["cycle_reservations"] = spending
    state["counts"]["proposals"] = total
    champion = _entry(core)
    if state["champion"] != champion:
        state["patch"] = None
    state["champion"] = champion
    state["ref"] = champion.get("ref") if champion else None
    if champion:
        if state["base_commit"] is None:
            state["base_commit"] = _entry(core, "seed")["commit"]
        if state["cycle_open"]:
            if state["cycle_champion"] is None:
                state["cycle_champion"] = champion["commit"]
            elif champion["commit"] != state["cycle_champion"]:
                state["last_outcome"] = "promoted"


def _saturated(core: dict | None, mission: dict) -> bool:
    champion = _entry(core)
    if not champion:
        return False
    for metric in ("search", "confirm"):
        score = (champion.get(metric) or {}).get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score < mission[metric + "_ceiling"]):
            return False
    return True


def _step(state: dict, core: dict | None, mission: dict) -> str | None:
    if core is not None:
        if core.get("phase") == "finalized":
            return None
        if core.get("phase") == "finalizing" or _saturated(core, mission):
            return "finalize"
    if not state["cycle_open"]:
        return None
    if core is None:
        return "seed"
    if (state["cycle_reservations"] + 2 > 4
            or core["next_round"] - state["round_baseline"] >= mission["rounds_per_cycle"]):
        return None
    return "round"


def _identity(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": pid, "starttime": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def _ownership(root: Path) -> str:
    try:
        owner = _read(root / "owner.json")
    except (OSError, ValueError):
        owner = {}
    try:
        with (root / "owner.lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "live" if owner and _identity(owner.get("pid")) == owner else "stale"
    except FileNotFoundError:
        pass
    return "stale" if owner else "none"


@contextlib.contextmanager
def _owner(root: Path):
    # Never replace this inode: independent ticks must contend on the same lock.
    with (root / "owner.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise _Overlap("mission already has a kernel owner") from exc
        identity = _identity(os.getpid())
        if identity is None:
            raise RuntimeError("cannot identify the mission owner")
        _put(root / "owner.json", identity)
        try:
            yield
        finally:
            _put(root / "owner.json", {})


def _control(root: Path, stop: bool) -> None:
    # STOP has its own lock/file, so an active owner's state save cannot erase it.
    with (root / "control.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if stop:
            _put(root / "STOP", {"requested_at": time.time()})
        else:
            (root / "STOP").unlink(missing_ok=True)


def _wait_core(root: Path) -> None:
    # A killed outer owner releases its lock before its guardian finishes draining.
    try:
        lock = (root / "rsi" / "run.lock").open("rb")
    except FileNotFoundError:
        return
    with lock:
        deadline = time.monotonic() + 5
        while True:
            if (root / "STOP").exists():
                raise _Stopped("STOP requested")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("previous core worker has not released its lock")
                time.sleep(0.05)


@contextlib.contextmanager
def _cancellation(root: Path):
    cancel, done = threading.Event(), threading.Event()

    def watch():
        while not done.is_set():
            if (root / "STOP").exists():
                cancel.set()
                return
            done.wait(0.05)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield cancel
    finally:
        done.set()
        thread.join(timeout=0.2)


def _inputs(path: Path, digest=None) -> dict:
    try:
        inputs = load_inputs(path)
    except ValueError as exc:
        if digest is not None:
            raise _Changed(f"frozen inputs are invalid or unavailable: {exc}") from exc
        raise
    if digest is not None and inputs["digest"] != digest:
        raise _Changed("frozen mission, RSI, bridge or controller inputs changed")
    return inputs


def _signature(core: dict | None) -> str:
    return hashlib.sha256(json.dumps(core, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _worker_timeout(config: dict) -> float:
    # Include generous room for both candidates, repeated measurements and cleanup.
    repeats = sum(config.get(k, 1) for k in ("search_repeats", "confirm_repeats", "sealed_repeats"))
    return 60 + 2 * config["timeout"] + 4 * config.get("evaluation_timeout", 300) * (
        len(config.get("checks", [])) + repeats)


def _worker() -> int:
    try:
        request = json.load(sys.stdin)
        state = request["state"]
        inputs = _inputs(Path(request["mission_path"]), state["digest"])
        mission = inputs["mission"]
        root = Path(mission["state_dir"])
        if (root / "STOP").exists():
            raise _Stopped("STOP requested before core launch")
        core = _core(root)
        if _signature(core) != request["checkpoint"]:
            raise RuntimeError("core checkpoint changed before worker launch")
        _reconcile(state, core)
        job = state["inflight"]
        kind = job["kind"]
        if _step(state, core, mission) != kind:
            raise RuntimeError("worker action is not authorized by the durable budget")
        directory = root / "workers" / job["id"]
        with (directory / "events.jsonl").open("x", encoding="utf-8") as events:
            def on_event(row):
                events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                events.flush()
                os.fsync(events.fileno())
                if kind == "seed" and row.get("event") == "seed":
                    # This event follows the ready checkpoint, before any reservation.
                    raise _SeedReady()

            with contextlib.redirect_stdout(sys.stderr):
                from gama.rsi import run_rsi
                try:
                    result = run_rsi(
                        inputs["rsi_config"], repo=mission["repo"], state_dir=root / "rsi",
                        rounds=1, resume=kind != "seed", finalize=kind == "finalize",
                        on_event=on_event,
                    )
                except _SeedReady:
                    if kind != "seed":
                        raise
                    result = {"seeded": True}
        _put(directory / "result.json", result)
        return 0
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 5 if isinstance(exc, _Changed) else 4 if isinstance(exc, _Stopped) else 2


def _settle(root: Path, state: dict, core: dict | None) -> bool:
    job = state["inflight"]
    if not job or core is None:
        return False
    kind = job["kind"]
    if kind == "round":
        # Receipts precede commit. Only checkpoint advancement proves completion.
        if core["next_round"] != job["round"] + 1 or core.get("pending") is not None:
            return False
        if core["reserved_proposals"] != job["reserved"] + 2:
            raise ValueError("completed round has an inconsistent reservation delta")
        path = root / "workers" / job["id"] / "events.jsonl"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            lines = []
        candidates = {}
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "candidate" and row.get("round", job["round"]) == job["round"]:
                candidates[(row.get("round"), row.get("attempt"), row.get("id"))] = row
        rows = list(candidates.values())
        if rows:
            rows = [row for row in rows if row.get("attempt") == rows[-1].get("attempt")]
        statuses = {row.get("status") for row in rows}
        grew = len(core["archive"]) > job["archive_size"]
        state["cycle_healthy"] |= grew or bool(statuses & {"viable", "duplicate"})
        failures = core.get("recent_failures", [])
        all_failed = (len(rows) == 2 and all(row.get("status") == "rejected" for row in rows)) or (
            len(failures) == 2 and all(row.get("stage") in ("proposal", "evaluation") for row in failures))
        if all_failed and not state["cycle_healthy"]:
            state["phase"] = "blocked"
            state["last_outcome"] = "rejected"
            state["error"] = "All proposals were rejected during generation or evaluation; explicit resume required."
        elif state["last_outcome"] != "promoted":
            state["last_outcome"] = ("viable" if grew or "viable" in statuses else
                                     "duplicate" if "duplicate" in statuses else "unchanged")
    elif kind == "seed":
        if core["reserved_proposals"] != job["reserved"] or core["next_round"] != job["round"]:
            return False
    elif kind == "finalize":
        if core.get("phase") != "finalized":
            return False
    else:
        raise ValueError("unknown durable worker action")
    state["inflight"] = None
    return True


def _dispatch(root: Path, path: Path, state: dict, inputs: dict, kind: str, cancel) -> str | None:
    core = _core(root)
    _reconcile(state, core)
    if _step(state, core, inputs["mission"]) != kind:
        raise RuntimeError("checkpoint no longer authorizes this dispatch")
    job = {"id": uuid.uuid4().hex, "kind": kind, "round": _count(core, "next_round"),
           "reserved": _count(core, "reserved_proposals"),
           "archive_size": len(core["archive"]) if core else 0}
    directory = root / "workers" / job["id"]
    directory.mkdir(mode=0o700, parents=True)
    state["inflight"] = job
    state["phase"] = "active"
    # This journal (including the cycle baseline) must precede even bootstrap.
    _put(root / "state.json", state)
    request = {"mission_path": str(path), "state": state, "checkpoint": _signature(core)}
    _put(directory / "request.json", request)
    try:
        result = run_guarded(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
            cwd=Path(inputs["mission"]["repo"]), input_text=json.dumps(request, allow_nan=False),
            timeout=_worker_timeout(inputs["rsi_config"]), artifact_dir=directory / "guard", cancel=cancel,
        )
    except (ProcessError, OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if result.returncode == 5:
        raise _Changed(result.stderr.strip() or "worker detected changed frozen inputs")
    if result.returncode:
        return result.stderr[-2000:].strip() or f"core worker exited {result.returncode}"
    return None


def _export(root: Path, state: dict, mission: dict) -> None:
    if not state["champion"]:
        return
    commit = state["champion"]["commit"]
    destination = root / "exports" / (commit + ".patch")
    if not destination.exists():
        result = run_guarded(
            ["git", "--no-pager", "diff", "--no-ext-diff", "--no-textconv", "--binary",
             "--full-index", state["base_commit"], commit, "--"],
            cwd=Path(mission["repo"]), timeout=15, artifact_dir=root / "workers" / uuid.uuid4().hex / "export",
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:] or "champion patch export failed")
        _write(destination, result.stdout)
    state["patch"] = str(destination)


def _drive(root: Path, path: Path, state: dict, inputs: dict, action: str) -> int:
    if action == "resume":
        _control(root, False)
    if (root / "STOP").exists():
        raise _Stopped("STOP requested")
    if action == "run" and state["phase"] == "blocked":
        return 4
    _wait_core(root)
    core = _core(root)
    _reconcile(state, core)
    _settle(root, state, core)
    if action == "resume":
        state["phase"], state["error"] = "waiting", None
    elif state["phase"] == "blocked":
        _put(root / "state.json", state)
        return 4
    mission = inputs["mission"]
    terminal = core is not None and core.get("phase") in ("finalizing", "finalized")
    if not state["cycle_open"] and action == "run" and not terminal:
        state.update(cycle_open=True, reserved_baseline=_count(core, "reserved_proposals"),
                     cycle_reservations=0, round_baseline=_count(core, "next_round"),
                     round_limit=mission["rounds_per_cycle"], cycle_healthy=False, inflight=None,
                     cycle_champion=state["champion"]["commit"] if state["champion"] else None,
                     last_outcome=None, error=None)
        state["counts"]["cycles"] += 1
    state["phase"] = "active" if state["cycle_open"] or terminal else "waiting"
    _put(root / "state.json", state)
    with _cancellation(root) as cancel:
        while True:
            inputs = _inputs(path, state["digest"])
            core = _core(root)
            _reconcile(state, core)
            if cancel.is_set() or (root / "STOP").exists():
                raise _Stopped("STOP requested")
            if state["phase"] == "blocked":
                _put(root / "state.json", state)
                return 0
            if core is not None and core.get("phase") == "finalized":
                state.update(phase="saturated", cycle_open=False, last_outcome="saturated", inflight=None)
                kind = None
            else:
                kind = _step(state, core, mission)
            if kind is None:
                if state["phase"] != "saturated":
                    state["last_outcome"] = state["last_outcome"] or ("exhausted" if state["cycle_open"] else None)
                    state.update(phase="waiting", cycle_open=False, inflight=None)
                _export(root, state, mission)
                _put(root / "state.json", state)
                return 0
            failure = _dispatch(root, path, state, inputs, kind, cancel)
            core = _core(root)
            _reconcile(state, core)
            completed = _settle(root, state, core)
            _put(root / "state.json", state)
            if cancel.is_set() or (root / "STOP").exists():
                raise _Stopped("STOP requested")
            _inputs(path, state["digest"])
            # A committed checkpoint wins even if worker stdout/export failed.
            if not completed:
                raise RuntimeError(failure or "core worker returned without checkpoint progress")


def _execute(root: Path, path: Path, action: str) -> int:
    with _owner(root):
        state = _state(root)
        inputs = _inputs(path, state.get("digest"))
        if "digest" not in state:
            if (root / "rsi").exists():
                raise ValueError("core directory exists without frozen mission metadata")
            state["digest"] = inputs["digest"]
            state["mission_id"] = inputs["mission"]["mission_id"]
            _put(root / "state.json", state)
        try:
            return _drive(root, path, state, inputs, action)
        except _Changed:
            raise
        except _Stopped:
            state["phase"], state["error"] = "stopped", None
            _put(root / "state.json", state)
            return 4
        except Exception as exc:
            state["phase"] = "blocked"
            state["error"] = f"{type(exc).__name__}: {exc}"
            _put(root / "state.json", state)
            return 2


def _root(path: Path) -> Path:
    # stop/status need only the locator; a broken provider/config cannot disable STOP.
    mission = _read(path)
    paths = []
    for key in ("state_dir", "repo"):
        value = mission.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"mission.{key} must be an absolute path")
        paths.append(Path(value).resolve())
    root, repo = paths
    if root.is_relative_to(repo) or repo.is_relative_to(root):
        raise ValueError("mission state must be external to the repository")
    return root


_PUBLIC = ("phase", "cycle_open", "reserved_baseline", "cycle_reservations", "last_outcome",
           "champion", "ref", "patch", "error", "counts")


def _view(root: Path) -> dict:
    state = _state(root)
    _reconcile(state, _core(root))
    result = {key: state[key] for key in _PUBLIC}
    result["ownership"] = _ownership(root)
    if (root / "STOP").exists():
        result["phase"] = "stopped"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "stop", "resume"))
    parser.add_argument("--mission", type=Path, required=True)
    args = parser.parse_args(argv)
    root, error, code = None, None, 0
    try:
        path = args.mission.resolve()
        root = _root(path)
        if args.action != "status":
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if args.action == "stop":
            _control(root, True)
        elif args.action != "status":
            code = _execute(root, path, args.action)
    except _Overlap as exc:
        code, error = 3, str(exc)
    except _Changed as exc:
        code, error = 5, str(exc)
    except (Exception, KeyboardInterrupt) as exc:
        code, error = 2, f"{type(exc).__name__}: {exc}"
    result = {key: _empty()[key] for key in _PUBLIC}
    result["ownership"] = "none"
    if root is not None:
        try:
            result = _view(root)
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            if code == 0 and args.action != "stop":
                code = 2
            if (root / "STOP").exists():
                result["phase"] = "stopped"
    if error:
        result["error"] = error
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(_worker() if sys.argv[1:] == ["--worker"] else main())
