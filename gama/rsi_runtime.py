"""Frozen control inputs shared by the RSI mission runner and bridge."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


def _constant(text: str):
    raise ValueError(f"nonfinite JSON value: {text}")


def _float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _input_path(value, label: str) -> Path:
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _absolute(value, label: str) -> Path:
    if (not isinstance(value, str) or not value or "\0" in value
            or not Path(value).is_absolute()):
        raise ValueError(f"{label} must be an absolute path")
    return _input_path(value, label)


def _read(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_float=_float,
                           parse_constant=_constant, object_pairs_hook=_object)
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _number(value, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _integer(value, label: str, choices: tuple[int, ...] | None = None) -> int:
    if type(value) is not int or value <= 0 or (choices is not None and value not in choices):
        raise ValueError(f"invalid {label}: expected " + (str(choices) if choices else "a positive integer"))
    return value


def _directory(path: Path, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")


def _overlap(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


def _bridge(data: dict) -> dict:
    result = dict(data)
    for key in ("astra_loop_root", "artifact_root"):
        path = _absolute(data.get(key), key)
        _directory(path, key)
        result[key] = str(path)
    if Path(result["artifact_root"]) == Path(result["artifact_root"]).parent:
        raise ValueError("artifact_root must not be the filesystem root")
    result["timeout"] = _number(data.get("timeout", 510), "bridge.timeout", positive=True)
    backend = data.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("bridge.backend must be an object")
    # Keep adapter-specific settings intact; the explicit package interprets them.
    result["backend"] = dict(backend)
    for key in ("max_context_bytes", "bedrock_max_tokens"):
        if key in backend:
            _integer(backend[key], f"backend.{key}")
    if "timeout_seconds" in backend:
        _number(backend["timeout_seconds"], "backend.timeout_seconds", positive=True)
    return result


def load_bridge_config(path: Path) -> dict:
    """Parse the bridge schema without loading an adapter or the bridge module."""
    data, _ = _read(_input_path(path, "bridge config path"))
    return _bridge(data)


def _controller(path: Path) -> bool:
    parts = path.parts
    return (len(parts) >= 2 and parts[0] == "gama"
            and parts[1].startswith("rsi") and parts[1].endswith(".py"))


def _targets(repo: Path, config: dict, frozen: set[Path]) -> None:
    targets = config.get("allowed_paths", [])
    if not isinstance(targets, list):
        raise ValueError("allowed_paths must be a list of individual repository files")
    for name in targets:
        if (not isinstance(name, str) or not name or "\0" in name
                or any(char in name for char in "*?[]\\")):
            raise ValueError("allowed_paths must name individual files without wildcards")
        path = Path(name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(f"unsafe mutation target: {name}")
        resolved = _input_path(repo / path, "mutation target")
        if not resolved.is_relative_to(repo) or resolved.is_dir():
            raise ValueError(f"mutation target must be a repository file: {name}")
        if _controller(path) or _controller(resolved.relative_to(repo)):
            raise ValueError(f"outer RSI modules are immutable: {name}")
        if resolved in frozen:
            raise ValueError(f"control input is immutable: {name}")


def _code_hashes() -> dict:
    # Missing parallel-node files are allowed during construction. Their later
    # presence, or a source edit on continuation, changes the frozen digest.
    hashes = {}
    for name in ("rsi_guard.py", "rsi_runtime.py", "rsi_bridge.py", "rsi_mission.py"):
        path = Path(__file__).resolve().with_name(name)
        try:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            hashes[str(path)] = None
        except OSError as exc:
            raise ValueError(f"cannot freeze controller {path}: {exc}") from exc
    return hashes


def load_inputs(mission_path: Path) -> dict:
    """Return normalized mission/configs and a digest of all immutable controls."""
    mission_path = _input_path(mission_path, "mission path")
    raw_mission, mission_sha = _read(mission_path)
    mission = dict(raw_mission)
    mission_id = mission.get("mission_id")
    if not isinstance(mission_id, str) or not mission_id.strip() or "\0" in mission_id:
        raise ValueError("mission_id must be a nonempty string")
    paths = {}
    for key in ("repo", "state_dir", "rsi_config", "bridge_config"):
        paths[key] = _absolute(mission.get(key), f"mission.{key}")
        mission[key] = str(paths[key])
    repo, state_dir = paths["repo"], paths["state_dir"]
    if not repo.is_dir():
        raise ValueError(f"mission.repo must be an existing directory: {repo}")
    _directory(state_dir, "state_dir")
    if _overlap(repo, state_dir):
        raise ValueError("state_dir must be external to and disjoint from the repository")
    for key in ("search_ceiling", "confirm_ceiling"):
        mission[key] = _number(mission.get(key), f"mission.{key}")
    mission["rounds_per_cycle"] = _integer(
        mission.get("rounds_per_cycle", 2), "rounds_per_cycle", (1, 2))
    for key, value in (("batch_size", 2), ("max_reservations_per_cycle", 4)):
        mission[key] = _integer(mission.get(key, value), key, (value,))

    raw_rsi, rsi_sha = _read(paths["rsi_config"])
    raw_bridge, bridge_sha = _read(paths["bridge_config"])
    bridge = _bridge(raw_bridge)
    artifacts = Path(bridge["artifact_root"])
    if _overlap(repo, artifacts):
        raise ValueError("artifact_root must be external to and disjoint from the repository")
    # Core owns this entire subtree; proposal evidence must not land in it.
    if _overlap(state_dir / "rsi", artifacts):
        raise ValueError("artifact_root must be disjoint from the core state directory")

    rsi = dict(raw_rsi)
    for key in ("workers", "batch_size"):
        rsi[key] = _integer(rsi.get(key, 2), f"rsi.{key}", (2,))
    rsi["timeout"] = _number(
        rsi.get("timeout", bridge["timeout"] + 5), "rsi.timeout", positive=True)
    if rsi["timeout"] < bridge["timeout"] + 5:
        raise ValueError("rsi.timeout must leave at least 5 seconds beyond bridge.timeout")
    if "evaluation_timeout" in rsi:
        rsi["evaluation_timeout"] = _number(
            rsi["evaluation_timeout"], "rsi.evaluation_timeout", positive=True)

    input_files = [mission_path, paths["rsi_config"], paths["bridge_config"]]
    _targets(repo, rsi, set(input_files))
    evaluation_files = rsi.get("evaluation_files", [])
    if (not isinstance(evaluation_files, list)
            or any(not isinstance(p, str) or not p or "\0" in p for p in evaluation_files)):
        raise ValueError("evaluation_files must be a list of file paths")
    rsi["evaluation_files"] = list(dict.fromkeys(
        [*evaluation_files, *(str(path) for path in input_files)]))
    # Preserve the venv executable spelling: resolving its symlink loses the venv.
    command = [
        sys.executable, "-I", "-B", str(Path(__file__).resolve().with_name("rsi_bridge.py")),
        "--config", str(paths["bridge_config"]),
    ]
    rsi["agents"] = [{"name": name, "command": list(command)} for name in ("astra-a", "astra-b")]

    frozen = {
        "schema_version": 1,
        "mission_path": str(mission_path),
        "mission": mission,
        "rsi_config": raw_rsi,
        "effective_rsi_config": rsi,
        "bridge_config": bridge,
        "file_sha256": [
            {"path": str(path), "sha256": sha}
            for path, sha in zip(input_files, (mission_sha, rsi_sha, bridge_sha))
        ],
        "python": {"executable": sys.executable, "major_minor": list(sys.version_info[:2])},
        "controller_sha256": _code_hashes(),
    }
    payload = json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "mission": mission,
        "rsi_config": rsi,
        "bridge_config": bridge,
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
