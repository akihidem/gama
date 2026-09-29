"""Offline source-RSI protocol example; not evidence of autonomous LLM improvement.

Run ``python3 examples/rsi_demo.py [--directory /tmp/my-demo]``. Artifacts persist
in a new or empty directory. Only the disposable toy repository receives commits.
The deterministic emitters and fixed evaluators make no model or network calls.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile


SCRIPT = Path(__file__).resolve()
SOURCE_NAME = "program.py"
SOURCE = """LIMIT = 1


def solve(value):
    return max(0, min(value, LIMIT))
"""
STEPS = {"raise-by-3": 3, "raise-by-6": 6}
MAX_LIMIT = 64
# Pairwise disjoint inputs: each split occupies a different residue modulo three.
INPUTS = {
    name: tuple(range(offset, 24, 3))
    for offset, name in enumerate(("search", "confirm", "sealed"))
}
NOTICE = (
    "Offline protocol example with deterministic patch emitters; "
    "this does not show autonomous LLM improvement."
)


def threshold(source: str) -> re.Match:
    match = re.search(r"(?m)^LIMIT = ([0-9]+)$", source)
    if match is None or not 1 <= int(match[1]) <= MAX_LIMIT:
        raise ValueError("toy source must declare a finite LIMIT between 1 and 64")
    return match


def emit_patch(name: str) -> None:
    request = json.load(sys.stdin)
    old = request["files"][SOURCE_NAME]
    # Read the selected parent's actual file, without importing the controller.
    if (Path.cwd() / SOURCE_NAME).read_text(encoding="utf-8") != old:
        raise ValueError("request source differs from the selected parent checkout")
    match = threshold(old)
    limit = min(MAX_LIMIT, int(match[1]) + STEPS[name])
    new = old[:match.start()] + f"LIMIT = {limit}" + old[match.end():]
    patch = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{SOURCE_NAME}", tofile=f"b/{SOURCE_NAME}",
    ))
    if not patch:
        raise ValueError("deterministic emitter has reached its finite limit")
    sys.stdout.write(patch)


def candidate() -> dict:
    # Absolute script argv does not put the candidate checkout on sys.path.
    # Load the target explicitly; run_path does not need a gama installation.
    return runpy.run_path(str(Path.cwd() / SOURCE_NAME))


def check() -> None:
    program = candidate()
    limit = program["LIMIT"]
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise ValueError("candidate must retain a finite integer clamp threshold")
    for value, expected in ((-100, 0), (-1, 0), (0, 0), (1, 1)):
        if program["solve"](value) != expected:
            raise ValueError(f"fixed baseline check failed for input {value}")
    print(json.dumps({"checks": "passed"}))


def score(split: str) -> None:
    solve = candidate()["solve"]
    cases = [
        {"input": value, "expected": value, "actual": solve(value)}
        for value in INPUTS[split]
    ]
    correct = sum(case["actual"] == case["expected"] for case in cases)
    print(json.dumps(
        {"score": correct / len(cases), "split": split, "cases": cases},
        allow_nan=False,
    ))


def demo_directory(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="gama-rsi-demo-"))
    path = requested.expanduser()
    if path.is_symlink():
        raise ValueError("--directory must not be a symlink")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("--directory must be absent or empty; nothing was overwritten")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def run_demo(requested: Path | None) -> dict:
    # Worker modes never reach these imports. The normal mode uses this checkout's
    # fixed controller, even though all candidate execution happens in the toy repo.
    sys.path.insert(0, str(SCRIPT.parent.parent))
    from gama.rsi import run_rsi
    from gama.rsi_process import run_process

    root = demo_directory(requested)
    print(NOTICE, file=sys.stderr)
    print(f"Persistent demo artifacts: {root}", file=sys.stderr)
    repo, state_dir = root / "repo", root / "run"
    repo.mkdir()
    (repo / SOURCE_NAME).write_text(SOURCE, encoding="utf-8")

    # Keep inherited Git routing/configuration from redirecting these operations
    # into the caller's repository. HOME and the caller's checkout are untouched.
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    git_env.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")

    def git(*args: str) -> str:
        result = run_process(
            ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "commit.gpgsign=false",
             "-c", "user.name=Gama offline demo", "-c", "user.email=demo@localhost", *args],
            cwd=repo, timeout=10, env=git_env,
        )
        if result.returncode:
            raise RuntimeError(f"toy Git command failed: {result.stderr[-2000:]}")
        return result.stdout

    git("init", "--quiet", "--template=")
    git("add", "--", SOURCE_NAME)
    git("commit", "--quiet", "-m", "Seed offline protocol example")

    worker = [str(Path(sys.executable).resolve()), "-I", "-B", str(SCRIPT)]
    config = {
        "goal": "Improve identity accuracy for nonnegative inputs while preserving "
                "the negative-input floor and finite clamp.",
        "allowed_paths": [SOURCE_NAME],
        "agents": [
            {"name": name, "command": [*worker, "--agent", name]} for name in STEPS
        ],
        "checks": [[*worker, "--check"]],
        "search_command": [*worker, "--score", "search"],
        "confirm_command": [*worker, "--score", "confirm"],
        "sealed_command": [*worker, "--score", "sealed"],
        "evaluation_files": [str(SCRIPT)],
        "workers": 2,
        "batch_size": 2,
        "timeout": 5,
        "evaluation_timeout": 5,
        "search_repeats": 1,
        "confirm_repeats": 2,
        "seed": 6,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    searched = run_rsi(config, repo=repo, state_dir=state_dir, rounds=2)
    if searched["rounds_completed"] != 2 or searched["sealed_verdict"] != "not_opened":
        raise RuntimeError("demo expected two search rounds with the sealed split unopened")
    (root / "search-result.json").write_text(
        json.dumps(searched, indent=2) + "\n", encoding="utf-8")
    # A separate invocation opens the sealed split and permanently closes search.
    finished = run_rsi(config, repo=repo, state_dir=state_dir, resume=True, finalize=True)
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    promotions = sum(
        event["event"] == "confirmation" and event.get("accepted", False) for event in events)
    champion_source = git("show", f"{finished['champion']['commit']}:{SOURCE_NAME}")
    if (promotions < 1 or champion_source == SOURCE or finished["phase"] != "finalized"
            or finished["sealed_verdict"] != "improved"):
        raise RuntimeError(f"demo did not produce the expected source improvement; inspect {root}")

    seed, champion = state["archive"][0], finished["champion"]
    result = {
        "demo_directory": str(root),
        "notice": NOTICE,
        "phase": finished["phase"],
        "rounds_completed": finished["rounds_completed"],
        "promotions": promotions,
        "base_limit": int(threshold(SOURCE)[1]),
        "champion_limit": int(threshold(champion_source)[1]),
        "base_scores": {"search": seed["search"]["score"], "confirm": seed["confirm"]["score"],
                        "sealed": state["sealed_base"]["score"]},
        "champion_scores": {"search": champion["search"]["score"],
                            "confirm": champion["confirm"]["score"],
                            "sealed": state["sealed_champion"]["score"]},
        "sealed_verdict": finished["sealed_verdict"],
        "base_commit": finished["base"],
        "champion_commit": champion["commit"],
        "champion_ref": finished["ref"],
        "patch": finished["patch"],
        "state_dir": str(state_dir),
    }
    (root / "demo-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=NOTICE)
    parser.add_argument("--directory", type=Path, help="absent or empty artifact directory")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--agent", choices=STEPS, help="internal patch-emitter mode")
    modes.add_argument("--score", choices=INPUTS, help="internal fixed-evaluator mode")
    modes.add_argument("--check", action="store_true", help="internal baseline-check mode")
    args = parser.parse_args()
    if args.directory is not None and (args.agent or args.score or args.check):
        parser.error("--directory applies only to the normal demo mode")
    try:
        if args.agent:
            emit_patch(args.agent)
        elif args.score:
            score(args.score)
        elif args.check:
            check()
        else:
            print(json.dumps(run_demo(args.directory), indent=2))
        return 0
    except Exception as exc:
        print(f"rsi_demo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
