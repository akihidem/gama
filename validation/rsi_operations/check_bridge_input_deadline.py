"""Run from a bridge checkout; stdin deliberately stays open until process exit."""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, os, subprocess as sp, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "validation/rsi_operations"))
from ops_common import Env
CASES = ("empty", "partial", "complete")
TIMEOUT, BOUND = .2, 6.0  # Includes five seconds for cleanup plus launch tolerance.

def snapshot(repo):
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in repo.rglob("*") if p.is_file()
            and ".git" not in p.relative_to(repo).parts}


def check(root, case):
    e = Env(root / case)
    config = json.loads(e.bridge.read_text())
    config["timeout"] = TIMEOUT
    e.bridge.write_text(json.dumps(config))
    payload = {"empty": b"", "partial": b'{"goal":',
               "complete": json.dumps(e.request()).encode()}[case]
    before = snapshot(e.repo)
    out_path, err_path = e.tmp / "stdout.bin", e.tmp / "stderr.bin"
    forced = False
    with out_path.open("xb") as out, err_path.open("xb") as err:
        started = time.monotonic()
        p = sp.Popen(e.bridge_cmd(), cwd=e.repo, stdin=sp.PIPE, stdout=out,
                     stderr=err, start_new_session=True,
                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        try:
            try:
                p.stdin.write(payload)
                p.stdin.flush()
            except BrokenPipeError:
                pass
            try:
                p.wait(timeout=max(0, BOUND - (time.monotonic() - started)))
            except sp.TimeoutExpired:
                forced = True
        finally:
            elapsed = time.monotonic() - started
            if p.poll() is None:
                p.kill()  # Only this test's Popen child; its guardian owns descendants.
            p.wait(timeout=2)
            try:
                p.stdin.close()
            except BrokenPipeError:
                pass
    with err_path.open("rb") as stream:
        stderr = stream.read(4096).decode("utf-8", errors="replace")
    proposals = sorted(e.artifacts.glob("proposal-*"))
    result = None
    if len(proposals) == 1:
        try:
            result = json.loads((proposals[0] / "result.json").read_text())
        except (OSError, ValueError):
            pass
    calls = e.events()
    checks = {
        "bounded_exit": not forced and elapsed <= BOUND,
        "nonzero_with_stderr": p.returncode != 0 and bool(stderr.strip()),
        "empty_stdout": out_path.stat().st_size == 0, "zero_calls": not calls,
        "caller_unchanged": snapshot(e.repo) == before,
        "failed_result": isinstance(result, dict) and result.get("status") == "failed"
                         and isinstance(result.get("error"), str) and bool(result["error"].strip()),
        "evidence": len(proposals) == 1 and all((proposals[0] / name).is_file()
                    for name in ("request.json", "identities.json", "usage.json", "result.json")),
    }
    return dict(case=case, passed=all(checks.values()), checks=checks, calls=calls,
                elapsed_s=elapsed, watchdog_killed=forced, returncode=p.returncode,
                stderr=stderr, result=result, artifacts=[str(p) for p in proposals])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--directory", type=Path, help="New directory; retained after execution")
    args = parser.parse_args()
    root = args.directory or Path(tempfile.mkdtemp(prefix="bridge-input-deadline-"))
    if args.directory:
        root.mkdir(parents=True)  # Refuse existing paths; never replace evidence.
    rows = []
    for case in (args.case,) if args.case else CASES:
        try:
            rows.append(check(root.resolve(), case))
        except Exception as exc:
            rows.append(dict(case=case, passed=False, error=f"{type(exc).__name__}: {exc}"))
    print(json.dumps(dict(passed=sum(r["passed"] for r in rows), total=len(rows),
                          timeout_s=TIMEOUT, bound_s=BOUND, directory=str(root.resolve()),
                          cases=rows), indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
