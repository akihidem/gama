"""Offline bridge acceptance."""
import json, os, subprocess as sp, tempfile, time
from pathlib import Path
from ops_common import Env, alive, wait_for


def snapshot(root):
    return {p: p.read_bytes() for p in root.rglob("*")
            if ".git" not in p.relative_to(root).parts and p.is_file()}


def make(root, mode, deadline=5):
    e = Env(root / mode, mode=mode)
    cfg = json.loads(e.bridge.read_text())
    cfg["timeout"] = deadline
    e.bridge.write_text(json.dumps(cfg))
    (e.repo / "caller-sentinel").write_text("x")
    return e


def roles(e): return [r["role"] for r in e.events()]


def invoke(e, request=None, expected=None):
    before = snapshot(e.repo)
    p, _ = e.bridge_run(request=request, timeout=12)
    assert snapshot(e.repo) == before, "caller changed"
    dirs = list(e.artifacts.glob("proposal-*"))
    assert dirs and all(snapshot(d) for d in dirs), "missing evidence"
    if expected is None:
        assert p.returncode and not p.stdout and p.stderr.strip(), p
    else:
        assert not p.returncode and p.stdout == expected, p
    return dirs


def cleanup(e):
    for p in (e.worker_pid, e.leaf_pid):
        try:
            os.kill(int(p.read_text()), 9)
        except (OSError, ValueError):
            pass


def check(root):
    e = make(root, "noise")
    diff = "--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    first = invoke(e, expected=diff)
    assert len(first) == 1 and roles(e) == ["builder", "reviewer"]
    records = {name: json.loads((first[0] / (name + ".json")).read_text())
               for name in ("request", "identities", "usage", "result")}
    assert records["request"] == e.request()
    for role, values in (
        ("builder", ("codex", "astra", "openai", "bedrock-astra")),
        ("reviewer", ("bedrock", "opus", "anthropic", "global.anthropic.claude-opus-5")),
    ):
        assert records["identities"][role] == dict(
            zip("provider model family resolved_model".split(), values), simulated=True)
        assert records["usage"][role]["fixture_tokens"] == 1
    saved = snapshot(first[0])
    both = invoke(e, expected=diff)
    assert len(both) == 2 and snapshot(first[0]) == saved
    assert roles(e) == ["builder", "reviewer"] * 2
    assert len({r["output_dir"] for r in e.events()}) == 4
    for r in e.events():
        d = Path(r["output_dir"])
        assert d.parent in both and d.name == r["role"]
        if r["role"] == "reviewer":
            raw = (d / "prompt.txt").read_text().split("\nREVIEW_JSON\n", 1)[1]
            assert json.loads(raw) == {"request": e.request(), "patch": diff}

    for mode, calls in (("reject", 2), ("invalid_review", 2), ("extra_review", 2),
                        ("invalid_constant", 2), ("invalid_patch", 1),
                        ("error", 1), ("bad_identity", 0)):
        e = make(root, mode)
        assert len(invoke(e)) == 1
        assert roles(e) == ["builder", "reviewer"][:calls], mode

    for cap, size in ((200000, 110001), (1000, 2000)):
        e = make(root / str(cap), "increment")
        cfg = json.loads(e.bridge.read_text())
        cfg["backend"]["max_context_bytes"] = cap
        e.bridge.write_text(json.dumps(cfg))
        request = e.request()
        request["goal"] = "x" * size
        invoke(e, request)
        assert not e.events()

    e = make(root, "hang", .5)
    started = time.monotonic()
    try:
        invoke(e)
        assert roles(e) == ["builder"]
        assert wait_for(lambda: not alive(e.worker_pid.read_text()), timeout=1)
        assert time.monotonic() - started < 6.5
    finally:
        cleanup(e)

    e = make(root, "detached", 20)
    before = snapshot(e.repo)
    outer = sp.Popen(e.bridge_cmd(), cwd=e.repo, stdin=sp.PIPE, text=True,
                     stdout=sp.DEVNULL, stderr=sp.DEVNULL, start_new_session=True)
    try:
        outer.stdin.write(json.dumps(e.request()))
        outer.stdin.close()
        files = (e.worker_pid, e.leaf_pid)
        assert wait_for(lambda: all(p.exists() and p.read_text().isdigit() for p in files))
        pids = [int(p.read_text()) for p in files]
        assert all(alive(pid) for pid in pids)
        outer.kill()
        outer.wait(timeout=5)
        assert wait_for(lambda: all(not alive(pid) for pid in pids), timeout=5)
        assert snapshot(e.repo) == before
        assert roles(e) == ["builder"]
    finally:
        if outer.poll() is None:
            outer.kill()
        outer.wait(timeout=5)
        cleanup(e)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bridge-acceptance-") as tmp:
        check(Path(tmp))
    print("bridge PASS")
