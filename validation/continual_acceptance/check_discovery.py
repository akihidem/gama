"""Run from the implementation checkout with the pinned Python; all models are fake."""
import sys
sys.dont_write_bytecode = True
import json, os, subprocess as sp, tempfile, time
from pathlib import Path
from campaign_common import PY, fixture, process_identity, wait_for
from discovery_common import install, snapshot


def invoke(e, mode, *, seen=None, cursor=0, limit=18):
    e.mode.write_text(mode)
    cfg = json.loads(e.bridge.read_text())
    cfg["timeout"] = limit
    e.bridge.write_text(json.dumps(cfg))
    index = len(list(e.tmp.glob("discovery-input-*.json")))
    directory = e.tmp / f"discovery-{index}"
    request = e.tmp / f"discovery-input-{index}.json"
    request.write_text(json.dumps({"config": json.loads(e.config.read_text()),
                                  "directory": str(directory), "seen": seen or [], "cursor": cursor}))
    before, head = snapshot(e.repo), e.git("rev-parse", "HEAD")
    start_calls, started = len(e.events()), time.monotonic()
    p = sp.Popen([PY, "-B", str(Path(__file__).resolve()), "--worker", str(request)],
                 cwd=e.repo, env=e.git_env, stdout=sp.PIPE, stderr=sp.PIPE,
                 text=True, start_new_session=True)
    e.children.append(p)
    try:
        out, err = p.communicate(timeout=limit + 7)
    except sp.TimeoutExpired:
        p.kill()
        p.communicate(timeout=5)
        raise AssertionError("discovery exceeded its whole-pair deadline")
    assert snapshot(e.repo) == before and e.git("rev-parse", "HEAD") == head, "caller changed"
    assert not (e.tmp / "sealed-was-executed").exists(), "discovery executed sealed source"
    assert wait_for(e.drained, timeout=5), "provider survived discovery"
    value = json.loads(out) if out.strip() else None
    calls = e.events()[start_calls:]
    assert len([r for r in calls if r["role"] == "builder"]) <= 2, calls
    assert len([r for r in calls if r["role"] == "reviewer"]) <= 2, calls
    assert len({r["output_dir"] for r in calls}) == len(calls), "reused evidence directory"
    if not p.returncode:
        assert isinstance(value, dict) and set(value) == {"goals", "cursor", "rejected"}, value
        assert type(value["cursor"]) is int and isinstance(value["rejected"], list)
    else:
        assert err.strip() and not out.strip(), (out, err)
    return p.returncode, value, directory, calls, time.monotonic() - started


def check():
    with fixture(empty=True) as e:
        install(e)
        rc, result, directory, calls, _ = invoke(e, "valid")
        assert rc == 0 and result["goals"], result
        assert sum(r["role"] == "builder" for r in calls) == 2
        assert 1 <= sum(r["role"] == "reviewer" for r in calls) <= 2
        # Exact fake identities/usage must survive; do not prescribe artifact layout.
        records = []
        for path in directory.rglob("*.json"):
            try:
                records.append(json.loads(path.read_bytes()))
            except ValueError:
                pass
        encoded = json.dumps(records)
        for token in ("bedrock-astra", "global.anthropic.claude-opus-5", "fixture_tokens", "simulated"):
            assert token in encoded, "missing actual adapter evidence: " + token
        for row in calls:
            prompt = Path(row["output_dir"]) / "prompt.txt"
            assert prompt.exists()
            if row["role"] == "builder":
                req = json.loads(prompt.read_text().split("\nDISCOVERY_JSON\n", 1)[1])
                assert req["files"] and all(name.startswith("gama/fixture_") for name in req["files"])
                assert sum(len(text.encode()) for text in req["files"].values()) <= 60000
        saved = snapshot(directory)
        # seen consists of portable descriptors, independent of history outcome.
        for mode in ("valid", "semantic_duplicate"):
            rc, repeat, _, _, _ = invoke(e, mode, seen=result["goals"])
            assert rc == 0 and not repeat["goals"] and repeat["rejected"], repeat
            assert snapshot(directory) == saved, "prior evidence overwritten"
        cursor = result["cursor"]
        rc, rotated, _, rotated_calls, _ = invoke(e, "invalid", cursor=cursor)
        assert rc == 0 and not rotated["goals"] and rotated["rejected"], rotated
        # With two eligible files a full turn may wrap the numeric cursor; source
        # selection must still expose both, instead of pinning the first forever.
        selections = set()
        for row in [*calls, *rotated_calls]:
            if row["role"] == "builder":
                req = json.loads((Path(row["output_dir"]) / "prompt.txt").read_text().split("\nDISCOVERY_JSON\n", 1)[1])
                selections.update(req["files"])
        assert {"gama/fixture_total.py", "gama/fixture_text.py"} <= selections
        for mode in ("duplicate_keys", "nan", "extra", "path", "no_tests", "broken",
                     "test_error", "satisfied", "review_json", "review_extra", "reject", "oversize"):
            rc, bad, _, _, _ = invoke(e, mode)
            assert rc != 0 or (not bad["goals"] and bad["rejected"]), (mode, bad)
        rc, bad, _, calls, _ = invoke(e, "identity")
        assert not calls and (rc != 0 or not bad["goals"]), "unverified identity dispatched"
        # A private new-session leaf must also be drained by the whole-pair guard.
        try:
            rc, bad, _, _, elapsed = invoke(e, "hang", limit=1.5)
            assert elapsed < 7.5 and (rc != 0 or (not bad["goals"] and bad["rejected"]))
            leaves = [json.loads(p.read_text()) for p in e.mode.parent.glob("leaf-*.json")]
            assert leaves, "hang fixture never launched"
            assert wait_for(lambda: all(process_identity(r["pid"]) != r["start"] for r in leaves), timeout=2)
        finally:
            for path in e.mode.parent.glob("leaf-*.json"):
                row = json.loads(path.read_text())
                if process_identity(row["pid"]) == row["start"]:
                    try:
                        os.kill(row["pid"], 9)
                    except ProcessLookupError:
                        pass
    print("discovery PASS (offline fake adapter)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.path.insert(0, str(Path.cwd()))
        import threading
        from gama.continual_discover import discover
        value = json.loads(Path(sys.argv[2]).read_text())
        result = discover(value["config"], directory=Path(value["directory"]),
                          seen=value["seen"], cursor=value["cursor"], cancel=threading.Event())
        print(json.dumps(result, allow_nan=False))
    else:
        check()
