"""Frozen offline finalized-report recovery acceptance; run from candidate cwd.

Uses the original campaign fixture unchanged. A test-only guard wrapper interrupts
the second goal after real core execution/drain, before the supervisor records its
result. It removes the report or restores an actual preceding ready report.
Only temporary fixture inputs/evidence change; product sources are never edited.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path.cwd() / "validation/continual_acceptance"))
from campaign_common import BRANCH, GOALS, NEW, PY, fixture, wait_for

NOW = "2031-04-12T09:01:00+09:00"

# Executed inside an isolated campaign caller, using the real guard and mission.
HARNESS = r'''
import json,os,sys
from datetime import datetime
from pathlib import Path
from gama import continual
config, control_name, variant, now = sys.argv[1:]
control_path = Path(control_name)
control = json.loads(control_path.read_text())
original = continual.run_guarded
def boundary(command, **kwargs):
    target = False
    if "--mission" in command:
        mission_path = Path(command[command.index("--mission")+1])
        descriptor = json.loads((mission_path.parent/"goal.json").read_text())
        target = descriptor["id"] == control["target"]
        if target:
            mission = json.loads(mission_path.read_text())
            control.update(mission_path=str(mission_path),
                           core=str(Path(mission["state_dir"])/"rsi"),
                           sealed_script=str(mission_path.parent/"sealed.py"))
            control_path.write_text(json.dumps(control))
    result = original(command, **kwargs)
    if target and result.returncode == 0:
        core_dir = Path(control["core"])
        state = json.loads((core_dir/"state.json").read_text())
        if state["phase"] == "finalized":
            assert state["sealed_verdict"] == "improved"
            report = core_dir/"result.json"
            final = json.loads(report.read_text())
            assert final["phase"] == "finalized" and final["sealed_verdict"] == "improved"
            Path(control["expected_report"]).write_bytes(report.read_bytes())
            previous = Path(control["previous_report"]).read_bytes()
            assert json.loads(previous)["phase"] == "ready"
            if variant == "missing":
                report.unlink()
            else:
                report.write_bytes(previous)
            receipt = Path(kwargs["artifact_dir"])/"process.json"
            assert json.loads(receipt.read_text()) == {"returncode": 0, "error": None}
            control.update(variant=variant, guard_receipt=str(receipt),
                           checkpoint=state)
            control_path.write_text(json.dumps(control))
            os._exit(91)
    return result
continual.run_guarded = boundary
continual._now = lambda: datetime.fromisoformat(now)
raise SystemExit(continual.main(["run", "--config", config]))
'''


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_source(control_path):
    # This observer is fixed before campaign initialization. It leaves all test
    # assertions intact and distinguishes core sealed calls from retained tests.
    return f'''
import json as _audit_json
from pathlib import Path as _AuditPath
_audit = _audit_json.loads(_AuditPath({str(control_path)!r}).read_text())
if str(_AuditPath(__file__).resolve()) == _audit.get("sealed_script"):
    with _AuditPath(_audit["sealed_calls"]).open("a") as _log:
        _log.write(_audit_json.dumps({{"script": __file__, "cwd": str(_AuditPath.cwd())}})+"\\n")
    _report = _AuditPath(_audit["core"])/"result.json"
    if _report.is_file() and _audit_json.loads(_report.read_text()).get("phase") == "ready":
        try:
            with _AuditPath(_audit["previous_report"]).open("xb") as _copy:
                _copy.write(_report.read_bytes())
        except FileExistsError:
            pass
'''


class FinalizedReportAcceptance(unittest.TestCase):
    maxDiff = 1600

    def exercise(self, variant):
        with fixture() as env:
            control_path = env.tmp / "report-boundary.json"
            control = dict(target=GOALS[1]["id"],
                           previous_report=str(env.tmp / "previous-report.json"),
                           expected_report=str(env.tmp / "expected-report.json"),
                           sealed_calls=str(env.tmp / "core-sealed.jsonl"))
            control_path.write_text(json.dumps(control))
            config = read(env.config)
            target_descriptor = Path(config["goals"][1])
            descriptor = read(target_descriptor)
            descriptor["tests"]["sealed"] += observation_source(control_path)
            target_descriptor.write_text(json.dumps(descriptor))

            child = subprocess.Popen(
                [PY, "-B", "-c", HARNESS, str(env.config), str(control_path), variant, NOW],
                cwd=env.repo, env=env.git_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True)
            env.children.append(child)
            stdout, stderr = child.communicate(timeout=90)
            self.assertEqual(child.returncode, 91, (child.returncode, stdout[-2000:], stderr[-2000:]))
            self.assertTrue(wait_for(env.drained, timeout=10), env.events())

            boundary = read(control_path)
            mission = Path(boundary["mission_path"])
            core_dir = Path(boundary["core"])
            report = core_dir / "result.json"
            checkpoint = read(core_dir / "state.json")
            self.assertEqual(checkpoint, boundary["checkpoint"])
            self.assertEqual(checkpoint["phase"], "finalized")
            self.assertEqual(checkpoint["sealed_verdict"], "improved")
            self.assertEqual(read(boundary["guard_receipt"]), {"returncode": 0, "error": None})
            if variant == "missing":
                self.assertFalse(report.exists())
            else:
                self.assertEqual(report.read_bytes(), Path(boundary["previous_report"]).read_bytes())
                self.assertEqual(read(report)["phase"], "ready")
            expected = read(boundary["expected_report"])
            champion = next(e for e in checkpoint["archive"] if e["id"] == checkpoint["champion"])
            self.assertEqual(expected["champion"], champion)
            before_calls = env.events()
            self.assertEqual(len(env.builders()), 4)
            self.assertEqual(sum(r["role"] == "reviewer" for r in before_calls), 4)
            sealed_log = Path(boundary["sealed_calls"])
            before_sealed = sealed_log.read_bytes()
            self.assertEqual(len(before_sealed.splitlines()), 6)
            mission_sha = digest(mission)
            frozen = checkpoint["contract"]["evaluation_files"]
            for path, sha in frozen.items():
                self.assertEqual(digest(path), sha, path)

            code, before, raw, err = env.run("status", now=NOW)
            self.assertEqual(code, 0, (raw, err))
            self.assertEqual(before["reserved"], 4, before)
            self.assertEqual(len(before["history"]), 1, before)
            self.assertEqual(before["active_goal"]["mission_path"], str(mission), before)
            self.assertEqual(env.git("rev-parse", "HEAD").strip(), checkpoint["base"])
            self.assertEqual(env.remote_head(), checkpoint["base"])
            print("BOUNDARY " + json.dumps(dict(
                variant=variant, core_phase=checkpoint["phase"], verdict=checkpoint["sealed_verdict"],
                report_phase=None if variant == "missing" else "ready", reserved=before["reserved"],
                provider_calls=len(before_calls), core_sealed_calls=6,
                source_commit=champion["commit"], mission_path=str(mission),
                guard=read(boundary["guard_receipt"]))), flush=True)

            code, after, raw, err = env.run("resume", now=NOW)
            observed = dict(variant=variant, returncode=code,
                            phase=after.get("phase") if isinstance(after, dict) else None,
                            error=after.get("error") if isinstance(after, dict) else err[-1200:],
                            reserved=after.get("reserved") if isinstance(after, dict) else None,
                            history=len(after.get("history", [])) if isinstance(after, dict) else None,
                            provider_calls=len(env.events()),
                            report_phase=read(report).get("phase") if report.exists() else None)
            print("RESUME " + json.dumps(observed), flush=True)
            self.assertIsInstance(after, dict, (raw[-1600:], err[-1600:]))
            self.assertEqual(env.events(), before_calls, observed)
            self.assertEqual((after["slot"], after["reserved"]), (before["slot"], 4), observed)
            self.assertEqual(read(core_dir / "state.json"), checkpoint, "sealed/source checkpoint changed")
            self.assertEqual(sealed_log.read_bytes(), before_sealed, "core sealed evaluator replayed")
            self.assertEqual(digest(mission), mission_sha, "original mission rewritten")
            for path, sha in frozen.items():
                self.assertEqual(digest(path), sha, path)
            self.assertEqual(code, 0, observed)

            self.assertEqual(read(report), expected, "actual finalized core report was not regenerated")
            self.assertEqual(after["phase"], "scheduled", after)
            self.assertIsNone(after["active_goal"], after)
            self.assertEqual(after["queue"], [], after)
            self.assertEqual(len(after["history"]), 2, after)
            completed = next(g for g in after["history"] if g["id"] == GOALS[1]["id"])
            self.assertEqual(completed["mission_path"], str(mission))
            self.assertEqual(completed["outcome"], "published")
            publication = completed["publication"]
            self.assertEqual(publication["source_commit"], champion["commit"])
            self.assertEqual(publication["release_commit"], env.remote_head())
            self.assertEqual(publication["release_commit"], env.git("rev-parse", "HEAD").strip())
            self.assertEqual(env.git("symbolic-ref", "--short", "HEAD").strip(), BRANCH)
            self.assertEqual(env.git("status", "--porcelain", "--untracked-files=all"), "")
            self.assertEqual((env.repo / GOALS[1]["allowed_paths"][0]).read_text(),
                             NEW[GOALS[1]["allowed_paths"][0]])
            self.assertEqual(read(publication["regression_descriptor"]), descriptor)
            for action in ("run", "resume"):
                code, status, raw, err = env.run(action, now=NOW)
                self.assertEqual(code, 0, (raw, err))
                self.assertEqual((status["slot"], status["reserved"]), (before["slot"], 4))
                self.assertEqual(env.events(), before_calls)
                self.assertEqual(read(report), expected)
                self.assertEqual(sealed_log.read_bytes(), before_sealed)

    def test_missing_finalized_report(self):
        self.exercise("missing")

    def test_stale_ready_report(self):
        self.exercise("stale")


if __name__ == "__main__":
    unittest.main(verbosity=2)
