"""Frozen behavioral acceptance: candidate cwd, real Git/core, fake models only."""
from datetime import datetime
import json
from pathlib import Path
import signal
import unittest

from campaign_common import BRANCH, GOALS, NEW, fixture, wait_for


class CampaignAcceptance(unittest.TestCase):
    def call(self, env, action, code=0, *, now=None):
        result = env.run(action, now=now)
        self.assertEqual(result[0], code, result)
        self.assertIsInstance(result[1], dict, result)
        status = result[1]
        for key in ("phase", "active_goal", "queue", "history", "slot",
                    "reserved", "next_run", "ownership", "error"):
            self.assertIn(key, status, result)
        self.assertNotEqual(status["phase"], "saturated", status)
        self.assertIs(type(status["reserved"]), int, status)
        self.assertGreaterEqual(status["reserved"], 0)
        self.assertLessEqual(status["reserved"], 4)
        return status

    def unchanged(self, env, action, code=0, *, now=None):
        before = env.events(), env.git("rev-parse", "HEAD"), env.remote_head()
        status = self.call(env, action, code, now=now)
        self.assertEqual(before, (env.events(), env.git("rev-parse", "HEAD"),
                                  env.remote_head()))
        return status

    def wait_builders(self, env, process):
        self.assertTrue(wait_for(lambda: len(env.builders()) >= 2 or process.poll() is not None,
                                 timeout=45))
        self.assertEqual(len(env.builders()), 2, (process.poll(), env.events()))

    def assert_published(self, env, goals):
        self.assertEqual(env.git("symbolic-ref", "--short", "HEAD").strip(), BRANCH)
        self.assertEqual(env.git("status", "--porcelain", "--untracked-files=all"), "")
        head = env.git("rev-parse", "HEAD").strip()
        self.assertNotEqual(head, env.base)
        self.assertEqual(head, env.remote_head())
        changed = env.git("diff", "--name-only", env.base, head).splitlines()
        permitted = {path for item in goals for path in item["allowed_paths"]}
        prefixes = tuple("validation/continual/" + item["id"] + "/" for item in goals)
        self.assertTrue(all(p in permitted or p.startswith(prefixes) for p in changed), changed)
        for item in goals:
            for path in item["allowed_paths"]:
                self.assertEqual((env.repo / path).read_text(), NEW[path])
            directory = env.repo / "validation/continual" / item["id"]
            files = [p for p in directory.rglob("*") if p.is_file()]
            self.assertTrue(files, str(directory))
            texts = [p.read_text() for p in files]
            for source in item["tests"].values():
                self.assertIn(source, texts)
            descriptors = []
            for text in texts:
                try:
                    descriptors.append(json.loads(text))
                except ValueError:
                    pass
            self.assertIn(item, descriptors, texts)

    def test_two_goals_real_core_publish_and_same_slot_noop(self):
        with fixture() as env:
            status = self.call(env, "run")
            self.assertEqual(status["phase"], "scheduled", status)
            self.assertEqual(status["reserved"], 4, status)
            self.assertEqual(status["queue"], [], status)
            self.assertIsNone(status["active_goal"], status)
            self.assertEqual(len(status["history"]), 2, status)
            for item in GOALS:
                self.assertIn(item["id"], json.dumps(status["history"]))
            self.assert_published(env, GOALS)
            self.assertEqual(len(env.builders()), 4)
            self.assertEqual(sum(r["role"] == "reviewer" for r in env.events()), 4)
            self.assertFalse(any(r["discovery"] for r in env.events()))
            next_run = datetime.fromisoformat(status["next_run"])
            self.assertIsNotNone(next_run.utcoffset())
            self.assertGreater(next_run, datetime.now(next_run.tzinfo))
            for action in ("run", "resume", "run", "status"):
                later = self.unchanged(env, action)
                self.assertEqual((later["slot"], later["reserved"]), (status["slot"], 4))
                self.assertEqual(later["phase"], "scheduled")

    def test_pending_retry_spends_fresh_pair_across_goals(self):
        with fixture() as env:
            env.mode.write_text("hang")
            process = env.start("run")
            self.wait_builders(env, process)
            status = self.call(env, "status")
            self.assertEqual(status["reserved"], 2, status)
            process.send_signal(signal.SIGKILL)
            process.communicate(timeout=10)
            self.assertTrue(wait_for(env.drained, timeout=10), env.events())
            env.mode.write_text("fix")
            resumed = self.call(env, "resume")
            self.assertEqual((resumed["slot"], resumed["reserved"]), (status["slot"], 4))
            self.assertEqual(len(env.builders()), 4, env.events())
            self.assert_published(env, GOALS[:1])
            # The second goal cannot borrow a new pair after retrying the first.
            self.assertEqual((env.repo / GOALS[1]["allowed_paths"][0]).read_text(),
                             "def canonical(text):\n    return text.strip()\n")
            self.assertEqual(len(resumed["history"]), 1, resumed)
            for action in ("run", "resume"):
                self.assertEqual(self.unchanged(env, action)["reserved"], 4)

    def test_stop_persists_overlap_is_free_and_resume_keeps_quota(self):
        with fixture() as env:
            env.mode.write_text("hang")
            process = env.start("run")
            self.wait_builders(env, process)
            status = self.call(env, "status")
            self.assertEqual((status["phase"], status["ownership"]), ("active", "live"))
            self.assertEqual(self.unchanged(env, "run", 3)["reserved"], 2)
            self.call(env, "stop")
            process.communicate(timeout=12)
            self.assertTrue(wait_for(env.drained, timeout=10), env.events())
            for _ in range(2):
                stopped = self.unchanged(env, "run", 4)
                self.assertEqual((stopped["phase"], stopped["reserved"]), ("stopped", 2))
            env.mode.write_text("fix")
            resumed = self.call(env, "resume")
            self.assertEqual(resumed["slot"], status["slot"])
            self.assertLessEqual(resumed["reserved"], 4)
            self.assertLessEqual(len(env.builders()), 4, env.events())

    def test_dirty_checkout_is_preserved_without_inference(self):
        with fixture() as env:
            source = env.repo / GOALS[0]["allowed_paths"][0]
            source.write_text(source.read_text() + "# caller edit\n")
            untracked = env.repo / "operator-notes.txt"
            untracked.write_text("keep my work\n")
            before = source.read_bytes(), untracked.read_bytes(), env.git("diff")
            self.unchanged(env, "run", 4)
            self.assertFalse(env.events())
            self.assertEqual(before, (source.read_bytes(), untracked.read_bytes(),
                                      env.git("diff")))

    def test_invalid_discovery_is_charged_then_same_slot_stays_closed(self):
        with fixture(empty=True) as env:
            status = self.call(env, "run")
            self.assertEqual(status["phase"], "scheduled", status)
            self.assertEqual(status["reserved"], 2, status)
            self.assertEqual(len(env.builders()), 2)
            self.assertTrue(all(r["discovery"] for r in env.builders()))
            self.assertEqual(env.git("rev-parse", "HEAD").strip(), env.base)
            for action in ("run", "resume"):
                self.assertEqual(self.unchanged(env, action)["reserved"], 2)

    def test_empty_queue_advances_cadence_without_becoming_terminal(self):
        with fixture(empty=True) as env:
            first = self.call(env, "run", now="2031-04-12T08:59:00+09:00")
            self.assertEqual((first["phase"], first["reserved"]), ("scheduled", 2))
            self.assertEqual(len(env.builders()), 2)
            self.assertEqual(datetime.fromisoformat(first["next_run"]),
                             datetime.fromisoformat("2031-04-12T09:00:00+09:00"))
            same = self.unchanged(env, "run", now="2031-04-12T08:59:59+09:00")
            self.assertEqual(same["slot"], first["slot"])
            second = self.call(env, "run", now="2031-04-12T09:00:00+09:00")
            self.assertNotEqual(second["slot"], first["slot"])
            self.assertEqual((second["phase"], second["reserved"]), ("scheduled", 2))
            self.assertEqual(len(env.builders()), 4)
            self.assertEqual(datetime.fromisoformat(second["next_run"]),
                             datetime.fromisoformat("2031-04-12T21:00:00+09:00"))
            self.unchanged(env, "run", now="2031-04-12T20:59:59+09:00")
            third = self.call(env, "run", now="2031-04-12T21:00:00+09:00")
            self.assertNotEqual(third["slot"], second["slot"])
            self.assertEqual(third["reserved"], 2)
            self.assertEqual(len(env.builders()), 6)
            self.assertEqual(env.git("rev-parse", "HEAD").strip(), env.base)
            self.assertEqual(env.remote_head(), env.base)

    def test_corrupt_persistent_state_does_not_reset_spending(self):
        with fixture(empty=True) as env:
            status = self.call(env, "run")
            self.assertEqual(status["reserved"], 2)
            # Corrupt persistent JSON records, retaining locks and all evidence.
            # No product-specific checkpoint filename or private schema is assumed.
            records = sorted(env.state.glob("*.json"))
            self.assertTrue(records, "campaign must have durable state after dispatch")
            for path in records:
                path.write_bytes(b'{"interrupted":')
            before = env.events(), env.git("rev-parse", "HEAD"), env.remote_head()
            for action in ("run", "resume"):
                result = env.run(action)
                self.assertIn(result[0], (2, 4, 5), result)
                self.assertEqual(before, (env.events(), env.git("rev-parse", "HEAD"),
                                          env.remote_head()))
                self.assertTrue(all(p.read_bytes() == b'{"interrupted":' for p in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
