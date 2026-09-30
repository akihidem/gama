import json
import tempfile
import unittest
from pathlib import Path
import ops_common as C
from gama.rsi_runtime import load_inputs, load_bridge_config


class Inputs(unittest.TestCase):
    def test_binding_and_frozen_digest(self):
        with tempfile.TemporaryDirectory() as d:
            e = C.Env(Path(d), stub_bridge=True)
            first = load_inputs(e.mission)
            self.assertEqual(first["digest"], load_inputs(e.mission)["digest"])
            cfg = first["rsi_config"]
            self.assertEqual((cfg["workers"], cfg["batch_size"]), (2, 2))
            self.assertEqual([a["name"] for a in cfg["agents"]], ["astra-a", "astra-b"])
            for a in cfg["agents"]:
                self.assertIn("--config", a["command"])
                self.assertIn(str(e.bridge), a["command"])
                self.assertTrue(any(x.endswith("/gama/rsi_bridge.py") for x in a["command"]))
            self.assertEqual(load_bridge_config(e.bridge), first["bridge_config"])
            for path, key, value in ((e.mission, "search_ceiling", .9),
                                     (e.rsi, "goal", "changed"),
                                     (e.bridge, "timeout", 19)):
                old = path.read_text()
                data = json.loads(old); data[key] = value
                path.write_text(json.dumps(data))
                self.assertNotEqual(first["digest"], load_inputs(e.mission)["digest"])
                path.write_text(old)

    def test_hard_limits_and_controller_protection(self):
        with tempfile.TemporaryDirectory() as d:
            e = C.Env(Path(d), stub_bridge=True)
            for path, key, value in (
                (e.rsi, "workers", 3), (e.rsi, "batch_size", 3),
                (e.rsi, "allowed_paths", ["gama/rsi_mission.py"]),
                (e.mission, "rounds_per_cycle", 3),
                (e.mission, "confirm_ceiling", float("nan")),
                (e.mission, "state_dir", str(e.repo / "state")),
                (e.bridge, "artifact_root", str(e.repo / "artifacts"))):
                old = path.read_text(); data = json.loads(old); data[key] = value
                path.write_text(json.dumps(data))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_inputs(e.mission)
                path.write_text(old)


if __name__ == "__main__":
    unittest.main()
