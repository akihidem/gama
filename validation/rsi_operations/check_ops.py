import json
from pathlib import Path
import ops_common as C


def section(text, wanted):
    current = None
    result = {}
    for line in text.replace("\\\n", "").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            current = line.strip("[]")
        elif current == wanted and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            result.setdefault(key, []).append(value)
    return result


root = C.ROOT
repo = "/home/akhd/work/gama-rsi"
py = repo + "/.venv/bin/python"
rsi = json.loads((root / "examples/rsi_aws.json").read_text())
for command in rsi["checks"] + [rsi[k] for k in
                               ("search_command", "confirm_command", "sealed_command")]:
    assert command[0] == py, command
assert rsi["allowed_paths"] == ["gama/_json.py"]
assert rsi["workers"] == rsi["batch_size"] == 2
mission = json.loads((root / "examples/rsi_aws_mission.json").read_text())
assert mission["repo"] == repo
assert mission["rsi_config"] == repo + "/examples/rsi_aws.json"
assert mission["bridge_config"] == repo + "/examples/rsi_bridge_aws.json"
bridge = json.loads((root / "examples/rsi_bridge_aws.json").read_text())
assert bridge["astra_loop_root"] == "/home/akhd/astra-loop"
assert bridge["backend"]["backend_python"] == "/usr/bin/python3"
timer = section((root / "deploy/gama-rsi.timer").read_text(), "Timer")
assert set(timer["OnCalendar"]) == {
    "*-*-* 09:00:00 Asia/Tokyo", "*-*-* 21:00:00 Asia/Tokyo"}
assert not any(k.startswith("On") and k != "OnCalendar" for k in timer)
assert timer["Persistent"] == ["false"]
assert timer.get("Unit", ["gama-rsi.service"]) == ["gama-rsi.service"]
unit = section((root / "deploy/gama-rsi.service").read_text(), "Service")
assert unit["Type"] == ["oneshot"] and unit["KillMode"] == ["control-group"]
assert unit.get("Restart", ["no"]) == ["no"]
assert py in unit["ExecStart"][0] and repo + "/examples/rsi_aws_mission.json" in unit["ExecStart"][0]
doc = (root / "docs/rsi_operations.md").read_text()
for command in ("run", "status", "stop", "resume", "systemctl"):
    assert command in doc, command
print("AWS examples, user unit, schedule, and command documentation agree.")
