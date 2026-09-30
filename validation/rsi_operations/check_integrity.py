"""Independent byte comparison against the committed pre-change baseline."""
from pathlib import Path
import subprocess
import ops_common as C

fixed = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", C.BASE, "--", "tests"],
    cwd=C.ROOT, text=True).splitlines()
fixed += ["gama/_json.py", "examples/rsi_json_score.py", "examples/rsi.example.json",
          "gama/rsi.py", "gama/rsi_agent.py", "gama/rsi_cli.py",
          "gama/rsi_workspace.py", "gama/rsi_evaluate.py", "gama/rsi_process.py"]
for name in fixed:
    expected = subprocess.check_output(["git", "show", f"{C.BASE}:{name}"], cwd=C.ROOT)
    assert (C.ROOT / name).read_bytes() == expected, f"protected source changed: {name}"
print(f"Verified {len(fixed)} existing test/controller/evaluator files byte-for-byte.")
