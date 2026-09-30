"""Independent generated validation for a standalone gama/_json.py candidate."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

SEED = 0x4A534F4E
REJECT = object()


def text(rng):
    alphabet = 'ab09 "\\\n\t\r\b\f{}[]/Ω漢🧭e\u0301'
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 33)))


def value(rng, depth):
    if depth and rng.randrange(3):
        if rng.randrange(2):
            return [value(rng, depth - 1) for _ in range(rng.randrange(1, 4))]
        return {text(rng): value(rng, depth - 1) for _ in range(rng.randrange(1, 4))}
    return rng.choice([None, bool(rng.randrange(2)), rng.randrange(-999, 1000),
                       rng.randrange(-10000, 10001) / 16, text(rng)])


def cases():
    rng = random.Random(SEED)
    roots = [
        {"sample": i, "data": value(rng, 3), "text": text(rng)} if i % 2
        else [text(rng), value(rng, 3), {"sample": i}]
        for i in range(24)
    ]
    result = []
    for i, expected in enumerate(roots):
        later = json.dumps({"alternative": i, "discard": True})
        for ascii_only in (False, True):
            body = json.dumps(expected, ensure_ascii=ascii_only, indent=2 if i % 3 == 0 else None)
            wrappers = {
                "raw": body,
                "prose": f"Observation:\n{body}\nEnd of record.",
                "fence": f"```json\n{body}\n```",
                "embedded-fence": f"Notes:\n```json\n{body}\n```\nEnd.",
                "first": f"Selected:\n{body}\nAlternative:\n{later}",
            }
            for name, wrapped in wrappers.items():
                result.append((f"roundtrip/{i:02}/{ascii_only}/{name}", wrapped, expected))
    for i, item in enumerate(roots[:6]):
        obj, array = json.dumps({"broken": item}), json.dumps([item])
        damaged = (obj[:-1] + ",}", obj.replace(":", "", 1), array[:-1] + "}", obj[:-1])
        for j, bad in enumerate(damaged):
            try:
                json.loads(bad)
            except json.JSONDecodeError:
                pass
            else:
                raise AssertionError("negative fixture is valid JSON")
            later = json.dumps({"later_valid": [i, j], "must_not_return": True})
            for name, wrapped in (
                ("prose", f"Notes:\n{bad}\nAlternative:\n{later}"),
                ("fence", f"```json\n{bad}\n```\nAlternative:\n{later}"),
            ):
                result.append((f"reject-first/{i}/{j}/{name}", wrapped, REJECT))
    assert len(result) <= 300
    return result


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False)


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    source = parser.parse_args().source.resolve()
    samples = cases()
    report = {"source": str(source), "seed": SEED, "passed": 0,
              "total": len(samples), "executed": 0, "failures": []}
    try:
        original = source.read_bytes()
        report["source_sha256"] = hashlib.sha256(original).hexdigest()
        spec = importlib.util.spec_from_file_location("_json_validation_target", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        with redirect_stdout(sys.stderr):
            # Compile the supplied bytes directly, avoiding reads of stale .pyc files.
            exec(compile(original, str(source), "exec"), module.__dict__)
            extract, rejection = module._extract_json, module.LLMDecompositionError
            for name, wrapped, expected in samples:
                report["executed"] += 1
                good = False
                try:
                    actual = extract(wrapped)
                    good = expected is not REJECT and canonical(actual) == canonical(expected)
                    observed = repr(actual)
                except rejection as exc:
                    good = expected is REJECT
                    observed = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    observed = f"{type(exc).__name__}: {exc}"
                report["passed"] += int(good)
                if not good and len(report["failures"]) < 8:
                    report["failures"].append({
                        "case": name, "input_preview": wrapped[:120],
                        "expected": "LLMDecompositionError" if expected is REJECT else repr(expected)[:120],
                        "observed": observed[:180],
                    })
        report["source_unchanged"] = source.read_bytes() == original
    except Exception as exc:
        report["failures"].append({"case": "load-or-harness", "error": str(exc)[:180]})
    ok = report["passed"] == report["total"] and report.get("source_unchanged", False)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
