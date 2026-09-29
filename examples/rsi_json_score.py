"""Score the current worktree's JSON extractor on small, public fixtures.

Run from a candidate checkout: python3 -B examples/rsi_json_score.py search
Splits contain distinct inputs, but all fixtures are public in this file. Scores
demonstrate source changes on this local task; they do not measure hidden-task
generalization.

Contract: accept complete JSON values and the first object/array in surrounding
prose or Markdown fences, respecting strings, escapes and nesting. A malformed
first container or text without JSON must raise LLMDecompositionError.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path


_ERROR = object()
_MAX_FAILURES = 8
_SPLITS = ("search", "confirm", "sealed")


def _cases(split: str) -> list[tuple[str, str, object]]:
    """Use distinct text and values for each split, without random sampling."""
    index = _SPLITS.index(split)
    token = ("atlas", "birch", "cedar")[index]
    prefix = ("Model reply: ", "Observation:\n", "Summary follows:\n")[index]
    suffix = (" End of reply.", "\nObservation ends.", "\nSummary ends.")[index]
    values = [
        ("object", {"label": token, "value": index + 2}),
        ("array", [token, index + 2, True, None]),
        ("delimiters", {"text": token + " {literal} [literal]"}),
        ("quoted-brace", {"text": token + ' says "close }" then continue'}),
        ("quoted-bracket", [token + ' says "close ]" then continue']),
        ("backslashes", {"path": token + "\\folder\\nested\\"}),
        ("slash-quote", {"text": token + ' \\"} still inside'}),
        ("nested", {"items": [
            {"text": token + ' says "}]}"', "ok": True},
            [index + 2, {"nothing": None}],
        ]}),
    ]
    cases = [
        ("prose-" + name, prefix + json.dumps(value) + suffix, value)
        for name, value in values
    ]
    for name, value in values[:2]:
        cases.append(("bare-" + name, json.dumps(value), value))
    nested = values[-1][1]
    scalar = (7, None, "cedar scalar")[index]
    cases.extend([
        ("fenced", "```json\n" + json.dumps(nested) + "\n```", nested),
        ("scalar", json.dumps(scalar), scalar),
        ("first-container", prefix + json.dumps(values[0][1]) + "\n"
         + json.dumps(values[1][1]) + suffix, values[0][1]),
        ("empty", ("", "\t\n", " \r\n ")[index], _ERROR),
        ("no-json", prefix + "only plain words" + suffix, _ERROR),
        ("truncated", prefix + json.dumps(nested)[:-1], _ERROR),
        ("trailing-comma", prefix + '{"label": ' + json.dumps(token) + ",}" + suffix, _ERROR),
        ("bad-escape", prefix + '{"label": ' + json.dumps(token)
         + r', "bad": "\q"}' + suffix, _ERROR),
        ("malformed-first", prefix + '{"label": ' + json.dumps(token)
         + ', "bad": } then {"valid": true}' + suffix, _ERROR),
        ("mismatched", prefix + '["' + token + '", {"x": 1]]' + suffix, _ERROR),
    ])
    return cases


def _load_candidate():
    # Load this exact file without importing gama or using an installed package.
    path = Path.cwd() / "gama" / "_json.py"
    spec = importlib.util.spec_from_file_location("_rsi_candidate_json", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate gama/_json.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    error = getattr(module, "LLMDecompositionError", None)
    if not callable(getattr(module, "_extract_json", None)):
        raise TypeError("candidate must define _extract_json")
    if not isinstance(error, type) or not issubclass(error, Exception) \
            or error.__name__ != "LLMDecompositionError":
        raise TypeError("candidate must preserve the LLMDecompositionError exception class")
    return module


def _same(actual, expected) -> bool:
    """Check JSON values without treating True as 1 or False as 0."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(left, right) for left, right in zip(actual, expected))
    return actual == expected


def score(split: str) -> dict:
    cases = _cases(split)
    passed = 0
    failures = []
    # Ordinary candidate debug prints must not enter the scoring protocol or leak
    # case details from confirm/sealed. The coordinator bounds process execution.
    with open(os.devnull, "w", encoding="utf-8") as sink, \
            contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            candidate = _load_candidate()
        except Exception as exc:
            failures.append({"case": "load", "reason": f"{type(exc).__name__}: {exc}"[:240]})
        else:
            for name, text, expected in cases:
                reason = ""
                try:
                    actual = candidate._extract_json(text)
                except Exception as exc:
                    if expected is _ERROR and isinstance(exc, candidate.LLMDecompositionError):
                        passed += 1
                        continue
                    reason = f"raised {type(exc).__name__}: {exc}"[:240]
                else:
                    if expected is not _ERROR and _same(actual, expected):
                        passed += 1
                        continue
                    reason = ("expected LLMDecompositionError" if expected is _ERROR
                              else "returned the wrong JSON value or type")
                if split == "search" and len(failures) < _MAX_FAILURES:
                    failures.append({"case": name, "input": text[:320], "reason": reason})
    result = {"score": passed / len(cases), "passed": passed, "total": len(cases)}
    if split == "search":
        result["failures"] = failures
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=_SPLITS)
    args = parser.parse_args()
    print(json.dumps(score(args.split), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
