"""Bounded external checks and scoring for source-code RSI.

The caller selects the commands and controls evaluation splits. Each repeat is a
fresh process; promotion compares the observed sample ranges conservatively and
provides no statistical confidence guarantee.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .rsi_process import ProcessError, ProcessResult, run_process


class EvaluationError(RuntimeError):
    """A check or evaluation could not produce valid evidence."""


@dataclass(frozen=True)
class Evaluation:
    score: float
    samples: tuple[float, ...]
    results: tuple[dict, ...]


_FEEDBACK_CHARS = 4096


def _excerpt(text: str) -> str:
    if len(text) <= _FEEDBACK_CHARS:
        return text
    marker = "\n...[truncated]...\n"
    head = (_FEEDBACK_CHARS - len(marker)) // 2
    tail = _FEEDBACK_CHARS - len(marker) - head
    return text[:head] + marker + text[-tail:]


def _result_error(message: str, result: ProcessResult) -> EvaluationError:
    return EvaluationError(
        f"{message}\nstderr:\n{_excerpt(result.stderr)}"
        f"\nstdout:\n{_excerpt(result.stdout)}"
    )


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_score(value: object) -> bool:
    return _finite_number(value) and 0 <= value <= 1


def _validate_timeout(timeout: float) -> None:
    if not _finite_number(timeout) or timeout <= 0:
        raise EvaluationError("timeout must be a finite positive number")


def _validate_command(command: list[str]) -> None:
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or "\0" in arg for arg in command)
        or not command[0]
    ):
        raise EvaluationError("command must be a non-empty list of strings without NULs")


def _run(
    command: list[str], *, cwd: Path, timeout: float, cancel, label: str,
) -> ProcessResult:
    try:
        result = run_process(command, cwd=cwd, timeout=timeout, cancel=cancel)
    except (ProcessError, OSError, ValueError) as exc:
        raise EvaluationError(f"{label} failed: {_excerpt(str(exc))}") from exc
    if result.returncode != 0:
        raise _result_error(f"{label} exited with code {result.returncode}", result)
    return result


def run_checks(
    commands: list[list[str]], *, cwd: Path, timeout: float, cancel=None,
) -> list[dict]:
    """Run checks in order, stopping at the first error or nonzero exit.

    Receipts contain command, returncode, stdout, stderr and elapsed_s. Output and
    runtime limits are enforced per command by run_process; no shell is used.
    """
    _validate_timeout(timeout)
    if not isinstance(commands, list):
        raise EvaluationError("commands must be a list of argument lists")
    for command in commands:
        _validate_command(command)

    results = []
    for index, command in enumerate(commands, 1):
        result = _run(
            command, cwd=cwd, timeout=timeout, cancel=cancel, label=f"check {index}",
        )
        results.append({
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed_s": result.elapsed_s,
        })
    return results


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def evaluate(
    command: list[str], *, cwd: Path, timeout: float, repeats: int = 1, cancel=None,
) -> Evaluation:
    """Average fresh command runs, each emitting one JSON object with a score.

    Scores must be finite numbers in [0, 1], excluding booleans. Stdout must be
    JSON only (surrounding whitespace is allowed); duplicate object keys and
    non-standard NaN/Infinity constants are rejected. Any failed repeat raises
    EvaluationError, with no partial Evaluation returned.
    """
    _validate_timeout(timeout)
    _validate_command(command)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise EvaluationError("repeats must be a positive integer")

    samples = []
    results = []
    for index in range(1, repeats + 1):
        label = f"evaluation repeat {index}/{repeats}"
        result = _run(command, cwd=cwd, timeout=timeout, cancel=cancel, label=label)
        try:
            details = json.loads(
                result.stdout, parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (ValueError, RecursionError) as exc:
            raise _result_error(
                f"{label} stdout must be exactly one JSON object: {_excerpt(str(exc))}",
                result,
            ) from exc
        if not isinstance(details, dict):
            raise _result_error(f"{label} stdout must be a JSON object", result)
        if not _valid_score(details.get("score")):
            raise _result_error(
                f"{label} must contain a numeric finite score in [0, 1] (no booleans)",
                result,
            )
        samples.append(float(details["score"]))
        results.append(details)
    return Evaluation(
        score=math.fsum(samples) / len(samples),
        samples=tuple(samples),
        results=tuple(results),
    )


def promotion(
    candidate: Evaluation, incumbent: Evaluation, min_gain: float = 0,
) -> tuple[bool, str]:
    """Require min(candidate.samples) > max(incumbent.samples) + min_gain.

    Invalid min_gain raises EvaluationError. Empty or invalid sample collections
    reject promotion with a reason. This conservative non-overlap rule uses
    observed extrema and provides no statistical confidence guarantee.
    """
    if not _finite_number(min_gain) or min_gain < 0:
        raise EvaluationError("min_gain must be a finite non-negative number")
    for label, samples in (("candidate", candidate.samples), ("incumbent", incumbent.samples)):
        if not isinstance(samples, (tuple, list)) or not samples:
            return False, f"{label} samples must be a non-empty sequence"
        if not all(_valid_score(sample) for sample in samples):
            return False, f"{label} samples must be numeric finite scores in [0, 1]"

    floor = min(candidate.samples)
    ceiling = max(incumbent.samples)
    promote = floor > ceiling + min_gain
    relation = ">" if promote else "<="
    return promote, (
        f"conservative non-overlap rule: candidate minimum {floor!r} {relation} "
        f"incumbent maximum {ceiling!r} + min_gain {min_gain!r}"
    )
