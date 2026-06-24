"""Backend benchmark — the *external anchor* for vendor routing.

Measures each model backend on a small suite of task-class cases with **deterministic
checkers** (exact numbers, executed code, required-element presence — never an
LLM judge), then proposes a ``routing_table`` mapping each task class to the backend
that scored best. The proposal is written to a file for a human to ratify and adopt
via ``--config`` (same discipline as ``tehai calibrate``): routing fires on *measured*
performance, not a model's self-report.

Honesty notes:
- The *writing* class is scored by a coarse deterministic proxy (does it contain the
  required elements / shape), NOT a quality judgement. Treat its score as a floor.
- The *code* class **executes model-generated code** in-process to check it. Only run
  the live bench on inputs you trust (this is opt-in, like ``tehai run --sandbox``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ._json import _extract_json
from .logger import ExecutionLogger, LogRecord
from .models import ModelTier


# --------------------------------------------------------------------------- #
# Case model
# --------------------------------------------------------------------------- #
@dataclass
class BenchCase:
    case_id: str
    task_type: str                       # a TaskType value -> becomes a routing_table key
    prompt: str
    checker: Callable[[str], object]     # returns bool or float in [0,1]


# --------------------------------------------------------------------------- #
# Deterministic checkers
# --------------------------------------------------------------------------- #
def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s


_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\n(.*?)```", re.DOTALL)


def _extract_code(s: str) -> str:
    """Pull runnable code out of a model reply: the longest fenced ``` block if any
    (so reasoning/verbose models that wrap code in prose are scored fairly), else the
    de-fenced text. Without this, a model that writes 'Here is the code:\\n```py...```'
    scores 0 purely on output format, not correctness."""
    blocks = _FENCE_RE.findall(s or "")
    if blocks:
        return max(blocks, key=len)
    return _strip_fences(s)


def _last_int(s: str) -> Optional[int]:
    nums = re.findall(r"-?\d+", (s or "").replace(",", ""))
    return int(nums[-1]) if nums else None


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _check_func(code: str, func_name: str, cases) -> float:
    """Exec model code in a fresh namespace and check a function against cases.

    SECURITY: executes model output. Only used by the opt-in live benchmark.
    """
    ns: dict = {}
    try:
        exec(compile(_extract_code(code), "<bench>", "exec"), ns)  # noqa: S102
    except Exception:
        return 0.0
    fn = ns.get(func_name)
    if not callable(fn):
        return 0.0
    ok = 0
    for args, expected in cases:
        try:
            if fn(*args) == expected:
                ok += 1
        except Exception:
            pass
    return ok / len(cases)


def _chk_palindrome(out: str) -> float:
    return _check_func(out, "is_palindrome", [
        (("A man, a plan, a canal: Panama",), True),
        (("hello",), False),
        (("racecar",), True),
    ])


def _chk_fizzbuzz(out: str) -> float:
    return _check_func(out, "fizzbuzz", [
        ((15,), "FizzBuzz"), ((3,), "Fizz"), ((5,), "Buzz"), ((7,), "7"),
    ])


def _chk_arith(out: str) -> float:
    return 1.0 if _last_int(out) == 396 else 0.0


def _chk_speed(out: str) -> float:
    return 1.0 if _last_int(out) == 80 else 0.0


def _chk_syllogism(out: str) -> float:
    return 1.0 if re.search(r"\byes\b", (out or "").lower()) else 0.0


def _chk_seq(out: str) -> float:
    return 1.0 if _last_int(out) == 42 else 0.0


def _chk_haiku(out: str) -> float:
    o = out or ""
    lines = [ln for ln in o.splitlines() if ln.strip()]
    return (int("rain" in o.lower()) + int(len(lines) >= 3)) / 2.0


def _chk_summary(out: str) -> float:
    o = (out or "").strip()
    sentences = [x for x in o.split(".") if x.strip()]
    one_sentence = o.endswith(".") and len(sentences) == 1
    has_terms = ("cat" in o.lower()) and ("mouse" in o.lower())
    return (int(one_sentence) + int(has_terms)) / 2.0


def _chk_tool_json(out: str) -> float:
    try:
        data = _extract_json(out)
    except Exception:
        return 0.0
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return 0.0
    return (int(str(data.get("name")) == "tehai") + int(data.get("count") == 3)) / 2.0


def _chk_tool_call(out: str) -> float:
    norm = _norm_ws(out).replace("add(8,34)", "add(8, 34)")
    return 1.0 if "add(8, 34)" in norm else 0.0


# --------------------------------------------------------------------------- #
# Default suite — 5 task classes x 2 cases. task_type values are real TaskType
# members so the proposed routing_table keys slot straight into config.
# --------------------------------------------------------------------------- #
DEFAULT_SUITE: list[BenchCase] = [
    BenchCase("code-palindrome", "code_implementation",
              "Write a Python function `is_palindrome(s)` that returns True iff s reads "
              "the same forwards and backwards, ignoring case and non-alphanumeric "
              "characters. Return ONLY the function definition, no prose.", _chk_palindrome),
    BenchCase("code-fizzbuzz", "code_implementation",
              "Write a Python function `fizzbuzz(n)` returning 'Fizz' if n is divisible by "
              "3, 'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, else str(n). "
              "Return ONLY the function definition.", _chk_fizzbuzz),
    BenchCase("math-arith", "qa",
              "Compute 17 * 23 + 5. Reply with ONLY the integer.", _chk_arith),
    BenchCase("math-speed", "qa",
              "A train travels 60 km in 45 minutes. What is its average speed in km/h? "
              "Reply with ONLY the integer.", _chk_speed),
    BenchCase("reason-syllogism", "research",
              "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops necessarily "
              "Lazzies? Answer ONLY 'yes' or 'no'.", _chk_syllogism),
    BenchCase("reason-seq", "research",
              "What number continues this sequence: 2, 6, 12, 20, 30, ? Reply with ONLY "
              "the integer.", _chk_seq),
    BenchCase("write-haiku", "content",
              "Write a haiku about autumn rain. It must mention 'rain' and be exactly "
              "three lines. Output only the haiku.", _chk_haiku),
    BenchCase("write-summary", "content",
              "Summarize the following in ONE sentence ending with a period: 'The cat sat "
              "on the mat, then chased a mouse across the kitchen.' Output only the "
              "sentence.", _chk_summary),
    BenchCase("tool-json", "integration",
              'Return a JSON object with keys "name" (string "tehai") and "count" '
              "(integer 3). Output ONLY the JSON.", _chk_tool_json),
    BenchCase("tool-call", "integration",
              "You can call the function add(a, b). To add 8 and 34, output ONLY the call "
              "exactly as: add(8, 34)", _chk_tool_call),
]


# --------------------------------------------------------------------------- #
# Hard suite — break the ceiling effect of DEFAULT_SUITE (everyone hits 1.0, so
# it can't tell backends apart). Ported from tehai's experiments/hard_bench.py:
# executed algorithms, modular arithmetic, non-obvious sequences, acrostic/format
# constraints, strict nested JSON — cases hard enough to *discriminate* backends,
# so gama's thesis ("a structured combination of small models ties a big one")
# becomes measurable rather than a ceiling. Same deterministic-checker discipline;
# task_type values stay within the 5 real classes so the proposed routing_table
# keys slot straight into a config.
# --------------------------------------------------------------------------- #
def _chk_longest_pal(out: str) -> float:
    ns: dict = {}
    try:
        exec(compile(_extract_code(out), "<bench>", "exec"), ns)  # noqa: S102
    except Exception:
        return 0.0
    fn = ns.get("longest_palindrome")
    if not callable(fn):
        return 0.0
    tests = {"babad": 3, "cbbd": 2, "a": 1, "forgeeksskeegfor": 10, "racecarx": 7}
    ok = 0
    for s, length in tests.items():
        try:
            r = fn(s)
            if isinstance(r, str) and r in s and r == r[::-1] and len(r) == length:
                ok += 1
        except Exception:
            pass
    return ok / len(tests)


def _chk_merge(out: str) -> float:
    return _check_func(out, "merge_intervals", [
        (([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]]),
        (([[1, 4], [4, 5]],), [[1, 5]]),
        (([[1, 4], [0, 4]],), [[0, 4]]),
        (([[1, 4], [2, 3]],), [[1, 4]]),
    ])


def _chk_mult(out: str) -> float:
    return 1.0 if _last_int(out) == 1059 else 0.0          # 37*43 - 28*19


def _chk_modexp(out: str) -> float:
    return 1.0 if _last_int(out) == 9 else 0.0             # 7^100 mod 13


def _chk_weekday(out: str) -> float:
    return 1.0 if re.search(r"\bfriday\b", (out or "").lower()) else 0.0


def _chk_lookandsay(out: str) -> float:
    return 1.0 if _last_int(out) == 312211 else 0.0


def _chk_acrostic(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if len(lines) != 4:
        return 0.0
    firsts = "".join(ln[0].upper() for ln in lines if ln)
    return (int(firsts == "CODE") + int(all("machine" in ln.lower() for ln in lines))) / 2.0


def _chk_primelist(out: str) -> float:
    return 1.0 if _norm_ws(out).replace(" ", "") == "53,59,61,67" else 0.0


def _chk_json_nested(out: str) -> float:
    try:
        d = _extract_json(out)
    except Exception:
        return 0.0
    if isinstance(d, list) and d:
        d = d[0]
    if not isinstance(d, dict):
        return 0.0
    a = d.get("args") or {}
    return 1.0 if (d.get("tool") == "search" and isinstance(a, dict)
                   and a.get("query") == "gama" and a.get("limit") == 5
                   and d.get("tags") == ["a", "b", "c"]) else 0.0


def _chk_json_squares(out: str) -> float:
    try:
        d = _extract_json(out)
    except Exception:
        return 0.0
    return 1.0 if d == [1, 4, 9, 16, 25] else 0.0


HARD_SUITE: list[BenchCase] = [
    BenchCase("hard-code-longpal", "code_implementation",
              "Write a Python function `longest_palindrome(s)` returning the longest "
              "contiguous palindromic substring of s (return any one if there is a tie). "
              "Return ONLY the function definition, no prose.", _chk_longest_pal),
    BenchCase("hard-code-merge", "code_implementation",
              "Write a Python function `merge_intervals(intervals)` that merges all "
              "overlapping intervals and returns them sorted ascending by start, as a list "
              "of [start, end] lists. Return ONLY the function definition, no prose.", _chk_merge),
    BenchCase("hard-math-mult", "qa",
              "Compute 37 * 43 - 28 * 19. Reply with ONLY the integer.", _chk_mult),
    BenchCase("hard-math-modexp", "qa",
              "What is the remainder when 7^100 is divided by 13? Reply with ONLY the "
              "integer.", _chk_modexp),
    BenchCase("hard-reason-weekday", "research",
              "The day before two days after the day before tomorrow is Saturday. What day "
              "is it today? Answer with ONLY the weekday name.", _chk_weekday),
    BenchCase("hard-reason-lookandsay", "research",
              "Give the next term of this sequence: 1, 11, 21, 1211, 111221, ? Reply with "
              "ONLY the integer.", _chk_lookandsay),
    BenchCase("hard-write-acrostic", "content",
              "Write exactly 4 lines. The first letters of the four lines must spell C, O, "
              "D, E in that order. Every line must contain the word 'machine'. Output ONLY "
              "the 4 lines.", _chk_acrostic),
    BenchCase("hard-write-primelist", "content",
              "Output every prime number strictly between 50 and 70 as a comma-separated, "
              "ascending list with no spaces and no other text.", _chk_primelist),
    BenchCase("hard-tool-json-nested", "integration",
              'Output ONLY a JSON object with: "tool"="search"; "args" an object with '
              '"query"="gama" and "limit"=5; "tags"=["a","b","c"]. No prose.', _chk_json_nested),
    BenchCase("hard-tool-json-squares", "integration",
              "Output ONLY a JSON array of the squares of the integers 1 through 5, as "
              "integers.", _chk_json_squares),
]


# --------------------------------------------------------------------------- #
# Brutal suite — a frontier-split probe: even strong models miss some of these.
# Ported from tehai's hard_bench.py BRUTAL_SUITE. Use when `hard` no longer
# discriminates your strongest backends.
# --------------------------------------------------------------------------- #
def _chk_trailzeros(out: str) -> float:
    return 1.0 if _last_int(out) == 24 else 0.0           # trailing zeros in 100!


def _chk_powmod(out: str) -> float:
    return 1.0 if _last_int(out) == 624 else 0.0          # 2^50 mod 1000


def _chk_distinct3(out: str) -> float:
    return 1.0 if _last_int(out) == 648 else 0.0          # 3-digit all-distinct count


def _chk_knights(out: str) -> float:
    toks = re.findall(r"[a-z]+", (out or "").lower())
    return 1.0 if toks and toks[-1] == "knight" else 0.0


def _chk_palindrome_sentence(out: str) -> float:
    t = re.sub(r"[^a-z0-9]", "", (out or "").lower())
    return 1.0 if len(t) >= 11 and t == t[::-1] else 0.0


def _chk_p_alliteration(out: str) -> float:
    words = re.findall(r"[A-Za-z']+", out or "")
    return 1.0 if len(words) == 8 and all(w[0].lower() == "p" for w in words) else 0.0


BRUTAL_SUITE: list[BenchCase] = [
    BenchCase("brutal-qa-trailzeros", "qa",
              "How many trailing zeros does 100! (100 factorial) have? Reply with ONLY the "
              "integer.", _chk_trailzeros),
    BenchCase("brutal-qa-powmod", "qa",
              "Compute 2^50 mod 1000. Reply with ONLY the integer.", _chk_powmod),
    BenchCase("brutal-research-knights", "research",
              "On an island each person is a knight (always tells the truth) or a knave "
              "(always lies). A says 'B is a knave'. B says 'A and I are the same type'. Is "
              "A a knight or a knave? Answer with ONLY the single word: knight or knave.",
              _chk_knights),
    BenchCase("brutal-research-distinct", "research",
              "How many 3-digit numbers (100-999) have all three digits distinct? Reply "
              "with ONLY the integer.", _chk_distinct3),
    BenchCase("brutal-content-palindrome", "content",
              "Write a sentence that is a palindrome (reads identically forwards and "
              "backwards when ignoring case, spaces and punctuation) and is at least 11 "
              "letters long. Output ONLY the sentence.", _chk_palindrome_sentence),
    BenchCase("brutal-content-alliteration", "content",
              "Write a sentence of exactly 8 words where every single word begins with the "
              "letter 'p'. Output ONLY the sentence.", _chk_p_alliteration),
]


# Named suites — `gama bench --suite {default,hard,brutal}`. DEFAULT_SUITE stays
# the default so public behavior is unchanged; hard/brutal break the ceiling.
SUITES: dict[str, list[BenchCase]] = {
    "default": DEFAULT_SUITE,
    "hard": HARD_SUITE,
    "brutal": BRUTAL_SUITE,
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def score_output(case: BenchCase, output: str) -> float:
    """Normalize a checker result to a float in [0, 1]; any error -> 0.0."""
    try:
        r = case.checker(output or "")
    except Exception:
        return 0.0
    if isinstance(r, bool):
        return 1.0 if r else 0.0
    try:
        return max(0.0, min(1.0, float(r)))
    except (TypeError, ValueError):
        return 0.0


def _limit_per_class(suite: list[BenchCase], limit: Optional[int]) -> list[BenchCase]:
    if not limit or limit <= 0:
        return list(suite)
    seen: dict[str, int] = {}
    out: list[BenchCase] = []
    for c in suite:
        if seen.get(c.task_type, 0) < limit:
            out.append(c)
            seen[c.task_type] = seen.get(c.task_type, 0) + 1
    return out


def _run_one(name: str, backend, case: BenchCase, tier: ModelTier, rep: int,
             unit_cost: dict) -> dict:
    error = None
    output = ""
    t0 = time.monotonic()
    try:
        if hasattr(backend, "last_usage"):
            backend.last_usage = None
        # Thread the case's external checker as `verify` so a MeshflowBackend gates its
        # cheap->expensive escalation on the SAME check the bench scores with (honest
        # measurement). Other backends accept **kwargs and ignore it.
        output = backend.complete(case.prompt, tier, task_type=case.task_type,
                                  verify=case.checker)
    except Exception as e:  # never let one backend abort the sweep
        error = f"{type(e).__name__}: {e}"[:200]
    latency = round(time.monotonic() - t0, 4)
    score = 0.0 if error else score_output(case, output)
    usage = getattr(backend, "last_usage", None) or {}
    tokens = usage.get("total_tokens") if usage else None
    uc = unit_cost.get(name, 0.0)
    cost = round((tokens / 1000.0) * uc, 6) if (tokens and uc) else None
    return {
        "backend": name, "task_type": case.task_type, "case_id": case.case_id, "rep": rep,
        "score": round(score, 4), "success": score >= 0.5, "latency_s": latency,
        "tokens": tokens, "cost": cost, "error": error, "output_preview": (output or "")[:200],
    }


def run_bench(backends: dict, suite: Optional[list[BenchCase]] = None,
              tier: ModelTier = ModelTier.LARGE, repeats: int = 1,
              limit_per_class: Optional[int] = None, unit_cost: Optional[dict] = None,
              logger: Optional[ExecutionLogger] = None, run_id: str = "bench") -> list[dict]:
    """Run every (backend x case x repeat) and return rich per-call records.

    If ``logger`` is given, each call is also appended as a LogRecord-compatible row
    so ``tehai evaluate <ledger>`` works on a bench ledger too.
    """
    suite = _limit_per_class(suite if suite is not None else DEFAULT_SUITE, limit_per_class)
    unit_cost = unit_cost or {}
    records: list[dict] = []
    for name, backend in backends.items():
        for case in suite:
            for rep in range(max(1, repeats)):
                rec = _run_one(name, backend, case, tier, rep, unit_cost)
                records.append(rec)
                if logger is not None:
                    logger.log(_bench_logrecord(rec, run_id))
    return records


def _bench_logrecord(rec: dict, run_id: str) -> LogRecord:
    return LogRecord(
        run_id=run_id, task_id=rec["case_id"], task_type=rec["task_type"],
        selected_model=rec["backend"], review_score=rec["score"], actual_cost=rec["cost"],
        elapsed_seconds=rec["latency_s"], judge_decision="accept" if rec["success"] else "revise",
        failure_reason=rec["error"],
    )


# --------------------------------------------------------------------------- #
# Aggregation + proposal
# --------------------------------------------------------------------------- #
def _agg(rs: list[dict]) -> dict:
    n = len(rs)
    costs = [r["cost"] for r in rs if r["cost"] is not None]
    return {
        "n": n,
        "score": round(sum(r["score"] for r in rs) / n, 4),
        "success_rate": round(sum(1 for r in rs if r["success"]) / n, 4),
        "latency_s": round(sum(r["latency_s"] for r in rs) / n, 4),
        "cost": round(sum(costs) / len(costs), 6) if costs else None,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate to per-(class x backend) and per-backend-overall stats."""
    by_class: dict[str, dict[str, list]] = {}
    by_backend: dict[str, list] = {}
    for r in records:
        by_class.setdefault(r["task_type"], {}).setdefault(r["backend"], []).append(r)
        by_backend.setdefault(r["backend"], []).append(r)
    return {
        "by_class": {t: {b: _agg(rs) for b, rs in per.items()} for t, per in by_class.items()},
        "overall": {b: _agg(rs) for b, rs in by_backend.items()},
    }


def propose_routing_table(records: list[dict]) -> dict:
    """Pick the winning backend per class: highest score, then lower latency, then
    lower cost, then name (deterministic). Returns the table plus the full ranking."""
    summary = summarize(records)
    table: dict[str, str] = {}
    ranking: dict[str, list] = {}
    for task_type, per_backend in summary["by_class"].items():
        ranked = sorted(
            per_backend.items(),
            key=lambda kv: (-kv[1]["score"], kv[1]["latency_s"],
                            kv[1]["cost"] if kv[1]["cost"] is not None else 0.0, kv[0]),
        )
        table[task_type] = ranked[0][0]
        ranking[task_type] = [{"backend": b, **agg} for b, agg in ranked]
    return {"routing_table": table, "ranking": ranking, "overall": summary["overall"]}
