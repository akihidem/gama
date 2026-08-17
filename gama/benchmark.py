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


# --------------------------------------------------------------------------- #
# Wide suite — same difficulty band as `hard`, but *broad*: 8 cases in each of the
# 5 classes. Difficulty was never the only thing missing. With 26 cases total the
# discriminating suites could rank two backends, but they could not carry a
# three-way split: `gama grow` ends up deciding promotions on 5 confirm cases,
# where one case is 0.2 of the score and a real 3-point improvement is invisible.
# Breadth is what buys *resolution*, so this suite exists to be split, not to be
# harder. The existing three suites are deliberately left untouched: the numbers in
# README / recipes were measured on them, and silently changing a suite would make
# every published score incomparable to the next one.
#
# Answers here were computed (not recalled) before being written down, and
# tests/test_suite_integrity.py re-derives every one of them from a reference
# solution — a case with a wrong expected answer silently penalises correct models,
# which is the one failure a benchmark must not have.
# --------------------------------------------------------------------------- #
def _eq_int(n: int) -> Callable[[str], float]:
    """Last integer in the reply equals n (the convention every suite here uses)."""
    return lambda out: 1.0 if _last_int(out) == n else 0.0


def _eq_norm(expected: str) -> Callable[[str], float]:
    """Reply equals expected after whitespace-normalising and lower-casing.

    Case and surrounding whitespace are free; *internal* spacing is not, so a prompt
    that demands "no spaces" is still scored on that.
    """
    want = _norm_ws(expected).lower()
    return lambda out: 1.0 if _norm_ws(out).lower() == want else 0.0


def _eq_nospace(expected: str) -> Callable[[str], float]:
    """Reply equals expected once ALL whitespace is deleted.

    For call-syntax answers, where `f(6,7)` and `f(6, 7)` are the same answer: the existing
    `tool-call` case makes the same allowance, and scoring the space would measure
    formatting pedantry rather than whether the model produced the call.
    """
    want = "".join(expected.split()).lower()
    return lambda out: 1.0 if "".join((out or "").split()).lower() == want else 0.0


def _eq_exact(expected: str) -> Callable[[str], float]:
    """Reply equals expected after stripping the ends only (inner whitespace is scored).

    For the machine-consumed formats: a prompt that says "one line, single space" is not
    satisfied by two lines, and a checker looser than its own prompt measures nothing in
    particular. Case is still free.
    """
    want = expected.strip().lower()
    return lambda out: 1.0 if (out or "").strip().lower() == want else 0.0


def _last_word(word: str) -> Callable[[str], float]:
    """The last alphabetic token equals word (one-word answers wrapped in prose)."""
    want = word.lower()

    def chk(out: str) -> float:
        toks = re.findall(r"[a-z]+", (out or "").lower())
        return 1.0 if toks and toks[-1] == want else 0.0
    return chk


def _func(name: str, cases) -> Callable[[str], float]:
    """Execute the reply as Python and check `name` against (args, expected) pairs."""
    return lambda out: _check_func(out, name, cases)


def _json_eq(expected) -> Callable[[str], float]:
    """The reply's JSON payload equals expected exactly.

    Deliberately no list-unwrapping (unlike ``_chk_tool_json``): some cases here ask for a
    JSON *array*, so unwrapping a single-element list would make "[{...}]" and "{...}"
    indistinguishable and silently pass the wrong shape.
    """
    def chk(out: str) -> float:
        try:
            d = _extract_json(out)
        except Exception:
            return 0.0
        return 1.0 if d == expected else 0.0
    return chk


def _chk_seven_words(out: str) -> float:
    o = (out or "").strip()
    words = re.findall(r"[A-Za-z']+", o)
    return 1.0 if (len(words) == 7 and o.endswith(".")) else 0.0


def _chk_acrostic_gama(out: str) -> float:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    if len(lines) != 4:
        return 0.0
    firsts = "".join(ln[0].upper() for ln in lines)
    return 1.0 if (firsts == "GAMA" and all("frog" in ln.lower() for ln in lines)) else 0.0


def _chk_lipogram(out: str) -> float:
    o = (out or "").strip()
    letters = re.findall(r"[A-Za-z]", o)
    return 1.0 if (len(letters) >= 25 and "e" not in o.lower()) else 0.0


def _chk_three_bullets(out: str) -> float:
    lines = [ln.rstrip() for ln in (out or "").splitlines() if ln.strip()]
    if len(lines) != 3:
        return 0.0
    return 1.0 if all(ln.startswith("- ") and "local" in ln.lower() for ln in lines) else 0.0


def _chk_s_alliteration(out: str) -> float:
    words = re.findall(r"[A-Za-z']+", out or "")
    return 1.0 if (len(words) == 6 and all(w[0].lower() == "s" for w in words)) else 0.0


def _chk_toad_thrice(out: str) -> float:
    return 1.0 if len(re.findall(r"\btoad\b", (out or "").lower())) == 3 else 0.0


WIDE_SUITE: list[BenchCase] = [
    # --- qa: one exact integer, no prose ---------------------------------- #
    BenchCase("wide-qa-pow2", "qa",
              "What is 2 to the power of 20? Reply with ONLY the integer.", _eq_int(1048576)),
    BenchCase("wide-qa-gcd", "qa",
              "What is the greatest common divisor of 1071 and 462? Reply with ONLY the "
              "integer.", _eq_int(21)),
    BenchCase("wide-qa-divisors", "qa",
              "How many positive divisors does 360 have? Reply with ONLY the integer.",
              _eq_int(24)),
    BenchCase("wide-qa-sum50", "qa",
              "What is the sum of all integers from 1 to 50 inclusive? Reply with ONLY the "
              "integer.", _eq_int(1275)),
    BenchCase("wide-qa-modpow", "qa",
              "What is 3^7 mod 11? Reply with ONLY the integer.", _eq_int(9)),
    BenchCase("wide-qa-binary", "qa",
              "Convert the binary number 110101 to decimal. Reply with ONLY the integer.",
              _eq_int(53)),
    BenchCase("wide-qa-seconds", "qa",
              "How many seconds are there in 3 hours and 45 minutes? Reply with ONLY the "
              "integer.", _eq_int(13500)),
    BenchCase("wide-qa-lcm", "qa",
              "What is the least common multiple of 14 and 21? Reply with ONLY the integer.",
              _eq_int(42)),

    # --- research: multi-step reasoning, single token answer --------------- #
    BenchCase("wide-research-choose", "research",
              "In how many ways can you choose 3 items from 8 distinct items when order does "
              "not matter? Reply with ONLY the integer.", _eq_int(56)),
    BenchCase("wide-research-seq", "research",
              "What number continues this sequence: 3, 8, 15, 24, 35, ? Reply with ONLY the "
              "integer.", _eq_int(48)),
    BenchCase("wide-research-digitsum", "research",
              "What is the sum of the digits of 2^10? Reply with ONLY the integer.", _eq_int(7)),
    BenchCase("wide-research-arrangements", "research",
              "How many distinct arrangements are there of the letters in the word 'level'? "
              "Reply with ONLY the integer.", _eq_int(30)),
    BenchCase("wide-research-ages", "research",
              "Mika is three times as old as her son. In 12 years she will be twice as old as "
              "he is then. How old is Mika now? Reply with ONLY the integer.", _eq_int(36)),
    BenchCase("wide-research-tallest", "research",
              "Ann is taller than Ben. Ceci is shorter than Ben. Dan is taller than Ann. Who "
              "is the shortest? Answer with ONLY the name.", _last_word("ceci")),
    BenchCase("wide-research-knave", "research",
              "On an island each person is a knight (always truthful) or a knave (always "
              "lying). A says 'B is a knight'. B says 'A and I are of different types'. What "
              "is B? Answer with ONLY the single word: knight or knave.", _last_word("knave")),
    BenchCase("wide-research-dice", "research",
              "Two fair six-sided dice are rolled. What is the probability that the sum is 7? "
              "Answer with ONLY a reduced fraction like a/b.", _eq_norm("1/6")),

    # --- code_implementation: the reply is executed ------------------------ #
    BenchCase("wide-code-twosum", "code_implementation",
              "Write a Python function `two_sum(nums, target)` returning the indices of the "
              "two numbers that add up to target, as a list [i, j] with i < j. Exactly one "
              "solution exists. Return ONLY the function definition.",
              _func("two_sum", [(([2, 7, 11, 15], 9), [0, 1]), (([3, 2, 4], 6), [1, 2]),
                                (([3, 3], 6), [0, 1])])),
    BenchCase("wide-code-anagram", "code_implementation",
              "Write a Python function `is_anagram(a, b)` returning True iff a and b are "
              "anagrams, ignoring case and spaces. Return ONLY the function definition.",
              _func("is_anagram", [(("Listen", "Silent"), True), (("hello", "world"), False),
                                   (("Dormitory", "dirty room"), True)])),
    BenchCase("wide-code-flatten", "code_implementation",
              "Write a Python function `flatten(x)` that flattens an arbitrarily nested list "
              "of integers into a flat list, preserving order. Return ONLY the function "
              "definition.",
              _func("flatten", [(([[1, [2, 3]], 4],), [1, 2, 3, 4]), (([],), []),
                                (([1, [2, [3, [4]]]],), [1, 2, 3, 4])])),
    BenchCase("wide-code-roman", "code_implementation",
              "Write a Python function `roman_to_int(s)` converting a Roman numeral string to "
              "an integer. Return ONLY the function definition.",
              _func("roman_to_int", [(("MCMXCIV",), 1994), (("LVIII",), 58), (("IX",), 9),
                                     (("MMXXVI",), 2026)])),
    BenchCase("wide-code-bsearch", "code_implementation",
              "Write a Python function `binary_search(arr, target)` returning the index of "
              "target in the ascending list arr, or -1 if absent. Return ONLY the function "
              "definition.",
              _func("binary_search", [(([1, 3, 5, 7, 9, 11], 7), 3), (([1, 3, 5], 4), -1),
                                      (([2], 2), 0), (([], 1), -1)])),
    BenchCase("wide-code-rotate", "code_implementation",
              "Write a Python function `rotate(lst, k)` returning lst rotated k positions to "
              "the right (k may exceed len(lst)). Return ONLY the function definition.",
              _func("rotate", [(([1, 2, 3, 4, 5], 2), [4, 5, 1, 2, 3]),
                               (([1, 2, 3], 3), [1, 2, 3]), (([1, 2, 3], 4), [3, 1, 2])])),
    BenchCase("wide-code-balanced", "code_implementation",
              "Write a Python function `is_balanced(s)` returning True iff the brackets in s "
              "(only ()[]{}) are correctly balanced and nested. Return ONLY the function "
              "definition.",
              _func("is_balanced", [(("([]{})",), True), (("(]",), False), (("(((",), False),
                                    (("",), True)])),
    BenchCase("wide-code-topword", "code_implementation",
              "Write a Python function `most_common_word(text)` returning the most frequent "
              "whitespace-separated word in text (lowercased). Return ONLY the function "
              "definition.",
              _func("most_common_word", [(("the cat the dog the bird",), "the"),
                                         (("a b b",), "b")])),

    # --- content: format constraints, checked structurally ----------------- #
    BenchCase("wide-content-sevenwords", "content",
              "Write one sentence of exactly 7 words ending with a period. Output ONLY the "
              "sentence.", _chk_seven_words),
    BenchCase("wide-content-acrostic", "content",
              "Write exactly 4 lines whose first letters spell G, A, M, A in that order. "
              "Every line must contain the word 'frog'. Output ONLY the 4 lines.",
              _chk_acrostic_gama),
    BenchCase("wide-content-lipogram", "content",
              "Write a sentence of at least 25 letters that does not contain the letter 'e' "
              "at all. Output ONLY the sentence.", _chk_lipogram),
    BenchCase("wide-content-weekdays", "content",
              "Output the five weekday names from Monday to Friday as a comma-separated list "
              "with no spaces and no other text.",
              _eq_norm("Monday,Tuesday,Wednesday,Thursday,Friday")),
    BenchCase("wide-content-bullets", "content",
              "Output exactly 3 lines. Every line must start with '- ' and contain the word "
              "'local'. Output ONLY those lines.", _chk_three_bullets),
    BenchCase("wide-content-alliteration", "content",
              "Write a sentence of exactly 6 words where every word begins with the letter "
              "'s'. Output ONLY the sentence.", _chk_s_alliteration),
    BenchCase("wide-content-toad", "content",
              "Write a short paragraph that contains the word 'toad' exactly three times. "
              "Output ONLY the paragraph.", _chk_toad_thrice),
    BenchCase("wide-content-reverse", "content",
              "Write the word 'gamabunta' backwards. Output ONLY that reversed word.",
              _eq_norm("atnubamag")),

    # --- integration: structured output another program must consume ------- #
    BenchCase("wide-tool-json-user", "integration",
              'Output ONLY a JSON object with: "name"="gama"; "tags"=["a","b"]; "active"=true.',
              _json_eq({"name": "gama", "tags": ["a", "b"], "active": True})),
    BenchCase("wide-tool-json-list", "integration",
              'Output ONLY a JSON array of two objects: the first {"id":1}, the second '
              '{"id":2}.', _json_eq([{"id": 1}, {"id": 2}])),
    BenchCase("wide-tool-call", "integration",
              "You can call the function multiply(a, b). To multiply 6 and 7, output ONLY the "
              "call exactly as: multiply(6, 7)", _eq_nospace("multiply(6, 7)")),
    BenchCase("wide-tool-csv", "integration",
              "Output ONLY one CSV line with three fields in this order: gama, toad, 3 — "
              "comma-separated, no spaces, no header, no other text.", _eq_exact("gama,toad,3")),
    BenchCase("wide-tool-json-nested", "integration",
              'Output ONLY a JSON object with "model" an object of "name"="gemma" and '
              '"tier"="large", plus "retries"=2.',
              _json_eq({"model": {"name": "gemma", "tier": "large"}, "retries": 2})),
    BenchCase("wide-tool-json-squares", "integration",
              "Output ONLY a JSON array of the squares of the integers 6 through 10, as "
              "integers.", _json_eq([36, 49, 64, 81, 100])),
    BenchCase("wide-tool-json-null", "integration",
              'Output ONLY a JSON object with "ok"=true and "error"=null.',
              _json_eq({"ok": True, "error": None})),
    BenchCase("wide-tool-kv", "integration",
              "Output ONLY the two key=value pairs separated by a single space, in this "
              "order: host=localhost port=11434", _eq_exact("host=localhost port=11434")),
]


# Named suites — `gama bench --suite {default,hard,brutal,wide}`. DEFAULT_SUITE stays
# the default so public behavior is unchanged; hard/brutal break the ceiling, and
# wide adds the breadth a three-way split (`gama grow`) needs.
SUITES: dict[str, list[BenchCase]] = {
    "default": DEFAULT_SUITE,
    "hard": HARD_SUITE,
    "brutal": BRUTAL_SUITE,
    "wide": WIDE_SUITE,
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
