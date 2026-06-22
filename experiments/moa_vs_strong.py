"""moa_vs_strong — compare model-combination *systems* (single / tool / ensemble / gama)
on a deterministic suite, THROUGH tehai (build_backend + benchmark checkers). Nothing is
hardcoded: hosts/models/strategies come from a JSON config, so the same harness drives the
"can light + structure match a big model?" study reproducibly.

Usage:
    python3 -m experiments.moa_vs_strong path/to/config.json

config = {
  "suite": "easy" | "hard" | "big",         # which built-in deterministic suite
  "limit_per_class": <int>,                  # optional cap of cases per class
  "systems": { "label": <backend-spec>, ... }  # each spec per tehai.config.build_backend
}
A spec composes real tehai backends, e.g. a sovereign stack:
  {"backend":"gama","kwargs":{"backends":{
      "tool":{"backend":"tool","kwargs":{"inner":{"backend":"ssh-openai","kwargs":{...7B}}}},
      "coder":{"backend":"ssh-openai","kwargs":{...32B}},
      "hetero":{"backend":"ensemble","kwargs":{"members":[...],"aggregator":{...}}}},
    "routing_table":{"qa":"tool","code_implementation":"coder","research":"hetero"},
    "default":"coder"}}
"""
import json
import statistics
import sys

from gama import benchmark as B
from gama.benchmark import BenchCase, _check_func, _last_int
from gama.config import build_backend
from gama.models import ModelTier

T = ModelTier.LARGE


def _eqi(a):
    return lambda o: 1.0 if _last_int(o) == a else 0.0


def _has(sub):
    return lambda o: 1.0 if sub in " ".join((o or "").split()) else 0.0


def _has_ci(sub):
    return lambda o: 1.0 if sub in " ".join((o or "").lower().split()) else 0.0


HARD = [
    BenchCase("m1", "qa", "Compute 47 * 53 + 89 * 17. Reply with ONLY the integer.", _eqi(4004)),
    BenchCase("m2", "qa", "A tank holds 480 liters. It drains at 12 L/min for 9 minutes, then "
              "is filled at 20 L/min for 6 minutes. How many liters now? Reply with ONLY the "
              "integer.", _eqi(492)),
    BenchCase("c1", "code_implementation", "Write a Python function nth_prime(n) returning the "
              "n-th prime (1-indexed; nth_prime(1)==2). Return ONLY the function.",
              lambda o: _check_func(o, "nth_prime", [((1,), 2), ((6,), 13), ((10,), 29)])),
    BenchCase("c2", "code_implementation", "Write a Python function lcs(a, b) returning the "
              "integer LENGTH of the longest common subsequence. Return ONLY the function.",
              lambda o: _check_func(o, "lcs", [(("ABCBDAB", "BDCAB"), 4), (("abc", "abc"), 3),
                                               (("", "x"), 0)])),
    BenchCase("r1", "research", "A drawer has 21 blue, 15 black and 17 red socks. In total "
              "darkness, how many socks must you take to be CERTAIN of a matching pair? Reply "
              "with ONLY the integer.", _eqi(4)),
    BenchCase("r2", "research", "Look-and-say: 1, 11, 21, 1211, 111221, then what? Reply with "
              "ONLY the digits of the next term.", _has("312211")),
]

BIG = HARD + [
    BenchCase("m3", "qa", "What is 2**13 - 3**5? Reply with ONLY the integer.", _eqi(7949)),
    BenchCase("m4", "qa", "A book has 350 pages. You read 40 pages/day for 5 days, then 25 "
              "pages/day for 4 days. How many pages are left? Reply with ONLY the integer.",
              _eqi(50)),
    BenchCase("c3", "code_implementation", "Write a Python function roman(n) converting an "
              "integer 1..3999 to a Roman numeral string. Return ONLY the function.",
              lambda o: _check_func(o, "roman", [((4,), "IV"), ((49,), "XLIX"),
                                                 ((1994,), "MCMXCIV")])),
    BenchCase("c4", "code_implementation", "Write a Python function is_balanced(s) returning "
              "True iff the brackets ()[]{} in s are balanced/properly nested. Return ONLY the "
              "function.", lambda o: _check_func(o, "is_balanced", [(("()[]{}",), True),
              (("([)]",), False), (("",), True), (("(]",), False)])),
    BenchCase("r3", "research", "If 5 machines make 5 widgets in 5 minutes, how many minutes "
              "do 100 machines need to make 100 widgets? Reply with ONLY the integer.", _eqi(5)),
    BenchCase("r4", "research", "If today is Monday, what day of the week is it in 100 days? "
              "Reply with ONLY the day name.", _has_ci("wednesday")),
]


def _easy():
    out, seen = [], set()
    for c in B.DEFAULT_SUITE:
        if c.task_type not in seen:
            out.append(c)
            seen.add(c.task_type)
    return out


def _suite(name):
    return {"easy": _easy(), "hard": HARD, "big": BIG}[name]


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: python3 -m experiments.moa_vs_strong <config.json>\n")
        return 2
    cfg = json.loads(open(argv[1], encoding="utf-8").read())
    suite = _suite(cfg.get("suite", "big"))
    lpc = cfg.get("limit_per_class")
    if lpc:
        seen, capped = {}, []
        for c in suite:
            if seen.get(c.task_type, 0) < lpc:
                capped.append(c)
                seen[c.task_type] = seen.get(c.task_type, 0) + 1
        suite = capped

    systems = {label: build_backend(spec) for label, spec in cfg["systems"].items()}
    rows = {c.case_id: {} for c in suite}
    for label, be in systems.items():
        for c in suite:
            try:
                o = be.complete(c.prompt, T, task_type=c.task_type)
            except Exception as e:
                o = f"[ERR {type(e).__name__}: {str(e)[:80]}]"
            rows[c.case_id][label] = B.score_output(c, o)
        print(f"[done] {label}: " + "  ".join(f"{c.case_id}={rows[c.case_id][label]}"
                                              for c in suite), flush=True)

    names = list(systems)
    w = max(10, max((len(n) for n in names), default=10) + 2)
    print("\n" + "TASK".ljust(12) + "".join(n.rjust(w) for n in names))
    for c in suite:
        print(c.case_id.ljust(12) + "".join(str(rows[c.case_id][n]).rjust(w) for n in names))
    print("AVG".ljust(12) + "".join(
        str(round(statistics.mean(rows[c.case_id][n] for c in suite), 2)).rjust(w)
        for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
