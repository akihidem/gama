"""L0 for the case suites themselves: a benchmark's own answers must be checked.

A case whose expected answer is wrong is worse than a missing case — it silently
penalises the models that got it right, and every score built on it (a routing table,
a grown champion, a recipe) inherits the error while looking perfectly measured.

So every case in every suite carries a reference answer here, and this file asserts:

  1. **completeness** — every case has one (a case with no reference is unverified,
     and "we forgot to check that one" is exactly how a bad case survives)
  2. **solvable** — the reference scores 1.0, so a perfect answer really passes
  3. **not trivially satisfiable** — junk and empty replies score < 1.0, so a checker
     that says yes to everything cannot hide behind (2)
  4. **splittable** — ids are unique across suites and classes are real TaskTypes

The reference answers for the code cases are working implementations: they are run by
the same `exec`-and-check path the benchmark uses, so this file also proves the code
checkers accept a genuinely correct solution rather than one particular style of it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.benchmark import SUITES, score_output
from gama.models import TaskType

JUNK = "I do not know the answer."

_PALINDROME = '''
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
'''
_FIZZBUZZ = '''
def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
'''
_LONGPAL = '''
def longest_palindrome(s):
    best = ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            sub = s[i:j + 1]
            if sub == sub[::-1] and len(sub) > len(best):
                best = sub
    return best
'''
_MERGE = '''
def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
'''
_TWOSUM = '''
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
'''
_ANAGRAM = '''
def is_anagram(a, b):
    norm = lambda s: sorted(s.lower().replace(" ", ""))
    return norm(a) == norm(b)
'''
_FLATTEN = '''
def flatten(x):
    out = []
    for item in x:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
'''
_ROMAN = '''
def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and vals[c] < vals[s[i + 1]]:
            total -= vals[c]
        else:
            total += vals[c]
    return total
'''
_BSEARCH = '''
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''
_ROTATE = '''
def rotate(lst, k):
    if not lst:
        return []
    k = k % len(lst)
    return lst[-k:] + lst[:-k] if k else list(lst)
'''
_BALANCED = '''
def is_balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack
'''
_TOPWORD = '''
def most_common_word(text):
    from collections import Counter
    words = [w.lower() for w in text.split()]
    return Counter(words).most_common(1)[0][0]
'''

REFERENCE: dict[str, str] = {
    # --- default ---------------------------------------------------------- #
    "code-palindrome": _PALINDROME,
    "code-fizzbuzz": _FIZZBUZZ,
    "math-arith": "396",
    "math-speed": "80",
    "reason-syllogism": "yes",
    "reason-seq": "42",
    "write-haiku": "Cold autumn rain falls\nonto the quiet rooftops\nnight softens the town",
    "write-summary": "The cat chased a mouse across the kitchen.",
    "tool-json": '{"name": "tehai", "count": 3}',
    "tool-call": "add(8, 34)",
    # --- hard -------------------------------------------------------------- #
    "hard-code-longpal": _LONGPAL,
    "hard-code-merge": _MERGE,
    "hard-math-mult": "1059",
    "hard-math-modexp": "9",
    "hard-reason-weekday": "Friday",
    "hard-reason-lookandsay": "312211",
    "hard-write-acrostic": ("Clever machine drafts\nOptimal machine routes\n"
                            "Deep machine reasoning\nEvery machine agrees"),
    "hard-write-primelist": "53,59,61,67",
    "hard-tool-json-nested": ('{"tool": "search", "args": {"query": "gama", "limit": 5}, '
                              '"tags": ["a", "b", "c"]}'),
    "hard-tool-json-squares": "[1, 4, 9, 16, 25]",
    # --- brutal ------------------------------------------------------------ #
    "brutal-qa-trailzeros": "24",
    "brutal-qa-powmod": "624",
    "brutal-research-knights": "knight",
    "brutal-research-distinct": "648",
    "brutal-content-palindrome": "A man, a plan, a canal, Panama",
    "brutal-content-alliteration": "Peter piled plump purple plums past pretty porches",
    # --- wide: qa ---------------------------------------------------------- #
    "wide-qa-pow2": "1048576",
    "wide-qa-gcd": "21",
    "wide-qa-divisors": "24",
    "wide-qa-sum50": "1275",
    "wide-qa-modpow": "9",
    "wide-qa-binary": "53",
    "wide-qa-seconds": "13500",
    "wide-qa-lcm": "42",
    # --- wide: research ---------------------------------------------------- #
    "wide-research-choose": "56",
    "wide-research-seq": "48",
    "wide-research-digitsum": "7",
    "wide-research-arrangements": "30",
    "wide-research-ages": "36",
    "wide-research-tallest": "Ceci",
    "wide-research-knave": "knave",
    "wide-research-dice": "1/6",
    # --- wide: code -------------------------------------------------------- #
    "wide-code-twosum": _TWOSUM,
    "wide-code-anagram": _ANAGRAM,
    "wide-code-flatten": _FLATTEN,
    "wide-code-roman": _ROMAN,
    "wide-code-bsearch": _BSEARCH,
    "wide-code-rotate": _ROTATE,
    "wide-code-balanced": _BALANCED,
    "wide-code-topword": _TOPWORD,
    # --- wide: content ----------------------------------------------------- #
    "wide-content-sevenwords": "The small toad watched the evening rain.",
    "wide-content-acrostic": ("Green frog waits\nAnother frog hops\nMossy frog blinks\n"
                              "And the frog sings"),
    "wide-content-lipogram": "A quick brown fox will jump across a lazy dog again.",
    "wide-content-weekdays": "Monday,Tuesday,Wednesday,Thursday,Friday",
    "wide-content-bullets": ("- runs on local models\n- keeps local data private\n"
                             "- stays local by default"),
    "wide-content-alliteration": "Seven small snails sang sweetly southward",
    "wide-content-toad": ("A toad sat by the pond, a second toad hopped past, "
                          "and the third toad simply blinked."),
    "wide-content-reverse": "atnubamag",
    # --- wide: integration -------------------------------------------------- #
    "wide-tool-json-user": '{"name": "gama", "tags": ["a", "b"], "active": true}',
    "wide-tool-json-list": '[{"id": 1}, {"id": 2}]',
    "wide-tool-call": "multiply(6, 7)",
    "wide-tool-csv": "gama,toad,3",
    "wide-tool-json-nested": '{"model": {"name": "gemma", "tier": "large"}, "retries": 2}',
    "wide-tool-json-squares": "[36, 49, 64, 81, 100]",
    "wide-tool-json-null": '{"ok": true, "error": null}',
    "wide-tool-kv": "host=localhost port=11434",
}


def _all_cases():
    for suite_name, suite in SUITES.items():
        for case in suite:
            yield suite_name, case


class TestSuiteIntegrity(unittest.TestCase):
    def test_every_case_has_a_reference_answer(self):
        missing = sorted(c.case_id for _, c in _all_cases() if c.case_id not in REFERENCE)
        self.assertEqual(missing, [], "cases with no verified reference answer")

    def test_reference_answers_score_full_marks(self):
        for suite_name, case in _all_cases():
            with self.subTest(suite=suite_name, case=case.case_id):
                self.assertEqual(score_output(case, REFERENCE[case.case_id]), 1.0)

    def test_junk_and_empty_answers_do_not_pass(self):
        # Without this, a checker that returns 1.0 unconditionally would look perfect above.
        for suite_name, case in _all_cases():
            with self.subTest(suite=suite_name, case=case.case_id):
                self.assertLess(score_output(case, JUNK), 1.0)
                self.assertLess(score_output(case, ""), 1.0)

    def test_case_ids_are_unique_across_suites(self):
        # `suite_pool` de-duplicates by case_id, so a collision would silently drop a case
        # from a pooled split rather than fail.
        ids = [c.case_id for _, c in _all_cases()]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        self.assertEqual(dupes, [])

    def test_task_types_are_real_routing_keys(self):
        valid = {t.value for t in TaskType}
        for suite_name, case in _all_cases():
            with self.subTest(suite=suite_name, case=case.case_id):
                self.assertIn(case.task_type, valid)

    def test_wide_suite_is_balanced_enough_to_split_three_ways(self):
        from collections import Counter
        counts = Counter(c.task_type for c in SUITES["wide"])
        self.assertEqual(len(counts), 5, counts)
        # 4 per class is the point at which the 2:1:1 split gives every class a case in
        # search, confirm AND sealed; below that a class stops being confirmable.
        self.assertTrue(all(n >= 4 for n in counts.values()), counts)


if __name__ == "__main__":
    unittest.main()
