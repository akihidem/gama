"""L0 for the source itself: an unconditional ``def``/``class`` repeated in one body is a defect.

Why this file exists (2026-09-02): ``gama/grow.py`` carried two ``_atomic_lane``
definitions. Python raises nothing for that; the later one silently replaced the one
the call sites were written against, and a whole class of candidates (every follow-up
to a deepened lane) stopped being proposed. The unit tests stayed green because the
surviving definition happened to agree with the intended one on the hand-built lanes
they used. No amount of testing behaviour finds a defect whose symptom is *absence*;
the shape of the source has to be checked directly, and a linter is not a dependency
this project has, so the check lives here on ``ast`` alone.

The rule: within one module body, and within one class body, every ``def``/``class``
name appears once. ``@property`` followed by ``@x.setter`` / ``@x.deleter`` is the one
legitimate reason for a repeated name inside a class, so those are counted as their
own kind of definition: a getter, one setter and one deleter of ``x`` are fine, and a
second setter of ``x`` is the same silent replacement this file is here to catch.
What this floor does *not* promise: definitions nested under ``if``/``try``/``with``
(a conditional definition is usually a deliberate fallback, and the two branches of
one ``try`` are not a redefinition), rebinding by assignment or import (``f = ...``),
and function bodies (a redefinition there is local and short-lived). The defect this
was written for is the plain, unconditional second ``def`` that an edit left behind.
"""
import ast
import collections
import pathlib
import tempfile
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "gama"
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _role(node, in_class):
    """``"setter"``/``"deleter"`` for the property idiom, ``""`` for a plain definition.

    The decorator has to name *this* function (`@y.setter` on `def x` is not the idiom)
    and the idiom only exists inside a class body. Giving the accessor its own role,
    rather than skipping it, is what lets a *second* setter still count as a duplicate.
    """
    if in_class:
        for d in getattr(node, "decorator_list", []):
            if (isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter")
                    and isinstance(d.value, ast.Name) and d.value.id == node.name):
                return d.attr
    return ""


def duplicate_definitions(path):
    """Return ``[(scope, name, count)]`` for every name defined more than once in a body.

    ``name`` carries the accessor role where one applies (``"x.setter"``), so the
    report says which of the definitions was repeated.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    def scan(body, scope, in_class):
        counts = collections.Counter()
        for n in body:
            if isinstance(n, DEFS):
                role = _role(n, in_class)
                counts[n.name + ("." + role if role else "")] += 1
        found.extend((scope, name, c) for name, c in counts.items() if c > 1)
        for n in body:
            if isinstance(n, ast.ClassDef):
                scan(n.body, f"{scope}.{n.name}", True)

    scan(tree.body, path.stem, False)
    return found


def stranded_tests(path):
    """Return ``[(line, name)]`` for every ``test_*`` function the runner would never see.

    A method pasted after a module-level function at that function's body indentation
    parses as a nested function of it (unreachable, after its ``return``): no error, no
    test, and the suite count moves by one less than expected, which nobody counts. The
    runner only collects ``test_*`` methods on ``TestCase`` classes, so a ``test_*`` that
    is not a direct child of a class body is a test that does not exist.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    def scan(body, in_class):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if n.name.startswith("test_") and not in_class:
                    found.append((n.lineno, n.name))
                scan(n.body, False)
            elif isinstance(n, ast.ClassDef):
                scan(n.body, True)
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    for m in getattr(n, attr, []) or []:
                        scan(getattr(m, "body", [m]), in_class)

    scan(tree.body, False)
    return found


class TestNoShadowedDefinitions(unittest.TestCase):
    def test_every_module_defines_each_name_once(self):
        modules = sorted(PACKAGE.rglob("*.py"))   # subpackages too, should any appear
        self.assertGreater(len(modules), 5, "the package moved; point PACKAGE at it")
        dups = [(str(p.name), *d) for p in modules for d in duplicate_definitions(p)]
        self.assertEqual(dups, [], f"a later definition silently wins: {dups}")

    def _scan(self, src, name="probe"):
        # A private directory per call: fixed /tmp names collide under parallel runs.
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d) / f"{name}.py"
            tmp.write_text(src, encoding="utf-8")
            return duplicate_definitions(tmp)

    def test_the_check_sees_a_redefinition(self):
        # The floor has to be shown to catch the defect it was written for.
        src = "def f():\n    return 1\n\ndef g():\n    return 2\n\ndef f():\n    return 3\n"
        self.assertEqual(self._scan(src), [("probe", "f", 2)])

    def test_a_redefined_method_is_seen_inside_its_class(self):
        src = "class A:\n    def m(self):\n        return 1\n    def m(self):\n        return 2\n"
        self.assertEqual(self._scan(src), [("probe.A", "m", 2)])

    def test_property_setters_are_not_flagged(self):
        src = ("class A:\n    @property\n    def x(self):\n        return 1\n"
               "    @x.setter\n    def x(self, v):\n        pass\n")
        self.assertEqual(self._scan(src), [])

    def test_a_setter_for_another_name_does_not_excuse_a_redefinition(self):
        src = ("class A:\n    @property\n    def y(self):\n        return 1\n"
               "    def x(self):\n        return 1\n"
               "    @y.setter\n    def x(self, v):\n        pass\n")
        self.assertEqual(self._scan(src), [("probe.A", "x", 2)])

    def test_a_second_setter_is_a_redefinition(self):
        src = ("class A:\n    @property\n    def x(self):\n        return 1\n"
               "    @x.setter\n    def x(self, v):\n        pass\n"
               "    @x.setter\n    def x(self, v):\n        pass\n")
        self.assertEqual(self._scan(src), [("probe.A", "x.setter", 2)])

    def test_the_setter_idiom_does_not_exist_at_module_level(self):
        src = ("class P:\n    pass\n\nx = P()\n\n"
               "def x():\n    return 1\n\n"
               "@x.setter\ndef x(v):\n    pass\n")
        self.assertEqual(self._scan(src), [("probe", "x", 2)])


class TestNoStrandedTests(unittest.TestCase):
    """2026-09-02: a regression test for ``promoted_gain_cases`` was inserted after a
    module-level helper at its body indentation. The file parsed, the suite was green,
    and the test had never run (``-k`` found 0 tests). codex read it as an
    IndentationError; the truth was quieter than that."""

    def test_every_test_function_in_the_suite_is_a_method_of_a_class(self):
        tests = sorted(pathlib.Path(__file__).resolve().parent.glob("test_*.py"))
        self.assertGreater(len(tests), 5, "the suite moved; point this at it")
        stranded = [(p.name, *t) for p in tests for t in stranded_tests(p)]
        self.assertEqual(stranded, [], f"tests the runner never collects: {stranded}")

    def _scan(self, src):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d) / "probe.py"
            tmp.write_text(src, encoding="utf-8")
            return stranded_tests(tmp)

    def test_the_check_sees_a_method_stranded_inside_a_helper(self):
        src = ("def helper():\n    return 1\n\n\n"
               "    def test_x(self):\n        pass\n\n"
               "class A:\n    def test_y(self):\n        pass\n")
        self.assertEqual(self._scan(src), [(5, "test_x")])

    def test_a_module_level_test_function_is_stranded_too(self):
        # unittest does not collect bare functions either
        self.assertEqual(self._scan("def test_x():\n    pass\n"), [(1, "test_x")])

    def test_methods_and_helpers_are_left_alone(self):
        src = ("def _make():\n    def inner():\n        pass\n    return inner\n\n"
               "class A:\n    def test_y(self):\n        def test_local():\n"
               "            pass\n        pass\n")
        # a test_* nested inside a test method is that method's business, not a stranded test
        self.assertEqual(self._scan(src), [(8, "test_local")])


if __name__ == "__main__":
    unittest.main()
