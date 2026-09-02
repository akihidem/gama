"""L0 for the source itself: a name defined twice in one scope is a defect, not a style nit.

Why this file exists (2026-09-02): ``gama/grow.py`` carried two ``_atomic_lane``
definitions. Python raises nothing for that; the later one silently replaced the one
the call sites were written against, and a whole class of candidates (every follow-up
to a deepened lane) stopped being proposed. The unit tests stayed green because the
surviving definition happened to agree with the intended one on the hand-built lanes
they used. No amount of testing behaviour finds a defect whose symptom is *absence*;
the shape of the source has to be checked directly, and a linter is not a dependency
this project has, so the check lives here on ``ast`` alone.

The rule: within one module body, and within one class body, every ``def``/``class``
name appears once. Decorated redefinitions (``@property`` then ``@x.setter``) are the
one legitimate reason for a repeated name inside a class, and are allowed as such.
"""
import ast
import collections
import pathlib
import tempfile
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "gama"
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_setter_pair(node):
    # `@x.setter` / `@x.deleter` on `def x` re-use the property's name by design. The
    # decorator has to name *this* function: `@y.setter` on `def x` is not that idiom,
    # and letting any setter through would hide a genuine redefinition behind one.
    for d in getattr(node, "decorator_list", []):
        if (isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter")
                and isinstance(d.value, ast.Name) and d.value.id == node.name):
            return True
    return False


def duplicate_definitions(path):
    """Return ``[(scope, name, count)]`` for every name defined more than once in a scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    def scan(body, scope):
        counts = collections.Counter(
            n.name for n in body if isinstance(n, DEFS) and not _is_setter_pair(n))
        found.extend((scope, name, c) for name, c in counts.items() if c > 1)
        for n in body:
            if isinstance(n, ast.ClassDef):
                scan(n.body, f"{scope}.{n.name}")

    scan(tree.body, path.stem)
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


if __name__ == "__main__":
    unittest.main()
