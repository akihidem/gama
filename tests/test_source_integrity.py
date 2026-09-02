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
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "gama"
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_setter_pair(node):
    # `@name.setter` / `@name.deleter` re-use the property's name by design.
    for d in getattr(node, "decorator_list", []):
        if isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter"):
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
        modules = sorted(PACKAGE.glob("*.py"))
        self.assertGreater(len(modules), 5, "the package moved; point PACKAGE at it")
        dups = [(str(p.name), *d) for p in modules for d in duplicate_definitions(p)]
        self.assertEqual(dups, [], f"a later definition silently wins: {dups}")

    def test_the_check_sees_a_redefinition(self):
        # The floor has to be shown to catch the defect it was written for.
        src = "def f():\n    return 1\n\ndef g():\n    return 2\n\ndef f():\n    return 3\n"
        tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "gama_dup_check.py"
        tmp.write_text(src, encoding="utf-8")
        try:
            self.assertEqual(duplicate_definitions(tmp), [("gama_dup_check", "f", 2)])
        finally:
            tmp.unlink()

    def test_property_setters_are_not_flagged(self):
        src = ("class A:\n    @property\n    def x(self):\n        return 1\n"
               "    @x.setter\n    def x(self, v):\n        pass\n")
        tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "gama_setter_check.py"
        tmp.write_text(src, encoding="utf-8")
        try:
            self.assertEqual(duplicate_definitions(tmp), [])
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()
