"""L0 for `gama grow` — the properties that make the loop's numbers mean something.

Every test runs on deterministic stand-in backends (no model calls), because the point of
these checks is the *loop's discipline*, not any model's skill:

  1. the three splits never overlap (a leak makes "held-out" a lie)
  2. a candidate that wins on `search` but not on `confirm` is NOT promoted (the overfit trap)
  3. the `sealed` split is not touched until after the last generation
  4. an improvement smaller than the champion's own measured drift is refused
  5. the loop terminates, is deterministic, and every proposed spec actually builds
"""
import contextlib
import io
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from gama import backends as backends_mod
from gama.backends import ModelBackend
from gama.benchmark import BenchCase
from gama.cli import build_parser, main
from gama.config import build_backend, system_from_config
from gama.grow import (
    canonical,
    measure,
    validate_pool,
    grow,
    promote_gate,
    propose,
    seed_champion,
    spec_hash,
    split_cases,
    suite_pool,
    write_recipe,
)


# --------------------------------------------------------------------------- #
# A deterministic stand-in for a local model: it solves exactly the cases listed
# in WINS[tag], and records every case it was shown (so a test can prove which
# split was consulted, and when).
# --------------------------------------------------------------------------- #
class Scripted(ModelBackend):
    name = "scripted"
    available = True
    WINS: dict = {}
    PARTIAL: dict = {}
    SEEN: list = []

    def __init__(self, tag: str = "a"):
        self.tag = tag
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        cid = prompt.split("case=")[1].split()[0] if "case=" in prompt else "?"
        Scripted.SEEN.append(cid)
        if cid in Scripted.PARTIAL.get(self.tag, set()):
            return "HALF"
        return "GOOD" if cid in Scripted.WINS.get(self.tag, set()) else "BAD"


def _cases(n=4, task_type="qa", prefix="qa"):
    """n cases of one class; the prompt carries the case id so Scripted can act on it."""
    return [BenchCase(f"{prefix}{i}", task_type, f"case={prefix}{i}",
                      lambda o: 1.0 if "GOOD" in o else 0.0) for i in range(1, n + 1)]


def _partial_cases(n=4):
    """Same, but a reply of "HALF" earns 0.4 — less than one whole case."""
    def chk(o):
        return 1.0 if "GOOD" in o else (0.4 if "HALF" in o else 0.0)
    return [BenchCase(f"qa{i}", "qa", f"case=qa{i}", chk) for i in range(1, n + 1)]


def _lane(tag):
    return {"backend": "scripted", "kwargs": {"tag": tag}}


class ScriptedCase(unittest.TestCase):
    def setUp(self):
        backends_mod._BACKENDS["scripted"] = Scripted
        Scripted.WINS, Scripted.SEEN, Scripted.PARTIAL = {}, [], {}

    def tearDown(self):
        backends_mod._BACKENDS.pop("scripted", None)
        Scripted.WINS, Scripted.SEEN, Scripted.PARTIAL = {}, [], {}


# --------------------------------------------------------------------------- #
# 1. Splits
# --------------------------------------------------------------------------- #
class TestSplit(unittest.TestCase):
    def test_disjoint_and_complete(self):
        pool = suite_pool(["hard", "brutal"])
        s = split_cases(pool)
        ids = {k: {c.case_id for c in v} for k, v in s.items()}
        self.assertEqual(ids["search"] & ids["confirm"], set())
        self.assertEqual(ids["search"] & ids["sealed"], set())
        self.assertEqual(ids["confirm"] & ids["sealed"], set())
        self.assertEqual(ids["search"] | ids["confirm"] | ids["sealed"],
                         {c.case_id for c in pool})

    def test_deterministic(self):
        pool = suite_pool(["hard"])
        a, b = split_cases(pool), split_cases(pool)
        self.assertEqual([c.case_id for c in a["confirm"]], [c.case_id for c in b["confirm"]])

    def test_classes_spread_across_splits(self):
        # A class that lands entirely in one split would make its mutations unconfirmable.
        s = split_cases(suite_pool(["default", "hard", "brutal"]))
        self.assertEqual({c.task_type for c in s["search"]}, {c.task_type for c in s["confirm"]})

    def test_empty_confirm_is_refused(self):
        # Positive-proof floor: refuse to grow rather than silently decide on the search split.
        with self.assertRaises(ValueError):
            split_cases(_cases(1))

    def test_ratio_validated(self):
        with self.assertRaises(ValueError):
            split_cases(_cases(4), ratio=(0, 0, 0))


# --------------------------------------------------------------------------- #
# 2. Spec canonicalisation
# --------------------------------------------------------------------------- #
class TestCanonical(ScriptedCase):
    def test_orphan_lane_dropped_so_hashes_match(self):
        champ = seed_champion({"a": _lane("a"), "b": _lane("b")}, "a")
        with_orphan = json.loads(json.dumps(champ))
        with_orphan["kwargs"]["backends"]["unused"] = _lane("b")
        self.assertEqual(spec_hash(champ), spec_hash(with_orphan))
        self.assertNotIn("unused", canonical(with_orphan)["kwargs"]["backends"])

    def test_routing_change_changes_hash(self):
        pool = {"a": _lane("a"), "b": _lane("b")}
        champ = seed_champion(pool, "a")
        other = json.loads(json.dumps(champ))
        other["kwargs"]["backends"]["b"] = _lane("b")
        other["kwargs"]["routing_table"]["qa"] = "b"
        self.assertNotEqual(spec_hash(champ), spec_hash(other))


# --------------------------------------------------------------------------- #
# 3. Candidate generation
# --------------------------------------------------------------------------- #
class TestPropose(ScriptedCase):
    def setUp(self):
        super().setUp()
        self.pool = {"a": _lane("a"), "b": _lane("b")}
        self.champ = seed_champion(self.pool, "a")

    def test_every_candidate_builds(self):
        for c in propose(self.champ, self.pool, ["qa"], width=12):
            build_backend(c.spec)                     # raises if the mutation emitted junk

    def test_mixes_kinds_and_respects_width(self):
        cands = propose(self.champ, self.pool, ["qa"], width=4)
        self.assertEqual(len(cands), 4)
        self.assertGreaterEqual(len({c.kind for c in cands}), 3)

    def test_spreads_across_task_classes_at_small_width(self):
        # Filling `width` class-by-class would leave every class after the first untouched.
        classes = ["code_implementation", "content", "integration", "qa", "research"]
        cands = propose(self.champ, self.pool, classes, width=5)
        touched = {c.label.split(":")[1].split("(")[0].split("->")[0] for c in cands}
        self.assertGreaterEqual(len(touched), 4, [c.label for c in cands])

    def test_lane_names_with_stray_parens_do_not_crash(self):
        # Pool keys come from user JSON; only grow's own derived names are `kind(...)`,
        # so an unbalanced paren in a lane name must read as a plain name, not crash.
        for name in ("we(ird", "foo(a)"):      # unbalanced, and the dangerous `kind(inner)` shape
            pool = {"a": _lane("a"), name: _lane("b")}
            champ = seed_champion(pool, name)
            labels = []
            for c in propose(champ, pool, ["qa"], width=12):
                build_backend(c.spec)
                labels.append(c.label)
            # `foo(a)` is a user lane, not a composite: unwrapping it to `a` would reroute to a
            # different backend while labelling the move "simplify".
            self.assertNotIn("simplify:qa->a", labels, name)

    def test_explicit_classes_still_cannot_escape_the_confirm_filter(self):
        # Asking for a class that has no confirm case must fail loudly: running it anyway
        # would report "0 promotions" as if that were a measured verdict.
        Scripted.WINS = {"a": set(), "b": {"cd1"}}
        with self.assertRaises(ValueError):
            grow({"a": _lane("a"), "b": _lane("b")},
                 cases=_cases(4) + _cases(1, "code_implementation", "cd"),
                 classes=["code_implementation"], generations=1, width=4)
        # ...and asking for a mix keeps only the confirmable half.
        res = grow({"a": _lane("a"), "b": _lane("b")},
                   cases=_cases(4) + _cases(1, "code_implementation", "cd"),
                   classes=["code_implementation", "qa"], generations=1, width=4)
        self.assertTrue(all("code_implementation" not in h["challenger"] for h in res["history"]))

    def test_seed_spec_routing_to_a_missing_lane_is_refused(self):
        # GamaBackend would quietly fall back to the default lane, making the recorded spec
        # and the measured backend disagree — the one thing this ledger must never do.
        broken = {"backend": "gama", "kwargs": {"backends": {"a": _lane("a")},
                                                "routing_table": {"qa": "ghost"}, "default": "a"}}
        with self.assertRaises(ValueError):
            grow({"a": _lane("a")}, cases=_cases(4), seed_spec=broken, generations=1, width=1)

    def test_composite_lanes_record_their_base_instead_of_encoding_it_in_the_name(self):
        pool = {"a": _lane("a"), "b": _lane("b")}
        champ = seed_champion(pool, "a")
        for c in propose(champ, pool, ["qa"], width=12):
            lane = c.spec["kwargs"]["routing_table"]["qa"]
            spec = c.spec["kwargs"]["backends"][lane]
            if spec["backend"] in ("tool", "ensemble", "meshflow"):
                self.assertEqual(spec["_grow_base"], "a")
                build_backend(spec)          # the extra key must not disturb construction

    def test_measure_refuses_an_empty_split(self):
        with self.assertRaises(ValueError):
            measure(seed_champion({"a": _lane("a")}, "a"), [])

    def test_pool_lane_named_like_a_composite_is_refused(self):
        # `tool(a)` as a user lane name would be silently overwritten by the tool mutation:
        # the config would then declare one backend and measure another under that name.
        bad = {"a": _lane("a"), "tool(a)": _lane("b")}
        with self.assertRaises(ValueError):
            validate_pool(bad)
        with self.assertRaises(ValueError):
            propose(seed_champion({"a": _lane("a")}, "a"), bad, ["qa"], width=4)
        with self.assertRaises(ValueError):
            grow(bad, cases=_cases(4), generations=1, width=2)

    CLASSES = ["code_implementation", "content", "integration", "qa", "research"]

    def test_the_frontier_rotates_across_generations(self):
        # Measured defect: with width < #kinds x #classes, an un-challenged candidate sits at
        # the head of its kind's queue forever, so later classes of that kind are never
        # proposed at all. A real run spent three generations re-offering `tool:integration`
        # and never once tried `tool:qa` — which reads in the ledger as "qa was not worth it".
        rounds = [propose(self.champ, self.pool, self.CLASSES, width=4, generation=g)
                  for g in range(3)]
        labels = {c.label for r in rounds for c in r}
        self.assertIn("tool:qa(a)", labels, sorted(labels))
        self.assertEqual([c.label for c in rounds[1]],   # still deterministic per generation
                         [c.label for c in propose(self.champ, self.pool, self.CLASSES,
                                                   width=4, generation=1)])

    def test_no_mutation_kind_is_starved_at_a_narrow_width(self):
        # The nastier half of the same defect: kinds were always tried in a fixed order, so
        # width=4 could never reach the 5th kind — and `simplify` becomes non-empty exactly
        # when a promotion happens, so the first promotion pushed `meshflow` off the list.
        champ = json.loads(json.dumps(self.champ))
        champ["kwargs"]["backends"]["mesh(a+b)"] = {
            "backend": "meshflow", "_grow_base": "a",
            "kwargs": {"tiers": [_lane("a"), _lane("b")], "mesh": "union"}}
        champ["kwargs"]["routing_table"]["research"] = "mesh(a+b)"
        champ = canonical(champ)
        seen = set()
        for g in range(5):
            seen |= {c.kind for c in propose(champ, self.pool, self.CLASSES, width=4,
                                             generation=g)}
        self.assertEqual(seen, {"simplify", "route", "tool", "ensemble", "meshflow"}, seen)

    def test_never_proposes_the_champion_or_excluded(self):
        cands = propose(self.champ, self.pool, ["qa"], width=12)
        hashes = {spec_hash(c.spec) for c in cands}
        self.assertNotIn(spec_hash(self.champ), hashes)
        drop = sorted(hashes)[0]
        again = {spec_hash(c.spec) for c in propose(self.champ, self.pool, ["qa"], width=12,
                                                    exclude={drop})}
        self.assertNotIn(drop, again)

    def test_composite_lanes_can_be_simplified_again(self):
        # A loop that can only ADD structure drifts one way forever. Every composite lane
        # shape must be reachable *backwards* to its atomic model.
        for lane_name, spec in (
            ("tool(a)", {"backend": "tool", "kwargs": {"inner": _lane("a")}}),
            ("ens(a+b)", {"backend": "ensemble",
                          "kwargs": {"members": [_lane("a"), _lane("b")], "strategy": "majority"}}),
            ("mesh(a->b)", {"backend": "meshflow",
                            "kwargs": {"tiers": [_lane("a"), _lane("b")], "mesh": "union"}}),
        ):
            champ = json.loads(json.dumps(self.champ))
            champ["kwargs"]["backends"][lane_name] = spec
            champ["kwargs"]["routing_table"]["qa"] = lane_name
            kinds = {c.kind: c for c in propose(canonical(champ), self.pool, ["qa"], width=12)}
            self.assertIn("simplify", kinds, f"{lane_name} is a dead end for shrinking")
            self.assertEqual(kinds["simplify"].label, "simplify:qa->a")


# --------------------------------------------------------------------------- #
# 4. The gate
# --------------------------------------------------------------------------- #
class TestPromoteGate(unittest.TestCase):
    def test_promotes_only_when_all_three_hold(self):
        ok, why = promote_gate(0.5, 0.8, 0.5, 0.9, 0.05)
        self.assertTrue(ok)
        self.assertEqual(why, "promote")

    def test_rejects_when_search_not_better(self):
        ok, why = promote_gate(0.8, 0.8, 0.5, 0.9, 0.05)
        self.assertFalse(ok)
        self.assertEqual(why, "search-not-better")

    def test_rejects_when_confirm_not_better(self):
        ok, why = promote_gate(0.5, 1.0, 0.5, 0.5, 0.05)
        self.assertFalse(ok)
        self.assertEqual(why, "confirm-not-better")

    def test_rejects_improvement_below_the_margin(self):
        ok, why = promote_gate(0.5, 1.0, 0.50, 0.52, 0.10)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("below-margin"))


# --------------------------------------------------------------------------- #
# 5. The loop
# --------------------------------------------------------------------------- #
class TestGrowLoop(ScriptedCase):
    """Landscape: 4 qa cases -> search {qa1,qa4}, confirm {qa2}, sealed {qa3}."""

    def _grow(self, pool, **kw):
        opts = dict(cases=_cases(4), generations=2, width=2, patience=5, min_margin=0.05)
        opts.update(kw)
        return grow(pool, **opts)

    def test_search_only_winner_is_not_promoted(self):
        # `b` aces the split used for selection and solves nothing else — exactly the
        # candidate a search-only loop would crown.
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa4"}}
        res = self._grow({"a": _lane("a"), "b": _lane("b")})
        self.assertEqual(res["promotions"], 0)
        self.assertEqual(res["champion_hash"], res["seed_hash"])
        gen0 = res["history"][0]
        self.assertEqual(gen0["challenger"], "route:qa->b")
        self.assertEqual(gen0["challenger_search"], 1.0)      # it did win the selection split
        self.assertEqual(gen0["reason"], "confirm-not-better")

    def test_genuine_winner_is_promoted(self):
        Scripted.WINS = {"a": set(), "c": {"qa1", "qa2", "qa3"}}
        res = self._grow({"a": _lane("a"), "c": _lane("c")})
        self.assertEqual(res["promotions"], 1)
        self.assertEqual(res["champion"]["kwargs"]["routing_table"]["qa"], "c")
        self.assertEqual(res["sealed"]["seed"]["score"], 0.0)
        self.assertEqual(res["sealed"]["champion"]["score"], 1.0)

    def test_champion_confirm_never_regresses(self):
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa4"}, "c": {"qa1", "qa2", "qa3"}}
        res = self._grow({"a": _lane("a"), "b": _lane("b"), "c": _lane("c")},
                         generations=4, width=6)
        best = 0.0
        for h in res["history"]:
            self.assertGreaterEqual(h["champion_confirm"] + 1e-9, best)
            best = max(best, h["champion_confirm"])

    def test_sealed_split_is_untouched_until_the_end(self):
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa4"}, "c": {"qa1", "qa2", "qa3"}}
        trace: list = []
        Scripted.SEEN = trace

        def on_event(row):
            trace.append(f"@{row['event']}")

        self._grow({"a": _lane("a"), "b": _lane("b"), "c": _lane("c")},
                   generations=3, width=4, on_event=on_event)
        last_decision = max(i for i, t in enumerate(trace)
                            if t in ("@generation", "@stop", "@seed"))
        seen_sealed_early = [t for t in trace[:last_decision] if t == "qa3"]
        self.assertEqual(seen_sealed_early, [], "sealed cases were used to decide something")
        self.assertIn("qa3", trace[last_decision:], "sealed split was never measured at all")

    def test_deterministic_across_runs(self):
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa4"}, "c": {"qa1", "qa2", "qa3"}}
        pool = {"a": _lane("a"), "b": _lane("b"), "c": _lane("c")}
        r1 = self._grow(pool, generations=3, width=4)
        Scripted.SEEN = []
        r2 = self._grow(pool, generations=3, width=4)
        self.assertEqual(r1["champion_hash"], r2["champion_hash"])
        self.assertEqual([(h["challenger"], h["verdict"]) for h in r1["history"]],
                         [(h["challenger"], h["verdict"]) for h in r2["history"]])

    def test_margin_floor_defaults_to_one_confirm_case(self):
        Scripted.WINS = {"a": set()}
        res = self._grow({"a": _lane("a")}, generations=1, width=1, min_margin=None)
        # confirm holds 1 of the 4 cases here, so one case is the whole score
        self.assertEqual(len(res["splits"]["confirm"]), 1)
        self.assertEqual(res["params"]["min_margin"], 1.0)
        self.assertEqual(res["params"]["min_margin_source"], "auto(one confirm case)")

    def test_a_coarse_margin_floor_is_declared_up_front(self):
        # A 5-case confirm split makes the floor 0.2, which refuses real improvements worth
        # less than a whole case. That limit has to be visible at the start of the run, not
        # inferred afterwards from a champion that never moved.
        Scripted.WINS = {"a": set()}
        events = []
        res = self._grow({"a": _lane("a")}, generations=1, width=1, min_margin=None,
                         on_event=events.append)
        seed = [e for e in events if e["event"] == "seed"][0]
        self.assertTrue(seed["margin_floor_coarse"])
        self.assertEqual(res["params"]["min_margin"], 1.0)

    def test_a_fine_margin_floor_is_not_flagged(self):
        Scripted.WINS = {}
        events = []
        grow({"a": _lane("a")}, cases=_cases(40), generations=1, width=1,
             on_event=events.append)
        seed = [e for e in events if e["event"] == "seed"][0]
        self.assertFalse(seed["margin_floor_coarse"])

    def test_gain_smaller_than_one_confirm_case_is_refused(self):
        # `b` genuinely improves the confirm case, but only by partial credit (0.0 -> 0.4).
        # Under the derived floor that is not a case, so it is not a promotion.
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"}}
        Scripted.PARTIAL = {"b": {"qa2"}}
        opts = dict(cases=_partial_cases(4), generations=1, width=1, patience=5)
        refused = grow({"a": _lane("a"), "b": _lane("b")}, **opts)
        self.assertEqual(refused["promotions"], 0)
        self.assertTrue(refused["history"][0]["reason"].startswith("below-margin"))
        # ...and an explicit, smaller floor still promotes it, so the refusal is the floor
        # doing its job rather than the candidate being weak.
        taken = grow({"a": _lane("a"), "b": _lane("b")}, min_margin=0.3, **opts)
        self.assertEqual(taken["promotions"], 1)

    def test_exactly_one_confirm_case_is_enough(self):
        # The floor is "at least one case", not "more than one case" — the README says the
        # same thing, and a gate that disagrees with its own docs is the bug either way.
        ok, why = promote_gate(0.5, 0.6, 0.60, 0.60 + 1 / 15, 1 / 15)
        self.assertTrue(ok, why)
        # and with the 4-decimal scores `summarize` actually emits (11/15 -> 12/15), where a
        # bare `>=` against the raw 1/n would land on the wrong side of the rounding
        ok, why = promote_gate(0.5, 0.6, 0.7333, 0.8, 1 / 15)
        self.assertTrue(ok, why)

    def test_negative_margin_is_refused(self):
        # max(negative, drift) collapses to drift, and drift can be 0 — that removes the
        # gate rather than loosening it.
        with self.assertRaises(ValueError):
            grow({"a": _lane("a")}, cases=_cases(4), generations=1, width=1, min_margin=-0.1)

    def test_patience_stops_a_loop_that_is_going_nowhere(self):
        Scripted.WINS = {"a": set(), "b": set()}
        res = self._grow({"a": _lane("a"), "b": _lane("b")}, generations=10, width=6, patience=2)
        self.assertEqual(res["promotions"], 0)
        self.assertLessEqual(res["generations_run"], 2)

    def test_ledger_is_written_and_replayable(self):
        Scripted.WINS = {"a": set(), "c": {"qa1", "qa2", "qa3"}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "grow.jsonl"
            res = self._grow({"a": _lane("a"), "c": _lane("c")}, ledger_path=str(path))
            rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
            events = [r["event"] for r in rows]
            self.assertEqual(events[0], "seed")
            self.assertEqual(events[-1], "final")
            gen = [r for r in rows if r["event"] == "generation"]
            self.assertEqual(gen[0]["verdict"], "promote")
            self.assertEqual(gen[0]["champion_after"], res["champion_hash"])

    def test_unconfirmable_classes_are_left_alone(self):
        # code_implementation exists only in the search split here -> mutating it could never
        # be confirmed, so the loop must not spend measurements on it.
        cases = _cases(4) + _cases(1, "code_implementation", "cd")
        Scripted.WINS = {"a": set(), "b": {"cd1"}}
        res = grow({"a": _lane("a"), "b": _lane("b")}, cases=cases, generations=1, width=6)
        labels = [h["challenger"] for h in res["history"]]
        self.assertTrue(all("code_implementation" not in ln for ln in labels), labels)


# --------------------------------------------------------------------------- #
# 6. Recipe emission
# --------------------------------------------------------------------------- #
class TestWriteRecipe(ScriptedCase):
    def test_writes_a_loadable_recipe_quoting_the_sealed_numbers(self):
        Scripted.WINS = {"a": set(), "c": {"qa1", "qa2", "qa3"}}
        res = grow({"a": _lane("a"), "c": _lane("c")}, cases=_cases(4), generations=2, width=2)
        with tempfile.TemporaryDirectory() as d:
            out = write_recipe(res, Path(d) / "grown", hardware="test box")
            cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
            build_backend(cfg["system"])              # `gama run --config` path must work
            self.assertEqual(cfg["grow"]["promotions"], 1)
            md = (out / "recipe.md").read_text(encoding="utf-8")
            self.assertIn("sealed score", md)
            self.assertIn("do not quote", md)         # the biased number is labelled as such

    def test_the_spot_check_command_measures_the_champion(self):
        # A recipe whose own reproduce line measures something else is worse than no recipe:
        # `--backends gama` reads flat routing keys this file does not have, and silently
        # benchmarks a default model under the champion's name.
        Scripted.WINS = {"a": set(), "c": {"qa1", "qa2", "qa3"}}
        res = grow({"a": _lane("a"), "c": _lane("c")}, cases=_cases(4), generations=2, width=2)
        with tempfile.TemporaryDirectory() as d:
            out = write_recipe(res, Path(d) / "grown")
            md = (out / "recipe.md").read_text(encoding="utf-8")
            cmd = re.search(r"spot-check the champion[^`]*`([^`]+)`", md, re.S).group(1)
            argv = [str(out / "config.json") if a == "config.json" else a
                    for a in shlex.split(cmd)[1:]]
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
            built = system_from_config(out / "config.json")
            self.assertEqual(built.routing_table,
                             res["champion"]["kwargs"]["routing_table"])
            self.assertEqual(sorted(built.backends),
                             sorted(res["champion"]["kwargs"]["backends"]))
            # the conditions the numbers came from must travel with them
            self.assertIn("grown with:", md)
            self.assertEqual(json.loads((out / "config.json").read_text(
                encoding="utf-8"))["grow"]["splits"], res["splits"])


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
class TestGrowCli(unittest.TestCase):
    def test_parser_wires_grow(self):
        args = build_parser().parse_args(["grow", "--models", "qwen2.5:7b", "--generations", "2"])
        self.assertEqual(args.command, "grow")
        self.assertEqual(args.generations, 2)

    def test_bad_pool_json_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "pool.json"
            bad.write_text("{not json", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["grow", "--pool", str(bad)]), 2)

    def test_bad_ratio_is_a_clean_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["grow", "--smoke", "--ratio", "0:0:0"]), 2)
            self.assertEqual(main(["grow", "--smoke", "--ratio", "2:1"]), 2)

    def test_too_few_cases_is_a_clean_error_not_a_traceback(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["grow", "--smoke", "--suites", "hard", "--ratio", "3:0:1"]), 2)

    def test_zero_width_is_rejected_not_silently_empty(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["grow", "--smoke", "--width", "0"])

    def test_grow_without_lanes_is_an_error(self):
        self.assertEqual(main(["grow", "--suites", "hard"]), 2)

    def test_smoke_runs_and_promotes_nothing(self):
        # echo/null solve none of the hard cases, so the only honest outcome is "no promotion".
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(["grow", "--smoke", "--suites", "hard,brutal", "--generations", "1",
                       "--width", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["promotions"], 0)


if __name__ == "__main__":
    unittest.main()
