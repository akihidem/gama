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

from gama.cli import main as cli_main
from gama import backends as backends_mod
from gama.grow import (Candidate, _challenger_key, _default_swap_viable,
                       _structure_size, class_headroom)
from gama.backends import ModelBackend
from gama.benchmark import BenchCase
from gama.cli import build_parser, main
from gama.config import build_backend, system_from_config
from gama.backends import note_served, reset_served, served_conflicts, served_map
from gama.grow import (
    MeasurementFailure,
    sealed_verdict,
    _guard_measurement,
    Measurement,
    paired_gain,
    sign_test,
    canonical,
    load_checkpoint,
    code_stamp,
    shrink_band,
    simplify_gate,
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
    def test_a_route_back_to_the_default_lane_is_dropped(self):
        # Shrinking a class back onto the default lane must land on the SAME design as the
        # seed, not a look-alike with a no-op route in it.
        pool = {"a": _lane("a")}
        seed = seed_champion(pool, "a")
        shrunk = json.loads(json.dumps(seed))
        shrunk["kwargs"]["routing_table"]["qa"] = "a"
        self.assertEqual(spec_hash(canonical(shrunk)), spec_hash(seed))

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
        # `default->X` changes the lane every unrouted class sits on, so it carries no class
        # in its label; only the per-class mutations are counted here.
        touched = {c.label.split(":")[1].split("(")[0].split("->")[0]
                   for c in cands if ":" in c.label}
        self.assertGreaterEqual(len(touched), 3, [c.label for c in cands])

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
            if c.kind == "default":            # not a per-class move: it has no qa route
                continue
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
        self.assertEqual(seen, {"simplify", "route", "tool", "ensemble", "meshflow",
                                "default", "deepen"}, seen)

    def test_ensemble_mutations_aggregate_with_the_other_member(self):
        # `majority` needs verbatim agreement, which free-text answers never give: Counter sees
        # all-ones and returns members[0]. Measured on graded: majority 0.705, below the bare
        # 3B's 0.830, at 14x the latency — the mutation was paying for three models to keep the
        # cheapest one's answer. synthesize scored 0.975 (8-0 by case).
        cands = propose(self.champ, self.pool, ["qa"], width=12)
        ens = [c for c in cands if c.kind == "ensemble"]
        self.assertTrue(ens)
        for c in ens:
            lane = c.spec["kwargs"]["routing_table"]["qa"]
            kw = c.spec["kwargs"]["backends"][lane]["kwargs"]
            self.assertEqual(kw["strategy"], "synthesize")
            self.assertEqual(kw["aggregator"], kw["members"][1])  # the other model, not base
            build_backend(c.spec)

    def test_non_synthesize_ensembles_carry_no_unused_aggregator(self):
        # build_backend constructs `aggregator` eagerly, so shipping one with a strategy that
        # never reads it builds a backend for nothing (and drags in whatever that backend needs).
        for strategy in ("majority", "first"):
            for c in propose(self.champ, self.pool, ["qa"], width=12,
                             ensemble_strategy=strategy):
                if c.kind != "ensemble":
                    continue
                lane = c.spec["kwargs"]["routing_table"]["qa"]
                self.assertNotIn("aggregator", c.spec["kwargs"]["backends"][lane]["kwargs"])
                build_backend(c.spec)

    def test_an_excluded_candidate_does_not_consume_its_kind_s_slot(self):
        # Third variant of the same starvation family, measured in run J: once `simplify:qa` was
        # decided, popping it consumed the simplify slot every generation, so `simplify:research`
        # was never proposed — the loop never once had to argue for keeping the research lane.
        # `width` counts candidates EMITTED, not pops attempted.
        champ = json.loads(json.dumps(self.champ))
        champ["kwargs"]["backends"]["tool(a)"] = {
            "backend": "tool", "_grow_base": "a", "kwargs": {"inner": _lane("a")}}
        champ["kwargs"]["backends"]["mesh(a+b)"] = {
            "backend": "meshflow", "_grow_base": "a",
            "kwargs": {"tiers": [_lane("a"), _lane("b")], "mesh": "union"}}
        champ["kwargs"]["routing_table"] = {"qa": "tool(a)", "research": "mesh(a+b)"}
        champ = canonical(champ)
        classes = ["code_implementation", "content", "integration", "qa", "research"]
        first = propose(champ, self.pool, classes, width=4, generation=0)
        decided = {spec_hash(c.spec) for c in first if c.kind == "simplify"}
        self.assertTrue(decided, [c.label for c in first])
        later = set()
        for g in range(1, 5):
            later |= {c.label for c in propose(champ, self.pool, classes, width=4,
                                               exclude=decided, generation=g)
                      if c.kind == "simplify"}
        self.assertTrue(later, "the second simplification was never proposed")

    def test_the_default_lane_itself_can_be_mutated(self):
        # Every other mutation is per class, so the lane that serves every UNROUTED class was
        # unreachable: a champion routing 2 of 5 classes had no move that touched the other 3.
        cands = propose(self.champ, self.pool, ["qa"], width=20)
        defaults = [c for c in cands if c.kind == "default"]
        self.assertTrue(defaults, [c.label for c in cands])
        for c in defaults:
            self.assertNotEqual(c.spec["kwargs"]["default"],
                                self.champ["kwargs"]["default"])
            build_backend(c.spec)

    def test_a_composite_lane_can_be_deepened(self):
        # Composites were only ever built from atomic lanes, so a nested stack like
        # mesh(tool(3b) -> 7b) — the shape gama's own README advertises — was structurally
        # unreachable by the loop.
        champ = json.loads(json.dumps(self.champ))
        champ["kwargs"]["backends"]["mesh(a+b)"] = {
            "backend": "meshflow", "_grow_base": "a",
            "kwargs": {"tiers": [_lane("a"), _lane("b")], "mesh": "union"}}
        champ["kwargs"]["routing_table"]["qa"] = "mesh(a+b)"
        deep = [c for c in propose(canonical(champ), self.pool, ["qa"], width=20)
                if c.kind == "deepen"]
        self.assertTrue(deep)
        lane = deep[0].spec["kwargs"]["routing_table"]["qa"]
        inner = deep[0].spec["kwargs"]["backends"][lane]["kwargs"]["tiers"][0]
        self.assertEqual(inner["backend"], "tool")     # the cheap tier now writes code
        build_backend(deep[0].spec)

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
class TestSimplifyGate(unittest.TestCase):
    def test_promotes_a_simpler_champion_that_is_not_measurably_worse(self):
        ok, why = simplify_gate(0.80, 0.79, 0.05, 2, 1)
        self.assertTrue(ok, why)

    def test_refuses_when_the_drop_is_measurable(self):
        ok, why = simplify_gate(0.80, 0.70, 0.05, 2, 1)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("measurably-worse"))

    def test_refuses_when_nothing_got_simpler(self):
        # "no worse" alone would let the loop churn between equivalent designs forever.
        self.assertFalse(simplify_gate(0.80, 0.80, 0.05, 1, 1)[0])

    def test_a_noisy_generation_does_not_licence_dropping_structure(self):
        # Real numbers from run I gen0: drift 0.1141 on a 23-case confirm split. Reusing the
        # additive delta there let a 0.065 drop (1.5 cases) through as "not measurable", and it
        # removed `tool:qa` — the change with 7 promotions in 8 attempts and a per-case verified
        # mechanism. Noise must not widen the licence to discard.
        floor, drift = 0.0435, 0.1141
        self.assertEqual(shrink_band(floor, drift), floor)
        self.assertFalse(simplify_gate(0.913, 0.8478, shrink_band(floor, drift), 2, 1)[0])

    def test_a_quiet_generation_narrows_the_band_too(self):
        # And when the measurement is stable, a sub-case drop is still a drop: the band is the
        # intersection of resolution and noise, not the maximum.
        self.assertEqual(shrink_band(0.0435, 0.01), 0.01)
        self.assertFalse(simplify_gate(0.90, 0.88, shrink_band(0.0435, 0.01), 2, 1)[0])
        self.assertTrue(simplify_gate(0.90, 0.895, shrink_band(0.0435, 0.01), 2, 1)[0])


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

    def test_the_ledger_records_which_code_produced_it(self):
        # This loop has changed its own gates while running experiments (frontier rotation,
        # ensemble aggregation, the shrink gate and its band). Conditions alone cannot tell two
        # runs apart when the judging changed between them.
        stamp = code_stamp()
        self.assertIn("version", stamp)
        self.assertIn("commit", stamp)
        Scripted.WINS = {"a": set()}
        events = []
        res = self._grow({"a": _lane("a")}, generations=1, width=1, on_event=events.append)
        seed = [e for e in events if e["event"] == "seed"][0]
        self.assertEqual(seed["code"]["version"], stamp["version"])
        self.assertEqual(res["params"]["code"], stamp)

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

    def test_a_champion_can_shrink_again(self):
        # Counted across six real runs: simplify was proposed 4 times, became the challenger 0
        # times, and the champion's cost only ever went up. Additions and removals are
        # different questions ("is it better?" vs "is it worse?") and need different gates.
        Scripted.WINS = {"a": {"qa1", "qa2", "qa4"}}
        pool = {"a": _lane("a")}
        champ = seed_champion(pool, "a")
        champ["kwargs"]["backends"]["tool(a)"] = {
            "backend": "tool", "_grow_base": "a", "kwargs": {"inner": _lane("a")}}
        champ["kwargs"]["routing_table"]["qa"] = "tool(a)"
        res = grow(pool, cases=_cases(4), seed_spec=canonical(champ), generations=2, width=6,
                   patience=5, min_margin=0.05)
        gen0 = res["history"][0]
        # the tool lane wraps the same scripted model, so removing it costs exactly nothing —
        # which is the only situation the tightened band allows a removal in
        self.assertEqual(gen0.get("simplify_challenger"), "simplify:qa->a")
        self.assertEqual(gen0.get("simplify_verdict"), "promote", gen0.get("simplify_reason"))
        self.assertEqual(res["champion"]["kwargs"]["routing_table"], {})
        self.assertEqual(res["history"][0]["structure_size"], 0)

    def test_a_shrink_counts_as_a_promotion(self):
        # The champion changed; a ledger that says "0 promotions" because the additive gate
        # said reject is a record that disagrees with what happened.
        Scripted.WINS = {"a": {"qa1", "qa2", "qa4"}}
        pool = {"a": _lane("a")}
        champ = seed_champion(pool, "a")
        champ["kwargs"]["backends"]["tool(a)"] = {
            "backend": "tool", "_grow_base": "a", "kwargs": {"inner": _lane("a")}}
        champ["kwargs"]["routing_table"]["qa"] = "tool(a)"
        res = grow(pool, cases=_cases(4), seed_spec=canonical(champ), generations=1, width=6,
                   patience=5, min_margin=0.05)
        self.assertEqual(res["promotions"], 1)
        self.assertNotEqual(res["champion_hash"], res["seed_hash"])

    def test_structure_is_counted_per_routed_class(self):
        # Two classes sharing one composite lane: returning one of them to the bare model is a
        # real reduction, and counting lanes instead of routes would call it "not simpler".
        from gama.grow import _structure_size
        shared = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"),
                         "tool(a)": {"backend": "tool", "_grow_base": "a",
                                     "kwargs": {"inner": _lane("a")}}},
            "routing_table": {"qa": "tool(a)", "research": "tool(a)"}, "default": "a"}}
        one_back = json.loads(json.dumps(shared))
        one_back["kwargs"]["routing_table"]["research"] = "a"
        self.assertEqual(_structure_size(canonical(shared)), 2)
        self.assertEqual(_structure_size(canonical(one_back)), 1)

    def test_an_interrupted_run_resumes_from_its_ledger(self):
        # A real run died to the OOM killer right after measuring 43 cases x 4 candidates, and
        # every one of those measurements was lost because nothing in the ledger carried the
        # state a continuation needs.
        # confirm 側に必ず伸びしろを残す(満点だとクラスが飽和して挑戦が一度も起きず、
        # 再開の試験が空回りする)。ここで見たいのは再開であって精度ではない。
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            first = grow(pool, cases=_cases(8), generations=1, width=2, patience=5,
                         ledger_path=str(led), min_margin=0.05)
            ck = load_checkpoint(led)
            self.assertIsNotNone(ck)
            self.assertEqual(ck["gen"], 0)

            # a half-written line is what a kill actually leaves behind
            with led.open("a", encoding="utf-8") as fh:
                fh.write('{"event": "gener')
            second = grow(pool, cases=_cases(8), generations=3, width=2, patience=5,
                          resume_from=str(led), min_margin=0.05)
        self.assertEqual(second["history"][0]["gen"], 1)          # continued, did not restart
        self.assertEqual(second["seed_hash"], first["champion_hash"])

    def test_a_run_that_dies_before_its_first_generation_still_resumes(self):
        # The gap the per-generation checkpoints did not cover, and the one that actually cost a
        # run: dying during the seed measurement left a zero-row ledger, so the 120 calls that
        # had already been paid for were unrecoverable.
        Scripted.WINS = {"a": set(), "b": {"qa1"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(4), generations=0, width=2, ledger_path=str(led))
            ck = load_checkpoint(led)
            self.assertIsNotNone(ck, "no checkpoint after the seed measurement")
            self.assertEqual(ck["gen"], -1)
            resumed = grow(pool, cases=_cases(4), generations=1, width=2,
                           resume_from=str(led), min_margin=0.05)
        self.assertEqual(resumed["history"][0]["gen"], 0)   # starts at the first generation

    def test_resuming_across_a_different_split_is_refused(self):
        Scripted.WINS = {"a": set()}
        pool = {"a": _lane("a")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(4), generations=1, width=1, ledger_path=str(led))
            with self.assertRaises(ValueError):    # sealed cases would not be sealed any more
                grow(pool, cases=_cases(8), generations=2, width=1, resume_from=str(led))

    def test_a_run_that_adds_then_removes_reports_net_zero(self):
        # Measured in run L: `ensemble:integration` was promoted in generation 4 and the shrink
        # gate took the same lane back out in generation 5 (removing it scored HIGHER). The run
        # ended with the seed's champion and "promotions: 2". Counting promotions alone reads
        # as two improvements.
        Scripted.WINS = {"a": {"qa1", "qa2", "qa3", "qa4"}}
        pool = {"a": _lane("a")}
        champ = seed_champion(pool, "a")
        champ["kwargs"]["backends"]["tool(a)"] = {
            "backend": "tool", "_grow_base": "a", "kwargs": {"inner": _lane("a")}}
        champ["kwargs"]["routing_table"]["qa"] = "tool(a)"
        res = grow(pool, cases=_cases(4), seed_spec=canonical(champ), generations=1, width=6,
                   patience=5, min_margin=0.05)
        self.assertEqual(res["promotions"], 1)          # the shrink changed the champion
        self.assertTrue(res["net_change"])              # ...and it differs from the seed
        again = grow(pool, cases=_cases(4), generations=1, width=6, patience=5)
        self.assertFalse(again["net_change"])           # nothing moved at all

    def test_the_ledger_says_which_constraint_set_the_bar(self):
        # Noise-bound and resolution-bound runs need OPPOSITE fixes (more repeats vs more
        # cases), and "delta was 0.05" does not tell them apart. Measured on the AWS runs the
        # drift went to exactly 0.0, so every bar was the floor — and nothing in the ledger
        # said so.
        Scripted.WINS = {"a": set(), "b": {"qa1"}}
        res = self._grow({"a": _lane("a"), "b": _lane("b")}, generations=2, width=2)
        for h in res["history"]:
            self.assertIn(h["bound_by"], ("floor", "drift"))
        self.assertEqual(sum(res["bound_by"].values()), len(res["history"]))
        # a deterministic backend has no drift, so the floor is what binds
        self.assertEqual(res["bound_by"]["drift"], 0, res["bound_by"])

    def test_the_gain_is_reported_in_whole_cases(self):
        # The floor is one case, so the only interpretable unit for a gain is cases. It also
        # corrects a mistake this project published: "more confirm cases" is NOT a general
        # lever, because the floor 1/n and a gain S/n scale together — the ratio is S, the
        # case-equivalents actually improved, and it does not move when you add cases the
        # mutation never touches.
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa3", "qa4"}}
        res = self._grow({"a": _lane("a"), "b": _lane("b")}, generations=1, width=2)
        gen0 = res["history"][0]
        self.assertIn("gain_cases", gen0)
        n_confirm = len(res["splits"]["confirm"])
        expected = round((gen0["challenger_confirm"] - gen0["champion_confirm"]) * n_confirm, 2)
        self.assertEqual(gen0["gain_cases"], expected)

    def test_a_broken_measurement_stops_the_run(self):
        # Run S: the served model was swapped out mid-run, every later call returned 503, and
        # `run_bench` turned each exception into a 0.0. To the loop that looked like a coherent
        # measurement — champion 0.0, challenger 0.0, drift 0.0 — so every gate worked
        # correctly and concluded that dropping a lane cost nothing. The champion and the
        # sealed score were both artefacts of a dead backend, and the ledger looked normal.
        class Broken(ModelBackend):
            available = True

            def __init__(self, tag="x"):
                self.tag, self.last_usage = tag, None

            def complete(self, prompt, tier, **kw):
                raise RuntimeError("503 Loading model")

        backends_mod._BACKENDS["broken"] = Broken
        try:
            with self.assertRaises(MeasurementFailure) as caught:
                grow({"x": {"backend": "broken"}}, cases=_cases(4), generations=1, width=1)
            self.assertIn("broken measurement", str(caught.exception))
        finally:
            backends_mod._BACKENDS.pop("broken", None)

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
            # provenance travels with the numbers, not just in the prose
            self.assertEqual(cfg["grow"]["params"]["repeats"], 1)
            self.assertIn("min_margin", cfg["grow"]["params"])
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

    def test_start_from_seeds_the_loop_with_an_existing_champion(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "champ.json"
            cfg.write_text(json.dumps({"system": {
                "backend": "gama",
                "kwargs": {"backends": {"echo": {"backend": "echo"}},
                           "routing_table": {}, "default": "echo"}}}), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = main(["grow", "--smoke", "--suites", "wide", "--generations", "1",
                           "--width", "1", "--start-from", str(cfg)])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out.getvalue())["seed"]["kwargs"]["default"], "echo")

    def test_start_from_a_broken_config_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "x.json"
            bad.write_text("{nope", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["grow", "--smoke", "--start-from", str(bad)]), 2)

    def test_a_ledger_that_will_not_survive_a_reboot_is_flagged(self):
        # Fourteen runs wrote their ledgers into a /tmp scratch dir. A reboot took every one of
        # them: the evidence behind every published number, and the only thing --resume can
        # read. The tool knew the path and said nothing.
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            main(["grow", "--smoke", "--suites", "wide", "--generations", "1", "--width", "1",
                  "--out", "/tmp/gama-test-ledger.jsonl"])
        self.assertIn("will not survive a reboot", err.getvalue())
        Path("/tmp/gama-test-ledger.jsonl").unlink(missing_ok=True)

    def test_no_ledger_at_all_is_flagged(self):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            main(["grow", "--smoke", "--suites", "wide", "--generations", "1", "--width", "1"])
        self.assertIn("leaves no ledger", err.getvalue())

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


class TestPairedEvidence(unittest.TestCase):
    """平均差の床は「どの問題が split に入ったか」を見ていない。対応のある比較の素材と
    その検定が、床を通った手の**証拠の強さ**を言えることを固定する。"""

    def _m(self, per_case, score=0.5):
        return Measurement(score=score, success_rate=score, latency_s=1.0,
                           n=len(per_case), cases=len(per_case), per_case=per_case)

    def test_sign_test_exact_values(self):
        self.assertAlmostEqual(sign_test(8, 0), 1 / 256)
        self.assertAlmostEqual(sign_test(3, 0), 0.125)
        self.assertAlmostEqual(sign_test(1, 1), 0.75)
        self.assertEqual(sign_test(0, 0), 1.0)          # 差が出た問題が無い = 何も言えない

    def test_sign_test_is_one_sided(self):
        # 負けに偏った側は「偶然でない」と言わない(片側)
        self.assertGreater(sign_test(0, 8), 0.9)

    def test_paired_gain_counts_only_shared_cases(self):
        a = self._m({"x": 0.0, "y": 1.0, "only_a": 0.0})
        b = self._m({"x": 1.0, "y": 1.0, "only_b": 1.0})
        self.assertEqual(paired_gain(a, b), (1, 0, 1))   # x=勝ち y=引き分け、片側だけの case は無視

    def test_paired_gain_uses_tolerance(self):
        a = self._m({"x": 0.5})
        b = self._m({"x": 0.5 + 1e-12})
        self.assertEqual(paired_gain(a, b), (0, 0, 1))   # 浮動小数の遊びは勝ちではない

    def test_gate_unchanged_when_paired_is_off(self):
        # 既定(None)は従来の判定をそのまま通す。この既定を変えると過去の走行と比較できなくなる
        ok, reason = promote_gate(0.5, 0.6, 0.5, 0.6, 0.05)
        self.assertTrue(ok, reason)

    def test_gate_blocks_weak_paired_evidence_when_asked(self):
        ok, reason = promote_gate(0.5, 0.6, 0.5, 0.6, 0.05,
                                  paired=(3, 0, 20), max_paired_p=0.05)
        self.assertFalse(ok)
        self.assertIn("paired-not-significant", reason)
        self.assertIn("3w-0l", reason)

    def test_gate_admits_strong_paired_evidence(self):
        ok, _ = promote_gate(0.5, 0.6, 0.5, 0.6, 0.05,
                             paired=(8, 0, 20), max_paired_p=0.05)
        self.assertTrue(ok)

    def test_paired_condition_cannot_rescue_a_failing_mean(self):
        # 対応のある証拠が強くても、平均差の床を割った手は通らない(条件は AND)
        ok, reason = promote_gate(0.5, 0.6, 0.5, 0.51, 0.05,
                                  paired=(8, 0, 0), max_paired_p=0.05)
        self.assertFalse(ok)
        self.assertIn("below-margin", reason)

    def test_measure_populates_per_case(self):
        spec = {"backend": "echo", "kwargs": {}}
        from gama.benchmark import SUITES
        cases = SUITES["default"][:3]
        m = measure(spec, cases)
        self.assertEqual(set(m.per_case), {c.case_id for c in cases})
        for v in m.per_case.values():
            self.assertIsInstance(v, float)


class TestPairedEvidenceHardening(unittest.TestCase):
    """codex review 由来。検定を要求したのに材料が無い経路と、測定失敗が証拠に化ける経路。"""

    def test_gate_refuses_to_silently_skip_the_paired_condition(self):
        # max_paired_p を渡したのに paired が無いのは fail-open。素通りさせず落とす
        with self.assertRaises(ValueError):
            promote_gate(0.5, 0.6, 0.5, 0.6, 0.05, paired=None, max_paired_p=0.05)

    def test_reason_names_the_most_basic_failure_first(self):
        # 平均差の床も割っている候補は below-margin と記録する(paired が根本原因を隠さない)
        _, reason = promote_gate(0.5, 0.6, 0.5, 0.51, 0.05,
                                 paired=(0, 8, 0), max_paired_p=0.05)
        self.assertIn("below-margin", reason)

    def test_unmeasurable_cases_are_not_counted_as_losses(self):
        champ = Measurement(0.5, 0.5, 1.0, 2, 2, per_case={"ok": 1.0, "boom": 1.0},
                            error_cases=frozenset())
        # challenger は "boom" で例外 -> 0.0。これは負けではなく「測れなかった」
        chal = Measurement(0.5, 0.5, 1.0, 2, 2, per_case={"ok": 1.0, "boom": 0.0},
                           error_cases=frozenset({"boom"}))
        self.assertEqual(paired_gain(champ, chal), (0, 0, 1))


class TestPairedEvidenceSurvivesResume(ScriptedCase):
    """codex は「resume 後も champ_confirm_now を測り直すから壊れない」と読んだ。実際に
    その通りだが、根拠は「毎世代 unconditional に測っている」1 行だけで、あとで測定を節約
    しようとキャッシュした瞬間に**resume した走行でだけ**証拠が消える。しかも p=1.0 が
    出るだけなので赤くならず、--max-paired-p を有効にした人の走行が黙って全却下になる。
    推論でなく実測で固定する。"""

    def test_resumed_run_still_produces_paired_counts(self):
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(8), generations=1, width=2, patience=5,
                 ledger_path=str(led), min_margin=0.05)
            second = grow(pool, cases=_cases(8), generations=3, width=2, patience=5,
                          resume_from=str(led), min_margin=0.05)
        challenged = [h for h in second["history"] if h.get("challenger")]
        self.assertTrue(challenged, "resumed run never challenged anything")
        for h in challenged:
            counts = (h.get("paired_wins"), h.get("paired_losses"), h.get("paired_ties"))
            self.assertNotIn(None, counts, "paired evidence missing after resume")
            # 共有 case が 1 つも見つからない = per_case が復元されていない証拠
            self.assertGreater(sum(counts), 0,
                               "no shared cases after resume: the champion's per-case scores "
                               "were restored from a checkpoint instead of re-measured")


class TestBackendIdentity(unittest.TestCase):
    """run S は載せ替えが 503 を返したから捕まった。200 を返しながら中身だけ入れ替わる
    載せ替えは何も鳴らさず、世代 0 と世代 5 が別モデルの比較になる。"""

    def setUp(self):
        reset_served()

    def tearDown(self):
        reset_served()

    def test_a_stable_backend_reports_no_conflict(self):
        for _ in range(3):
            note_served("aws:8000/kimi", "/models/kimi-48b-IQ2_M.gguf")
        self.assertEqual(served_conflicts(), {})

    def test_a_swapped_backend_is_detected(self):
        note_served("aws:8000/kimi", "/models/kimi-48b-IQ2_M.gguf")
        note_served("aws:8000/kimi", "/models/qwen-7b.gguf")
        self.assertIn("aws:8000/kimi", served_conflicts())

    def test_the_same_model_name_on_two_hosts_is_not_a_conflict(self):
        # 鍵が要求名だけだと、同じ名前を別ホストで出しているだけで偽陽性になる
        note_served("hostA:8000/kimi", "/models/kimi.gguf")
        note_served("hostB:8000/kimi", "/models/kimi-other.gguf")
        self.assertEqual(served_conflicts(), {})

    def test_a_new_lane_adding_a_model_is_not_a_conflict(self):
        # 変異が新しいレーンを足せば新しいモデルが正当に増える。集合の一致で見てはいけない
        note_served("aws:8000/kimi", "/models/kimi.gguf")
        note_served("aws:8000/coder", "/models/coder.gguf")
        self.assertEqual(served_conflicts(), {})

    def test_the_guard_refuses_to_decide_after_a_swap(self):
        m = Measurement(0.9, 0.9, 1.0, 10, 10)
        _guard_measurement(m, "champion")           # 平常時は通る
        note_served("aws:8000/kimi", "/models/a.gguf")
        note_served("aws:8000/kimi", "/models/b.gguf")
        with self.assertRaises(MeasurementFailure) as cm:
            _guard_measurement(m, "champion on confirm")
        self.assertIn("changed under the run", str(cm.exception))

    def test_a_backend_that_cannot_identify_itself_records_nothing(self):
        # echo/null は名乗らない。ここで嘘の保証を作らないこと(空 = 確認できなかった)
        note_served("x", None)
        note_served(None, "y")
        self.assertEqual(served_map(), {})


class SwappingBackend(ModelBackend):
    """途中で中身が入れ替わるサーバの模型。200 を返し続けるので error_rate では鳴らない。"""
    name = "swapping"
    available = True
    CALLS = 0
    SWAP_AFTER = 10_000

    def __init__(self, tag: str = "a"):
        self.tag = tag
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        SwappingBackend.CALLS += 1
        served = ("/models/first.gguf" if SwappingBackend.CALLS <= SwappingBackend.SWAP_AFTER
                  else "/models/second.gguf")
        # レーンごとに別の要求先を名乗る。要求先が 1 つしかない模型だと、複数レーンが
        # 混ざったときに鍵の取り違えで起きる偽陽性/偽陰性を一切踏まない。
        note_served(f"box:8000/{self.tag}", served if self.tag == "a" else "/models/other.gguf")
        # わざと外す。全問正解だと**伸びしろ 0 でクラスが飽和し**、足す変異が提案されなく
        # なって lane b が一度も測られない(= 同一性の試験そのものが空回りする)。ここで
        # 見たいのは載せ替え検知であって精度ではないので、常に伸びしろのある側に置く。
        return "BAD"


class TestBackendIdentityThroughTheLoop(ScriptedCase):
    """helper を直接叩く単体試験は、**検査が呼ばれる場所**の間違いを一切捕まえない。

    実際 `reset_served()` を `grow()` ではなく `propose()` に置いてしまい、世代ごとに観測が
    消えて検査が空回りしていたが、単体試験は全部緑のままだった(codex review が発見)。
    ここでは載せ替えを起こしたうえで **grow() を通して** 止まることを確かめる。
    """

    def setUp(self):
        super().setUp()
        backends_mod._BACKENDS["swapping"] = SwappingBackend
        SwappingBackend.CALLS = 0
        reset_served()

    def tearDown(self):
        backends_mod._BACKENDS.pop("swapping", None)
        SwappingBackend.SWAP_AFTER = 10_000
        reset_served()
        super().tearDown()

    def test_a_mid_run_swap_stops_the_run(self):
        SwappingBackend.SWAP_AFTER = 6        # seed 測定のあと、世代の途中で入れ替わる
        pool = {"a": {"backend": "swapping", "kwargs": {"tag": "a"}},
                "b": {"backend": "swapping", "kwargs": {"tag": "b"}}}
        with self.assertRaises(MeasurementFailure) as cm:
            grow(pool, cases=_cases(4), generations=3, width=2, patience=3, min_margin=0.05)
        self.assertIn("changed under the run", str(cm.exception))

    def test_a_stable_backend_runs_to_completion_and_records_what_it_measured(self):
        # 床が「止める対象自身」を巻き込んでいないことを同時に確かめる(偽陽性ゼロ)
        pool = {"a": {"backend": "swapping", "kwargs": {"tag": "a"}},
                "b": {"backend": "swapping", "kwargs": {"tag": "b"}}}
        result = grow(pool, cases=_cases(4), generations=2, width=2, patience=3, min_margin=0.05)
        self.assertEqual(result["served"], {"box:8000/a": ["/models/first.gguf"],
                                            "box:8000/b": ["/models/other.gguf"]})
        self.assertTrue(result["identity_verified"])

    def test_observations_survive_a_resume(self):
        # resume を跨いだ載せ替えが見えなければ、長い走行ほど検査の抜け道になる
        pool = {"a": {"backend": "swapping", "kwargs": {"tag": "a"}},
                "b": {"backend": "swapping", "kwargs": {"tag": "b"}}}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(4), generations=1, width=2, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            ck = load_checkpoint(led)
            self.assertEqual(ck["served"]["box:8000/a"], ["/models/first.gguf"])
            SwappingBackend.SWAP_AFTER = 0        # 再開後は別モデルが応える
            with self.assertRaises(MeasurementFailure):
                grow(pool, cases=_cases(4), generations=3, width=2, patience=3,
                     resume_from=str(led), min_margin=0.05)

    def test_resuming_a_ledger_without_identity_says_so_instead_of_claiming_it(self):
        # この機能より前に書かれた台帳(run O〜T)には served が無い。空から再開して黙って
        # 「検査済み」を名乗ると、未検証が検証済みに化ける。肯定形で持つ。
        pool = {"a": {"backend": "swapping", "kwargs": {"tag": "a"}},
                "b": {"backend": "swapping", "kwargs": {"tag": "b"}}}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "old.jsonl"
            grow(pool, cases=_cases(4), generations=1, width=2, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            # served を落とした古い形式の台帳に書き換える
            rows = [json.loads(l) for l in led.read_text(encoding="utf-8").splitlines() if l.strip()]
            for r in rows:
                r.pop("served", None)
            led.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                           encoding="utf-8")
            reset_served()
            out = grow(pool, cases=_cases(4), generations=2, width=2, patience=3,
                       resume_from=str(led), min_margin=0.05)
        self.assertFalse(out["identity_verified"])

    def test_an_unidentifiable_pool_is_not_marked_unverified(self):
        # echo だけの走行は「名乗れない」のであって「怪しい」のではない。偽の警告を出さない
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "old.jsonl"
            pool = {"e1": {"backend": "echo", "kwargs": {}}, "e2": {"backend": "echo", "kwargs": {}}}
            grow(pool, cases=_cases(4), generations=1, width=2, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            rows = [json.loads(l) for l in led.read_text(encoding="utf-8").splitlines() if l.strip()]
            for r in rows:
                r.pop("served", None)
            led.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                           encoding="utf-8")
            out = grow(pool, cases=_cases(4), generations=2, width=2, patience=3,
                       resume_from=str(led), min_margin=0.05)
        self.assertTrue(out["identity_verified"])

    def test_a_crash_after_a_blind_resume_does_not_launder_it_into_verified(self):
        # 未検証の境界は系統に属する。blind な再開のあと落ちて再々開すると、その checkpoint
        # には以降の世代の served が載っているので、引き継がないと「確かめた」ことにされる。
        pool = {"a": {"backend": "swapping", "kwargs": {"tag": "a"}},
                "b": {"backend": "swapping", "kwargs": {"tag": "b"}}}
        with tempfile.TemporaryDirectory() as d:
            old_led = Path(d) / "old.jsonl"
            grow(pool, cases=_cases(4), generations=1, width=2, patience=3,
                 ledger_path=str(old_led), min_margin=0.05)
            rows = [json.loads(l) for l in old_led.read_text(encoding="utf-8").splitlines() if l.strip()]
            for r in rows:                                    # 古い形式(served 無し)にする
                r.pop("served", None)
            old_led.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                               encoding="utf-8")
            reset_served()
            mid = Path(d) / "mid.jsonl"                       # blind な再開。ここで落ちたとする
            out1 = grow(pool, cases=_cases(4), generations=2, width=2, patience=3,
                        resume_from=str(old_led), ledger_path=str(mid), min_margin=0.05)
            self.assertFalse(out1["identity_verified"])
            reset_served()
            out2 = grow(pool, cases=_cases(4), generations=3, width=2, patience=3,
                        resume_from=str(mid), min_margin=0.05)
        self.assertFalse(out2["identity_verified"],
                         "an unverified boundary was laundered away by a second resume")


class TestSealedVerdict(unittest.TestCase):
    """昇格数は「何手通したか」であって「良くなったか」ではない。run T は sealed が下がって
    いるのに『昇格 1・成功』として champion を出した。数字は台帳に在ったが誰も判定していない。"""

    def _s(self, seed, champ, n=28):
        return {"seed": {"score": seed, "cases": n}, "champion": {"score": champ, "cases": n}}

    def test_a_real_gain_is_called_improved(self):
        # 実測(WSL llama3.2:3b): 0.6375 -> 0.8542 on 20 = +4.33 問
        v = sealed_verdict(self._s(0.6375, 0.8542, n=20))
        self.assertEqual(v["verdict"], "improved")
        self.assertAlmostEqual(v["delta_cases"], 4.33, places=1)

    def test_a_drop_beyond_resolution_is_called_regressed(self):
        v = sealed_verdict(self._s(0.90, 0.80))
        self.assertEqual(v["verdict"], "regressed")
        self.assertIn("Do not adopt", v["note"])

    def test_a_sub_case_move_is_not_separable_in_either_direction(self):
        # run T: -0.5 問。run R: +0.33 問。どちらも「分からない」であって成功でも失敗でもない
        self.assertEqual(sealed_verdict(self._s(0.8393, 0.8214))["verdict"], "not-separable")
        self.assertEqual(sealed_verdict(self._s(0.8611, 0.8750, n=24))["verdict"], "not-separable")

    def test_no_sealed_split_is_reported_as_unsealed_not_as_success(self):
        v = sealed_verdict(None)
        self.assertEqual(v["verdict"], "unsealed")
        self.assertIsNone(v["delta_cases"])

    def test_the_verdict_reaches_the_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            result = {"champion": {"backend": "echo", "kwargs": {}},
                      "seed": {"backend": "echo", "kwargs": {}},
                      "champion_hash": "deadbeef", "seed_hash": "cafe",
                      "promotions": 0, "net_change": False, "bound_by": {},
                      "generations_run": 0, "archive_size": 0,
                      "sealed": self._s(0.90, 0.80), "sealed_verdict": sealed_verdict(self._s(0.90, 0.80)),
                      "splits": {"search": [], "confirm": [], "sealed": []},
                      "search": {"score": 0.9}, "confirm": {"score": 0.9}, "params": {}}
            out = write_recipe(result, Path(d) / "r")
            text = (out / "recipe.md").read_text(encoding="utf-8")
        self.assertIn("REGRESSED", text)
        self.assertIn("do not adopt", text.lower())

    def test_a_malformed_sealed_block_is_not_judged_instead_of_crashing(self):
        # write_recipe は任意の result に対してこれを呼ぶ。古い台帳で例外を投げると
        # recipe が書けなくなる。判定できないものは判定しない
        for bad in ({"seed": {}}, {"seed": {"score": 1.0}, "champion": {"score": 1.0}},
                    {"seed": {"score": "x", "cases": 4}, "champion": {"score": 1.0, "cases": 4}},
                    {"seed": "not a dict", "champion": {"score": 1.0, "cases": 4}},
                    {"seed": [1], "champion": [2]},
                    {"seed": {"score": 1.0, "cases": 0}, "champion": {"score": 1.0, "cases": 0}},
                    {"seed": {"score": float("nan"), "cases": 4},
                     "champion": {"score": 1.0, "cases": 4}},
                    {"seed": {"score": True, "cases": 4}, "champion": {"score": 1.0, "cases": 4}}):
            with self.subTest(bad=bad):
                self.assertEqual(sealed_verdict(bad)["verdict"], "unjudgeable")
        # 「split が無い」と「在るが比べられない」は別物として残す
        self.assertEqual(sealed_verdict(None)["verdict"], "unsealed")
        self.assertEqual(sealed_verdict({})["verdict"], "unsealed")

    def test_a_mismatched_case_count_is_refused_not_scaled(self):
        # 分母が違えば差は問数に直せない。丸めて出すと嘘の「何問差」になる
        v = sealed_verdict({"seed": {"score": 0.5, "cases": 20},
                            "champion": {"score": 0.9, "cases": 28}})
        self.assertEqual(v["verdict"], "unjudgeable")
        self.assertIsNone(v["delta_cases"])
        self.assertIn("not a comparison", v["note"])


def _lane_spec(tag):
    return {"backend": "gama", "kwargs": {
        "backends": {tag: {"backend": "echo", "kwargs": {}}},
        "routing_table": {"qa": tag}, "default_backend": tag}}


def _tool_spec(tag):
    return {"backend": "gama", "kwargs": {
        "backends": {tag: {"backend": "echo", "kwargs": {}},
                     "t": {"backend": "tool", "kwargs": {"inner": {"backend": "echo", "kwargs": {}}},
                           "_grow_base": tag}},
        "routing_table": {"qa": "t"}, "default_backend": tag}}


class TestSelectionIsDeterministic(unittest.TestCase):
    """同点の候補を実測レイテンシで割ると、走行が決定的でなくなる。壁時計は走るたび違い、
    共有 GPU なら他人の負荷でも動くので、「たまたま空いている時に測られた」候補が勝つ。
    determinism テストは 21 回に 1 回しか落ちず、通常の 1 回実行では見えなかった。"""

    def _cand(self, label, spec):
        return Candidate(label=label, kind="route", spec=spec)

    def _m(self, score, latency):
        return Measurement(score=score, success_rate=score, latency_s=latency, n=4, cases=4)

    def test_the_key_is_unchanged_by_measured_latency(self):
        # 文字列検査ではなく振る舞いで見る(別名で実測値を読んでも捕まえられるように)
        c = self._cand("route:qa->b", _lane_spec("b"))
        self.assertEqual(_challenger_key(c, self._m(0.9, 0.001)),
                         _challenger_key(c, self._m(0.9, 12.5)))

    def test_latency_cannot_outrank_structure_or_label(self):
        cheap_slow = (self._cand("z:bare", _lane_spec("b")), self._m(0.9, 99.0))
        rich_fast = (self._cand("a:tool", _tool_spec("b")), self._m(0.9, 0.001))
        # 速く測れた方ではなく、構造の小さい方が勝つ
        self.assertEqual(min([cheap_slow, rich_fast], key=lambda t: _challenger_key(*t))[0].label,
                         "z:bare")

    def test_equal_score_and_structure_fall_back_to_label(self):
        a = (self._cand("a:one", _lane_spec("b")), self._m(0.9, 5.0))
        b = (self._cand("b:two", _lane_spec("b")), self._m(0.9, 0.1))
        self.assertEqual(min([a, b], key=lambda t: _challenger_key(*t))[0].label, "a:one")

    def test_a_higher_score_still_wins_over_a_smaller_structure(self):
        # タイブレークは同点のときだけ。点の差を構造で覆さない
        better = (self._cand("z:tool", _tool_spec("b")), self._m(0.95, 9.0))
        smaller = (self._cand("a:bare", _lane_spec("b")), self._m(0.90, 0.1))
        self.assertEqual(min([better, smaller], key=lambda t: _challenger_key(*t))[0].label,
                         "z:tool")

    def test_structure_size_ranks_a_bare_lane_below_a_composite_one(self):
        self.assertLess(_structure_size(_lane_spec("b")), _structure_size(_tool_spec("b")))


class TestSaturatedClasses(ScriptedCase):
    """伸びしろの尽きたクラスに変異を当てるのは、確実に無駄と分かっている測定に実モデルを
    焚くこと。実測(run U seed): integration は 8/8 満点=伸びしろ 0.00 問なのに
    `tool:integration` が候補に出ていた。床(1 問)を越えられないことは算術で言える。"""

    def test_headroom_is_counted_per_class_in_cases(self):
        cases = _cases(4, "qa", "qa") + _cases(4, "research", "re")
        m = Measurement(0.5, 0.5, 1.0, 8, 8,
                        per_case={**{f"qa{i}": 1.0 for i in range(1, 5)},
                                  **{f"re{i}": 0.5 for i in range(1, 5)}})
        h = class_headroom(m, cases)
        self.assertAlmostEqual(h["qa"], 0.0)          # 満点 = 伸びしろ無し
        self.assertAlmostEqual(h["research"], 2.0)    # 0.5 x 4 問

    def test_no_per_case_means_no_exclusion(self):
        # 飽和を**証明できない**ときは除外しない(古い checkpoint からの再開など)
        self.assertEqual(class_headroom(Measurement(0.9, 0.9, 1.0, 4, 4), _cases(4)), {})

    def test_a_saturated_class_is_not_mutated(self):
        # qa 全問正解 = 伸びしろ 0。qa への変異は 1 問の床を越えようがないので提案しない
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)},
                         "b": {f"qa{i}" for i in range(1, 9)}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        seen = []
        grow(pool, cases=_cases(8), generations=2, width=4, patience=3, min_margin=0.05,
             on_event=lambda r: seen.append(r))
        kinds = {r["kind"] for r in seen if r["event"] == "candidate"}
        self.assertNotIn("route", kinds, "measured an additive mutation on a saturated class")
        self.assertNotIn("tool", kinds)
        self.assertNotIn("ensemble", kinds)
        self.assertTrue([r for r in seen if r["event"] == "saturated"])

    def test_a_saturated_class_can_still_be_shrunk(self):
        # 門が非対称なので対象も非対称。削る側は「悪くならないこと」しか要求せず、満点の
        # クラスこそ「その構造は何も買っていない」と言える場所になる。ここを塞ぐと、
        # 一度足した構造が二度と外れなくなる。
        champ = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"),
                         "tool(a)": {"backend": "tool", "kwargs": {"inner": _lane("a")},
                                     "_grow_base": "a"}},
            "routing_table": {"qa": "tool(a)"}, "default": "a"}}
        cands = propose(champ, {"a": _lane("a"), "b": _lane("b")}, ["qa"], width=6,
                        additive_classes=[])          # qa は飽和している
        kinds = {c.kind for c in cands}
        self.assertIn("simplify", kinds, "a saturated class could not be shrunk")
        self.assertFalse(kinds & {"route", "tool", "ensemble", "meshflow", "deepen"},
                         f"additive mutation proposed for a saturated class: {kinds}")

    def test_a_class_with_room_is_still_mutated(self):
        # 床が「止める対象自身」を巻き込んでいないことを確かめる(偽陽性ゼロ)
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        seen = []
        grow(pool, cases=_cases(8), generations=1, width=4, patience=3, min_margin=0.05,
             on_event=lambda r: seen.append(r))
        self.assertTrue([r for r in seen if r["event"] == "candidate"],
                        "a class with headroom was skipped")

    def test_a_default_swap_needs_the_SUM_of_headroom_under_the_default(self):
        # レーン変異は 1 クラスしか触らないのでクラス単位で切れるが、既定の差し替えは既定に
        # 落ちている全クラスへ同時に効く。個別には門を越えなくても合計で越えるなら消さない。
        champ = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"), "b": _lane("b")},
            "routing_table": {}, "default": "a"}}
        cls = ["qa", "research"]
        # 各 1.5 問・門 2 問 -> 個別は飽和だが合計 3 問なので通す
        self.assertTrue(_default_swap_viable(champ, cls, {"qa": 1.5, "research": 1.5}, 2.0))
        # 合計 1.2 問 < 門 2 問 -> どう転んでも越えない
        self.assertFalse(_default_swap_viable(champ, cls, {"qa": 0.6, "research": 0.6}, 2.0))

    def test_a_default_swap_ignores_headroom_that_is_not_under_the_default(self):
        # qa は明示ルートなので既定を替えても動かない。その伸びしろを足してはいけない
        champ = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"), "b": _lane("b")},
            "routing_table": {"qa": "b"}, "default": "a"}}
        self.assertFalse(_default_swap_viable(champ, ["qa", "research"],
                                              {"qa": 9.0, "research": 0.1}, 2.0))

    def test_unmeasurable_headroom_keeps_the_default_swap(self):
        champ = {"backend": "gama", "kwargs": {"backends": {"a": _lane("a")},
                                               "routing_table": {}, "default": "a"}}
        self.assertTrue(_default_swap_viable(champ, ["qa"], {}, 2.0))

    def test_saturation_still_applies_after_a_resume(self):
        # 飽和判定を復元状態から作っていた版では、再開のたびにこの除外が黙って無効化されて
        # いた(codex 指摘)。confirm 側は毎世代測り直すので復元値に依存しない。
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)},
                         "b": {f"qa{i}" for i in range(1, 9)}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(8), generations=1, width=4, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            seen = []
            grow(pool, cases=_cases(8), generations=2, width=4, patience=3,
                 resume_from=str(led), min_margin=0.05, on_event=lambda r: seen.append(r))
        self.assertTrue([r for r in seen if r["event"] == "saturated"],
                        "saturation was not detected after a resume")
        kinds = {r["kind"] for r in seen if r["event"] == "candidate"}
        self.assertFalse(kinds & {"route", "tool", "ensemble", "meshflow", "deepen"},
                         f"additive mutation measured on a saturated class after resume: {kinds}")

    def test_a_default_swap_survives_when_headroom_only_adds_up_across_classes(self):
        # 既定レーンの差し替えは既定に落ちている全クラスへ同時に効く。各クラス単独では門を
        # 越えなくても、合計で越えるなら消してはいけない(codex 指摘)。
        champ = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"), "b": _lane("b")},
            "routing_table": {}, "default": "a"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        self.assertIn("default",
                      {c.kind for c in propose(champ, pool, ["qa", "research"], width=8,
                                               additive_classes=[], allow_default=True)})
        self.assertNotIn("default",
                         {c.kind for c in propose(champ, pool, ["qa", "research"], width=8,
                                                  additive_classes=[], allow_default=False)})

    def test_an_unmeasured_class_under_the_default_is_not_read_as_zero_headroom(self):
        # 欠損を 0 と読むと「測っていない」が「伸びしろ無し」に化ける
        champ = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a")}, "routing_table": {}, "default": "a"}}
        self.assertTrue(_default_swap_viable(champ, ["qa", "research"], {"qa": 0.1}, 2.0))

    def test_stopping_for_want_of_candidates_still_writes_a_checkpoint(self):
        # 変数を更新しただけでは再開で古い値に戻る(load_checkpoint が読むのは checkpoint 行)
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(8), generations=3, width=4, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            rows = [json.loads(l) for l in led.read_text(encoding="utf-8").splitlines() if l.strip()]
        # 前提そのものを assert する。条件付き assert は、条件に入らなかった走行で**何も
        # 確かめずに緑**になる(赤くならずに黙って死ぬ試験になる)。
        stops = [r for r in rows if r["event"] == "stop"]
        self.assertTrue(stops, "the run never stopped, so this test checked nothing")
        self.assertEqual(stops[0]["reason"], "no-new-candidates")
        gens = [r["gen"] for r in rows if r["event"] == "checkpoint"]
        self.assertIn(stops[0]["gen"], gens,
                      "stopped without checkpointing the confirm measurement it paid for")

    def test_the_run_reports_where_the_headroom_is(self):
        # 「変えているクラスに case を足せ」を一般論でなく実測で言えるようにする
        Scripted.WINS = {"a": {"qa1", "qa2"}, "b": {"qa1", "qa2", "qa3"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        res = grow(pool, cases=_cases(8), generations=1, width=3, patience=3, min_margin=0.05)
        self.assertIn("qa", res["headroom"])
        self.assertGreater(res["headroom"]["qa"], 0.0)

    def test_headroom_only_covers_classes_the_loop_can_mutate(self):
        # confirm には居るが search に居ないクラスは触れない。そこに余地が無いと言われても
        # 打てる手が無いので、警告の対象にしない
        cases = _cases(8, "qa", "qa") + _cases(2, "research", "re")
        Scripted.WINS = {"a": {"qa1"}, "b": {"qa1", "qa2"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        res = grow(pool, cases=cases, generations=1, width=3, patience=3, min_margin=0.05)
        self.assertTrue(set(res["headroom"]) <= set(res["params"].get("classes", res["headroom"])
                                                    or res["headroom"]))
        for cls in res["headroom"]:
            self.assertIn(cls, {c.task_type for c in cases})

    def test_a_class_the_champion_aces_on_search_is_not_mutated_either(self):
        # 挑戦権は search で**厳密に**上回ること。チャンピオンがそのクラスの search case を
        # 取り切っていれば、どの変異も 1 点も上げられず必ず search-not-better で落ちる。
        # confirm 側の床とは別の理由なので、両方見ないと片方から漏れる(run U gen1 で実測)。
        cases = _cases(8, "qa", "qa") + _cases(8, "research", "re")
        # qa は search でも confirm でも満点、research は落とす
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)},
                         "b": {f"qa{i}" for i in range(1, 9)}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        seen = []
        grow(pool, cases=cases, generations=1, width=6, patience=3, min_margin=0.05,
             on_event=lambda r: seen.append(r))
        sat = [r for r in seen if r["event"] == "saturated"]
        self.assertTrue(sat, "a class the champion aces was not reported saturated")
        self.assertIn("qa", sat[0]["classes"])
        for r in (r for r in seen if r["event"] == "candidate"):
            self.assertFalse(r["label"].endswith(":qa") or ":qa(" in r["label"]
                             or r["label"].startswith("route:qa"),
                             f"measured an additive mutation on an aced class: {r['label']}")

    def test_search_saturation_still_applies_after_a_resume(self):
        # per_case を checkpoint から落としていたせいで、再開後に静かに死ぬ機能を 3 回作った。
        # 3 回目は「見せる形」と「続きを走らせる形」を分けて直したので、そこを固定する。
        cases = _cases(8, "qa", "qa") + _cases(8, "research", "re")
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)},
                         "b": {f"qa{i}" for i in range(1, 9)}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=cases, generations=1, width=6, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            ck = load_checkpoint(led)
            self.assertTrue(ck["champion_search"].get("per_case"),
                            "the checkpoint dropped the per-case scores a resume needs")
            seen = []
            grow(pool, cases=cases, generations=2, width=6, patience=3,
                 resume_from=str(led), min_margin=0.05, on_event=lambda r: seen.append(r))
        sat = [r for r in seen if r["event"] == "saturated"]
        self.assertTrue(sat, "saturation was not detected after a resume")
        self.assertIn("qa", sat[0]["classes"])

    def test_a_default_swap_also_needs_search_headroom(self):
        champ = {"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"), "b": _lane("b")},
            "routing_table": {}, "default": "a"}}
        cls = ["qa", "research"]
        room = {"qa": 3.0, "research": 3.0}
        self.assertTrue(_default_swap_viable(champ, cls, room, 2.0, {"qa": 1.0, "research": 0.0}))
        # search で取り切っているなら挑戦権が取れない = confirm に余地があっても通らない
        self.assertFalse(_default_swap_viable(champ, cls, room, 2.0, {"qa": 0.0, "research": 0.0}))

    def test_the_cli_says_at_gen0_when_no_class_can_grow(self):
        # 走行の最後に「区別できなかった」と知るのは高い(実走で数時間)。打つ手が無いことは
        # 候補を1つも測る前に分かるので、その時点で言う。実 suite は scripted backend では
        # 満点にできないので、この試験だけ registry に一時 suite を挿す。
        from gama.benchmark import SUITES
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)},
                         "b": {f"qa{i}" for i in range(1, 9)}}
        had = "_sat_probe" in SUITES
        prior = SUITES.get("_sat_probe")
        SUITES["_sat_probe"] = _cases(8)
        try:
            pool = {"a": _lane("a"), "b": _lane("b")}
            with tempfile.TemporaryDirectory() as d:
                Path(d, "pool.json").write_text(json.dumps(pool), encoding="utf-8")
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    cli_main(["grow", "--pool", str(Path(d, "pool.json")),
                              "--suites", "_sat_probe", "--generations", "2", "--width", "3",
                              "--min-margin", "0.05", "--out", str(Path(d, "l.jsonl"))])
            out = err.getvalue()
        finally:
            if had:
                SUITES["_sat_probe"] = prior
            else:
                SUITES.pop("_sat_probe", None)
        self.assertIn("saturated", out)
        self.assertIn("EVERY class this run can mutate", out)
        # **候補を1つも測る前に**言っていること。ここが機能の全部(最後に分かっても遅い)。
        # `out.index(...) < len(out)` のような常に真の比較にしないこと —— それは何も
        # 確かめずに緑になる(このセッションで一度やった)。
        # candidate 行は "[gama]  gen0 <label>  search=" 形式(空白2つ)。seed 行にも
        # "search=" は出るので、そちらに当たると常に真の比較になって試験が死ぬ。
        first_candidate = out.find("[gama]  gen")
        if first_candidate != -1:
            self.assertLess(out.index("saturated"), first_candidate,
                            "the saturation warning came after candidates were measured")

    def test_a_promoted_cached_candidate_keeps_its_per_case_scores(self):
        # archive は「表示」でなく「状態」。痩せた形で入れると、キャッシュ由来の候補が昇格した
        # 瞬間に champ_search の per_case が消え、search 側の飽和判定が残り全部で効かなくなる。
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa3"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        seen = []
        grow(pool, cases=_cases(8), generations=3, width=3, patience=3, min_margin=0.05,
             on_event=lambda r: seen.append(r))
        ck = [r for r in seen if r["event"] == "checkpoint" and r.get("archive")]
        self.assertTrue(ck, "no checkpoint carried an archive to check")
        for entry in ck[-1]["archive"].values():
            self.assertTrue(entry["search"].get("per_case"),
                            "an archived candidate lost its per-case scores")
