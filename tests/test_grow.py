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
from gama.grow import (Candidate, _challenger_key, _default_swap_viable, _prescribed,
                       _structure_size, class_headroom, propose, search_gate, spec_hash,
                       split_cases)
from gama.backends import ModelBackend
from gama.benchmark import BenchCase
from gama.cli import build_parser, main
from gama.config import build_backend, system_from_config
from gama.backends import (note_served, note_tool, reset_served, reset_tool_stats,
                           served_conflicts, served_map, tool_stats)
from gama.grow import (
    MeasurementFailure,
    sealed_verdict,
    promoted_gain_cases,
    confirm_claim,
    _guard_measurement,
    Measurement,
    paired_gain,
    remeasure_sd,
    sign_test,
    canonical,
    load_checkpoint,
    _ledger_splits,
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
    SEEN_BY: list = []          # (tag, cid): which lane was asked which case

    def __init__(self, tag: str = "a"):
        self.tag = tag
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        cid = prompt.split("case=")[1].split()[0] if "case=" in prompt else "?"
        Scripted.SEEN.append(cid)
        Scripted.SEEN_BY.append((self.tag, cid))
        if cid in Scripted.PARTIAL.get(self.tag, set()):
            return "HALF"
        return "GOOD" if cid in Scripted.WINS.get(self.tag, set()) else "BAD"


class Flaky(Scripted):
    """A stand-in for a temperature>0 lane: on every case it answers GOOD on odd calls and
    BAD on even ones, so a 2-repeat measurement sees [1, 0] on each case (mean 0.5, and a
    known spread), while a 1-repeat measurement sees a coin whose noise it cannot estimate."""
    name = "flaky"
    CALLS: dict = {}

    def complete(self, prompt, tier, **kw):
        cid = prompt.split("case=")[1].split()[0] if "case=" in prompt else "?"
        Scripted.SEEN.append(cid)
        Scripted.SEEN_BY.append((self.tag, cid))
        k = Flaky.CALLS.get((self.tag, cid), 0) + 1
        Flaky.CALLS[(self.tag, cid)] = k
        return "GOOD" if k % 2 == 1 else "BAD"


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
        backends_mod._BACKENDS["flaky"] = Flaky
        Scripted.WINS, Scripted.SEEN, Scripted.PARTIAL = {}, [], {}
        Scripted.SEEN_BY = []
        Flaky.CALLS = {}

    def tearDown(self):
        backends_mod._BACKENDS.pop("scripted", None)
        backends_mod._BACKENDS.pop("flaky", None)
        Scripted.WINS, Scripted.SEEN, Scripted.PARTIAL = {}, [], {}
        Scripted.SEEN_BY = []
        Flaky.CALLS = {}


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

    def test_empty_search_is_refused(self):
        # A ratio with no search share leaves the split empty; the search band (1 / n_search)
        # would then divide by zero. Refuse here, in positive form, without relying on how
        # _allocate rounds a one-case class.
        with self.assertRaisesRegex(ValueError, "search split is empty"):
            split_cases(_cases(4), ratio=(0, 1, 0))

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

    def test_a_tool_lane_gets_a_prefill_as_its_next_step_only_where_it_can_land(self):
        # Kimi-48B on crux research never opened a fence (0/3 at any temperature / max_tokens);
        # opening it for the model did. That is a one-step refinement OF a tool lane, offered
        # only when the inner backend declares it honours a prefill — a candidate the loop can
        # propose but not build is worse than none.
        class Chatty(Scripted):
            name = "chatty"
            supports_prefill = True

        backends_mod._BACKENDS["chatty"] = Chatty
        try:
            pool = {"a": {"backend": "chatty", "kwargs": {"tag": "a"}}, "b": _lane("b")}
            champ = seed_champion(pool, "a")
            champ["kwargs"]["backends"]["tool(a)"] = {
                "backend": "tool", "_grow_base": "a", "kwargs": {"inner": pool["a"]}}
            champ["kwargs"]["routing_table"]["qa"] = "tool(a)"
            tools = {c.label: c for c in propose(canonical(champ), pool, ["qa"], width=20)
                     if c.kind == "tool"}
            self.assertEqual(list(tools), ["tool:qa(a)+prefill"])
            self.assertEqual(tools["tool:qa(a)+prefill"].remedy, "qa")   # what _prescribed reads
            spec = tools["tool:qa(a)+prefill"].spec
            lane = spec["kwargs"]["routing_table"]["qa"]
            self.assertEqual(lane, "tool(a)+pf")
            self.assertEqual(spec["kwargs"]["backends"][lane]["kwargs"]["prefill"], "```python\n")
            self.assertEqual(spec["kwargs"]["backends"][lane]["_grow_base"], "a")
            be = build_backend(spec)
            self.assertEqual(be.backends[lane].prefill, "```python\n")
            # a lane that already carries it is not offered it again, but can still be simplified
            kinds = {c.kind: c for c in propose(spec, pool, ["qa"], width=20)}
            self.assertNotIn("tool", kinds)
            self.assertEqual(kinds["simplify"].label, "simplify:qa->a")
            # and a user pool lane in that minted namespace is refused like the others
            with self.assertRaises(ValueError):
                validate_pool({"a": _lane("a"), "tool(a)+pf": _lane("b")})
        finally:
            backends_mod._BACKENDS.pop("chatty", None)
        # an inner that does not declare support (Scripted) gets no prefill candidate at all
        champ = seed_champion(self.pool, "a")
        champ["kwargs"]["backends"]["tool(a)"] = {
            "backend": "tool", "_grow_base": "a", "kwargs": {"inner": _lane("a")}}
        champ["kwargs"]["routing_table"]["qa"] = "tool(a)"
        self.assertFalse([c for c in propose(canonical(champ), self.pool, ["qa"], width=20)
                          if c.kind == "tool"])

    def test_the_diagnosis_puts_the_prefill_for_the_symptomatic_class_first(self):
        # run W: the seed's confirm measurement said no_code 8/72 (all research), and gen1's tool
        # slot still went to qa's prefill by rotation; replaying propose() from the checkpoint
        # put the research prefill in gen2. The measurement already names the class; the queue
        # must read it rather than the position the rotation happened to start at.
        class Chatty(Scripted):
            name = "chatty"
            supports_prefill = True

        backends_mod._BACKENDS["chatty"] = Chatty
        try:
            pool = {"a": {"backend": "chatty", "kwargs": {"tag": "a"}}, "b": _lane("b")}
            champ = seed_champion(pool, "a")
            champ["kwargs"]["backends"]["tool(a)"] = {
                "backend": "tool", "_grow_base": "a", "kwargs": {"inner": pool["a"]}}
            for cls in ("qa", "research", "content"):
                champ["kwargs"]["routing_table"][cls] = "tool(a)"
            champ = canonical(champ)
            classes = ["content", "qa", "research"]

            def tool_order(**kw):
                return [c.label for c in propose(champ, pool, classes, width=30, **kw)
                        if c.kind == "tool"]

            plain = tool_order()
            self.assertEqual(sorted(plain), ["tool:content(a)+prefill", "tool:qa(a)+prefill",
                                             "tool:research(a)+prefill"])
            # no diagnosis, or one with nothing in it: the rotation is untouched
            self.assertEqual(tool_order(no_code_by_class={}), plain)
            self.assertEqual(tool_order(no_code_by_class={"qa": 0, "research": False}), plain)
            # a symptom on one class moves that class's prefill to the front, nothing else moves
            got = tool_order(no_code_by_class={"research": 3})
            self.assertEqual(got[0], "tool:research(a)+prefill")
            self.assertEqual(got[1:], [l for l in plain if l != "tool:research(a)+prefill"])
            # more symptoms first; equal counts by class name
            got = tool_order(no_code_by_class={"qa": 2, "research": 5, "content": 2})
            self.assertEqual(got, ["tool:research(a)+prefill", "tool:content(a)+prefill",
                                   "tool:qa(a)+prefill"])
            # a diagnosed class that has no prefill to offer (not on a tool lane) changes nothing
            self.assertEqual(tool_order(no_code_by_class={"integration": 9}), plain)
            # the other kinds keep their order: the diagnosis decides within the tool slot only
            other = lambda **kw: [c.label for c in propose(champ, pool, classes, width=30, **kw)
                                  if c.kind != "tool"]
            self.assertEqual(other(no_code_by_class={"research": 3}), other())
            # and it does not wait for the tool kind's turn in the rotation: at width 1 the seat
            # goes to simplify (gen0), route (gen1), ensemble (gen3); the prescription takes it
            # first, a measured one is listed without taking it, a settled one is gone
            pres = next(c for c in propose(champ, pool, classes, width=30)
                        if c.label == "tool:research(a)+prefill")
            for gen, seat in ((0, "simplify:content->a"), (1, "route:research->a"),
                              (3, "ensemble:content(a+b)")):
                one = lambda **kw: [c.label for c in propose(champ, pool, classes, width=1,
                                                             generation=gen, **kw)]
                self.assertEqual(one(), [seat])
                self.assertEqual(one(no_code_by_class={"research": 3}),
                                 ["tool:research(a)+prefill"])
                self.assertEqual(one(no_code_by_class={"research": 3},
                                     archived={spec_hash(pres.spec)}),
                                 ["tool:research(a)+prefill", seat])
                self.assertEqual(one(no_code_by_class={"research": 3},
                                     exclude={spec_hash(pres.spec)}), [seat])
        finally:
            backends_mod._BACKENDS.pop("chatty", None)

    def test_a_deepened_lane_is_not_a_dead_end(self):
        # Until 2026-09-02 a name-parsing copy of _atomic_lane shadowed the spec-reading one, and
        # `mesh(a->b)+tool` matched neither pattern: a class that had been deepened could never be
        # simplified back, wrapped, or recombined — only routed away. Every direction must stay open.
        champ = json.loads(json.dumps(self.champ))
        champ["kwargs"]["backends"]["mesh(a->b)+tool"] = {
            "backend": "meshflow", "_grow_base": "a",
            "kwargs": {"tiers": [{"backend": "tool", "kwargs": {"inner": _lane("a")}}, _lane("b")],
                       "mesh": "union"}}
        champ["kwargs"]["routing_table"]["qa"] = "mesh(a->b)+tool"
        cands = propose(canonical(champ), self.pool, ["qa"], width=20)
        labels = {c.label for c in cands}
        self.assertIn("simplify:qa->a", labels)
        self.assertIn("tool:qa(a)", labels)
        self.assertIn("ensemble:qa(a+b)", labels)
        # and it is not wrapped a second time (the guard reads the spec, not the lane name)
        self.assertFalse([c for c in cands if c.kind == "deepen"], labels)
        # a user-named lane outside grow's namespace is never decomposed by name
        champ["kwargs"]["backends"]["foo(a)"] = _lane("b")
        champ["kwargs"]["routing_table"]["qa"] = "foo(a)"
        self.assertNotIn("simplify:qa->a",
                         {c.label for c in propose(canonical(champ), self.pool, ["qa"], width=20)})

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
    def test_a_measured_design_takes_no_width_slot_and_every_one_comes_back(self):
        # run W replayed from its checkpoint: measured ties took the slots, new designs per
        # generation went 4,3,2,3,1,0, and a stepping stone came back only when the rotation
        # happened to land on it (the research prefill never did in six generations).
        pool = {"a": _lane("a"), "b": _lane("b"), "c": _lane("c")}
        champ = seed_champion(pool, "a")
        everything = propose(champ, pool, ["qa"], width=50)
        hashes = [spec_hash(c.spec) for c in everything]
        self.assertGreater(len(hashes), 5)
        measured = set(hashes[:3])
        out = propose(champ, pool, ["qa"], width=2, archived=measured)
        got = [spec_hash(c.spec) for c in out]
        self.assertEqual(len([h for h in got if h not in measured]), 2)   # width counts new ones
        self.assertEqual(set(got) & measured, measured)                   # all measured come back
        self.assertEqual(len(got), len(set(got)))
        # a measured design that is settled or challenged stays out
        out = propose(champ, pool, ["qa"], width=2, archived=measured, exclude={hashes[0]})
        self.assertNotIn(hashes[0], {spec_hash(c.spec) for c in out})
        # nothing new left: the pool still comes back whole, whatever the width
        out = propose(champ, pool, ["qa"], width=1, archived=set(hashes))
        self.assertEqual({spec_hash(c.spec) for c in out}, set(hashes))
        # no archive: unchanged
        self.assertEqual([c.label for c in propose(champ, pool, ["qa"], width=2)],
                         [c.label for c in propose(champ, pool, ["qa"], width=2, archived=set())])


class TestMeasurementNoise(ScriptedCase):
    """The gate's delta is max(floor, the champion's own re-measurement drift). That is the
    champion's noise, not the challenger's: on the AWS pool the champion runs at temperature
    0 (drift 0.0, so delta fell to the one-case floor) while a challenger with a
    temperature-0.8 lane measured +1.1 cases in run V and -0.9 in run W on the same 65
    confirm cases. The repeats already paid for carry the estimate; record it."""

    def test_one_repeat_cannot_estimate_the_noise_so_it_says_so(self):
        m = measure({"backend": "flaky", "kwargs": {"tag": "f"}}, _cases(4), repeats=1)
        self.assertIsNone(m.sem)          # not 0.0: "unknown" is not "steady"

    def test_a_deterministic_lane_has_no_noise(self):
        Scripted.WINS = {"a": {"qa1", "qa2"}}
        m = measure(_lane("a"), _cases(4), repeats=2)
        self.assertEqual(m.sem, 0.0)

    def test_the_noise_is_the_standard_error_built_from_the_repeats(self):
        # each case sees [1, 0]: unbiased variance 0.5, divided by 2 repeats = 0.25 per case;
        # summed over 4 independent cases = 1.0; sqrt = 1.0; divided by n = 4 -> 0.25
        m = measure({"backend": "flaky", "kwargs": {"tag": "f"}}, _cases(4), repeats=2)
        self.assertEqual(m.score, 0.5)
        self.assertEqual(set(m.per_case.values()), {0.5})
        self.assertAlmostEqual(m.sem, 0.25)

    def test_the_noise_of_a_comparison_adds_in_quadrature_and_is_unknown_if_either_side_is(self):
        Scripted.WINS = {"a": set()}
        det = measure(_lane("a"), _cases(4), repeats=2)
        fl = measure({"backend": "flaky", "kwargs": {"tag": "f"}}, _cases(4), repeats=2)
        self.assertAlmostEqual(remeasure_sd(det, fl), 0.25)
        fl2 = measure({"backend": "flaky", "kwargs": {"tag": "g"}}, _cases(4), repeats=2)
        self.assertAlmostEqual(remeasure_sd(fl, fl2), (0.25 ** 2 * 2) ** 0.5)
        once = measure({"backend": "flaky", "kwargs": {"tag": "h"}}, _cases(4), repeats=1)
        self.assertIsNone(remeasure_sd(det, once))

    def test_a_checkpoint_without_the_estimate_restores_as_unknown(self):
        from gama.grow import _restore, _state
        m = measure(_lane("a"), _cases(2), repeats=2)
        self.assertEqual(_restore(_state(m)).sem, m.sem)
        old = _state(m)
        del old["sem"]
        self.assertIsNone(_restore(old).sem)


class TestPromoteGate(unittest.TestCase):
    def test_exactly_one_case_passes_every_gate_whichever_way_the_scores_rounded(self):
        # scores reach the gates rounded to 4 digits; the widths (1/n) are exact. 11/15 − 10/15
        # arrives as 0.0666 against a delta of 0.066667, and 1 − 5/6 as 0.1667 against a band of
        # 0.166667. A one-case difference must not pass or fail on the rounding direction.
        r = lambda x: round(x, 4)
        self.assertEqual(promote_gate(1.0, 1.0, r(10 / 15), r(11 / 15), 1 / 15)[1], "promote")
        self.assertTrue(search_gate(1.0, r(5 / 6), 1 / 6)[0])
        self.assertTrue(search_gate(r(29 / 32), r(28 / 32), 1 / 32)[0])
        self.assertTrue(simplify_gate(r(11 / 15), r(10 / 15), 1 / 15, 2, 1)[0])
        # and a real extra case is still a real difference on the other side of the line
        self.assertFalse(search_gate(1.0, r(4 / 6), 1 / 6)[0])
        self.assertFalse(simplify_gate(r(12 / 15), r(10 / 15), 1 / 15, 2, 1)[0])
        self.assertTrue(promote_gate(1.0, 1.0, r(10 / 15), r(11 / 15) - 0.001, 1 / 15)[1]
                        .startswith("below-margin"))

    def test_promotes_only_when_all_three_hold(self):
        ok, why = promote_gate(0.5, 0.8, 0.5, 0.9, 0.05)
        self.assertTrue(ok)
        self.assertEqual(why, "promote")

    def test_a_tie_on_search_keeps_the_right_to_challenge(self):
        # search は選抜用で分解能が 1 問しかない。同点は負けではないので、決めるのは confirm。
        ok, why = promote_gate(0.8, 0.8, 0.5, 0.9, 0.05)
        self.assertTrue(ok)
        self.assertEqual(why, "promote")

    def test_rejects_when_search_is_worse_beyond_the_band(self):
        ok, why = promote_gate(0.8, 0.79, 0.5, 0.9, 0.05)          # band 0: 少しでも負ければ落ちる
        self.assertFalse(ok)
        self.assertTrue(why.startswith("search-worse"), why)
        # band = 1 問ぶん(1/32): 0.5 問の負けは通し、1 問を超える負けは落とす(run V gen3 が
        # 0.83 問負けで捨てられた 4勝0敗の候補を、この帯なら confirm に回せる)
        band = 1 / 32
        self.assertTrue(search_gate(0.901, 0.875, band)[0])
        self.assertTrue(search_gate(0.901, 0.901 - band, band)[0])     # band ちょうどは通す
        self.assertFalse(search_gate(0.901, 0.8646, band)[0])          # 1.17 問負け
        self.assertTrue(promote_gate(0.901, 0.875, 0.76, 0.79, 0.0154, search_band=band)[0])

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
        # The verdict is told what the run claimed on confirm (1 of 1 confirm case here).
        # A one-case sealed split cannot clear its own band even on a sweep, and the verdict
        # now says that this was fixed before the split was opened, instead of a bare
        # "cannot tell".
        sv = res["sealed_verdict"]
        self.assertEqual(sv["verdict"], "not-separable")
        self.assertEqual(sv["claimed_confirm_cases"], 1.0)
        self.assertEqual(sv["expected_cases"], 1.0)
        self.assertEqual(sv["power"], "underpowered")
        self.assertIn("never could have", sv["note"])

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

    def test_the_stamp_is_taken_once_so_the_recipe_names_the_code_that_ran(self):
        # Run W: the seed row said e7e5e63, the recipe said ffdb5bf. Commits landed while it
        # ran and the final row re-read HEAD. The process was running the tree it imported at
        # the start; the later SHA contributed nothing to the numbers.
        import unittest.mock as mock
        grow_mod = sys.modules["gama.grow"]      # `from gama import grow` is the function
        stamps = iter([{"version": "v", "commit": "start00", "dirty": False},
                       {"version": "v", "commit": "later00", "dirty": False}])
        Scripted.WINS = {"a": set()}
        events = []
        with mock.patch.object(grow_mod, "code_stamp", side_effect=lambda: next(stamps)) as cs:
            res = self._grow({"a": _lane("a")}, generations=2, width=1,
                             on_event=events.append)
        seed = [e for e in events if e["event"] == "seed"][0]
        self.assertEqual(cs.call_count, 1)
        self.assertEqual(seed["code"]["commit"], "start00")
        self.assertEqual(res["params"]["code"], seed["code"])

    def test_the_stamp_says_when_the_tree_is_not_the_commit(self):
        # Run X imported HEAD ffdb5bf plus an uncommitted grow.py and its stamp said ffdb5bf.
        # The SHA alone cannot say "this is not all of it"; `dirty` can.
        import shutil
        import subprocess
        if shutil.which("git") is None:
            self.skipTest("git not available")
        with tempfile.TemporaryDirectory() as d:
            def git(*args):
                return subprocess.run(["git", "-C", d, *args], capture_output=True, text=True,
                                      check=True)
            git("init", "-q")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
                "-m", "seed")
            Path(d, "mod.py").write_text("x = 1\n")
            git("add", "mod.py")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
            clean = code_stamp(where=Path(d))
            self.assertIsInstance(clean["commit"], str)
            self.assertIs(clean["dirty"], False)
            Path(d, "mod.py").write_text("x = 2\n")
            self.assertIs(code_stamp(where=Path(d))["dirty"], True)
            self.assertEqual(code_stamp(where=Path(d))["commit"], clean["commit"])
            git("checkout", "-q", "--", "mod.py")
            Path(d, "new.py").write_text("y = 1\n")            # untracked counts too
            self.assertIs(code_stamp(where=Path(d))["dirty"], True)
        # Not a checkout at all: nothing is claimed, and nothing is invented.
        with tempfile.TemporaryDirectory() as d:
            bare = code_stamp(where=Path(d))
            self.assertEqual((bare["commit"], bare["dirty"]), (None, None))

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

    def test_resuming_into_the_same_ledger_keeps_the_first_segment(self):
        # --resume L --out L is the natural way to continue. The ledger used to be truncated
        # right after its checkpoint was read: the seed row and every generation before the
        # crash were gone, and the ledger, "the only evidence behind the numbers", kept only
        # the second half of the run.
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(8), generations=1, width=2, patience=5,
                 ledger_path=str(led), min_margin=0.05)
            splits = _ledger_splits(led)
            with led.open("a", encoding="utf-8") as fh:
                fh.write('{"event": "gener')          # the half-written line a kill leaves
            grow(pool, cases=_cases(8), generations=3, width=2, patience=5,
                 ledger_path=str(led), resume_from=str(led), min_margin=0.05)
            rows, broken = [], 0
            for line in led.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    broken += 1
            self.assertEqual(broken, 1, "the half-written line is kept as is, nothing else breaks")
            events = [r["event"] for r in rows]
            self.assertEqual(events.count("seed"), 1)
            self.assertEqual(events.count("resumed"), 1)
            self.assertLess(events.index("seed"), events.index("resumed"))
            gens = [r["gen"] for r in rows if r["event"] == "generation"]
            self.assertEqual(gens, [0, 1, 2])              # both segments, in order
            self.assertEqual(load_checkpoint(led)["gen"], 2)  # the newest checkpoint wins
            self.assertEqual(_ledger_splits(led), splits)  # and the split guard still has its row
            # the row that follows the broken one must start on its own line
            text = led.read_text(encoding="utf-8")
            self.assertIn('{"event": "gener\n{"event": "resumed"', text)

    def test_a_resume_path_that_does_not_exist_is_refused_before_the_ledger_is_touched(self):
        Scripted.WINS = {"a": set(), "b": {"qa1"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(8), generations=1, width=2, patience=5,
                 ledger_path=str(led), min_margin=0.05)
            before = led.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no checkpoint"):
                grow(pool, cases=_cases(8), generations=2, width=2, patience=5,
                     ledger_path=str(led), resume_from=str(Path(d) / "missing.jsonl"),
                     min_margin=0.05)
            # a typo in --resume must not cost the ledger that --out points at
            self.assertEqual(led.read_text(encoding="utf-8"), before)

    def test_a_ledger_that_begins_with_a_resumed_row_still_guards_its_split(self):
        # resuming into a different file leaves a ledger whose first row is "resumed", and
        # resuming from that one skipped the split check (it looked for a seed row only)
        Scripted.WINS = {"a": set(), "b": {"qa1"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            first, second = Path(d) / "one.jsonl", Path(d) / "two.jsonl"
            grow(pool, cases=_cases(4), generations=1, width=1, ledger_path=str(first))
            grow(pool, cases=_cases(4), generations=2, width=1,
                 ledger_path=str(second), resume_from=str(first))
            self.assertEqual(json.loads(second.read_text().splitlines()[0])["event"], "resumed")
            self.assertIsNotNone(_ledger_splits(second))
            with self.assertRaises(ValueError):
                grow(pool, cases=_cases(8), generations=3, width=1, resume_from=str(second))

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

    def test_the_ledger_puts_the_comparison_noise_beside_the_gain(self):
        # The champion `a` is deterministic (drift 0, delta = floor); the challenger routes
        # qa to a coin. With 2 repeats the coin's confirm score is 0.5 on the one confirm
        # case with a standard error of 0.5, so the row says the gain of 0.5 cases sits
        # inside a re-measurement noise of 0.5 cases. The verdict is untouched: this is a
        # record, not a gate, until a run has carried the number.
        Scripted.WINS = {"a": set()}
        res = self._grow({"a": _lane("a"), "f": {"backend": "flaky", "kwargs": {"tag": "f"}}},
                         generations=1, width=2, repeats=2)
        gen0 = res["history"][0]
        self.assertEqual(gen0["challenger"], "route:qa->f")
        self.assertEqual(gen0["gain_cases"], 0.5)
        self.assertEqual(gen0["confirm_noise_cases"], 0.5)
        self.assertEqual(gen0["bound_by"], "floor")            # delta did not move
        # the sealed comparison carries its own noise: seed 0.0 vs the coin's 0.5 on one case
        self.assertEqual(res["sealed_verdict"]["noise_cases"], 0.5)
        # a 1-repeat run cannot estimate it and says so instead of writing 0
        Flaky.CALLS = {}
        res1 = self._grow({"a": _lane("a"), "f": {"backend": "flaky", "kwargs": {"tag": "f"}}},
                          generations=1, width=2, repeats=1)
        self.assertIsNone(res1["history"][0]["confirm_noise_cases"])
        self.assertIsNone(res1["sealed_verdict"]["noise_cases"])

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

    def test_the_table_names_what_the_sealed_difference_is_against(self):
        # Run W was seeded from the shipped recipe (two routed classes) and its recipe.md said
        # "seed (no structure)": a reader takes +2.34 sealed cases as a gain over the bare
        # model, which was never measured. The heading must say what the seed was.
        Scripted.WINS = {"a": set(), "c": {"qa1", "qa2", "qa3"}}
        res = grow({"a": _lane("a"), "c": _lane("c")}, cases=_cases(4), generations=2, width=2)
        self.assertEqual(res["promotions"], 1)
        with tempfile.TemporaryDirectory() as d:
            md = (write_recipe(res, Path(d) / "bare") / "recipe.md").read_text(encoding="utf-8")
            self.assertIn("| | seed (no structure) | grown champion |", md)
            again = grow({"a": _lane("a"), "c": _lane("c")}, cases=_cases(4), generations=1,
                         width=1, seed_spec=res["champion"])
            md2 = (write_recipe(again, Path(d) / "continued") / "recipe.md").read_text(
                encoding="utf-8")
            self.assertIn("| | seed (routes 1 class; not the bare model) | grown champion |",
                          md2)
            self.assertNotIn("no structure", md2)
        # No routing at all but a composite default lane is still not the bare model (codex).
        from gama.grow import _seed_label
        ens = {"backend": "ensemble", "kwargs": {"members": [_lane("a"), _lane("c")]}}
        self.assertEqual(_seed_label({"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a"), "ens": ens}, "routing_table": {}, "default": "ens"}}),
            "seed (a composite default lane; not the bare model)")
        self.assertEqual(_seed_label({"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a")}, "routing_table": {}, "default": "a"}}),
            "seed (no structure)")
        # A route to the default lane is a no-op, not a decision (canonical drops it).
        self.assertEqual(_seed_label({"backend": "gama", "kwargs": {
            "backends": {"a": _lane("a")}, "routing_table": {"qa": "a"}, "default": "a"}}),
            "seed (no structure)")
        self.assertEqual(_seed_label(None), "seed")


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

    def test_a_seed_whose_tool_lane_writes_no_code_says_so_by_class(self):
        # echo never opens a fence, so a seed with a tool(echo) lane fails every tool call
        # with "no code"; the seed line names the classes, so the prefill ordering that
        # follows is explained before it happens rather than read off the ledger afterwards
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "champ.json"
            cfg.write_text(json.dumps({"system": {
                "backend": "gama",
                "kwargs": {"backends": {"echo": {"backend": "echo"},
                                        "tool(echo)": {"backend": "tool", "_grow_base": "echo",
                                                       "kwargs": {"inner": {"backend": "echo"}}}},
                           "routing_table": {"qa": "tool(echo)"}, "default": "echo"}}}),
                encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = main(["grow", "--smoke", "--suites", "wide", "--generations", "1",
                           "--width", "1", "--start-from", str(cfg)])
            self.assertEqual(rc, 0)
            self.assertRegex(err.getvalue(), r"seed tool lanes returned no code on: qa \d+")
            # and the measurement itself carries the diagnosis per class (nothing promotes on
            # echo, so the final champion's confirm measurement is the seed's)
            result = json.loads(out.getvalue())
            self.assertEqual(result["promotions"], 0)
            self.assertEqual(list(result["confirm"]["tool_no_code_by_class"]), ["qa"])

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
        # 1 走目で route:qa->b が昇格し、再開後に c(b の上位互換)が挑戦して confirm まで進む。
        # search で帯を超えて負けた候補は confirm を測らない(対応のある証拠も出ない)ので、
        # 見るのは confirm まで進んだ行だけ —— ただし 1 行も無ければ試験は空振りなので落とす。
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"},
                         "c": {"qa1", "qa2", "qa3", "qa4", "qa5", "qa6"}}
        pool = {"a": _lane("a"), "b": _lane("b"), "c": _lane("c")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=_cases(8), generations=1, width=1, patience=5,
                 ledger_path=str(led), min_margin=0.05)
            second = grow(pool, cases=_cases(8), generations=3, width=3, patience=5,
                          resume_from=str(led), min_margin=0.05)
        challenged = [h for h in second["history"] if "challenger_confirm" in h]
        self.assertTrue(challenged, "resumed run never measured a challenger on confirm")
        for h in challenged:
            counts = (h.get("paired_wins"), h.get("paired_losses"), h.get("paired_ties"))
            self.assertNotIn(None, counts, "paired evidence missing after resume")
            # 共有 case が 1 つも見つからない = per_case が復元されていない証拠
            self.assertGreater(sum(counts), 0,
                               "no shared cases after resume: the champion's per-case scores "
                               "were restored from a checkpoint instead of re-measured")


class TestConfirmClaim(unittest.TestCase):
    """sealed が検定する仮説の confirm 側の大きさ。どの測定で出すかで ±1 問(帯そのもの)動くので、
    選択に使っていない測定の平均で出す。数字は run T/V/R の台帳そのまま。"""

    def test_run_t_the_seed_drifted_down_and_the_claim_is_the_mean_not_the_first_pair(self):
        c = confirm_claim([0.8619, 0.8485, 0.844], [0.8664, 0.8664, 0.8664], 56,
                          promotion_score=0.8664)
        self.assertAlmostEqual(c["cases"], 0.84, places=2)      # first pair would say +0.25
        self.assertEqual((c["seed_measurements"], c["champion_measurements"]), (3, 3))
        self.assertFalse(c["promotion_only"])
        self.assertFalse(c["same_as_seed"])

    def test_run_v_the_promotion_measurement_is_excluded_because_it_was_selected(self):
        c = confirm_claim([0.7669, 0.7669], [0.7792, 0.7637, 0.7608, 0.7592], 65,
                          promotion_score=0.7838)
        self.assertLess(c["cases"], 0)                           # -0.08: the gain evaporated
        self.assertAlmostEqual(c["cases"], -0.08, places=2)
        self.assertEqual(c["champion_mean"], 0.7657)
        # counting the selected measurement would have turned that into a positive claim
        self.assertGreater(confirm_claim([0.7669, 0.7669],
                                         [0.7838, 0.7792, 0.7637, 0.7608, 0.7592], 65)["cases"], 0)

    def test_run_r_the_claim_is_close_to_the_certified_total_when_nothing_regressed(self):
        c = confirm_claim([0.7712, 0.7712], [0.8493, 0.8597], 48, promotion_score=0.8597)
        self.assertAlmostEqual(c["cases"], 4.0, places=2)       # certified +4.37 on the way

    def test_a_promotion_in_the_last_generation_has_only_its_selected_measurement(self):
        c = confirm_claim([0.80, 0.80], [], 40, promotion_score=0.85)
        self.assertAlmostEqual(c["cases"], 2.0, places=2)
        self.assertTrue(c["promotion_only"])
        self.assertEqual(c["champion_measurements"], 1)

    def test_a_champion_identical_to_the_seed_claims_nothing_whatever_it_re_measured(self):
        c = confirm_claim([0.80, 0.85, 0.75], [], 40)
        self.assertEqual(c["cases"], 0.0)
        self.assertTrue(c["same_as_seed"])
        self.assertEqual(c["champion_mean"], c["seed_mean"])
        # the seed's measurements are not counted a second time as the champion's (codex r4)
        self.assertEqual((c["seed_measurements"], c["champion_measurements"]), (3, 0))

    def test_no_readable_seed_measurement_is_an_unreadable_claim_not_a_crash(self):
        # a total function: sealed_verdict reads cases=None as "not recorded" (codex r4)
        for seed in ([], None, [None, "x", True]):
            c = confirm_claim(seed, [0.9, 0.8], 40, promotion_score=0.9)
            self.assertIsNone(c["cases"])
            self.assertIsNone(c["seed_mean"])
            self.assertEqual(c["seed_measurements"], 0)
            # the champion's side is still recorded (codex r5)
            self.assertEqual((c["champion_mean"], c["champion_measurements"]), (0.85, 2))
            self.assertFalse(c["same_as_seed"])
        # no champion material at all is the seed itself, readable seed or not (codex r6)
        c = confirm_claim([], [], 40)
        self.assertTrue(c["same_as_seed"])
        self.assertIsNone(c["cases"])
        sealed = {"seed": {"score": 0.85, "cases": 32}, "champion": {"score": 0.85, "cases": 32}}
        v = sealed_verdict(sealed, c["cases"], 40, promoted_gain_cases=1.0)
        self.assertIsNone(v["power"])
        self.assertIn("not recorded", v["note"])
        # unreadable entries are dropped, not averaged as zero
        c = confirm_claim([0.8, None, "x"], [0.9, True], 40)
        self.assertEqual((c["seed_measurements"], c["champion_measurements"]), (1, 1))
        self.assertAlmostEqual(c["cases"], 4.0, places=6)

    def test_the_claim_is_not_rounded(self):
        c = confirm_claim([0.8], [0.8 + 1 / 3 / 40], 40, promotion_score=0.9)
        self.assertNotEqual(c["cases"], round(c["cases"], 2))


class TestTheClaimSealedTests(ScriptedCase):
    """sealed の検定力は「最終形が confirm で種より何問上か」で決まる(sealed が比べるのがその
    二つだから)。種は走行ごとに一つで、再開した走行の種は**再開点のチャンピオン**。主張の基準も
    それに合わせる: 元の種まで遡ると、sealed が比べていない相手に対する主張になる。"""

    def test_the_claim_is_the_final_champion_over_the_seed_on_confirm(self):
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"},
                         "c": {"qa1", "qa2", "qa3", "qa4", "qa5", "qa6"}}
        pool = {"a": _lane("a"), "b": _lane("b"), "c": _lane("c")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            first = grow(pool, cases=_cases(8), generations=1, width=1, patience=5,
                         ledger_path=str(led), min_margin=0.05)
            seed_row = [json.loads(l) for l in led.read_text().splitlines()
                        if json.loads(l).get("event") == "seed"][0]
            self.assertTrue(first["net_change"])
            claim = first["confirm_claim"]
            # one generation: the seed was measured twice (seed + gen0 re-measure), the champion
            # only at its promotion
            self.assertEqual(claim["seed_measurements"], 2)
            self.assertTrue(claim["promotion_only"])
            self.assertEqual(claim["champion_mean"], first["confirm"]["score"])
            self.assertEqual(claim["seed_mean"], seed_row["confirm"]["score"])   # deterministic
            n_confirm = len(first["splits"]["confirm"])
            sv = first["sealed_verdict"]
            self.assertAlmostEqual(
                sv["claimed_confirm_cases"],
                round((claim["champion_mean"] - claim["seed_mean"]) * n_confirm, 2), places=2)
            self.assertIsNotNone(sv["power"])
            self.assertIn("measured once, at its promotion", sv["note"])
            # the certified total is reported beside the claim
            self.assertAlmostEqual(sv["promoted_confirm_cases"],
                                   round(sum(h["gain_cases"] for h in first["history"]
                                             if h["champion_hash"] != h["champion_after"]), 2),
                                   places=2)
            # a resumed run's seed is the resume point, and so is the baseline of its claim
            resume_point = load_checkpoint(led)["champion_confirm"]["score"]
            second = grow(pool, cases=_cases(8), generations=3, width=3, patience=5,
                          ledger_path=str(led), resume_from=str(led), min_margin=0.05)
            self.assertEqual(second["seed_hash"], first["champion_hash"])
            claim = second["confirm_claim"]
            self.assertGreaterEqual(claim["seed_measurements"], 1)
            if second["net_change"]:
                self.assertFalse(claim["same_as_seed"])
                self.assertAlmostEqual(
                    second["sealed_verdict"]["claimed_confirm_cases"],
                    round((claim["champion_mean"] - claim["seed_mean"])
                          * len(second["splits"]["confirm"]), 2), places=2)
            else:
                self.assertTrue(claim["same_as_seed"])
                self.assertEqual(second["sealed_verdict"]["claimed_confirm_cases"], 0.0)

    def test_re_measurements_after_a_promotion_feed_the_claim_not_the_promotion_score(self):
        # b is promoted at gen0; gens 1-2 re-measure it (deterministic here, so the mean is the
        # same number, but the count shows which measurements were used)
        Scripted.WINS = {"a": set(), "b": {"qa1", "qa2", "qa4"}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        result = grow(pool, cases=_cases(8), generations=3, width=1, patience=5, min_margin=0.05)
        self.assertTrue(result["net_change"])
        claim = result["confirm_claim"]
        self.assertFalse(claim["promotion_only"])
        self.assertEqual(claim["champion_measurements"], 2)
        self.assertIn("means of 2 champion and 2 seed measurements",
                      result["sealed_verdict"]["note"])

    def test_a_run_that_stayed_on_its_seed_claims_nothing(self):
        # re-measuring the seed is drift, not a claim: a champion identical to the seed must not
        # read as "underpowered" or "powered" from its own noise
        Scripted.WINS = {"a": {"qa1", "qa2"}, "b": set()}
        pool = {"a": _lane("a"), "b": _lane("b")}
        result = grow(pool, cases=_cases(8), generations=1, width=1, patience=5, min_margin=0.05)
        self.assertFalse(result["net_change"])
        sv = result["sealed_verdict"]
        self.assertEqual(sv["verdict"], "not-separable")
        self.assertEqual(sv["claimed_confirm_cases"], 0.0)
        self.assertEqual(sv["power"], "nothing-claimed")
        self.assertTrue(result["confirm_claim"]["same_as_seed"])
        # the seed's own measurements are not presented as a separately measured champion
        with tempfile.TemporaryDirectory() as d:
            out = write_recipe(result, Path(d) / "r")
            md = (Path(out) / "recipe.md").read_text(encoding="utf-8")
        self.assertIn("the champion is the seed: 2 measurements", md)
        self.assertNotIn("champion measurements", md)


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
        self.assertIsNone(v["noise_cases"])

    def test_a_comparison_noisier_than_the_band_is_said_beside_the_verdict_not_instead_of_it(self):
        # seed at temperature 0 (sem 0), champion with a hot lane: sem 0.05 on 32 cases is
        # 1.6 cases of re-measurement noise, coarser than the one-case band. The verdict
        # stays "improved" (the band is the rule until the number has been carried by real
        # runs); the note says what the band does not cover.
        sealed = {"seed": {"score": 0.7484, "cases": 32, "sem": 0.0},
                  "champion": {"score": 0.8214, "cases": 32, "sem": 0.05}}
        v = sealed_verdict(sealed)
        self.assertEqual(v["verdict"], "improved")
        self.assertEqual(v["noise_cases"], 1.6)
        self.assertIn("would move about 1.6 cases", v["note"])
        # inside the band: nothing to add, the band already covers it
        sealed["champion"]["sem"] = 0.02          # 0.64 cases
        v = sealed_verdict(sealed)
        self.assertEqual(v["noise_cases"], 0.64)
        self.assertNotIn("would move", v["note"])
        # an old ledger (no sem) does not get a made-up zero
        v = sealed_verdict(self._s(0.7484, 0.8214, n=32))
        self.assertIsNone(v["noise_cases"])
        self.assertNotIn("would move", v["note"])
        # the recipe carries the sentence through the note
        with tempfile.TemporaryDirectory() as d:
            result = {"champion": {"backend": "echo", "kwargs": {}},
                      "seed": {"backend": "echo", "kwargs": {}},
                      "champion_hash": "deadbeef", "seed_hash": "cafe",
                      "promotions": 1, "net_change": True, "bound_by": {},
                      "generations_run": 1, "search": {"score": 0.9},
                      "confirm": {"score": 0.8}, "archive_size": 1,
                      "sealed": {"seed": {"score": 0.7484, "cases": 32, "sem": 0.0},
                                 "champion": {"score": 0.8214, "cases": 32, "sem": 0.05}},
                      "splits": {}, "history": [], "params": {}}
            text = (write_recipe(result, d) / "recipe.md").read_text(encoding="utf-8")
            self.assertIn("IMPROVED", text)
            self.assertIn("would move about 1.6 cases", text)

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

    # 「分からない」の三つの中身。7 走の台帳を並べて出た区別で、行動が違う(split を広げる /
    # 手を疑う / 門を疑う)ので一語に丸めない。判定の軸は「最終形が confirm で種より何問上か」
    # (sealed が比べるのがその二つだから)。昇格時の認定の合計は並記されるだけ。
    def test_a_gain_too_small_for_the_split_was_never_going_to_separate(self):
        # a champion standing +1.1 of 65 confirm over the seed → 0.54 of 32 sealed, inside the band
        v = sealed_verdict(self._s(0.7484, 0.7471, n=32), claimed_gain_cases=1.1, confirm_cases=65,
                           promoted_gain_cases=1.1)
        self.assertEqual(v["verdict"], "not-separable")
        self.assertEqual(v["power"], "underpowered")
        self.assertEqual(v["claimed_confirm_cases"], 1.1)
        self.assertEqual(v["promoted_confirm_cases"], 1.1)
        self.assertAlmostEqual(v["expected_cases"], 0.54, places=2)
        self.assertIn("never could have", v["note"])
        self.assertIn("certified +1.10 at promotion time", v["note"])
        self.assertIn("at least 60 cases", v["note"])          # int(65/1.1)+1
        self.assertIn("more than 2.03 confirm cases", v["note"])  # 65/32

    def test_a_gain_the_split_could_see_but_did_not_is_called_non_transfer(self):
        # run R: champion +4.25 of 48 confirm over the seed → 2.12 of 24 sealed expected,
        # the promotions certified +4.37 on the way, sealed shows +0.33
        v = sealed_verdict(self._s(0.8611, 0.8750, n=24), claimed_gain_cases=4.25, confirm_cases=48,
                           promoted_gain_cases=4.37)
        self.assertEqual(v["verdict"], "not-separable")
        self.assertEqual(v["power"], "powered")
        self.assertAlmostEqual(v["expected_cases"], 2.12, places=2)
        self.assertIn("did not transfer", v["note"])
        self.assertIn("stands +4.25 of 48", v["note"])
        self.assertIn("certified +4.37", v["note"])
        self.assertIn("+0.33", v["note"])

    def test_a_gain_the_run_itself_retracted_is_called_evaporated(self):
        # run V: the gate certified +1.10 at promotion time; re-measured every generation
        # after, the champion ended -0.50 under the seed on confirm. Sealed had nothing to test,
        # and that is not "nothing was claimed": the gate did certify a gain.
        v = sealed_verdict(self._s(0.7484, 0.7471, n=32), claimed_gain_cases=-0.5, confirm_cases=65,
                           promoted_gain_cases=1.1)
        self.assertEqual(v["verdict"], "not-separable")
        self.assertEqual(v["power"], "evaporated")
        self.assertEqual(v["claimed_confirm_cases"], -0.5)
        self.assertEqual(v["promoted_confirm_cases"], 1.1)
        self.assertIn("certified +1.10 of 65 confirm cases", v["note"])
        self.assertIn("stands -0.50 over the seed", v["note"])
        self.assertIn("evaporated", v["note"])
        self.assertIn("no better than the seed", v["note"])
        # a champion that gained nothing, promoted by simplification only, claims nothing
        v = sealed_verdict(self._s(0.7484, 0.7471, n=32), claimed_gain_cases=-0.5, confirm_cases=65,
                           promoted_gain_cases=-0.5)
        self.assertEqual(v["power"], "nothing-claimed")

    def test_the_claim_decides_power_even_when_the_certified_total_was_larger(self):
        # certified +4.0 on the way, but the champion stands only +0.3 over the seed at the end:
        # sealed tests the +0.3, and that is inside the band
        v = sealed_verdict(self._s(0.85, 0.85, n=32), claimed_gain_cases=0.3, confirm_cases=64,
                           promoted_gain_cases=4.0)
        self.assertEqual(v["power"], "underpowered")
        self.assertIn("stands +0.30 of 64", v["note"])
        self.assertIn("certified +4.00", v["note"])

    def test_an_unreadable_claim_with_a_recorded_certification_says_which_is_missing(self):
        # a resumed old ledger: the promotions are readable, the seed's confirm score is not
        v = sealed_verdict(self._s(0.85, 0.85, n=32), None, 64, promoted_gain_cases=2.0)
        self.assertEqual(v["verdict"], "not-separable")
        self.assertIsNone(v["power"])
        self.assertIsNone(v["claimed_confirm_cases"])
        self.assertEqual(v["promoted_confirm_cases"], 2.0)
        self.assertIn("not recorded", v["note"])
        self.assertIn("certified +2.00 of 64 confirm cases", v["note"])
        # without a confirm count the note still reads, and never prints a None (codex r3)
        v = sealed_verdict(self._s(0.85, 0.85, n=32), None, None, promoted_gain_cases=2.0)
        self.assertIn("certified +2.00 on confirm", v["note"])
        self.assertNotIn("None", v["note"])

    def test_a_gain_that_would_show_as_exactly_one_case_is_still_underpowered(self):
        # sealed calls "improved" only beyond the band, so an expectation of exactly one case
        # cannot be separated either; the note must not say "less than" (codex r3)
        v = sealed_verdict(self._s(0.85, 0.85, n=32), claimed_gain_cases=2.0, confirm_cases=64)
        self.assertEqual(v["expected_cases"], 1.0)
        self.assertEqual(v["power"], "underpowered")
        self.assertIn("not beyond the one case", v["note"])
        self.assertNotIn("less than", v["note"])
        v = sealed_verdict(self._s(0.85, 0.85, n=32), claimed_gain_cases=2.0 + 1e-9,
                           confirm_cases=64)
        self.assertEqual(v["power"], "powered")

    def test_no_claimed_gain_keeps_the_plain_reading(self):
        v = sealed_verdict(self._s(0.85, 0.85), claimed_gain_cases=0.0, confirm_cases=56)
        self.assertEqual(v["power"], "nothing-claimed")
        self.assertIn("neither proved nor disproved", v["note"])
        # a simplify-only run claims a small loss; that is still nothing to prove
        v = sealed_verdict(self._s(0.85, 0.85), claimed_gain_cases=-0.5, confirm_cases=56)
        self.assertEqual(v["power"], "nothing-claimed")

    def test_without_a_claim_the_verdict_is_unchanged_and_says_so(self):
        # old ledgers and callers that pass nothing: same verdict, power unknown, never "powered"
        v = sealed_verdict(self._s(0.8393, 0.8214))
        self.assertEqual(v["verdict"], "not-separable")
        self.assertIsNone(v["power"])
        self.assertIsNone(v["expected_cases"])
        self.assertIsNone(v["promoted_confirm_cases"])
        self.assertIn("neither proved nor disproved", v["note"])
        for bad in (float("nan"), True, "3"):
            self.assertIsNone(sealed_verdict(self._s(0.85, 0.85), bad, 56)["power"])
        self.assertIsNone(sealed_verdict(self._s(0.85, 0.85), 2.0, 0)["power"])

    def test_power_is_decided_on_the_exact_expectation_not_the_rounded_one(self):
        # 1.004 sealed cases expected (rounds to 1.00) is still more than the band
        v = sealed_verdict(self._s(0.85, 0.85, n=32), claimed_gain_cases=2.0, confirm_cases=63)
        self.assertGreater(2.0 * 32 / 63, 1.0)
        self.assertEqual(v["expected_cases"], 1.02)
        self.assertEqual(v["power"], "powered")
        v = sealed_verdict(self._s(0.85, 0.85, n=32), claimed_gain_cases=2.0, confirm_cases=64)
        self.assertEqual(v["power"], "underpowered")       # exactly one case is inside the band

    def test_power_does_not_override_a_real_verdict(self):
        v = sealed_verdict(self._s(0.6375, 0.8542, n=20), claimed_gain_cases=0.5, confirm_cases=40)
        self.assertEqual(v["verdict"], "improved")           # sealed moved; that stands
        self.assertEqual(v["power"], "underpowered")         # the claim was small; also stands
        v = sealed_verdict(self._s(0.90, 0.80), claimed_gain_cases=5.0, confirm_cases=40)
        self.assertEqual(v["verdict"], "regressed")
        v = sealed_verdict(self._s(0.90, 0.80), claimed_gain_cases=-1.0, confirm_cases=40,
                           promoted_gain_cases=3.0)
        self.assertEqual(v["verdict"], "regressed")
        self.assertEqual(v["power"], "evaporated")

    def test_the_certified_total_is_reported_rounded_but_the_claim_is_judged_exact(self):
        v = sealed_verdict(self._s(0.85, 0.85, n=32), claimed_gain_cases=2.0 + 1e-9,
                           confirm_cases=64, promoted_gain_cases=2.123456)
        self.assertEqual(v["promoted_confirm_cases"], 2.12)
        self.assertEqual(v["claimed_confirm_cases"], 2.0)
        self.assertEqual(v["power"], "powered")              # 2.000000001*32/64 > 1 exactly


class TestPromotedGain(unittest.TestCase):
    """門が昇格時に認めた伸びの合計。判定には使わず、最終形の主張の横に並記される。"""

    def _gen(self, gen, changed, **kw):
        row = {"gen": gen, "champion_hash": "h0", "champion_after": "h1" if changed else "h0",
               "champion_confirm": 0.80}
        row.update(kw)
        return row

    def test_adds_promotions_and_subtracts_removals(self):
        hist = [self._gen(0, True, gain_cases=1.25),
                self._gen(1, False, gain_cases=3.0),                    # rejected: not claimed
                self._gen(2, True, simplify_verdict="promote", simplify_confirm=0.79)]
        # +1.25, then a removal that cost 0.01*56 = 0.56; returned unrounded (display rounds)
        self.assertAlmostEqual(promoted_gain_cases(hist, 56), 0.69, places=2)
        self.assertNotEqual(promoted_gain_cases(hist, 56), 0.69)

    def test_nothing_changed_is_zero_not_none(self):
        self.assertEqual(promoted_gain_cases([self._gen(0, False, gain_cases=2.0)], 56), 0.0)
        self.assertEqual(promoted_gain_cases([], 56), 0.0)

    def test_an_unreadable_promotion_makes_the_claim_unknown(self):
        # a changed generation with no readable gain (an old ledger) must not count as 0:
        # "nothing claimed" would read as a run that could never separate, which is a lie
        self.assertIsNone(promoted_gain_cases([self._gen(0, True)], 56))
        self.assertIsNone(promoted_gain_cases(
            [self._gen(0, True, simplify_verdict="promote")], 56))
        self.assertIsNone(promoted_gain_cases([self._gen(0, True, gain_cases=1.0)], 0))
        self.assertIsNone(promoted_gain_cases([self._gen(0, True, gain_cases=1.0)], None))

    def test_booleans_and_non_finite_values_are_not_numbers(self):
        # a ledger's true/false must not be read as 1/0 cases (bool is an int in Python)
        self.assertIsNone(promoted_gain_cases([self._gen(0, True, gain_cases=True)], 56))
        self.assertIsNone(promoted_gain_cases([self._gen(0, True, gain_cases=float("nan"))], 56))
        self.assertIsNone(promoted_gain_cases(
            [self._gen(0, True, simplify_verdict="promote", simplify_confirm=True)], 56))
        self.assertIsNone(promoted_gain_cases([self._gen(0, True, gain_cases=1.0)], True))

    def test_a_row_without_the_champion_keys_is_unreadable_not_unchanged(self):
        # None == None must not read as "the champion did not change" (codex r4)
        hist = [{"gen": 0, "gain_cases": 1.0}]
        self.assertIsNone(promoted_gain_cases(hist, 56))
        hist = [{"gen": 0, "champion_hash": "a", "champion_after": "b", "gain_cases": 1.0},
                {"gen": 1, "champion_hash": "b"}]
        self.assertIsNone(promoted_gain_cases(hist, 56))


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

    def test_the_prescription_goes_first_among_ties_but_never_over_a_higher_score(self):
        # the diagnosis (no-code tool calls on qa) names the prefill for qa as the prescription;
        # search cannot show its payoff on a class it has nearly solved, confirm holds the misses
        key = lambda t: _challenger_key(*t, prescribed=_prescribed(t[0], {"qa": 3}))
        pres = (Candidate("tool:qa(b)+prefill", "tool", _tool_spec("b"), remedy="qa"),
                self._m(0.9, 1.0))
        bare = (self._cand("a:bare", _lane_spec("b")), self._m(0.9, 1.0))
        self.assertEqual(min([bare, pres], key=key)[0].label, "tool:qa(b)+prefill")
        # a prefill for a class without a symptom is not a prescription
        other = (Candidate("tool:research(b)+prefill", "tool", _tool_spec("b"),
                           remedy="research"), self._m(0.9, 1.0))
        self.assertEqual(min([bare, other], key=key)[0].label, "a:bare")
        # the judgement reads what the minting side wrote, not the label
        relabel = (Candidate("tool:qa(b)+prefill", "tool", _tool_spec("b")), self._m(0.9, 1.0))
        self.assertFalse(_prescribed(relabel[0], {"qa": 3}))
        # the score still decides first
        better = (self._cand("z:bare", _lane_spec("b")), self._m(0.95, 1.0))
        self.assertEqual(min([better, pres], key=key)[0].label, "z:bare")
        # without a diagnosis the key is what it was
        self.assertEqual(_challenger_key(*pres), _challenger_key(*pres, prescribed=False))

    def test_structure_size_ranks_a_bare_lane_below_a_composite_one(self):
        self.assertLess(_structure_size(_lane_spec("b")), _structure_size(_tool_spec("b")))


class TestSearchIsAFilterNotARace(ScriptedCase):
    """run V の台帳: 候補が search で 0.5 問・0.83 問負けただけで、confirm で 4勝1敗・4勝0敗
    (+1.6 問・+1.7 問)だったのに confirm を見ずに捨てられた。チャンピオンの search 点は昇格時の
    一回きりの最大値で、同じ設計を測り直すと 1 問ぶん動く(0.8958 と 0.8646)。search は選抜と
    足切りに使い、決めるのは confirm。"""

    def _split(self):
        cases = _cases(8)
        sp = split_cases(cases)
        return cases, [c.case_id for c in sp["search"]], [c.case_id for c in sp["confirm"]]

    def test_trailing_by_less_than_a_case_on_search_still_reaches_confirm_and_can_win(self):
        cases, search, confirm = self._split()
        # a: search 全勝・confirm 全敗。b: search の 1 問だけ部分点(0.4)= 0.6 問負け、confirm 全勝
        Scripted.WINS = {"a": set(search), "b": set(search[1:]) | set(confirm)}
        Scripted.PARTIAL = {"b": {search[0]}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        res = grow(pool, cases=cases, generations=1, width=1, patience=3, min_margin=0.05)
        gen0 = res["history"][0]
        self.assertEqual(gen0["challenger"], "route:qa->b")
        self.assertLess(gen0["challenger_search"], gen0["champion_search"])
        self.assertEqual(gen0["search_band"], round(1 / len(search), 4))
        self.assertIn("challenger_confirm", gen0, "a within-band candidate was not measured")
        self.assertEqual(gen0["reason"], "promote")
        self.assertEqual(res["promotions"], 1)

    def test_trailing_by_more_than_a_case_skips_confirm_and_settles_the_design(self):
        cases, search, confirm = self._split()
        # b は search で 2 問落とす(帯 1 問を超える)。confirm では全勝だが、そこは測らない
        Scripted.WINS = {"a": set(search), "b": set(search[2:]) | set(confirm)}
        pool = {"a": _lane("a"), "b": _lane("b")}
        seen = []
        res = grow(pool, cases=cases, generations=2, width=1, patience=3, min_margin=0.05,
                   on_event=lambda r: seen.append(r))
        gen0 = res["history"][0]
        self.assertEqual(gen0["challenger"], "route:qa->b")
        self.assertTrue(gen0["reason"].startswith("search-worse"), gen0["reason"])
        self.assertNotIn("challenger_confirm", gen0)
        self.assertNotIn("paired_wins", gen0)
        # b の confirm は一度も焚かれていない(search で決着した候補に confirm を払わない)
        self.assertEqual([c for t, c in Scripted.SEEN_BY if t == "b" and c in confirm], [])
        # 決着済み: gen0 の checkpoint で settled に入り(confirm を測っていないので challenged
        # ではない)、gen1 では提案されない
        ck0 = next(r for r in seen if r["event"] == "checkpoint" and r["gen"] == 0)
        self.assertIn(gen0["challenger_hash"], ck0["settled"])
        self.assertNotIn(gen0["challenger_hash"], ck0["challenged"])
        gen1_labels = [r["label"] for r in seen if r["event"] == "candidate" and r["gen"] == 1]
        self.assertNotIn("route:qa->b", gen1_labels)
        self.assertNotEqual(res["history"][1]["challenger"], "route:qa->b")
        self.assertEqual(res["promotions"], 0)

    def test_a_search_settlement_is_against_one_champion_and_clears_on_promotion(self):
        # 新チャンピオンの search は旧より帯 1 つぶん低くてよい。旧に帯超えで負けて決着した設計が
        # 新には帯の内側、ということが起きるので、決着は昇格で解ける(archive の点で再判定される)。
        cases = _partial_cases(12)
        sp = split_cases(cases)
        S = [c.case_id for c in sp["search"]]
        C = [c.case_id for c in sp["confirm"]]
        # a: search 6/6・confirm 0。c: search 5/6(帯の内側)・confirm 0.6。b: search 4/6・confirm 1.0
        Scripted.WINS = {"a": set(S), "c": set(S[1:]) | {C[0]}, "b": set(S[2:]) | set(C)}
        Scripted.PARTIAL = {"c": set(C[1:])}
        pool = {"a": _lane("a"), "b": _lane("b"), "c": _lane("c")}
        seen = []
        res = grow(pool, cases=cases, generations=2, width=2, patience=3, min_margin=0.05,
                   on_event=lambda r: seen.append(r))
        gen0, gen1 = res["history"][0], res["history"][1]
        cands0 = {r["label"]: r for r in seen if r["event"] == "candidate" and r["gen"] == 0}
        b_hash = cands0["route:qa->b"]["hash"]
        ck = {r["gen"]: r for r in seen if r["event"] == "checkpoint"}
        # gen0: b は旧チャンピオン(1.0)に 2 問負けて決着、tool:qa(a) は同点で挑戦して confirm で落ちる
        self.assertEqual(ck[0]["settled"], [b_hash])
        self.assertEqual(ck[0]["challenged"], [gen0["challenger_hash"]])
        # gen1: c が 1 問負けの帯の内側から挑戦して昇格。決着は空になる
        self.assertEqual((gen1["challenger"], gen1["reason"]), ("route:qa->c", "promote"))
        self.assertEqual(ck[1]["settled"], [])
        # 新チャンピオンの下では b は帯の内側(0.6667 vs 0.8333)で、次の propose に戻ってくる
        band = 1 / len(S)
        self.assertFalse(search_gate(ck[0]["champion_search"]["score"],
                                     ck[1]["archive"][b_hash]["search"]["score"], band)[0])
        self.assertTrue(search_gate(ck[1]["champion_search"]["score"],
                                    ck[1]["archive"][b_hash]["search"]["score"], band)[0])
        again = propose(ck[1]["champion"], pool, ["qa"], width=12,
                        exclude=set(ck[1]["challenged"]) | set(ck[1]["settled"]), generation=2)
        self.assertIn(b_hash, {spec_hash(c.spec) for c in again})

    def test_a_within_band_runner_up_is_not_settled(self):
        # gen0: search 同点の tool:qa(a) が最高点で挑戦し、confirm で落ちる。帯の内側にいた
        # 次点 route:qa->c は決着済みにならず、gen1 に archive の点のまま挑戦して昇格する。
        cases, search, confirm = self._split()
        Scripted.WINS = {"a": set(search), "c": set(search[1:]) | set(confirm)}
        pool = {"a": _lane("a"), "c": _lane("c")}
        seen = []
        res = grow(pool, cases=cases, generations=2, width=2, patience=3, min_margin=0.05,
                   on_event=lambda r: seen.append(r))
        gen0, gen1 = res["history"][0], res["history"][1]
        self.assertEqual(gen0["challenger"], "tool:qa(a)", gen0)
        self.assertNotEqual(gen0["reason"], "promote", gen0)
        cands0 = {r["label"]: r for r in seen if r["event"] == "candidate" and r["gen"] == 0}
        self.assertIn("route:qa->c", cands0)
        self.assertLess(cands0["route:qa->c"]["search"]["score"], gen0["champion_search"])
        ck0 = next(r for r in seen if r["event"] == "checkpoint" and r["gen"] == 0)
        self.assertIn(gen0["challenger_hash"], ck0["challenged"])           # 落ちた方は決着
        self.assertNotIn(cands0["route:qa->c"]["hash"], ck0["challenged"])  # 次点は踏み石
        self.assertNotIn(cands0["route:qa->c"]["hash"], ck0["settled"])
        # gen1: 踏み石は archive の点のまま挑戦者の候補に並ぶ(candidate イベントは新規測定の
        # ときだけ出るので、gen1 に route:qa->c の行が無い = 測り直していない)。席は使わないので
        # 新顔は width ぶん測られ、そのうち search で上回った meshflow が点で勝って昇格する。
        gen1_fresh = [r["label"] for r in seen if r["event"] == "candidate" and r["gen"] == 1]
        self.assertNotIn("route:qa->c", gen1_fresh, gen1_fresh)
        self.assertEqual((gen1["new_candidates"], gen1["candidates"]), (2, 3), gen1)
        self.assertEqual((gen1["challenger"], gen1["challenger_from"], gen1["reason"]),
                         ("meshflow:qa(a->c)", "new", "promote"), gen1)
        self.assertGreater(gen1["challenger_search"], cands0["route:qa->c"]["search"]["score"])

    def test_a_measured_tie_keeps_its_place_without_taking_a_width_slot(self):
        # 全部 search 同点・confirm は全敗: 挑戦者は毎世代 1 本ずつ却下される。測定済みの同点は
        # 席を使わず(新顔が毎世代 width 本測られる)、archive の点のまま挑戦者になる。以前は
        # 測定済みが席を 1 つ使い、この 4 世代で新顔は 2,1,1,2 本だった(run W の再生では
        # 4,3,2,3,1,0 で、踏み石が出直せるのは巡回が同じ設計を再び出した時だけだった)。
        cases, search, confirm = self._split()
        Scripted.WINS = {"a": set(search), "b": set(search), "c": set(search)}
        pool = {"a": _lane("a"), "b": _lane("b"), "c": _lane("c")}
        seen = []
        res = grow(pool, cases=cases, generations=4, width=2, patience=9, min_margin=0.05,
                   on_event=lambda r: seen.append(r))
        gens = res["history"]
        self.assertEqual(len(gens), 4)
        self.assertEqual([g["new_candidates"] for g in gens], [2, 2, 2, 2])
        self.assertEqual([g["candidates"] for g in gens], [2, 3, 4, 5])   # the pool grows by
        self.assertTrue(all(g["reason"] == "confirm-not-better" for g in gens), gens)  # one
        fresh = {}
        for r in seen:
            if r["event"] == "candidate":
                fresh.setdefault(r["gen"], []).append(r["label"])
        self.assertEqual({g: len(v) for g, v in fresh.items()}, {0: 2, 1: 2, 2: 2, 3: 2})
        from_archive = [g for g in gens if g["challenger_from"] == "archive"]
        self.assertTrue(from_archive, [g["challenger"] for g in gens])
        for g in from_archive:
            # measured in an earlier generation, not this one, and never re-measured
            self.assertNotIn(g["challenger"], fresh[g["gen"]])
            self.assertEqual(sum(g["challenger"] in v for v in fresh.values()), 1)
            self.assertTrue(any(g["challenger"] in fresh[e] for e in range(g["gen"])))
        self.assertEqual(gens[0]["challenger_from"], "new")

    def test_the_prescription_is_challenged_in_the_generation_it_is_proposed(self):
        # 種の tool レーンがコードを書かない(Scripted は文だけ返す)ので診断は qa に出る。width 2
        # の gen0 の席は巡回なら simplify と route で、処方はその席を待たずに先に取る。search 同点
        # の simplify:qa->a は構造が小さくラベル順も先なので、処方でなければそちらが挑戦していた。
        class Chatty(Scripted):
            name = "chatty"
            supports_prefill = True

        backends_mod._BACKENDS["chatty"] = Chatty
        try:
            cases, search, confirm = self._split()
            Scripted.WINS = {"a": set(search), "b": set(search)}
            pool = {"a": {"backend": "chatty", "kwargs": {"tag": "a"}}, "b": _lane("b")}
            seed = seed_champion(pool, "a")
            seed["kwargs"]["backends"]["tool(a)"] = {
                "backend": "tool", "_grow_base": "a", "kwargs": {"inner": pool["a"]}}
            seed["kwargs"]["routing_table"]["qa"] = "tool(a)"
            seen = []
            res = grow(pool, cases=cases, generations=1, width=2, patience=3, min_margin=0.05,
                       seed_spec=seed, on_event=lambda r: seen.append(r))
            seed_row = next(r for r in seen if r["event"] == "seed")
            self.assertGreater(seed_row["search"]["tool_no_code_by_class"].get("qa", 0), 0)
            cands = {r["label"]: r for r in seen if r["event"] == "candidate" and r["gen"] == 0}
            self.assertEqual(set(cands), {"simplify:qa->a", "tool:qa(a)+prefill"}, cands)
            self.assertEqual(cands["simplify:qa->a"]["search"]["score"],
                             cands["tool:qa(a)+prefill"]["search"]["score"])
            specs = {c.label: c.spec for c in propose(canonical(seed), pool, ["qa"], width=2,
                                                      no_code_by_class={"qa": 4})}
            self.assertLessEqual(_structure_size(specs["simplify:qa->a"]),
                                 _structure_size(specs["tool:qa(a)+prefill"]))
            gen0 = res["history"][0]
            self.assertEqual((gen0["challenger"], gen0["challenger_from"]),
                             ("tool:qa(a)+prefill", "new"), gen0)
            # without the diagnosis the same width gives the rotation's seats and no prescription
            plain = [c.label for c in propose(canonical(seed), pool, ["qa"], width=2)]
            self.assertNotIn("tool:qa(a)+prefill", plain, plain)
        finally:
            backends_mod._BACKENDS.pop("chatty", None)


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

    def test_a_class_the_champion_aces_on_search_is_still_mutated_when_confirm_has_room(self):
        # 以前は「search を取り切ったクラスは変異が search を 1 点も上げられず、挑戦権が取れない」
        # として飽和に数えていた。挑戦権が「厳密に上回ること」だった頃の算術で、今の門 ① は
        # 同点を通す。run V では qa が confirm に 1.25 問残しながら search 側の理由だけで
        # 5 世代とも試されなかった。取り切ったクラスでも confirm に余地があれば試す。
        cases = _cases(8)
        sp = split_cases(cases)
        search_ids = {c.case_id for c in sp["search"]}
        # a は search を取り切り confirm を全部落とす。b は全部取る(search は同点、confirm で勝つ)
        Scripted.WINS = {"a": set(search_ids), "b": {c.case_id for c in cases}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        seen = []
        res = grow(pool, cases=cases, generations=1, width=1, patience=3, min_margin=0.05,
                   on_event=lambda r: seen.append(r))
        self.assertFalse([r for r in seen if r["event"] == "saturated"],
                         "an aced-on-search class with confirm headroom was called saturated")
        gen0 = res["history"][0]
        self.assertEqual(gen0["challenger"], "route:qa->b")
        self.assertEqual(gen0["challenger_search"], gen0["champion_search"])   # 同点で挑戦
        self.assertEqual(gen0["reason"], "promote")

    def test_saturation_still_applies_after_a_resume(self):
        # per_case を checkpoint から落としていたせいで、再開後に静かに死ぬ機能を 3 回作った。
        # 3 回目は「見せる形」と「続きを走らせる形」を分けて直したので、そこを固定する。
        # 飽和は confirm の伸びしろで決まる(search 側の判定は挑戦権が同点を通すようになって
        # 撤去した)。再開後もそれが効くことと、checkpoint が per_case を運ぶことを見る。
        cases = _cases(8, "qa", "qa") + _cases(8, "research", "re")
        Scripted.WINS = {"a": {f"qa{i}" for i in range(1, 9)},
                         "b": {f"qa{i}" for i in range(1, 9)}}
        pool = {"a": _lane("a"), "b": _lane("b")}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "run.jsonl"
            grow(pool, cases=cases, generations=1, width=6, patience=3,
                 ledger_path=str(led), min_margin=0.05)
            ck = load_checkpoint(led)
            for side in ("champion_search", "champion_confirm"):
                self.assertTrue(ck[side].get("per_case"),
                                f"the checkpoint dropped the per-case {side} scores a resume needs")
            seen = []
            grow(pool, cases=cases, generations=2, width=6, patience=3,
                 resume_from=str(led), min_margin=0.05, on_event=lambda r: seen.append(r))
        sat = [r for r in seen if r["event"] == "saturated"]
        self.assertTrue(sat, "saturation was not detected after a resume")
        self.assertIn("qa", sat[0]["classes"])

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
        # 本命の性質: **昇格したあとのチャンピオン**が per_case を持ち続けていること
        # (保存形式が正しくても、昇格の受け渡しで落ちたら search 側の飽和判定は死ぬ)。
        promoted = [r for r in seen if r["event"] == "generation" and r["verdict"] == "promote"]
        self.assertTrue(promoted, "nothing was promoted, so this test checked nothing")
        after = [r for r in ck if r["gen"] >= promoted[0]["gen"]]
        self.assertTrue(after, "no checkpoint after the promotion")
        self.assertTrue(after[0]["champion_search"].get("per_case"),
                        "the champion lost its per-case search scores when it was promoted")


class TestToolLaneHealth(unittest.TestCase):
    """tool レーンはコードを取り出せないと**黙って**素の返答を返す。例外にならないので
    error_rate にも出ず、低い得点として data に混ざる。実測(2026-09-02, Kimi-48B): crux の
    research 4 問はすべて ```python が一度も出ず、生の思考文が採点されていた。"""

    def setUp(self):
        reset_tool_stats()

    def tearDown(self):
        reset_tool_stats()

    def test_a_tool_lane_that_never_runs_code_is_counted(self):
        from gama.benchmark import SUITES
        # echo はコードを書かないので、tool レーンは毎回 fall back する
        m = measure({"backend": "tool", "kwargs": {"inner": {"backend": "echo", "kwargs": {}}}},
                    SUITES["default"][:3])
        self.assertEqual(m.tool_calls, 3)
        self.assertEqual(m.tool_ran, 0)
        # echo はコードを一切書かないので、原因は「コードが出てこない」側に立つ
        self.assertEqual(m.tool_no_code, 3)
        self.assertEqual(m.tool_empty_out, 0)

    def test_the_no_code_count_is_kept_per_class(self):
        from gama.benchmark import SUITES
        cases = SUITES["default"][:3]
        m = measure({"backend": "tool", "kwargs": {"inner": {"backend": "echo", "kwargs": {}}}},
                    cases)
        expect = {}
        for c in cases:
            expect[c.task_type] = expect.get(c.task_type, 0) + 1
        self.assertEqual(m.tool_no_code_by_class, expect)
        self.assertEqual(sum(m.tool_no_code_by_class.values()), m.tool_no_code)
        # a lane that writes no code because it is not a tool lane has no symptom to report
        m = measure({"backend": "echo", "kwargs": {}}, cases)
        self.assertEqual(m.tool_no_code_by_class, {})

    def test_a_checkpoint_written_before_the_diagnosis_restores_without_one(self):
        from gama.grow import _restore, _state
        m = measure({"backend": "echo", "kwargs": {}},
                    __import__("gama.benchmark", fromlist=["SUITES"]).SUITES["default"][:2])
        d = _state(m)
        d.pop("tool_no_code_by_class")
        self.assertEqual(_restore(d).tool_no_code_by_class, {})

    def test_a_lane_without_a_tool_reports_zero_calls(self):
        from gama.benchmark import SUITES
        m = measure({"backend": "echo", "kwargs": {}}, SUITES["default"][:3])
        self.assertEqual(m.tool_calls, 0)     # 0 は「道具を通っていない」であって失敗ではない

    def test_the_counter_separates_ran_from_fell_back(self):
        note_tool(ran=True, had_code=True)
        note_tool(ran=False, had_code=False)     # コードが出てこなかった
        note_tool(ran=False, had_code=True)      # コードは走ったが出力が空
        self.assertEqual(tool_stats(),
                         {"calls": 3, "ran": 1, "no_code": 1, "empty_out": 1})


class TestSshExtraBody(unittest.TestCase):
    """サーバ固有の sampling パラメータ(llama.cpp の repeat_penalty 等)が config から
    届くこと。届かないせいで、temperature 0 の反復ループに手が出せなかった。"""

    def test_extra_body_reaches_the_payload_but_cannot_break_required_fields(self):
        import json as _json
        from gama.backends import SshOpenAIBackend
        from gama.models import ModelTier
        be = SshOpenAIBackend(ssh_host="h", extra_body={"repeat_penalty": 1.15,
                                                        "model": "hijacked",
                                                        "stream": True})
        captured = {}

        class _Proc:
            returncode = 0
            stdout = _json.dumps({"choices": [{"message": {"content": "ok"}}]})
            stderr = ""

        import gama.backends as b
        real = b.subprocess.run
        b.subprocess.run = lambda *a, **k: (captured.update(_json.loads(k["input"])), _Proc())[1]
        try:
            be.complete("hi", ModelTier.LARGE)
        finally:
            b.subprocess.run = real
        self.assertEqual(captured["repeat_penalty"], 1.15)
        self.assertNotEqual(captured["model"], "hijacked")   # 必須フィールドは奪われない
        self.assertFalse(captured["stream"])
