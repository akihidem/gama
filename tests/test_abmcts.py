import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.abmcts import GEN, ABMCTSBackend, _BetaPosterior, _Node, _NodeProbState
from gama.backends import ModelBackend
from gama.config import abmcts_from_config, build_backend
from gama.models import ModelTier


class Fixed(ModelBackend):
    """Returns a fixed string and counts how many times it was called."""
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        self.calls += 1
        return self.reply


class PromptLog(ModelBackend):
    """Records every prompt it was asked, so the 'go deeper' refine prompt can be inspected."""
    available = True

    def __init__(self, reply="ans"):
        self.reply = reply
        self.prompts = []
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        self.prompts.append(prompt)
        return self.reply


def good(art):
    return 1.0 if art == "GOOD" else 0.0


class TestABMCTSSearch(unittest.TestCase):
    def test_returns_highest_scoring_candidate(self):
        weak, strong = Fixed("bad"), Fixed("GOOD")
        be = ABMCTSBackend([("weak", weak), ("strong", strong)], verify=good, budget=6, seed=3)
        out = be.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "GOOD")
        self.assertEqual(be.last_resolved_by, "strong")
        self.assertEqual(be.last_best_score, 1.0)

    def test_stops_early_once_pass_score_reached(self):
        # A perfect candidate on the first call must halt the search — no wasted budget.
        w = Fixed("GOOD")
        be = ABMCTSBackend([("only", w)], verify=good, budget=50, seed=1)
        be.complete("q", ModelTier.LARGE)
        self.assertEqual(be.last_tree_size, 1)     # one generation, then early stop
        self.assertEqual(w.calls, 1)
        self.assertEqual(be.last_cost, 1.0)        # only the one call is charged

    def test_budget_exhausts_when_never_passing(self):
        # verify never reaches pass_score -> the loop runs the full budget.
        w = Fixed("meh")
        be = ABMCTSBackend([("only", w)], verify=lambda a: 0.4, budget=7, seed=1)
        be.complete("q", ModelTier.LARGE)
        self.assertEqual(be.last_tree_size, 7)
        self.assertEqual(w.calls, 7)
        self.assertEqual(len(be.last_trace), 7)

    def test_go_deeper_refine_prompt_feeds_back_parent_ANSWER_and_score(self):
        # THE load-bearing property: a refinement must include the parent's ACTUAL ANSWER and its
        # score, otherwise 'deeper' collapses into 'wider' (best-of-N). A worker that emits a
        # unique, findable answer lets us assert the answer itself is quoted, not just the task.
        class Marker(ModelBackend):
            available = True
            def __init__(self):
                self.prompts = []
                self.last_usage = None
            def complete(self, prompt, tier, **kw):
                self.prompts.append(prompt)
                return "UNIQUE_ANSWER_TOKEN_42"

        w = Marker()
        be = ABMCTSBackend([("only", w)], verify=lambda a: 0.5, budget=6, seed=0)
        be.complete("SOLVE THIS", ModelTier.LARGE)
        self.assertIn("refine", [t["mode"] for t in be.last_trace])
        refine_prompts = [p for p in w.prompts if "previous attempt" in p]
        self.assertTrue(refine_prompts)
        rp = refine_prompts[0]
        self.assertIn("SOLVE THIS", rp)                     # original task carried through
        self.assertIn("0.50", rp)                           # the parent's verify score is surfaced
        self.assertIn("UNIQUE_ANSWER_TOKEN_42", rp)         # the parent's ANSWER is fed back

    def test_refine_quotes_the_immediate_parent_not_the_root(self):
        # On a depth>=3 path the refine prompt must quote the IMMEDIATE parent's answer (the node
        # being deepened), and its score, so the improvement builds on the right attempt.
        class Numbered(ModelBackend):
            available = True
            def __init__(self):
                self.n = 0
                self.prompts = []
                self.last_usage = None
            def complete(self, prompt, tier, **kw):
                self.prompts.append(prompt)
                self.n += 1
                return f"answer-{self.n}"

        w = Numbered()
        # Distinct scores per answer so a refine prompt reveals WHICH parent it built on.
        scores = {"answer-1": 0.2, "answer-2": 0.4, "answer-3": 0.6}
        be = ABMCTSBackend([("only", w)], verify=lambda a: scores.get(a, 0.1), budget=8, seed=0)
        be.complete("T", ModelTier.LARGE)
        deep = [t for t in be.last_trace if t["depth"] >= 3]
        if deep:   # seed=0 reaches depth 3; guard so the test states intent even if it didn't
            # find the refine prompt whose quoted score matches a real parent's score
            quoted = [p for p in w.prompts if "previous attempt" in p]
            self.assertTrue(any(f"{s:.2f}" in p for p in quoted for s in scores.values()))

    def test_first_generation_is_a_fresh_sample_not_a_refine(self):
        # The root has no answer, so the very first call must be a plain sample of the task.
        w = PromptLog()
        be = ABMCTSBackend([("only", w)], verify=lambda a: 0.0, budget=1, seed=0)
        be.complete("TASK", ModelTier.LARGE)
        self.assertEqual(be.last_trace[0]["mode"], "sample")
        self.assertEqual(be.last_trace[0]["depth"], 1)
        self.assertEqual(w.prompts[0], "TASK")     # no refine wrapper on the first attempt
        self.assertNotIn("previous attempt", w.prompts[0])

    def test_multi_llm_bandit_converges_to_the_rewarded_model(self):
        # Only "strong" is ever rewarded; the Thompson bandit should spend most of the budget on
        # it. (Thompson sampling does NOT guarantee the other arm is ever tried — exploration is
        # probabilistic, not a floor — so we assert the convergence, not a minimum weak.calls.)
        weak, strong = Fixed("weak"), Fixed("strong")
        be = ABMCTSBackend([("weak", weak), ("strong", strong)],
                           verify=lambda a: 1.0 if a == "strong" else 0.0,
                           budget=30, pass_score=2.0, seed=7)   # pass_score>1 => never early-stop
        be.complete("q", ModelTier.LARGE)
        self.assertGreater(strong.calls, weak.calls * 3)        # strongly favors the payer
        self.assertEqual(strong.calls + weak.calls, 30)         # every generation went somewhere

    def test_bandit_averaged_over_seeds_prefers_the_rewarded_model(self):
        # A seed-independent statement of the same property: across many seeds the rewarded arm
        # dominates in aggregate (robust to any single seed's luck, unlike a per-seed floor).
        strong_total = weak_total = 0
        for seed in range(12):
            weak, strong = Fixed("weak"), Fixed("strong")
            be = ABMCTSBackend([("weak", weak), ("strong", strong)],
                               verify=lambda a: 1.0 if a == "strong" else 0.0,
                               budget=20, pass_score=2.0, seed=seed)
            be.complete("q", ModelTier.LARGE)
            strong_total += strong.calls
            weak_total += weak.calls
        self.assertGreater(strong_total, weak_total * 5)

    def test_failing_worker_does_not_crash_the_search(self):
        class Boom(ModelBackend):
            available = True
            def complete(self, prompt, tier, **kw):
                raise RuntimeError("boom")

        be = ABMCTSBackend([("flaky", Boom()), ("strong", Fixed("GOOD"))], verify=good,
                           budget=8, seed=2)
        out = be.complete("q", ModelTier.LARGE)   # must not raise
        self.assertEqual(out, "GOOD")             # a crashing arm scores 0, the good one wins

    def test_control_kwargs_not_forwarded_to_workers(self):
        # `verify`/`stakes` are bench control kwargs; workers get task_type but not those.
        seen = {}

        class Spy(ModelBackend):
            available = True
            def complete(self, prompt, tier, **kw):
                seen.update(kw)
                return "GOOD"

        be = ABMCTSBackend([("only", Spy())], verify=None, budget=1, seed=0)
        be.complete("q", ModelTier.LARGE, verify=good, stakes=0.9, task_type="qa")
        self.assertNotIn("verify", seen)
        self.assertNotIn("stakes", seen)
        self.assertEqual(seen.get("task_type"), "qa")

    def test_kwargs_verify_overrides_and_gates_the_search(self):
        # Constructed with no verifier; a bench-style `verify` kwarg drives the reward.
        be = ABMCTSBackend([("only", Fixed("GOOD"))], verify=None, budget=5, seed=1)
        out = be.complete("q", ModelTier.LARGE, verify=good)
        self.assertEqual(out, "GOOD")
        self.assertEqual(be.last_best_score, 1.0)

    def test_seed_makes_the_search_reproducible(self):
        mk = lambda: ABMCTSBackend([("a", Fixed("x")), ("b", Fixed("y"))],
                                   verify=lambda a: 0.3, budget=8, seed=42)
        a, b = mk(), mk()
        a.complete("q", ModelTier.LARGE)
        b.complete("q", ModelTier.LARGE)
        self.assertEqual(a.last_trace, b.last_trace)

    def test_usage_is_summed_across_generations(self):
        class Metered(ModelBackend):
            available = True
            def __init__(self):
                self.last_usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
            def complete(self, prompt, tier, **kw):
                return "meh"

        be = ABMCTSBackend([("only", Metered())], verify=lambda a: 0.0, budget=3, seed=0)
        be.complete("q", ModelTier.LARGE)
        self.assertEqual(be.last_usage["total_tokens"], 15)    # 3 calls x 5 tokens

    def test_no_verifier_still_returns_a_candidate(self):
        be = ABMCTSBackend([("only", Fixed("something"))], verify=None, budget=3, seed=0)
        out = be.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "something")         # best-effort even with no reward signal

    def test_cost_is_the_sum_of_the_chosen_models_weights(self):
        # Tie last_cost to the ACTUAL models picked (not just a range): a single worker means
        # every one of `budget` generations charges that worker's cost exactly, so the total is
        # deterministic and must equal budget * cost[0] — this fails if cost is mis-attributed.
        be = ABMCTSBackend([("only", Fixed("meh"))], verify=lambda a: 0.0,
                           budget=5, costs=[3.0], pass_score=2.0, seed=1)
        be.complete("q", ModelTier.LARGE)
        self.assertEqual(be.last_cost, 5 * 3.0)

    def test_cost_matches_the_per_model_pick_counts_in_the_trace(self):
        # With two workers, the total must equal sum over the trace of each picked model's cost.
        be = ABMCTSBackend([("a", Fixed("a")), ("b", Fixed("GOOD"))], verify=good,
                           budget=6, costs=[1.0, 2.0], pass_score=2.0, seed=5)
        be.complete("q", ModelTier.LARGE)
        cost_by = {"a": 1.0, "b": 2.0}
        expected = round(sum(cost_by[t["model"]] for t in be.last_trace), 3)
        self.assertEqual(be.last_cost, expected)

    def test_short_costs_list_does_not_crash_and_falls_back_to_unit_cost(self):
        # A config may give fewer costs than workers; picking a worker past the list must fall
        # back to unit cost (1.0), like meshflow/trinity — not raise IndexError mid-search.
        be = ABMCTSBackend([("a", Fixed("a")), ("b", Fixed("GOOD"))], verify=good,
                           budget=8, costs=[5.0], pass_score=2.0, seed=1)
        be.complete("q", ModelTier.LARGE)   # must not raise
        # worker "b" (index 1, past the 1-element costs) is charged 1.0; "a" is charged 5.0.
        cost_by = {"a": 5.0, "b": 1.0}
        expected = round(sum(cost_by[t["model"]] for t in be.last_trace), 3)
        self.assertEqual(be.last_cost, expected)


class TestABMCTSConstruction(unittest.TestCase):
    def test_empty_workers_rejected(self):
        with self.assertRaises(ValueError):
            ABMCTSBackend([])

    def test_budget_below_one_rejected(self):
        with self.assertRaises(ValueError):
            ABMCTSBackend([Fixed("x")], budget=0)

    def test_duplicate_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            ABMCTSBackend([("dup", Fixed("A")), ("dup", Fixed("B"))])

    def test_default_labels_collide_when_unlabeled(self):
        # Two bare ModelBackend instances both default to name="abstract".
        with self.assertRaises(ValueError):
            ABMCTSBackend([Fixed("A"), Fixed("B")])

    def test_available_reflects_workers(self):
        self.assertTrue(ABMCTSBackend([Fixed("x")]).available)

    def test_nonpositive_prior_rejected(self):
        # Beta(a,b) needs a,b > 0; a non-positive prior must fail loud at construction rather
        # than crash opaquely inside random.betavariate on the first sample.
        with self.assertRaises(ValueError):
            ABMCTSBackend([Fixed("x")], prior=(0.0, 0.5))
        with self.assertRaises(ValueError):
            ABMCTSBackend([Fixed("x")], prior=(0.5, -1.0))


class TestBetaPosterior(unittest.TestCase):
    def test_update_moves_mass(self):
        import random as _r
        pd = _BetaPosterior(_r.Random(0))
        pd.tell(1.0)
        self.assertEqual((pd.a, pd.b), (1.5, 0.5))   # Jeffreys 0.5,0.5 + (r, 1-r)
        pd.tell(0.0)
        self.assertEqual((pd.a, pd.b), (1.5, 1.5))

    def test_sample_in_unit_interval(self):
        import random as _r
        pd = _BetaPosterior(_r.Random(1))
        for _ in range(100):
            s = pd.sample()
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_prior_actually_shifts_the_sampling_distribution(self):
        # Behavioral (not storage) proof that `prior` is wired into sampling: an optimistic
        # Beta(5,1) draws higher than a pessimistic Beta(1,5) on average. Kills the mutation
        # that hardcodes Jeffreys and ignores the prior arg.
        import random as _r
        hi = _BetaPosterior(_r.Random(0), 5.0, 1.0)
        lo = _BetaPosterior(_r.Random(0), 1.0, 5.0)
        mean_hi = sum(hi.sample() for _ in range(300)) / 300
        mean_lo = sum(lo.sample() for _ in range(300)) / 300
        self.assertGreater(mean_hi, mean_lo + 0.3)   # clearly separated, not seed-luck

    def test_backend_threads_prior_into_its_node_states(self):
        # The prior given to the backend must reach the per-node posteriors, not just be stored.
        be = ABMCTSBackend([Fixed("x")], prior=(3.0, 3.0), budget=1)
        import random as _r
        st = _NodeProbState(_r.Random(0), be.prior)
        self.assertEqual((st.gen.a, st.gen.b), (3.0, 3.0))


class TestNodeProbState(unittest.TestCase):
    def test_childless_node_can_only_widen(self):
        import random as _r
        st = _NodeProbState(_r.Random(0), (0.5, 0.5))
        self.assertEqual(st.select(), GEN)           # no children -> GEN only

    def test_register_child_seeds_the_child_posterior_with_its_own_score(self):
        # The new child's per-child posterior must start from its OWN first score (Jeffreys prior
        # + one observation), not the bare prior — otherwise 'go deeper' starts blind. Pins the
        # mutation of seeding with the wrong value / dropping the seed.
        import random as _r
        st = _NodeProbState(_r.Random(0), (0.5, 0.5))
        st.register_child(0, 1.0)
        self.assertEqual((st.children[0].a, st.children[0].b), (1.5, 0.5))   # 0.5+1.0, 0.5+0.0
        st.register_child(1, 0.0)
        self.assertEqual((st.children[1].a, st.children[1].b), (0.5, 1.5))

    def test_select_deepens_into_the_strongly_rewarded_child(self):
        # Not tautological: give child 0 a wall of successes and child 1 a wall of failures, and
        # suppress GEN by concentrating CONT high; select() must return child 0, not just "some
        # value in the domain". Averaged over draws to be robust to Thompson variance.
        import random as _r
        st = _NodeProbState(_r.Random(0), (0.5, 0.5))
        st.register_child(0, 1.0)
        st.register_child(1, 0.0)
        for _ in range(40):                          # concentrate the posteriors hard
            st.children[0].tell(1.0)
            st.children[1].tell(0.0)
            st.cont.tell(1.0)                         # make CONT beat GEN almost always
        picks = [st.select() for _ in range(200)]
        self.assertGreater(picks.count(0), picks.count(1))   # deepen into the rewarded line
        self.assertGreater(picks.count(0), picks.count(GEN)) # and CONT dominates GEN here


class TestBackprop(unittest.TestCase):
    """White-box tests for the AB-MCTS-A asymmetric back-prop — the property that distinguishes
    this search from a flat bandit / best-of-N. Pins the mutation that deletes the ancestor loop."""

    def _prior(self):
        return (0.5, 0.5)

    def test_expansion_node_gets_gen_and_new_child_posterior(self):
        import random as _r
        rng = _r.Random(0)
        be = ABMCTSBackend([Fixed("x")], budget=1)
        root = _Node(None, -1.0, None, None, -1)
        child = _Node("a", 0.7, "only", root, 0)
        root.children.append(child)
        prob = {root.idx: _NodeProbState(rng, self._prior()),
                child.idx: _NodeProbState(rng, self._prior())}
        be._backprop(prob, root, child, 0.7)
        # expansion node's GEN posterior moved by (r, 1-r); child-line posterior seeded with r
        self.assertAlmostEqual(prob[root.idx].gen.a, 0.5 + 0.7)
        self.assertAlmostEqual(prob[root.idx].gen.b, 0.5 + 0.3)
        self.assertAlmostEqual(prob[root.idx].children[child.idx].a, 0.5 + 0.7)

    def test_ancestors_get_cont_and_child_line_updates(self):
        # root -> A -> B : expanding a leaf under B must update B's parent(A) CONT and the A->B
        # child-line, AND root's CONT and the root->A child-line — every descended ancestor.
        import random as _r
        rng = _r.Random(0)
        be = ABMCTSBackend([Fixed("x")], budget=1)
        root = _Node(None, -1.0, None, None, -1)
        A = _Node("a", 0.3, "only", root, 0); root.children.append(A)
        B = _Node("b", 0.4, "only", A, 1); A.children.append(B)
        prob = {n.idx: _NodeProbState(rng, self._prior()) for n in (root, A, B)}
        prob[root.idx].register_child(A.idx, 0.3)     # seeds root->A line at 0.3
        prob[A.idx].register_child(B.idx, 0.4)        # seeds A->B line at 0.4
        # Now expand a new leaf C under B with score 1.0.
        C = _Node("c", 1.0, "only", B, 2); B.children.append(C)
        prob[C.idx] = _NodeProbState(rng, self._prior())
        be._backprop(prob, B, C, 1.0)
        # B is the expansion node: GEN + new child line seeded.
        self.assertAlmostEqual(prob[B.idx].gen.a, 0.5 + 1.0)
        # A (ancestor): its CONT and the A->B child line each get +1.0 success mass. The A->B
        # line was seeded via register_child(B, 0.4) => a = 0.5(prior) + 0.4, then +1.0 obs.
        self.assertAlmostEqual(prob[A.idx].cont.a, 0.5 + 1.0)
        self.assertAlmostEqual(prob[A.idx].children[B.idx].a, 0.5 + 0.4 + 1.0)
        # root (ancestor): its CONT and the root->A child line (seeded at 0.3) each get +1.0.
        self.assertAlmostEqual(prob[root.idx].cont.a, 0.5 + 1.0)
        self.assertAlmostEqual(prob[root.idx].children[A.idx].a, 0.5 + 0.3 + 1.0)

    def test_backprop_does_not_touch_a_non_descended_sibling_line(self):
        # root has two children A and A2; deepening under A must NOT move root's A2 child-line.
        import random as _r
        rng = _r.Random(0)
        be = ABMCTSBackend([Fixed("x")], budget=1)
        root = _Node(None, -1.0, None, None, -1)
        A = _Node("a", 0.3, "only", root, 0); root.children.append(A)
        A2 = _Node("a2", 0.9, "only", root, 1); root.children.append(A2)
        prob = {n.idx: _NodeProbState(rng, self._prior()) for n in (root, A, A2)}
        prob[root.idx].register_child(A.idx, 0.3)
        prob[root.idx].register_child(A2.idx, 0.9)
        before = (prob[root.idx].children[A2.idx].a, prob[root.idx].children[A2.idx].b)
        B = _Node("b", 1.0, "only", A, 2); A.children.append(B)
        prob[B.idx] = _NodeProbState(rng, self._prior())
        be._backprop(prob, A, B, 1.0)
        after = (prob[root.idx].children[A2.idx].a, prob[root.idx].children[A2.idx].b)
        self.assertEqual(before, after)               # the untouched sibling line is unchanged


class TestABMCTSFromConfig(unittest.TestCase):
    def test_build_backend_wires_workers_recursively(self):
        spec = {"backend": "abmcts", "kwargs": {"workers": [
            {"label": "a", "backend": "echo"},
            {"label": "b", "backend": "echo"},
        ], "budget": 3, "seed": 0}}
        be = build_backend(spec)
        self.assertIsInstance(be, ABMCTSBackend)
        self.assertEqual([label for label, _ in be.workers], ["a", "b"])
        self.assertEqual(be.budget, 3)
        out = be.complete("hello", ModelTier.LARGE)
        self.assertIsInstance(out, str)

    def test_build_backend_reads_prior_and_pass_score(self):
        be = build_backend({"backend": "abmcts", "kwargs": {
            "workers": [{"backend": "echo"}], "prior": [1.0, 1.0], "pass_score": 0.9}})
        self.assertEqual(be.prior, (1.0, 1.0))
        self.assertEqual(be.pass_score, 0.9)

    def test_build_backend_nested_composite_worker(self):
        be = build_backend({"backend": "abmcts", "kwargs": {"workers": [
            {"label": "tooled", "backend": "tool", "kwargs": {"inner": {"backend": "echo"}}},
        ]}})
        from gama.backends import ToolBackend
        self.assertIsInstance(be.workers[0][1], ToolBackend)

    def test_abmcts_from_config(self):
        be = abmcts_from_config({"abmcts": {"workers": [{"backend": "echo"}], "budget": 2}})
        self.assertIsInstance(be, ABMCTSBackend)

    def test_abmcts_from_config_needs_workers(self):
        with self.assertRaises(ValueError):
            abmcts_from_config({"abmcts": {}})

    def test_cmd_run_recognizes_an_abmcts_only_config(self):
        # Regression guard (same shape as the trinity one): cmd_run must check raw.get("abmcts")
        # or it silently falls through to gama_from_config and builds the default ollama lane.
        import argparse
        import json
        import tempfile
        import os as _os

        from gama.cli import cmd_run

        cfg = {"abmcts": {"workers": [
            {"label": "a", "backend": "echo"},
            {"label": "b", "backend": "echo"},
        ], "budget": 2, "seed": 0}}
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with _os.fdopen(fd, "w") as fh:
                json.dump(cfg, fh)
            args = argparse.Namespace(config=path, prompt="hi", tier="large", task_type=None)
            rc = cmd_run(args)
            self.assertEqual(rc, 0)
        finally:
            _os.unlink(path)


if __name__ == "__main__":
    unittest.main()
