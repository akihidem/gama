r"""abmcts — AB-MCTS-A (Adaptive Branching MCTS, node-aggregation variant) as a gama backend.

Source: Sakana AI, "Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive
Branching Tree Search" (Inoue, Misaki, Imajuku, Kuroki, Nakamura, Akiba; arXiv:2503.04412,
NeurIPS 2025 spotlight; blog https://sakana.ai/ab-mcts/). Reference OSS: SakanaAI/treequest
(Apache-2.0). This is a clean-room stdlib-only reimplementation of the algorithm from the
paper's equations — no treequest/numpy/scipy/PyMC code or dependency is used.

WHY this is gama's natural 5th composite mode. gama already has three ways to *combine*
small models statically — route (``GamaBackend``), ensemble (``EnsembleBackend``), escalate
under verification (``MeshflowBackend``) — plus one-shot predictive routing (``TrinityBackend``).
What none of them has is an *adaptive inference-time search*: at every node decide, from the
rewards seen so far, whether the next model call is better spent **going wider** (a brand-new
candidate) or **going deeper** (refining an existing one). AB-MCTS is exactly that decision,
and it slots onto gama's existing seams with zero new machinery:

  * reward         = gama's external ``verify(artifact) -> score in [0,1]`` (the SAME signal
                     ``MeshflowBackend`` gates on and ``gama bench`` threads in as the case
                     checker — honest, not model self-report).
  * generators     = any gama ``ModelBackend`` children (ollama / ssh-openai / tool / a whole
                     nested ``gama`` stack).
  * multi-LLM axis = each child backend is one bandit "action"; on a "go wider" step we
                     Thompson-sample WHICH model to call, so the search learns online which
                     model is paying off on *this* problem = Multi-LLM AB-MCTS, for free.

Reward model = Beta. gama's verify score is already in [0,1], which is the exact conjugate
match for a Beta posterior with the Jeffreys prior ``Beta(0.5, 0.5)``: observing reward ``r``
updates ``a += r; b += (1 - r)`` (soft-Bernoulli success mass), and a Thompson sample is just
``random.betavariate(a, b)`` — pure stdlib, no numpy/scipy. This is why AB-MCTS-**A** (closed-form
per-node posteriors) is implemented and AB-MCTS-**M** (a hierarchical mixed model fit by MCMC)
is deliberately NOT: -M needs PyMC/numpyro, which would break gama's stdlib-only contract, and
the paper's own -A results capture nearly all of the benefit. (Scope note in the spirit of
``trinity.py`` not reproducing openfugu's CMA-ES hidden-state head: we port the algorithm, not
a heavy-dependency variant of it. The Gaussian Normal-Inv-χ² reward model the paper offers for
*unbounded* scores is omitted for the same reason — gama scores are bounded to [0,1].)

The one detail that carries the whole method (prior art: Reflexion / LATS): "go deeper" MUST
feed the parent's answer *and* the verifier's score back into the refine prompt. If refinement
were a blind resample, depth would collapse into width and AB-MCTS would be a dressed-up
best-of-N. See ``_refine_prompt``.
"""
from __future__ import annotations

import random

from .backends import ModelBackend
# Reuse gama's single source of truth for the reward signal (same normalization + verifier
# resolution MeshflowBackend uses), so a config's `"verify": "code_runs"` and a bench's
# threaded case-checker behave identically here and there — no second, drifting copy.
from .meshflow import _normalize_score, resolve_verifier
from .models import ModelTier

# Cap how much of a parent answer is quoted back into a refine prompt. A tree can deepen many
# times; without a bound the refine prompt would grow with the whole answer each level and blow
# past context limits. 2000 mirrors ``backends.synthesize``'s per-candidate truncation.
_MAX_REFINE_CONTEXT = 2000

GEN = "GEN"  # sentinel: at a node, the "go wider" action (generate a fresh child here)


class _BetaPosterior:
    r"""Beta(a, b) conjugate posterior over a reward in [0, 1] — AB-MCTS-A's stdlib-friendly
    reward model. Jeffreys prior ``a = b = 0.5``. A reward ``r`` is treated as soft-Bernoulli
    success mass: ``a += r; b += (1 - r)`` (treequest ``ab_mcts_a/prob_state.py``). A Thompson
    sample is a draw from the posterior predictive, i.e. ``random.betavariate(a, b)`` — the only
    RNG primitive the whole algorithm needs, so seeding ``random.Random`` makes a run reproducible.
    """

    __slots__ = ("rng", "a", "b")

    def __init__(self, rng: random.Random, a: float = 0.5, b: float = 0.5):
        self.rng = rng
        self.a = a
        self.b = b

    def tell(self, reward: float) -> None:
        self.a += reward
        self.b += 1.0 - reward

    def sample(self) -> float:
        return self.rng.betavariate(self.a, self.b)


class _Node:
    r"""One tree node = one generated answer (or the synthetic root). ``idx`` is creation order
    and the stable id (root = -1); ``score`` is the external reward in [0, 1] (root = -1.0 so it
    never wins ``top_k``); ``action`` is the model label that produced this answer (None at root)."""

    __slots__ = ("answer", "score", "action", "parent", "children", "idx")

    def __init__(self, answer, score, action, parent, idx):
        self.answer = answer
        self.score = score
        self.action = action
        self.parent = parent
        self.children: list = []
        self.idx = idx


class _NodeProbState:
    r"""AB-MCTS-A per-node posteriors. A shared GEN-vs-CONT pair models the *wider-vs-deeper*
    bet at this node (GEN = value of a brand-new child here; CONT = value of descending into an
    existing child); one posterior per existing child models *which* child is worth deepening.

    GEN is never emptied and keeps its prior mass, so a fresh child can always be sampled —
    that is what makes branching unbounded/adaptive (vs. a fixed width in ToT/LATS/UCT-MCTS)."""

    __slots__ = ("rng", "prior", "gen", "cont", "children")

    def __init__(self, rng: random.Random, prior: tuple):
        self.rng = rng
        self.prior = prior
        self.gen = _BetaPosterior(rng, *prior)
        self.cont = _BetaPosterior(rng, *prior)
        self.children: dict = {}   # child.idx -> _BetaPosterior (that child line's value)

    def register_child(self, child_idx: int, score: float) -> None:
        """Seed a new child's per-child posterior with its own first score (treequest
        ``register_new_child_node``)."""
        pd = _BetaPosterior(self.rng, *self.prior)
        pd.tell(score)
        self.children[child_idx] = pd

    def select(self):
        """Thompson-sample the wider-vs-deeper decision -> ``GEN`` (expand a new child here) or a
        child idx (descend into it). A node with no children can only widen."""
        if not self.children:
            return GEN
        # Draw one sample from GEN and from CONT; the larger draw wins. Exploration comes from
        # posterior variance (a diffuse GEN occasionally beats a concentrated CONT and vice
        # versa) — no UCB constant to tune. Ties favor GEN, keeping the tree able to widen.
        if self.gen.sample() >= self.cont.sample():
            return GEN
        # CONT won: Thompson-sample among existing children which line to deepen.
        best_idx, best_val = None, None
        for idx, pd in self.children.items():
            v = pd.sample()
            if best_val is None or v > best_val:
                best_idx, best_val = idx, v
        return best_idx


class ABMCTSBackend(ModelBackend):
    r"""Adaptive Branching MCTS (AB-MCTS-A) over child model-backends, with the external
    ``verify`` score as the reward — gama's inference-time-search composite.

    Runs a ``budget`` of model calls. Each call: descend from the root, at every node
    Thompson-sampling GEN (go wider) vs CONT (go deeper into a child); at the chosen node,
    Thompson-sample WHICH child backend to call (the Multi-LLM bandit — skipped when there is
    only one worker); generate a candidate (a fresh answer at the root, else a refinement of the
    parent's answer that is fed the parent answer + its score); score it with ``verify``; and
    back-propagate the score up the path with the AB-MCTS-A asymmetry (a fresh child updates the
    expansion node's GEN posterior + its own per-child posterior; every ancestor we descended
    through updates its CONT posterior + that child line's posterior). The returned artifact is
    the highest-scoring node found.

    Args:
      workers: list of ``ModelBackend`` (the candidate generators), or ``(label, backend)``
        tuples. Each is one bandit action; labels must be unique (like ``TrinityBackend``).
      verify: ``(artifact) -> score in [0,1]`` callable, a built-in name (``"code_runs"`` /
        ``"nonempty"``), or None. A ``verify`` in ``complete()`` kwargs overrides this, so
        ``gama bench`` gates the search on its own case checker (honest measurement). None
        means no reward signal — the search still runs but has nothing to steer on, so it
        returns a best-effort candidate; the method is only worth it *with* a real verifier
        (prior art: repeated-sampling gains vanish without one).
      budget: number of model calls (candidate generations) = the compute budget. >= 1.
      prior: Beta prior ``(a, b)`` for every posterior; default Jeffreys ``(0.5, 0.5)``.
      pass_score: stop early once a candidate scores >= this (the verifier is fully satisfied —
        no point spending more budget); default 1.0. Set > 1.0 to always exhaust the budget.
      costs: per-worker cost weights (price proxy), by list order; default ``1, 2, 3, ...``
        (same proxy ``MeshflowBackend`` / ``TrinityBackend`` use). ``last_cost`` sums the cost
        of every generation actually made.
      seed: RNG seed. None (default) = a fresh nondeterministic search each call (so
        ``gama bench --repeats`` gets real variance); an int = reproducible (used by tests).
    """

    name = "abmcts"

    def __init__(self, workers, verify=None, budget: int = 8, prior=(0.5, 0.5),
                 pass_score: float = 1.0, costs=None, seed=None):
        if not workers:
            raise ValueError("ABMCTSBackend needs at least one worker")
        if budget < 1:
            raise ValueError("ABMCTSBackend budget must be >= 1")
        self.workers = []
        for i, w in enumerate(workers):
            if isinstance(w, (tuple, list)):
                label, be = w
            else:
                label, be = getattr(w, "name", f"worker{i}"), w
            self.workers.append((str(label), be))
        labels = [label for label, _ in self.workers]
        if len(set(labels)) != len(labels):
            # A label is both the bandit action key and the reward-history key; a duplicate would
            # silently merge two models' posteriors. Fail loud (mirrors TrinityBackend / gama's
            # fail-closed style).
            raise ValueError(f"ABMCTSBackend worker labels must be unique, got {labels}")
        self.by_label = dict(self.workers)
        self.action_index = {label: i for i, (label, _) in enumerate(self.workers)}
        self.verify = resolve_verifier(verify)
        self.budget = int(budget)
        self.prior = (float(prior[0]), float(prior[1]))
        # Beta(a, b) requires a > 0 and b > 0; a non-positive prior would not fail here but
        # deep inside random.betavariate on the first sample (an opaque crash mid-search). Fail
        # loud at construction instead (gama's fail-closed style, like the duplicate-label check).
        if self.prior[0] <= 0.0 or self.prior[1] <= 0.0:
            raise ValueError(f"ABMCTSBackend prior must be a pair of positive floats, got {prior}")
        self.pass_score = pass_score
        self.costs = list(costs) if costs else [float(i + 1) for i in range(len(self.workers))]
        self.seed = seed
        self.available = any(getattr(be, "available", False) for _, be in self.workers)
        self.last_usage = None
        self.last_trace = None          # [{"iter","model","depth","mode","score"}] — the search path
        self.last_resolved_by = None    # label of the model that produced the winning artifact
        self.last_cost = None           # summed cost of every generation made (price proxy)
        self.last_best_score = None     # the winning verify score
        self.last_tree_size = None      # number of candidates generated (<= budget; < on early stop)

    # ------------------------------------------------------------------ #
    def _reward(self, verify, artifact) -> float:
        """External reward in [0,1] (0.0 when there is no verifier or it errors) — the same
        contract as ``MeshflowBackend._score`` / ``benchmark.score_output``."""
        if verify is None:
            return 0.0
        try:
            return _normalize_score(verify(artifact))
        except Exception:
            return 0.0

    def _refine_prompt(self, prompt: str, parent_answer: str, parent_score: float) -> str:
        """Build the "go deeper" prompt. THIS is the load-bearing detail: it feeds the parent
        answer AND its verifier score back in, so a refinement is informed by the prior attempt
        rather than a blind resample. Without it, "deeper" degenerates into "wider" and the whole
        search collapses to best-of-N (prior art: Reflexion / LATS). The verifier only yields a
        scalar (no message), so the score itself is the feedback we can honestly surface."""
        quoted = (parent_answer or "")[:_MAX_REFINE_CONTEXT]
        return (f"{prompt}\n\n"
                f"A previous attempt scored {parent_score:.2f} out of 1.0:\n"
                f"--- previous attempt ---\n{quoted}\n--- end of previous attempt ---\n\n"
                "Improve on it to score higher. Output only the improved answer, "
                "following the original task's format exactly.")

    def _select_action(self, rng: random.Random, all_rewards: dict) -> str:
        """Multi-LLM bandit: Thompson-sample which model generates the next candidate. Build a
        fresh Beta posterior per action from its whole-tree reward history and take the argmax
        draw; an unseen action keeps the diffuse prior, giving it a high probability (not a
        guarantee) of being drawn, so exploration stays alive. With a single worker there is
        nothing to choose."""
        if len(self.workers) == 1:
            return self.workers[0][0]
        best_label, best_val = None, None
        for label, _ in self.workers:
            pd = _BetaPosterior(rng, *self.prior)
            for r in all_rewards[label]:
                pd.tell(r)
            v = pd.sample()
            if best_val is None or v > best_val:
                best_label, best_val = label, v
        return best_label

    def _backprop(self, prob: dict, expansion_node, child, score: float) -> None:
        """AB-MCTS-A asymmetric back-propagation of a new leaf's ``score`` — the single property
        that distinguishes this search from a flat bandit / best-of-N (so it is factored out to
        be pinned white-box by a test). At the EXPANSION node, the score updates its GEN posterior
        (how good a brand-new attempt here is) and seeds the new child's own per-child posterior.
        Then, for every ANCESTOR the descent passed through, the score updates that ancestor's CONT
        posterior (how good *continuing* this line is) and that specific child-line's posterior."""
        prob[expansion_node.idx].gen.tell(score)          # a fresh attempt at the expansion node
        prob[expansion_node.idx].register_child(child.idx, score)
        cur = expansion_node                              # walk expansion node -> root
        while cur.parent is not None:                     # every ancestor we DESCENDED through
            pstate = prob[cur.parent.idx]
            pstate.cont.tell(score)                       # "continuing" this line yielded `score`
            pstate.children[cur.idx].tell(score)          # this specific child line's value
            cur = cur.parent

    # ------------------------------------------------------------------ #
    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        # `verify`/`stakes` are control kwargs threaded in by `gama bench`; a kwargs `verify`
        # wins (gate the search on the case checker), and neither is forwarded to children.
        kw_verify = kwargs.get("verify", None)
        verify = resolve_verifier(kw_verify) if kw_verify is not None else self.verify
        sub = {k: v for k, v in kwargs.items() if k not in ("verify", "stakes")}

        rng = random.Random(self.seed)   # seed=None -> nondeterministic; an int -> reproducible
        root = _Node(answer=None, score=-1.0, action=None, parent=None, idx=-1)
        prob = {root.idx: _NodeProbState(rng, self.prior)}     # node.idx -> its posteriors
        node_by_idx = {root.idx: root}
        all_rewards = {label: [] for label, _ in self.workers}  # per-action reward history (bandit)

        trace: list = []
        total_cost = 0.0
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        saw_usage = False
        best: _Node | None = None
        next_idx = 0

        for it in range(self.budget):
            # --- SELECT: descend from root, Thompson-sampling wider vs deeper at each node ---
            node, depth = root, 0
            while True:
                sel = prob[node.idx].select()
                if sel == GEN:
                    break                     # widen here: generate a fresh child of `node`
                node = node_by_idx[sel]        # deeper: descend into the chosen child
                depth += 1
            child_depth = depth + 1

            # --- pick the model (Multi-LLM bandit) and generate a candidate ---
            action = self._select_action(rng, all_rewards)
            be = self.by_label[action]
            if node is root:                   # a child of the root = a first-shot answer (width)
                gen_prompt, mode = prompt, "sample"
            else:                              # a child of an answer = a refinement (depth)
                gen_prompt, mode = self._refine_prompt(prompt, node.answer, node.score), "refine"
            try:
                artifact = be.complete(gen_prompt, tier, **sub)
            except Exception:
                artifact = ""                  # a failing model -> empty draft, never abort the search
            score = self._reward(verify, artifact)
            # Guard a short `costs` list the same way meshflow (meshflow.py:179) and trinity
            # (trinity.py:129) do: a config may supply fewer costs than workers, and an
            # unguarded index would raise mid-search (uncaught here — the try/except above wraps
            # only be.complete). Fall back to a unit cost for any worker past the list.
            act_idx = self.action_index[action]
            total_cost += self.costs[act_idx] if act_idx < len(self.costs) else 1.0
            u = getattr(be, "last_usage", None)
            if u:                              # sum tokens across the whole search = honest bench cost
                saw_usage = True
                for k in usage_total:
                    usage_total[k] += u.get(k, 0) or 0

            # --- register the new child + AB-MCTS-A asymmetric back-propagation ---
            child = _Node(answer=artifact, score=score, action=action, parent=node, idx=next_idx)
            next_idx += 1
            node.children.append(child)
            node_by_idx[child.idx] = child
            prob[child.idx] = _NodeProbState(rng, self.prior)
            all_rewards[action].append(score)
            self._backprop(prob, node, child, score)

            trace.append({"iter": it, "model": action, "depth": child_depth,
                          "mode": mode, "score": round(score, 3)})
            if best is None or score > best.score:
                best = child
            if best.score >= self.pass_score:           # verifier fully satisfied -> stop early
                break

        self.last_trace = trace
        self.last_resolved_by = best.action if best is not None else None
        self.last_cost = round(total_cost, 3)
        self.last_best_score = round(best.score, 4) if best is not None else None
        self.last_tree_size = next_idx
        self.last_usage = usage_total if saw_usage else None
        return best.answer if best is not None else ""
