r"""trinity — a structural analogue of openfugu's TRINITY component
(https://github.com/trotsky1997/openfugu, Sakana AI "Fugu" OSS reimplementation),
as a gama composite backend.

openfugu's TRINITY scores each worker from a small model's HIDDEN STATES, via a head
trained with CMA-ES (gradient-free). Conductor then builds an execution DAG. What we
take from it here is the structural core, not the training machinery:

    predict ONCE, dispatch to that ONE worker, don't retry.

Honesty note (scope boundary): the Mac Studio serving stack (``mlx_lm.server``,
OpenAI-compatible) does not expose hidden states, so the CMA-ES-trained head is not
reproduced here. ``TrinityBackend``'s scorer picks a worker via a single cheap PROMPTED
classification call, not a trained router. This measures whether "one-shot predictive
routing" (the structural shape) beats gama's own ``MeshflowBackend`` (sequential
verification escalation, ``meshflow.py``) on gama's benchmark suites -- not whether
openfugu's own trained router beats gama (a different, unmeasured claim).

Contrast with ``MeshflowBackend``:
  - meshflow: tries tiers cheap -> expensive, gated by an external ``verify``; escalates
    past any tier whose artifact fails the gate. Cost grows with how far it climbs.
  - trinity : the scorer picks one worker up front; that worker is called exactly once.
    No verify, no escalation -- a wrong prediction is not retried. Cost is fixed
    (scorer_cost + the chosen worker's cost), regardless of outcome.
"""
from __future__ import annotations

import re

from .backends import ModelBackend
from .models import ModelTier

_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")


class TrinityBackend(ModelBackend):
    """One-shot predictive routing: a single classification call picks ONE worker;
    that worker is dispatched to exactly once.

    Args:
      workers: list of ``ModelBackend``, or ``(label, ModelBackend)`` tuples (any
        order; unlike meshflow's tiers this isn't cheap->expensive escalation order,
        but ``costs`` still follows list order for a stable price proxy).
      scorer: a ``ModelBackend`` for the one classification call. Defaults to the
        first worker -- the pool's own cheapest lane doubles as the "small backbone
        that scores", mirroring openfugu's TRINITY using a small model for scoring.
        Pass a dedicated small-``max_tokens`` backend instance if call-time overrides
        aren't supported by the underlying adapter (most aren't; configure it on the
        scorer backend itself).
      costs: per-worker cost weights (price proxy), cheap->expensive by list order;
        default ``1, 2, 3, ...`` (same proxy ``MeshflowBackend`` uses).
      scorer_cost: cost of the one classification call; default = ``costs[0]``.
    """

    name = "trinity"

    def __init__(self, workers, scorer=None, costs=None, scorer_cost: float | None = None):
        if not workers:
            raise ValueError("TrinityBackend needs at least one worker")
        self.workers = []
        for i, w in enumerate(workers):
            if isinstance(w, (tuple, list)):
                label, be = w
            else:
                label, be = getattr(w, "name", f"worker{i}"), w
            self.workers.append((str(label), be))
        self.scorer = scorer if scorer is not None else self.workers[0][1]
        self.costs = list(costs) if costs else [float(i + 1) for i in range(len(self.workers))]
        self.scorer_cost = scorer_cost if scorer_cost is not None else self.costs[0]
        self.available = any(getattr(be, "available", False) for _, be in self.workers)
        self.last_usage = None
        self.last_trace = None        # [{"role": "scorer", "picked", "raw", "fell_back"}, {"tier"}]
        self.last_resolved_by = None  # the worker label actually dispatched to
        self.last_cost = None         # scorer_cost + chosen worker's cost (fixed, no retry)
        self.last_fallback = False    # True iff the scorer's reply didn't parse -> cheapest fallback

    def _pick(self, prompt: str, tier: ModelTier, **kwargs) -> tuple:
        """-> (label, raw_scorer_reply, fell_back). ``fell_back=True`` on an unparseable
        or erroring scorer -- fails closed to the CHEAPEST worker (``workers[0]``), never
        silently to the most expensive one."""
        labels = [label for label, _ in self.workers]
        if len(labels) == 1:
            return labels[0], "", False
        # The instruction goes AFTER the query and repeats "don't answer it" -- small
        # instruction-following models otherwise just answer the query itself instead
        # of naming a label (measured: this ordering fixed a 100% misfire rate on a
        # real 7B worker; see recipes/mac-studio-trinity/recipe.md).
        ask = (f"Query:\n{prompt}\n\n"
               "Do NOT answer the query above. Your ONLY job: pick which worker should "
               f"answer it. Respond with EXACTLY one word, no punctuation, no explanation: "
               f"either {' or '.join(labels)}.\nWorker:")
        try:
            raw = self.scorer.complete(ask, tier, task_type=kwargs.get("task_type"))
        except Exception:
            return labels[0], "", True
        # Some servers leave the chat-template's end-of-turn marker (e.g. `<|im_end|>`)
        # glued onto content with no separating whitespace (measured on a real MLX
        # server) -- strip `<|...|>`-style tokens before matching, not just whitespace.
        cleaned = _SPECIAL_TOKEN_RE.sub("", raw or "").strip().strip(".:\"'").lower()
        words = cleaned.split()
        for label in labels:
            if cleaned == label.lower() or (words and words[0] == label.lower()):
                return label, raw, False
        return labels[0], raw, True

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        # `verify`/`stakes` are meshflow-style control kwargs; trinity has no escalation
        # gate, so they're accepted (bench threads `verify=case.checker` unconditionally)
        # but not forwarded down.
        sub = {k: v for k, v in kwargs.items() if k not in ("verify", "stakes")}
        label, raw, fell_back = self._pick(prompt, tier, **sub)
        by_label = dict(self.workers)
        be = by_label[label]
        idx = [l for l, _ in self.workers].index(label)
        try:
            out = be.complete(prompt, tier, **sub)
        except Exception:
            out = ""
        cost = self.scorer_cost + (self.costs[idx] if idx < len(self.costs) else 1.0)
        self.last_trace = [
            {"role": "scorer", "picked": label, "raw": (raw or "")[:80], "fell_back": fell_back},
            {"tier": label},
        ]
        self.last_resolved_by = label
        self.last_cost = round(cost, 3)
        self.last_fallback = fell_back
        self.last_usage = getattr(be, "last_usage", None)
        return out
