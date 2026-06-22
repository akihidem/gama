r"""meshflow — soshiki-genron の「採用すべき組織図」を gama の合成 backend にしたもの.

出典: ``~/Projects/soshiki-genron`` ``experiments/meshflow.py`` / PAPER §6.5
「採用すべき組織図」. 人間組織(チーム/役割/管理者/階層)を起点にせず、AIネイティブな
「組織の形」を gama の **4つ目の合成モード** として移植する —— route(``GamaBackend``) /
ensemble(``EnsembleBackend``) / tool(``ToolBackend``) に並ぶ「**段階委譲**」.

  ① 検証エスカレーション … cheap->expensive を外部検証で gate(自己申告でなく verify->score で昇格)
  ② 縁の mesh           … どの単独ティアも通らない縁でだけ全ティアの試行を合成(相補的な誤りを束ねる)
  ③ 薄い人間統治膜       … stakes 高×未解決なら黙って ship せず human gate(``<<NEEDS_HUMAN>>``)
  ④ 外部検証が背骨       … 合否は verify(artifact)->score in [0,1]. gama の ``score_output`` と同型.

"Structure, not scale"(gama のテーゼ)を*組織レベル*で具体化したもの: 外部検証で gate した
段階委譲は、常に最強ティアを使うより**低コストで同じ正答**に届きうる —— soshiki-genron の
市場支配定理 p\*=w/s と同じ向きで、gama 自身の ``gama bench`` で再現できる(``_run_one`` が
case checker を ``verify`` として渡すため、エスカレーションは bench と同じ外部検証で gate される).
"""
from __future__ import annotations

import re
import subprocess
import sys

from .backends import ModelBackend, synthesize
from .models import ModelTier

NEEDS_HUMAN = "<<NEEDS_HUMAN>>"


def _normalize_score(r) -> float:
    """checker の戻り値 -> [0,1] の float(gama ``score_output`` と同じ規約; 例外/不正は 0.0)."""
    if isinstance(r, bool):
        return 1.0 if r else 0.0
    try:
        return max(0.0, min(1.0, float(r)))
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Built-in external verifiers — verify(artifact) -> score in [0, 1].
# Same shape as a benchmark.BenchCase.checker, so `gama bench` can thread its own
# checker straight in as the escalation gate (honest measurement).
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\n(.*?)```", re.DOTALL)


def verify_code_runs(artifact: str, timeout: int = 15) -> float:
    """Extract the longest fenced code block and run it; 1.0 iff it executes without
    error (exit 0), else 0.0. Mirrors ``ToolBackend``'s PAL execution. SECURITY: runs
    model-generated code in a subprocess (opt-in, like ``ToolBackend`` / ``--sandbox``)."""
    blocks = _FENCE_RE.findall(artifact or "")
    code = max(blocks, key=len) if blocks else (artifact or "")
    if not code.strip():
        return 0.0
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout)
        return 1.0 if proc.returncode == 0 else 0.0
    except Exception:
        return 0.0


def verify_nonempty(artifact: str) -> float:
    """A trivial floor verifier: 1.0 for any non-empty artifact that isn't the human gate."""
    a = (artifact or "").strip()
    return 1.0 if (a and a != NEEDS_HUMAN) else 0.0


_BUILTIN_VERIFIERS = {"code_runs": verify_code_runs, "nonempty": verify_nonempty}


def resolve_verifier(verify):
    """Normalize ``verify`` to a ``(artifact)->score`` callable (or None).

    Accepts a callable, a built-in name (``"code_runs"`` / ``"nonempty"``), or None.
    None means *no early stop* — escalation runs every tier, then mesh/membrane.
    """
    if verify is None or callable(verify):
        return verify
    if isinstance(verify, str):
        try:
            return _BUILTIN_VERIFIERS[verify]
        except KeyError:
            raise ValueError(
                f"unknown verifier {verify!r}; choose from {sorted(_BUILTIN_VERIFIERS)} "
                "or pass a callable")
    raise TypeError("verify must be a callable, a built-in name, or None")


def _mesh_union(drafts: list) -> str:
    """Deterministic mesh: union of distinct non-empty draft lines — a stand-in for
    cross-checking diverse agents whose value comes from COMPLEMENTARY errors (soshiki-genron
    ``meshflow._mesh_combine``). Real deployments use ``mesh='synthesize'`` (an aggregator)."""
    seen, out = set(), []
    for d in drafts:
        for ln in (d or "").splitlines():
            k = ln.strip()
            if k and k not in seen:
                seen.add(k)
                out.append(ln)
    return "\n".join(out)


class MeshflowBackend(ModelBackend):
    """段階委譲 — verification-routed escalation over capability tiers, mesh at the edge,
    a thin human governance membrane. The AI-native "organizational form" from
    soshiki-genron (``experiments/meshflow.py``, PAPER §6.5), as a gama composite backend.

    Where ``GamaBackend`` *routes* (1 task -> 1 vendor) and ``EnsembleBackend`` *combines*
    (N -> 1, blindly), this *escalates under external verification*: try the cheapest tier,
    accept its artifact only if ``verify(artifact)->score`` passes; otherwise climb to a
    stronger tier. At the edge (no single tier passes) it **meshes** the tier drafts; if it
    is *still* unresolved and ``stakes`` are high it returns ``NEEDS_HUMAN`` rather than
    silently shipping. Tiers are any gama backends, ordered cheap -> expensive.

    Args:
      tiers: list of ``ModelBackend`` (cheap->expensive), or list of ``(label, backend)``.
      verify: ``(artifact)->score in [0,1]`` callable, a built-in name (``"code_runs"`` /
        ``"nonempty"``), or None. A ``verify`` in ``complete()`` kwargs overrides this
        (so ``gama bench`` can gate escalation on its own case checker).
      mesh: ``"union"`` (deterministic, default) | ``"synthesize"`` (LLM aggregator) | False.
      aggregator: backend for ``mesh="synthesize"`` (defaults to the strongest tier).
      stakes / stakes_threshold: high unresolved stakes -> human gate (the membrane).
      pass_score: a score >= this stops escalation (external verification satisfied).
      costs: per-tier cost weights (price proxy) for ``last_cost``; default 1,2,3,...
    """

    name = "meshflow"

    def __init__(self, tiers, verify=None, mesh="union", aggregator=None,
                 stakes: float = 0.0, stakes_threshold: float = 0.7,
                 pass_score: float = 1.0, costs=None):
        if not tiers:
            raise ValueError("MeshflowBackend needs at least one tier")
        self.tiers = []
        for i, t in enumerate(tiers):
            if isinstance(t, (tuple, list)):
                label, be = t
            else:
                label, be = getattr(t, "name", f"tier{i}"), t
            self.tiers.append((str(label), be))
        self.verify = resolve_verifier(verify)
        self.mesh = mesh
        self.aggregator = aggregator
        self.stakes = stakes
        self.stakes_threshold = stakes_threshold
        self.pass_score = pass_score
        self.costs = list(costs) if costs else [float(i + 1) for i in range(len(self.tiers))]
        self.available = any(getattr(be, "available", False) for _, be in self.tiers)
        self.last_usage = None
        self.last_trace = None          # [{"tier", "score"}] — the escalation ladder
        self.last_resolved_by = None    # tier label / "mesh" / "best-effort" / None(=human gate)
        self.last_cost = None           # summed tier cost up to resolution (price proxy)
        self.last_human_gate = False

    def _score(self, verify, artifact) -> float:
        if verify is None:
            return 0.0
        try:
            return _normalize_score(verify(artifact))
        except Exception:
            return 0.0

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        # A `verify` passed via kwargs wins (lets `gama bench` thread the case checker =
        # honest gating). `verify`/`stakes` are control kwargs — don't forward them down.
        kw_verify = kwargs.get("verify", None)
        verify = resolve_verifier(kw_verify) if kw_verify is not None else self.verify
        stakes = kwargs.get("stakes", self.stakes)
        sub = {k: v for k, v in kwargs.items() if k not in ("verify", "stakes")}

        attempts, drafts, cost = [], [], 0.0
        for i, (label, be) in enumerate(self.tiers):        # cheap -> expensive
            try:
                art = be.complete(prompt, tier, **sub)
            except Exception:
                art = ""                                    # a failing tier -> empty draft, escalate
            cost += self.costs[i] if i < len(self.costs) else 1.0
            score = self._score(verify, art)
            attempts.append({"tier": label, "score": round(score, 3)})
            drafts.append(art)
            if score >= self.pass_score:                    # external verify satisfied -> stop
                return self._finish(art, label, cost, attempts, be, human=False)

        # ② edge: no single tier passed -> mesh the tier drafts (complementary capability).
        if self.mesh and len(self.tiers) > 1:
            if self.mesh == "synthesize":
                agg = self.aggregator or self.tiers[-1][1]
                merged, usage_src = synthesize(agg, prompt, tier, drafts, **sub), agg
            else:                                           # "union" (deterministic)
                merged, usage_src = _mesh_union(drafts), None
            cost += _MESH_COST
            ms = self._score(verify, merged)
            attempts.append({"tier": "mesh", "score": round(ms, 3)})
            if ms >= self.pass_score:
                return self._finish(merged, "mesh", cost, attempts, usage_src, human=False)

        # ③ membrane: still unresolved. high stakes -> human gate; else best-effort (flagged).
        if verify is not None and stakes >= self.stakes_threshold:
            return self._finish(NEEDS_HUMAN, None, cost, attempts, None, human=True)
        return self._finish(drafts[-1], "best-effort", cost, attempts, self.tiers[-1][1],
                            human=False)

    def _finish(self, artifact, resolved_by, cost, attempts, usage_src, human):
        self.last_trace = attempts
        self.last_resolved_by = resolved_by
        self.last_cost = round(cost, 3)
        self.last_human_gate = human
        self.last_usage = getattr(usage_src, "last_usage", None) if usage_src is not None else None
        return artifact


_MESH_COST = 0.5
