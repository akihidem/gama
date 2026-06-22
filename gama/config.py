"""Config + composite builders for gama (stdlib-only).

A config is a dict / JSON file with: ``default_backend``, ``routing_table``
(task_type -> backend name), ``backends`` (name -> constructor kwargs), ``ensemble``
(a spec), ``unit_cost`` (name -> USD/1k tokens). ``build_backend`` constructs any
composite (gama / ensemble / tool) from a nested spec.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DEFAULTS: dict[str, Any] = {
    "default_backend": "ollama",   # fallback lane for unmapped/absent task types
    "routing_table": {},           # task_type value -> backend name (deterministic)
    "backends": {},                # backend name -> constructor kwargs
    "unit_cost": {},               # backend name -> USD per 1k tokens (bench cost; default free)
    "ensemble": {},                # EnsembleBackend spec (members / member+n, strategy, aggregator)
}


def load_config(source: Optional[Any]) -> dict[str, Any]:
    """Return a config dict (defaults overlaid with known keys from source).

    source may be None, a dict, or a path to a JSON file."""
    cfg: dict[str, Any] = dict(DEFAULTS)
    cfg["routing_table"] = dict(DEFAULTS["routing_table"])
    cfg["backends"] = dict(DEFAULTS["backends"])
    cfg["unit_cost"] = dict(DEFAULTS["unit_cost"])
    cfg["ensemble"] = dict(DEFAULTS["ensemble"])
    if source is None:
        return cfg
    raw = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(raw.get("default_backend"), str):
        cfg["default_backend"] = raw["default_backend"]
    if isinstance(raw.get("routing_table"), dict):
        cfg["routing_table"] = {str(k): str(v) for k, v in raw["routing_table"].items()
                                if isinstance(v, str)}
    if isinstance(raw.get("backends"), dict):
        cfg["backends"] = {str(k): v for k, v in raw["backends"].items() if isinstance(v, dict)}
    if isinstance(raw.get("unit_cost"), dict):
        cfg["unit_cost"] = {str(k): float(v) for k, v in raw["unit_cost"].items()
                            if isinstance(v, (int, float))}
    if isinstance(raw.get("ensemble"), dict):
        cfg["ensemble"] = raw["ensemble"]
    return cfg


def gama_from_config(source: Optional[Any], backend_names: Optional[list[str]] = None):
    """Build the deterministic gama vendor router (GamaBackend) from config.

    Instantiates the sub-backends named in the routing table (plus the default),
    forwarding each backend's constructor kwargs from cfg["backends"].
    """
    from .backends import GamaBackend, get_backend

    c = load_config(source)
    names = backend_names or sorted(set(c["routing_table"].values()) | {c["default_backend"]})
    backends = {n: get_backend(n, **c["backends"].get(n, {})) for n in names}
    return GamaBackend(backends, routing_table=c["routing_table"], default=c["default_backend"])


def ensemble_from_config(source: Optional[Any]):
    """Build an EnsembleBackend (the model-combination loop) from cfg['ensemble'].

    cfg["ensemble"] accepts an explicit ``members`` list, or a single ``member`` spec
    repeated ``n`` times. Each spec is ``{"backend": <name>, "kwargs": {...}}``. Optional
    ``aggregator`` (same spec), ``strategy`` ("synthesize" | "majority" | "first").
    """
    from .backends import EnsembleBackend, get_backend

    spec = load_config(source)["ensemble"] or {}

    def build(s):
        return get_backend(s["backend"], **(s.get("kwargs") or {}))

    if spec.get("members"):
        members = [build(m) for m in spec["members"]]
    elif spec.get("member"):
        members = [build(spec["member"]) for _ in range(int(spec.get("n", 3)))]
    else:
        raise ValueError("ensemble config needs 'members' or 'member'+'n'")
    aggregator = build(spec["aggregator"]) if spec.get("aggregator") else None
    return EnsembleBackend(members, strategy=spec.get("strategy", "synthesize"),
                           aggregator=aggregator, aggregator_prompt=spec.get("aggregator_prompt"))


def build_backend(spec: Any):
    """Recursively build a backend from a spec ``{"backend": name, "kwargs": {...}}``.

    Composites compose: ``tool`` wraps ``kwargs['inner']`` (a spec); ``ensemble`` takes
    ``kwargs['members']`` (list of specs) or ``kwargs['member']`` + ``kwargs['n']`` plus
    optional ``aggregator``/``strategy``; ``gama`` takes ``kwargs['backends']`` (name ->
    spec) + ``routing_table`` + ``default``. Any other name goes to ``get_backend``. Lets
    a config declare a full stack (gama over tool / ensemble / coder lanes) as nested JSON.
    """
    from .backends import EnsembleBackend, GamaBackend, ToolBackend, get_backend

    name = spec["backend"]
    kw = dict(spec.get("kwargs") or {})
    if name == "tool":
        inner = build_backend(kw.pop("inner"))
        return ToolBackend(inner, **kw)
    if name == "ensemble":
        if kw.get("members"):
            members = [build_backend(m) for m in kw["members"]]
        else:
            members = [build_backend(kw["member"]) for _ in range(int(kw.get("n", 3)))]
        agg = build_backend(kw["aggregator"]) if kw.get("aggregator") else None
        return EnsembleBackend(members, strategy=kw.get("strategy", "synthesize"),
                               aggregator=agg, aggregator_prompt=kw.get("aggregator_prompt"))
    if name == "gama":
        backends = {n: build_backend(s) for n, s in (kw.get("backends") or {}).items()}
        return GamaBackend(backends, routing_table=kw.get("routing_table"),
                           default=kw.get("default"))
    return get_backend(name, **kw)
