"""gama — a community-grown "combination local LLM" toolkit.

Route each task class to the right local model, combine small models into a mixture
of agents, give them tools (program-aided), and benchmark which combination matches a
big model — all local, stdlib-only. Discovered finding: a *structured* combination of
small sovereign models can tie a much larger model on hard tasks (see README).
"""
from .backends import (
    EnsembleBackend,
    GamaBackend,
    ModelBackend,
    OllamaBackend,
    SshOpenAIBackend,
    ToolBackend,
    get_backend,
)
from .benchmark import (
    BRUTAL_SUITE,
    DEFAULT_SUITE,
    HARD_SUITE,
    SUITES,
    BenchCase,
    propose_routing_table,
    run_bench,
    score_output,
    summarize,
)
from .config import (
    build_backend,
    ensemble_from_config,
    gama_from_config,
    load_config,
    meshflow_from_config,
)
from .market import (
    analyze,
    dominates,
    escalation_cost,
    ladder,
    market_over_records,
    p_star,
)
from .meshflow import NEEDS_HUMAN, MeshflowBackend
from .models import ModelTier, TaskType

__version__ = "0.1.0"
__all__ = [
    "ModelTier", "TaskType",
    "ModelBackend", "GamaBackend", "EnsembleBackend", "ToolBackend", "MeshflowBackend",
    "OllamaBackend", "SshOpenAIBackend", "get_backend", "NEEDS_HUMAN",
    "build_backend", "gama_from_config", "ensemble_from_config", "meshflow_from_config",
    "load_config",
    "run_bench", "summarize", "propose_routing_table", "BenchCase",
    "DEFAULT_SUITE", "HARD_SUITE", "BRUTAL_SUITE", "SUITES", "score_output",
    "escalation_cost", "p_star", "dominates", "ladder", "market_over_records", "analyze",
    "__version__",
]
