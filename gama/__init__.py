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
    GRADED_SUITE,
    HARD_SUITE,
    WIDE_SUITE,
    SUITES,
    BenchCase,
    propose_routing_table,
    run_bench,
    score_output,
    summarize,
)
from .abmcts import ABMCTSBackend
from .config import (
    abmcts_from_config,
    build_backend,
    ensemble_from_config,
    gama_from_config,
    load_config,
    meshflow_from_config,
    system_from_config,
    trinity_from_config,
)
from .decorrelation import analyze as mesh_analyze
from .grow import (
    Candidate,
    canonical,
    grow,
    measure,
    ollama_pool,
    promote_gate,
    simplify_gate,
    propose,
    seed_champion,
    spec_hash,
    split_cases,
    suite_pool,
    write_recipe,
)
from .decorrelation import (
    failure_correlation,
    ignites,
    mesh_correctness,
    mesh_gain,
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
from .trinity import TrinityBackend

__version__ = "0.1.0"
__all__ = [
    "ModelTier", "TaskType",
    "ModelBackend", "GamaBackend", "EnsembleBackend", "ToolBackend", "MeshflowBackend",
    "TrinityBackend", "ABMCTSBackend",
    "OllamaBackend", "SshOpenAIBackend", "get_backend", "NEEDS_HUMAN",
    "build_backend", "gama_from_config", "ensemble_from_config", "meshflow_from_config",
    "trinity_from_config", "abmcts_from_config", "system_from_config",
    "load_config",
    "run_bench", "summarize", "propose_routing_table", "BenchCase",
    "DEFAULT_SUITE", "HARD_SUITE", "BRUTAL_SUITE", "WIDE_SUITE", "GRADED_SUITE", "SUITES",
    "score_output",
    "escalation_cost", "p_star", "dominates", "ladder", "market_over_records", "analyze",
    "mesh_gain", "mesh_correctness", "ignites", "failure_correlation", "mesh_analyze",
    "grow", "propose", "promote_gate", "simplify_gate", "measure", "split_cases", "suite_pool",
    "seed_champion", "ollama_pool", "canonical", "spec_hash", "Candidate", "write_recipe",
    "__version__",
]
