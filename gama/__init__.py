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
    DEFAULT_SUITE,
    BenchCase,
    propose_routing_table,
    run_bench,
    score_output,
    summarize,
)
from .config import build_backend, ensemble_from_config, gama_from_config, load_config
from .models import ModelTier, TaskType

__version__ = "0.1.0"
__all__ = [
    "ModelTier", "TaskType",
    "ModelBackend", "GamaBackend", "EnsembleBackend", "ToolBackend",
    "OllamaBackend", "SshOpenAIBackend", "get_backend",
    "build_backend", "gama_from_config", "ensemble_from_config", "load_config",
    "run_bench", "summarize", "propose_routing_table", "BenchCase", "DEFAULT_SUITE",
    "score_output", "__version__",
]
