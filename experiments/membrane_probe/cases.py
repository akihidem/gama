"""Deterministic tiers + synthetic cases for the membrane probe.

A case is solvable at some tier rank (the cheapest tier that produces "CORRECT")
or *hard* (no tier solves it). Hard cases carry high stakes — exactly the corner
where the membrane decides between holding for a human (m>0) and shipping a wrong
answer (m=0). Everything is deterministic so the comparison is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from gama.backends import ModelBackend
from gama.models import ModelTier

CORRECT = "CORRECT"
WRONG = "WRONG"


@dataclass(frozen=True)
class Case:
    cid: str
    stakes: float
    solve_rank: int | None  # cheapest tier rank that solves it; None = hard/unsolvable


def checker(artifact: str) -> float:
    """External verifier: 1.0 only for a verified-correct artifact."""
    return 1.0 if artifact == CORRECT else 0.0


class StubTier(ModelBackend):
    """A tier that returns CORRECT iff its rank can solve the case, else WRONG."""

    available = True

    def __init__(self, rank: int, cases_by_prompt: dict[str, "Case"]):
        self.rank = rank
        self.name = f"tier{rank}"
        self._cases = cases_by_prompt

    def complete(self, prompt: str, tier: ModelTier, **kwargs) -> str:
        case = self._cases[prompt]
        if case.solve_rank is not None and self.rank >= case.solve_rank:
            return CORRECT
        return WRONG


def default_cases() -> list[Case]:
    """Mix of solvable (low stakes) and hard (high stakes) cases.

    Solvable cases behave identically under both gates (both ship CORRECT); the
    membrane only changes behaviour on the hard, high-stakes corner.
    """
    cases = []
    for i in range(6):                 # solvable at the cheapest tier, low stakes
        cases.append(Case(f"easy-{i}", stakes=0.3, solve_rank=0))
    for i in range(3):                 # solvable only at the strongest tier
        cases.append(Case(f"mid-{i}", stakes=0.4, solve_rank=2))
    for i in range(4):                 # HARD: no tier solves it, high stakes
        cases.append(Case(f"hard-{i}", stakes=0.9, solve_rank=None))
    return cases


def build_tiers(cases: list[Case], n_tiers: int = 3) -> list[StubTier]:
    by_prompt = {c.cid: c for c in cases}
    return [StubTier(rank, by_prompt) for rank in range(n_tiers)]
