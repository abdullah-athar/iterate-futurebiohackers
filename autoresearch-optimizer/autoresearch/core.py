"""Core data types and the interfaces that every swappable autoresearch component implements."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Candidate:
    """A proposed solver: a full Python module defining one BaseSolver subclass."""

    code: str
    parent_id: str | None = None
    rationale: str = ""
    id: str = field(default_factory=new_id)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Outcome of evaluating one candidate on one suite (lower score is better)."""

    candidate_id: str
    suite: str
    score: float
    valid: bool
    error: str = ""
    baseline_score: float | None = None
    elapsed: float = 0.0
    per_instance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteSpec:
    """Which benchmark instances to use: a tier plus a seed offset (0 = the canonical suite)."""

    tier: str = "medium"
    seed_offset: int = 0

    def __str__(self) -> str:
        return f"{self.tier}+{self.seed_offset}"


@dataclass
class Record:
    """One step of the research history."""

    iteration: int
    candidate: Candidate
    result: Result
    accepted: bool


@dataclass
class ResearchContext:
    """Everything a proposer may look at when proposing the next candidate."""

    iteration: int
    parents: list[Record]
    history_summary: str
    suite: SuiteSpec


class Proposer(Protocol):
    def propose(self, ctx: ResearchContext) -> Candidate: ...


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: Candidate, suite: SuiteSpec) -> Result: ...


class Memory(Protocol):
    records: list[Record]

    def record(self, rec: Record) -> None: ...
    def champion(self) -> Record | None: ...
    def summary(self, k: int = 10) -> str: ...


class Selector(Protocol):
    def accept(self, memory: Memory, result: Result) -> bool: ...
    def parents(self, memory: Memory) -> list[Record]: ...


class Budget(Protocol):
    def exhausted(self, memory: Memory) -> bool: ...
