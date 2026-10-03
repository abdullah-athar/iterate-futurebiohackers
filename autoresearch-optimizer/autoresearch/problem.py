"""Problem adapter contract.

A `Problem` tells the research loop how to evaluate a candidate solver source file on a
named split and how to describe itself to the proposer. Scores are *minimised*.
Register new problems in `PROBLEMS` (see `autoresearch/problems/median_string.py`).
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class InstanceDiag:
    """Per-instance feedback: the 'actionable side information' shown to the proposer."""

    name: str
    score: float
    baseline: float
    best_known: float | None = None
    valid: bool = True
    error: str = ""
    elapsed: float = 0.0
    info: str = ""
    cpu_ms: float = 0.0

    @property
    def vs_baseline(self) -> float:
        return self.baseline - self.score

    @property
    def gap_to_best_known(self) -> float | None:
        return None if self.best_known is None else self.score - self.best_known

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstanceDiag:
        return cls(**d)


@dataclass
class EvalResult:
    split: str
    score: float
    baseline: float
    instances: list[InstanceDiag] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and all(i.valid for i in self.instances)

    @property
    def improvement_pct(self) -> float:
        return 100.0 * (self.baseline - self.score) / max(self.baseline, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "score": self.score,
            "baseline": self.baseline,
            "elapsed": self.elapsed,
            "error": self.error,
            "instances": [i.to_dict() for i in self.instances],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        return cls(
            split=d["split"],
            score=d["score"],
            baseline=d["baseline"],
            elapsed=d.get("elapsed", 0.0),
            error=d.get("error", ""),
            instances=[InstanceDiag.from_dict(i) for i in d.get("instances", [])],
        )


class Problem(Protocol):
    """Interface a benchmark must implement to be driven by the research loop."""

    name: str
    splits: tuple[str, ...]           # cascade order, e.g. ("screen", "validate", "holdout")
    objective_split: str              # split whose score is the search objective
    confirm_split: str | None  # fresh instances used only to confirm a claimed new global best
    allowed_imports: tuple[str, ...]
    timeouts: dict[str, float]        # wall-clock limit (s) per split for one evaluation (outer backstop)

    def describe(self) -> str:        # problem statement + solver contract for the proposer
        ...

    def seed_source(self) -> str:     # starting solver source code
        ...

    def evaluate(self, source: str, split: str, budget_ms: int | None = None) -> EvalResult:
        """Runs in the worker process. `budget_ms` is the CPU budget per instance (None = unlimited)."""
        ...


PROBLEMS: dict[str, str] = {
    "median_string": "autoresearch.problems.median_string:MedianStringProblem",
}


def get_problem(name: str) -> Problem:
    target = PROBLEMS.get(name, name)
    module_name, _, cls_name = target.partition(":")
    if not cls_name:
        raise KeyError(f"Unknown problem '{name}'. Known: {sorted(PROBLEMS)} or 'module:Class'.")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls()
