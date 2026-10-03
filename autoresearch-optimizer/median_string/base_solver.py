"""Base solver interface for Median String / Steiner String algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .instance import ProblemInstance


class BaseSolver(ABC):
    """Abstract base class that all median string solvers should inherit from.

    To implement a new solver:
    1. Subclass BaseSolver
    2. Define `name` and optional `description`
    3. Implement `solve(self, instance: ProblemInstance) -> str`
    """

    name: str = "base_solver"
    description: str = "Base solver template."

    @abstractmethod
    def solve(self, instance: ProblemInstance) -> str:
        """Find a median/Steiner candidate string for the given problem instance.

        Args:
            instance: ProblemInstance containing target strings, alphabet, and target length.

        Returns:
            The proposed median string candidate.
        """
        raise NotImplementedError

    def get_metadata(self) -> dict[str, Any]:
        """Return solver hyperparameters or configuration metadata."""
        return {
            "name": getattr(self, "name", self.__class__.__name__),
            "class": self.__class__.__name__,
            "description": getattr(self, "description", ""),
        }


class FunctionalSolver(BaseSolver):
    """Wrapper to allow plain functions `fn(instance: ProblemInstance) -> str` to act as solvers."""

    def __init__(
        self,
        func: Callable[[ProblemInstance], str],
        name: str | None = None,
        description: str = "",
    ) -> None:
        self.func = func
        self.name = name or getattr(func, "__name__", "custom_function_solver")
        self.description = description or func.__doc__ or "Functional solver wrapper."

    def solve(self, instance: ProblemInstance) -> str:
        return self.func(instance)


def normalize_solver(solver: BaseSolver | Callable[[ProblemInstance], str]) -> BaseSolver:
    """Ensure that the given solver object is a BaseSolver instance."""
    if isinstance(solver, BaseSolver):
        return solver
    if callable(solver):
        return FunctionalSolver(solver)
    raise TypeError(
        f"Expected BaseSolver instance or callable(ProblemInstance) -> str, got {type(solver).__name__}"
    )
