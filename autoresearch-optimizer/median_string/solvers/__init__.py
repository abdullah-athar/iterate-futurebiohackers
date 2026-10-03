"""Solvers registry and built-in solvers for the Median String problem.

To add your own solver:
1. Create a new python file in this `solvers/` folder (e.g. `my_solver.py`).
2. Subclass `BaseSolver` and implement `solve(self, instance: ProblemInstance) -> str`.
3. Register your solver using `@register_solver("my_solver_name")` or call `register_solver("my_solver_name", MySolver)`.
"""

from __future__ import annotations

from typing import Callable, Type, Union
from ..base_solver import BaseSolver, FunctionalSolver, normalize_solver

_SOLVER_REGISTRY: dict[str, BaseSolver | Type[BaseSolver] | Callable] = {}


def register_solver(name: str):
    """Decorator or function to register a solver in the global registry."""
    def decorator(cls_or_fn: Union[Type[BaseSolver], BaseSolver, Callable]):
        _SOLVER_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return decorator


def get_solver(name: str) -> BaseSolver:
    """Retrieve an instantiated solver by its registered name."""
    if name not in _SOLVER_REGISTRY:
        available = ", ".join(sorted(_SOLVER_REGISTRY.keys()))
        raise KeyError(f"Solver '{name}' not found. Available solvers: [{available}]")

    obj = _SOLVER_REGISTRY[name]
    if isinstance(obj, type) and issubclass(obj, BaseSolver):
        return obj()
    elif isinstance(obj, BaseSolver):
        return obj
    elif callable(obj):
        return FunctionalSolver(obj, name=name)
    else:
        raise TypeError(f"Registered object for '{name}' is not callable or BaseSolver.")


def list_solvers() -> list[str]:
    """List all registered solver names."""
    return sorted(_SOLVER_REGISTRY.keys())


# Import built-in solvers to register them
from .set_median import SetMedianSolver
from .frequency_consensus import FrequencyConsensusSolver
from .random_solver import RandomCandidateSolver
from .template_solver import TemplateSolver

register_solver("set_median")(SetMedianSolver)
register_solver("frequency_consensus")(FrequencyConsensusSolver)
register_solver("random_baseline")(RandomCandidateSolver)
register_solver("template")(TemplateSolver)

__all__ = [
    "register_solver",
    "get_solver",
    "list_solvers",
    "SetMedianSolver",
    "FrequencyConsensusSolver",
    "RandomCandidateSolver",
    "TemplateSolver",
]
