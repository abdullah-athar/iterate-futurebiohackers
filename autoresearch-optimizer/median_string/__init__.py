"""Median String (Steiner String) evaluation harness and benchmark framework."""

from .instance import ProblemInstance
from .metrics import (
    calculate_distance,
    compute_set_median,
    hamming_distance,
    levenshtein_distance,
    mean_distance,
    normalized_distance,
    sum_distance,
)
from .base_solver import BaseSolver, FunctionalSolver, normalize_solver
from .benchmarks import BenchmarkTier, get_benchmark_suite, get_confirm_suite, get_holdout_suite
from .evaluator import Evaluator, EvaluationSummary, InstanceResult
from .solvers import get_solver, list_solvers, register_solver

__all__ = [
    "ProblemInstance",
    "BaseSolver",
    "FunctionalSolver",
    "normalize_solver",
    "Evaluator",
    "EvaluationSummary",
    "InstanceResult",
    "BenchmarkTier",
    "get_benchmark_suite",
    "get_confirm_suite",
    "get_holdout_suite",
    "get_solver",
    "list_solvers",
    "register_solver",
    "calculate_distance",
    "compute_set_median",
    "hamming_distance",
    "levenshtein_distance",
    "mean_distance",
    "normalized_distance",
    "sum_distance",
]
