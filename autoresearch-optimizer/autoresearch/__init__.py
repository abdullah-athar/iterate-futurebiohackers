"""autoresearch: a small, reusable research loop for algorithm discovery.

The loop proposes a new solver (via an LLM API or a coding agent), screens it for
novelty, evaluates it in a sandboxed subprocess with a cascade of benchmark splits,
keeps a Pareto-per-instance archive, and records everything in an append-only ledger.
"""

from .problem import EvalResult, InstanceDiag, Problem, get_problem

__all__ = ["EvalResult", "InstanceDiag", "Problem", "get_problem"]
