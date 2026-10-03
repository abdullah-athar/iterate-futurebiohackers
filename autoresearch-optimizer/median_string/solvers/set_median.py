"""Set Median solver (baseline).

Selects the string from the input set S that minimizes the sum of distances
to all other strings in S. Under the triangle inequality, this provides a
guaranteed 2-approximation to the optimal Steiner string.
"""

from __future__ import annotations

from ..base_solver import BaseSolver
from ..instance import ProblemInstance
from ..metrics import compute_set_median


class SetMedianSolver(BaseSolver):
    """Computes the optimal Set Median string (best candidate chosen from S)."""

    name = "set_median"
    description = "Standard baseline: picks the string in S that minimizes sum of distances."

    def solve(self, instance: ProblemInstance) -> str:
        best_str, _ = compute_set_median(
            instance.strings,
            metric=instance.metric,
            target_length=instance.target_length,
            alphabet=instance.alphabet,
        )
        return best_str

