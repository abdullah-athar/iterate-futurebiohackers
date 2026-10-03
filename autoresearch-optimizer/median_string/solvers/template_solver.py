"""Template solver: Use this file as a starting point to implement your own algorithm.

Instructions:
1. Copy this file or modify it directly.
2. Choose a unique name and set `name = "your_solver_name"`.
3. Implement your search algorithm in `solve(self, instance: ProblemInstance) -> str`.
4. Register it in `solvers/__init__.py` using `@register_solver("your_solver_name")` or register dynamically.
5. Benchmark your solver using:
   `python -m median_string.evaluator --solver your_solver_name`
"""

from __future__ import annotations

import random
from ..base_solver import BaseSolver
from ..instance import ProblemInstance
from ..metrics import sum_distance


class TemplateSolver(BaseSolver):
    """Example starter solver implementing greedy 1-point local search from the set median."""

    name = "template_local_search"
    description = "Starter template: starts from set median and applies greedy 1-edit local search."

    def __init__(self, max_iterations: int = 50, seed: int = 42) -> None:
        self.max_iterations = max_iterations
        self.seed = seed

    def solve(self, instance: ProblemInstance) -> str:
        """Find a Steiner string candidate for the given instance."""
        rng = random.Random(self.seed)

        # Step 1: Start with a reasonable initial candidate (e.g. set median or frequency consensus)
        # Here we pick the shortest string matching target_length or adapt the first string
        length = instance.target_length or int(round(instance.mean_length))
        alpha = list(instance.alphabet)

        # Baseline start: pick best among the input strings
        current_best = instance.strings[0]
        if len(current_best) != length:
            current_best = (current_best + alpha[0] * length)[:length]

        current_score = sum_distance(current_best, instance.strings, metric=instance.metric)

        for s in instance.strings:
            candidate = (s + alpha[0] * length)[:length]
            score = sum_distance(candidate, instance.strings, metric=instance.metric)
            if score < current_score:
                current_score = score
                current_best = candidate

        # Step 2: Local search / optimization loop (e.g. mutate 1 character if it improves score)
        improved = True
        iterations = 0

        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1

            # Try mutating positions
            positions = list(range(len(current_best)))
            rng.shuffle(positions)

            for pos in positions:
                original_char = current_best[pos]
                for new_char in alpha:
                    if new_char == original_char:
                        continue

                    # Propose single-point substitution
                    neighbor = current_best[:pos] + new_char + current_best[pos + 1 :]
                    neighbor_score = sum_distance(neighbor, instance.strings, metric=instance.metric)

                    if neighbor_score < current_score:
                        current_best = neighbor
                        current_score = neighbor_score
                        improved = True
                        break  # Greedily accept first improvement

                if improved:
                    break

        return current_best
