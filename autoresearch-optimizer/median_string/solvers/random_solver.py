"""Random sampling baseline solver."""

from __future__ import annotations

import random
from ..base_solver import BaseSolver
from ..instance import ProblemInstance
from ..metrics import sum_distance


class RandomCandidateSolver(BaseSolver):
    """Generates random candidates over the alphabet and retains the best."""

    name = "random_baseline"
    description = "Generates N random candidates and returns the best."

    def __init__(self, num_samples: int = 50, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.seed = seed

    def solve(self, instance: ProblemInstance) -> str:
        rng = random.Random(self.seed)
        alpha = list(instance.alphabet)
        length = instance.target_length or int(round(instance.mean_length))

        best_str = "".join(rng.choice(alpha) for _ in range(length))
        best_score = sum_distance(best_str, instance.strings, metric=instance.metric)

        for _ in range(self.num_samples - 1):
            cand = "".join(rng.choice(alpha) for _ in range(length))
            score = sum_distance(cand, instance.strings, metric=instance.metric)
            if score < best_score:
                best_score = score
                best_str = cand

        return best_str
