"""Seed solver: set-median start + greedy single-substitution local search.

This is the starting point the research loop improves. Keep the `solve(instance) -> str`
contract; everything else may change.
"""

import random
import time

from median_string.metrics import sum_distance


def solve(instance) -> str:
    # stop at 80% of the CPU budget and return the best string found so far
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    rng = random.Random(42)
    strings = instance.strings
    metric = instance.metric
    alphabet = list(instance.alphabet)

    best = min(strings, key=lambda s: sum_distance(s, strings, metric=metric))
    best_score = sum_distance(best, strings, metric=metric)

    for _ in range(50):
        improved = False
        positions = list(range(len(best)))
        rng.shuffle(positions)
        for pos in positions:
            if time.process_time() > deadline:
                return best
            for ch in alphabet:
                if ch == best[pos]:
                    continue
                cand = best[:pos] + ch + best[pos + 1:]
                score = sum_distance(cand, strings, metric=metric)
                if score < best_score:
                    best, best_score, improved = cand, score, True
                    break
            if improved:
                break
        if not improved:
            break
    return best
