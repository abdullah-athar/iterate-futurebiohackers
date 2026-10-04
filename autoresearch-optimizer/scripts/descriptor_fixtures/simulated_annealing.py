"""Simulated annealing with random edit moves and geometric cooling."""

import math
import random
import time

from median_string.metrics import sum_distance


def solve(instance) -> str:
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    rng = random.Random(0)
    strings, metric, alphabet = instance.strings, instance.metric, instance.alphabet
    cur = min(strings, key=lambda s: sum_distance(s, strings, metric=metric))
    cur_score = sum_distance(cur, strings, metric=metric)
    best, best_score = cur, cur_score
    temp = 2.0
    while time.process_time() < deadline:
        pos = rng.randrange(len(cur) + 1)
        op = rng.choice("sid") if pos < len(cur) else "i"
        if op == "s":
            cand = cur[:pos] + rng.choice(alphabet) + cur[pos + 1:]
        elif op == "i":
            cand = cur[:pos] + rng.choice(alphabet) + cur[pos:]
        else:
            cand = cur[:pos] + cur[pos + 1:]
        score = sum_distance(cand, strings, metric=metric)
        if score <= cur_score or rng.random() < math.exp((cur_score - score) / temp):
            cur, cur_score = cand, score
            if score < best_score:
                best, best_score = cand, score
        temp = max(temp * 0.995, 1e-3)
    return best
