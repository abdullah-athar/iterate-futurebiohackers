"""Tabu search: best-improvement over the full substitution neighbourhood, recently changed positions are tabu."""

import time
from collections import deque

from median_string.metrics import sum_distance


def solve(instance) -> str:
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    strings, metric, alphabet = instance.strings, instance.metric, instance.alphabet
    cur = min(strings, key=lambda s: sum_distance(s, strings, metric=metric))
    best, best_score = cur, sum_distance(cur, strings, metric=metric)
    tabu = deque(maxlen=7)
    while time.process_time() < deadline:
        move, move_score = None, float("inf")
        for pos in range(len(cur)):
            for ch in alphabet:
                if ch == cur[pos]:
                    continue
                cand = cur[:pos] + ch + cur[pos + 1:]
                score = sum_distance(cand, strings, metric=metric)
                # aspiration: a tabu move is allowed if it beats the best ever
                if (pos not in tabu or score < best_score) and score < move_score:
                    move, move_score = (pos, cand), score
        if move is None:
            break
        tabu.append(move[0])
        cur = move[1]
        if move_score < best_score:
            best, best_score = cur, move_score
    return best
