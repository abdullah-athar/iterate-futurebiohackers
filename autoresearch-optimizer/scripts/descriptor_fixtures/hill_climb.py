"""First-improvement hill climbing over single-character substitutions, insertions and deletions."""

import time

from median_string.metrics import sum_distance


def solve(instance) -> str:
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    strings, metric, alphabet = instance.strings, instance.metric, instance.alphabet
    best = min(strings, key=lambda s: sum_distance(s, strings, metric=metric))
    best_score = sum_distance(best, strings, metric=metric)
    improved = True
    while improved and time.process_time() < deadline:
        improved = False
        for pos in range(len(best) + 1):
            moves = [best[:pos] + c + best[pos:] for c in alphabet]
            if pos < len(best):
                moves += [best[:pos] + c + best[pos + 1:] for c in alphabet if c != best[pos]]
                moves.append(best[:pos] + best[pos + 1:])
            for cand in moves:
                score = sum_distance(cand, strings, metric=metric)
                if score < best_score:
                    best, best_score, improved = cand, score, True
                    break
            if improved:
                break
    return best
