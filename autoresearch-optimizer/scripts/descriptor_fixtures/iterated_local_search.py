"""Iterated local search: substitution descent, then random multi-position kicks; accept if not worse."""

import random
import time

from median_string.metrics import sum_distance


def solve(instance) -> str:
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    rng = random.Random(0)
    strings, metric, alphabet = instance.strings, instance.metric, instance.alphabet
    f = lambda s: sum_distance(s, strings, metric=metric)

    def descend(s, score):
        improved = True
        while improved and time.process_time() < deadline:
            improved = False
            for pos in range(len(s)):
                for ch in alphabet:
                    cand = s[:pos] + ch + s[pos + 1:]
                    cs = f(cand)
                    if cs < score:
                        s, score, improved = cand, cs, True
        return s, score

    cur, cur_score = descend(min(strings, key=f), f(min(strings, key=f)))
    best, best_score = cur, cur_score
    while time.process_time() < deadline:
        kicked = list(cur)
        for pos in rng.sample(range(len(kicked)), min(3, len(kicked))):
            kicked[pos] = rng.choice(alphabet)
        cand, cand_score = descend("".join(kicked), f("".join(kicked)))
        if cand_score <= cur_score:
            cur, cur_score = cand, cand_score
            if cand_score < best_score:
                best, best_score = cand, cand_score
    return best
