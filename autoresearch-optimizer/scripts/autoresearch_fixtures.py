"""Fixture candidates for the tests: a near-duplicate of the seed, a syntax error, and two real solvers."""

SEED_DUPLICATE = '''
import random
from median_string.metrics import sum_distance

def solve(instance):
    # same algorithm as the seed with cosmetic changes -> should be rejected by the novelty gate
    rng = random.Random(42)
    S = instance.strings
    m = instance.metric
    A = list(instance.alphabet)
    best = min(S, key=lambda s: sum_distance(s, S, metric=m))
    best_score = sum_distance(best, S, metric=m)
    for _ in range(50):
        improved = False
        pos = list(range(len(best)))
        rng.shuffle(pos)
        for p in pos:
            for c in A:
                if c == best[p]:
                    continue
                cand = best[:p] + c + best[p + 1:]
                sc = sum_distance(cand, S, metric=m)
                if sc < best_score:
                    best, best_score, improved = cand, sc, True
                    break
            if improved:
                break
        if not improved:
            break
    return best
'''

BROKEN = '''
def solve(instance):
    return instance.strings[0] +
'''

CONSENSUS_LOCAL_SEARCH = '''
from collections import Counter
from median_string.metrics import sum_distance

def solve(instance):
    S, m, A = instance.strings, instance.metric, list(instance.alphabet)
    L = round(sum(map(len, S)) / len(S))
    consensus = "".join(Counter(s[i] for s in S if i < len(s)).most_common(1)[0][0] for i in range(L))
    starts = [consensus] + sorted(S, key=lambda s: sum_distance(s, S, metric=m))[:2]
    best, best_score = None, float("inf")
    for start in starts:
        cur, cur_score = start, sum_distance(start, S, metric=m)
        improved = True
        while improved:
            improved = False
            for p in range(len(cur)):
                for c in A:
                    if c == cur[p]:
                        continue
                    cand = cur[:p] + c + cur[p + 1:]
                    sc = sum_distance(cand, S, metric=m)
                    if sc < cur_score:
                        cur, cur_score, improved = cand, sc, True
        if cur_score < best_score:
            best, best_score = cur, cur_score
    return best
'''

INDEL_MOVES = '''
from collections import Counter
from median_string.metrics import sum_distance

def solve(instance):
    S, m, A = instance.strings, instance.metric, list(instance.alphabet)
    L = round(sum(map(len, S)) / len(S))
    cur = "".join(Counter(s[i] for s in S if i < len(s)).most_common(1)[0][0] for i in range(L))
    cur_score = sum_distance(cur, S, metric=m)
    improved = True
    while improved:
        improved = False
        moves = []
        for p in range(len(cur)):
            for c in A:
                if c != cur[p]:
                    moves.append(cur[:p] + c + cur[p + 1:])
            if m == "levenshtein":
                moves.append(cur[:p] + cur[p + 1:])
                for c in A:
                    moves.append(cur[:p] + c + cur[p:])
        for cand in moves:
            sc = sum_distance(cand, S, metric=m)
            if sc < cur_score:
                cur, cur_score, improved = cand, sc, True
                break
    return cur
'''
