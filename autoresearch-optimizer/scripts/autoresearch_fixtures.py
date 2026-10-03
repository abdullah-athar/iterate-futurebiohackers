"""Fixture candidates for the tests: a near-duplicate of the seed, a syntax error, and two real solvers."""

from autoresearch.problem import get_problem

# the seed with renamed locals and an extra comment -> must be rejected by the novelty gate
SEED_DUPLICATE = "# cosmetic rewrite of the seed\n" + (
    get_problem("median_string").seed_source().replace("best_score", "incumbent_score").replace("rng", "gen"))

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

# set median + first-improvement search over substitutions/insertions/deletions; stops at 80% of budget
FAST_INDEL_SEARCH = '''
import time
from median_string.metrics import sum_distance

def solve(instance):
    deadline = time.process_time() + 0.8 * (instance.time_budget_ms or 1000) / 1000
    S, A, m = instance.strings, instance.alphabet, instance.metric
    best = min(S, key=lambda s: sum_distance(s, S, metric=m))
    score = sum_distance(best, S, metric=m)
    improved = True
    while improved and time.process_time() < deadline:
        improved = False
        for i in range(len(best) + 1):
            if time.process_time() > deadline:
                break
            moves = [best[:i] + c + best[i:] for c in A]
            if i < len(best):
                moves += [best[:i] + c + best[i + 1:] for c in A if c != best[i]] + [best[:i] + best[i + 1:]]
            for cand in moves:
                sc = sum_distance(cand, S, metric=m)
                if sc < score:
                    best, score, improved = cand, sc, True
    return best
'''


# passes screen (returns the set median) but spins past the CPU budget on the validate instances
SLOW_ON_VALIDATE = '''
import time
from median_string.metrics import compute_set_median

def solve(instance):
    if len(instance.strings[0]) > 18:
        t = time.process_time()
        while time.process_time() - t < 2:
            pass
    return compute_set_median(instance.strings, metric=instance.metric)[0]
'''
