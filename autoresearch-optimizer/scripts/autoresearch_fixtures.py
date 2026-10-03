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

# steepest descent over substitutions/insertions/deletions with incremental DP; fits the 1000 ms budget
FAST_DESCENT = '''
"""Set-median start + steepest-descent local search over the full single-edit neighbourhood.

Every substitution, insertion and deletion is scored incrementally: for each input string s we
keep forward DP rows F[i] = d(t[:i], s[:j]) and backward rows B[i] = d(t[i:], s[j:]), so
d(t', s) for a single-edit neighbour t' is a min over j of a stitched row, O(len(s)).
"""

from operator import add

from median_string.metrics import sum_distance


def _forward(t, s):
    m = len(s)
    prev = list(range(m + 1))
    rows = [prev]
    for i, a in enumerate(t, 1):
        cur = [i] * (m + 1)
        for j in range(1, m + 1):
            v = prev[j - 1] + (a != s[j - 1])
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            cur[j] = v
        rows.append(cur)
        prev = cur
    return rows


def _backward(t, s):
    n, m = len(t), len(s)
    prev = [m - j for j in range(m + 1)]
    rows = [None] * (n + 1)
    rows[n] = prev
    for i in range(n - 1, -1, -1):
        a = t[i]
        cur = [n - i] * (m + 1)
        for j in range(m - 1, -1, -1):
            v = prev[j + 1] + (a != s[j])
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j + 1] + 1 < v:
                v = cur[j + 1] + 1
            cur[j] = v
        rows[i] = cur
        prev = cur
    return rows


def _step(prev, c, s):
    cur = [prev[0] + 1] * (len(prev))
    for j in range(1, len(prev)):
        v = prev[j - 1] + (c != s[j - 1])
        if prev[j] + 1 < v:
            v = prev[j] + 1
        if cur[j - 1] + 1 < v:
            v = cur[j - 1] + 1
        cur[j] = v
    return cur


def _best_move(t, strings, alphabet):
    """Return (total, new_string) of the best single-edit neighbour of t."""
    n = len(t)
    sub = [[0] * len(alphabet) for _ in range(n)]
    ins = [[0] * len(alphabet) for _ in range(n + 1)]
    dele = [0] * n
    for s in strings:
        F = _forward(t, s)
        B = _backward(t, s)
        for i in range(n + 1):
            Fi = F[i]
            Bi = B[i]
            Bn = B[i + 1] if i < n else None
            if Bn is not None:
                dele[i] += min(map(add, Fi, Bn))
            for k, c in enumerate(alphabet):
                G = _step(Fi, c, s)
                ins[i][k] += min(map(add, G, Bi))
                if Bn is not None:
                    sub[i][k] += min(map(add, G, Bn))
    best = None
    for i in range(n + 1):
        for k, c in enumerate(alphabet):
            if i < n and c != t[i]:
                cand = (sub[i][k], t[:i] + c + t[i + 1:])
                if best is None or cand[0] < best[0]:
                    best = cand
            cand = (ins[i][k], t[:i] + c + t[i:])
            if best is None or cand[0] < best[0]:
                best = cand
        if i < n and n > 1:
            cand = (dele[i], t[:i] + t[i + 1:])
            if cand[0] < best[0]:
                best = cand
    return best


def _hamming_consensus(strings, alphabet):
    out = []
    for col in zip(*strings):
        out.append(max(alphabet, key=lambda c: (col.count(c), -alphabet.index(c))))
    return "".join(out)


def solve(instance) -> str:
    strings = instance.strings
    alphabet = instance.alphabet
    if instance.metric == "hamming":
        return _hamming_consensus(strings, alphabet)

    t = min(strings, key=lambda s: sum_distance(s, strings, metric="levenshtein"))
    score = sum_distance(t, strings, metric="levenshtein")
    while True:
        total, cand = _best_move(t, strings, alphabet)
        if total >= score:
            return t
        t, score = cand, total
'''
