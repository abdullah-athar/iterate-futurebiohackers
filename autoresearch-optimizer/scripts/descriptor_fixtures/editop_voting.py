"""Edit-operation voting: repeatedly apply the single edit operation proposed by the most input strings."""

from collections import Counter

from median_string.metrics import levenshtein_editops, sum_distance


def solve(instance) -> str:
    strings = instance.strings
    cur = min(strings, key=lambda s: sum_distance(s, strings))
    cur_score = sum_distance(cur, strings)
    for _ in range(4 * len(cur)):
        votes = Counter()
        for s in strings:
            for op, i, j in levenshtein_editops(cur, s):
                votes[(op, i, s[j] if op != "delete" else "")] += 1
        applied = False
        for (op, i, ch), _ in votes.most_common(10):
            if op == "replace":
                cand = cur[:i] + ch + cur[i + 1:]
            elif op == "insert":
                cand = cur[:i] + ch + cur[i:]
            else:
                cand = cur[:i] + cur[i + 1:]
            score = sum_distance(cand, strings)
            if score < cur_score:
                cur, cur_score, applied = cand, score, True
                break
        if not applied:
            break
    return cur
