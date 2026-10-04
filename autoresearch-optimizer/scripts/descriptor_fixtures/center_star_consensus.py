"""Center-star alignment: align every string to the set median, then majority-vote each alignment column."""

from collections import Counter

from median_string.metrics import sum_distance


def _align(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    # traceback: for each center position, the aligned char of b (or '-') and insertions before it
    cols, ins = [None] * n, [[] for _ in range(n + 1)]
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            cols[i - 1] = b[j - 1]; i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            cols[i - 1] = "-"; i -= 1
        else:
            ins[i].append(b[j - 1]); j -= 1
    return cols, ins


def solve(instance) -> str:
    strings = instance.strings
    center = min(strings, key=lambda s: sum_distance(s, strings, metric=instance.metric))
    aligned = [_align(center, s) for s in strings]
    out = []
    for pos in range(len(center) + 1):
        # keep an insertion slot only if most strings insert something there
        inserted = [ins[pos][-1] for _, ins in aligned if ins[pos]]
        if len(inserted) > len(strings) / 2:
            out.append(Counter(inserted).most_common(1)[0][0])
        if pos < len(center):
            ch = Counter(cols[pos] for cols, _ in aligned).most_common(1)[0][0]
            if ch != "-":
                out.append(ch)
    return "".join(out)
