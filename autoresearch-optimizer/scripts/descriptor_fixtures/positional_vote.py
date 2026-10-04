"""Column-wise majority vote over the raw (unaligned) input strings."""

from collections import Counter


def solve(instance) -> str:
    length = round(sum(len(s) for s in instance.strings) / len(instance.strings))
    out = []
    for pos in range(length):
        counts = Counter(s[pos] for s in instance.strings if pos < len(s))
        out.append(counts.most_common(1)[0][0] if counts else instance.alphabet[0])
    return "".join(out)
