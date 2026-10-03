"""Left-to-right beam search: extend prefixes one character at a time, scored by incremental edit-distance DP rows."""

def solve(instance) -> str:
    strings, alphabet, width = instance.strings, instance.alphabet, 8
    max_len = max(len(s) for s in strings)

    def step(row, s, ch):
        new = [row[0] + 1]
        for j, c in enumerate(s, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (c != ch)))
        return new

    def lower_bound(rows):
        return sum(min(r) for r in rows)

    def total(rows):
        return sum(r[-1] for r in rows)

    start = [list(range(len(s) + 1)) for s in strings]
    beam = [("", start)]
    best, best_score = "", total(start)
    for _ in range(max_len + 2):
        expanded = []
        for prefix, rows in beam:
            for ch in alphabet:
                new_rows = [step(r, s, ch) for r, s in zip(rows, strings)]
                expanded.append((prefix + ch, new_rows))
                if total(new_rows) < best_score:
                    best, best_score = prefix + ch, total(new_rows)
        beam = sorted(expanded, key=lambda e: lower_bound(e[1]))[:width]
    return best
