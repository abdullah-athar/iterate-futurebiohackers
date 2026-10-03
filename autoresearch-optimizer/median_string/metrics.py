"""Distance metrics and scoring functions for the Median String problem."""

from __future__ import annotations


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein (edit) distance between two strings using 2-row DP.

    Operations: insertion (cost 1), deletion (cost 1), substitution (cost 1).
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Keep s2 as the shorter string to optimize memory
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev_row = list(range(len2 + 1))
    curr_row = [0] * (len2 + 1)

    for i, char1 in enumerate(s1):
        curr_row[0] = i + 1
        for j, char2 in enumerate(s2):
            cost = 0 if char1 == char2 else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,        # Insertion
                prev_row[j + 1] + 1,    # Deletion
                prev_row[j] + cost,     # Substitution
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len2]


def hamming_distance(s1: str, s2: str) -> int:
    """Calculate the Hamming distance (number of mismatched positions).

    If lengths differ, mismatches are counted up to min length, plus the length difference.
    """
    min_len = min(len(s1), len(s2))
    mismatches = sum(1 for a, b in zip(s1, s2) if a != b)
    return mismatches + abs(len(s1) - len(s2))


def calculate_distance(s1: str, s2: str, metric: str = "levenshtein") -> int:
    """Compute distance between two strings according to the chosen metric."""
    if metric == "levenshtein":
        return levenshtein_distance(s1, s2)
    elif metric == "hamming":
        return hamming_distance(s1, s2)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'levenshtein' or 'hamming'.")


def sum_distance(candidate: str, strings: list[str], metric: str = "levenshtein") -> int:
    """Calculate the total Steiner/median objective: sum of distances to all strings in S.

    Objective to MINIMIZE: f(candidate) = sum_{s in S} d(candidate, s)
    """
    return sum(calculate_distance(candidate, s, metric=metric) for s in strings)


def mean_distance(candidate: str, strings: list[str], metric: str = "levenshtein") -> float:
    """Calculate the average distance between the candidate and the target strings."""
    if not strings:
        return 0.0
    return sum_distance(candidate, strings, metric=metric) / len(strings)


def normalized_distance(candidate: str, strings: list[str], metric: str = "levenshtein") -> float:
    """Calculate the average normalized distance in [0, 1].

    For each string s, normalized_d = d(candidate, s) / max(len(candidate), len(s), 1).
    """
    if not strings:
        return 0.0
    total = 0.0
    for s in strings:
        d = calculate_distance(candidate, s, metric=metric)
        max_len = max(len(candidate), len(s), 1)
        total += d / max_len
    return total / len(strings)


def compute_set_median(
    strings: list[str],
    metric: str = "levenshtein",
    target_length: int | None = None,
    alphabet: str = "ACGT",
) -> tuple[str, int]:
    """Find the best representative string within the input set S (Set Median baseline).

    The set median is guaranteed to achieve a 2-approximation factor for the
    generalized median string under any metric space (triangle inequality).

    If target_length is provided, candidates are filtered or adapted to satisfy
    the target length constraint.

    Returns:
        (best_string, minimum_total_distance)
    """
    if not strings:
        raise ValueError("Cannot compute set median of empty collection.")

    candidates = strings
    if target_length is not None:
        exact_match_candidates = [s for s in candidates if len(s) == target_length]
        if exact_match_candidates:
            candidates = exact_match_candidates
        else:
            adapted: list[str] = []
            for s in strings:
                if len(s) > target_length:
                    adapted.append(s[:target_length])
                elif len(s) < target_length:
                    pad_char = s[-1] if s else alphabet[0]
                    adapted.append(s + pad_char * (target_length - len(s)))
                else:
                    adapted.append(s)
            candidates = adapted

    best_str = candidates[0]
    best_dist = float("inf")

    for s_cand in candidates:
        total_dist = sum_distance(s_cand, strings, metric=metric)
        if total_dist < best_dist:
            best_dist = total_dist
            best_str = s_cand

    return best_str, int(best_dist)

