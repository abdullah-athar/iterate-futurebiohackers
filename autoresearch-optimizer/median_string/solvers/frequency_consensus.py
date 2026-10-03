"""Frequency Consensus solver.

Constructs a consensus sequence by taking the most frequent character
at each position across the input strings (majority vote).
"""

from __future__ import annotations

from collections import Counter
from ..base_solver import BaseSolver
from ..instance import ProblemInstance


class FrequencyConsensusSolver(BaseSolver):
    """Positional majority-vote consensus solver."""

    name = "frequency_consensus"
    description = "Majority vote: selects the most frequent character at each position."

    def solve(self, instance: ProblemInstance) -> str:
        # Determine the length of the consensus string
        if instance.target_length is not None:
            length = instance.target_length
        else:
            length = int(round(instance.mean_length))

        consensus_chars: list[str] = []
        default_char = instance.alphabet[0]

        for pos in range(length):
            counts: Counter[str] = Counter()
            for s in instance.strings:
                if pos < len(s) and s[pos] in instance.alphabet_set:
                    counts[s[pos]] += 1

            if counts:
                # Pick the most common character at this position
                most_common_char = counts.most_common(1)[0][0]
                consensus_chars.append(most_common_char)
            else:
                consensus_chars.append(default_char)

        return "".join(consensus_chars)
