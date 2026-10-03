"""Prompt text for LLM proposers."""

from __future__ import annotations

import json

from .core import ResearchContext

SYSTEM_PROMPT = """\
You are an algorithms researcher improving a solver for the Median String (Steiner String) problem.

Problem: given a set of strings over an alphabet, find a string (any length, alphabet symbols only)
that minimises the SUM of distances to all input strings. The distance is usually Levenshtein
(unit-cost insert/delete/substitute); some instances use Hamming (mismatches + length difference).
The score is the total distance summed over all benchmark instances. Lower is better. The baseline
is the Set Median (the best input string).

Solver contract: return ONE complete Python module that defines exactly one subclass of
`median_string.base_solver.BaseSolver`, with a no-argument constructor and a method
`solve(self, instance) -> str`. `instance` is a `median_string.instance.ProblemInstance` with fields
`strings: list[str]`, `alphabet: str`, `metric: str` ("levenshtein" or "hamming"),
`target_length: int | None` (if set, the answer must have exactly that length), and the property
`mean_length`. Do not read `planted_consensus` or `known_best_score`; that counts as cheating.

Allowed imports: the Python standard library, plus `median_string.metrics`
(`levenshtein_distance`, `hamming_distance`, `calculate_distance(s1, s2, metric)`,
`sum_distance(candidate, strings, metric)`, `compute_set_median`). Import everything with absolute
imports. The whole benchmark suite has a hard wall-clock limit of {timeout_s:.0f} seconds, and the
distance functions are pure Python, so keep the search efficient.

Reply with a short rationale (2-4 sentences: what you changed and why), followed by the full module
in a single ```python code block."""


def build_prompt(ctx: ResearchContext) -> str:
    parts = [f"Iteration {ctx.iteration}. Benchmark suite: {ctx.suite}."]
    for rec in ctx.parents:
        per_instance = json.dumps(rec.result.per_instance, indent=1)
        parts.append(
            f"## Current champion (id {rec.candidate.id})\n"
            f"Total score: {rec.result.score} (set-median baseline {rec.result.baseline_score}), "
            f"time {rec.result.elapsed:.1f}s.\nPer instance:\n{per_instance}\n\n"
            f"```python\n{rec.candidate.code}\n```"
        )
    parts.append(f"## Recent attempts\n{ctx.history_summary}")
    parts.append("Propose an improved solver that lowers the total score. Do not repeat failed ideas.")
    return "\n\n".join(parts)
