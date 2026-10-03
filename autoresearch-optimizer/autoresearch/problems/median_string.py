"""Median String (Steiner String) problem adapter for the autoresearch loop."""

from __future__ import annotations

import time
from pathlib import Path

from median_string import Evaluator, get_benchmark_suite
from median_string.base_solver import FunctionalSolver

from ..problem import EvalResult, InstanceDiag
from ..sandbox import load_solve_function

SEED_PATH = Path(__file__).resolve().parent.parent / "seeds" / "median_string_seed.py"

DESCRIPTION = """\
PROBLEM: Median String / Steiner String (NP-hard).
Given strings S over an alphabet (DNA "ACGT" or 20 amino acids), return a string t
minimising sum_{s in S} d(t, s). d is Levenshtein distance, or Hamming distance when
instance.metric == "hamming" (then keep len(t) == len of inputs). Lower is better.
Baseline per instance = set median (best input string). Instances are planted motifs with
substitution noise 15-38% and indel noise 0-10%; the planted string's score is reported as
"best_known" (a strong but not necessarily optimal reference).

SOLVER CONTRACT: a single Python file defining `def solve(instance) -> str`.
  instance.strings: list[str]      instance.alphabet: str
  instance.metric: "levenshtein" | "hamming"   instance.target_length: None (any length ok)
Allowed imports: Python stdlib and `from median_string.metrics import sum_distance,
levenshtein_distance, hamming_distance, calculate_distance, compute_set_median`.
No other third-party packages. Must be deterministic (seed any RNG).
TIME BUDGET: whole 'screen' split < 20 s, whole 'validate' split (5 instances, 10-15 strings
of 20-50 chars) < 90 s. Levenshtein in pure Python is slow: a 40x40 DP is ~1 ms; budget
the number of objective evaluations accordingly (roughly <= 20k per instance).
"""


class MedianStringProblem:
    name = "median_string"
    splits = ("screen", "validate", "confirm", "holdout")
    objective_split = "validate"
    confirm_split = "confirm"
    allowed_imports = ("median_string.metrics",)
    timeouts = {"screen": 30.0, "validate": 120.0, "confirm": 120.0, "holdout": 300.0}  # noqa: RUF012
    _tiers = {"screen": "small", "validate": "medium", "confirm": "confirm", "holdout": "holdout"}  # noqa: RUF012

    def describe(self) -> str:
        return DESCRIPTION

    def seed_source(self) -> str:
        return SEED_PATH.read_text()

    def evaluate(self, source: str, split: str) -> EvalResult:
        tier = self._tiers[split]
        t0 = time.perf_counter()
        try:
            solve = load_solve_function(source)
        except Exception as e:  # noqa: BLE001 - any load error is a whole-split failure
            instances = get_benchmark_suite(tier)
            baseline = _baseline_total(instances)
            return EvalResult(split, baseline + 1000 * len(instances), baseline,
                              elapsed=time.perf_counter() - t0,
                              error=f"Failed to load solver: {type(e).__name__}: {e}")
        summary = Evaluator().evaluate_solver(
            FunctionalSolver(solve, name="candidate"), benchmark=tier, verbose=False
        )
        by_name = {i.name: i for i in get_benchmark_suite(tier)}
        diags = []
        for r in summary.instance_results:
            inst = by_name[r.instance_name]
            lens = sorted(len(s) for s in inst.strings)
            diags.append(InstanceDiag(
                name=r.instance_name,
                score=r.score,
                baseline=r.baseline_score,
                best_known=r.planted_score,
                valid=r.is_valid,
                error=r.error_message,
                elapsed=round(r.elapsed_seconds, 3),
                info=(f"k={inst.num_strings} len={lens[0]}-{lens[-1]} |alphabet|={len(inst.alphabet)} "
                      f"metric={inst.metric} mut={inst.metadata.get('mutation_rate')} "
                      f"indel={inst.metadata.get('indel_rate')}"),
            ))
        return EvalResult(split, summary.total_score, summary.total_baseline_score,
                          instances=diags, elapsed=time.perf_counter() - t0)


def _baseline_total(instances) -> int:
    from median_string.metrics import compute_set_median
    return sum(compute_set_median(i.strings, metric=i.metric)[1] for i in instances)
