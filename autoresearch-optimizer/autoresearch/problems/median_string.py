"""Median String (Steiner String) problem adapter for the autoresearch loop."""

from __future__ import annotations

import signal
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
levenshtein_distance, levenshtein_editops, hamming_distance, calculate_distance, compute_set_median`.
No other third-party packages. Must be deterministic (seed any RNG).
TIME BUDGET: each solve() call gets instance.time_budget_ms of CPU time (default 1000 ms).
Going over by more than 25% makes that instance invalid (score = baseline + 1000), so check
time.process_time() against the budget and return your best string so far.
median_string.metrics.levenshtein_distance is a fast C++ implementation (~2 us at 40x40,
~0.1 ms at 1500x1500); levenshtein_editops(a, b) returns an optimal edit script
[(op, i, j), ...] for alignment-based methods. A DP written in pure Python is ~3000x slower.
"""

LONG_DESCRIPTION = DESCRIPTION.replace(
    "Instances are planted motifs with\nsubstitution noise 15-38% and indel noise 0-10%",
    "Instances are MSA-scale: four 1500 bp DNA\ninstances (k=10-20 strings, substitution noise 10-30%, indel noise 2-8%) and one\n"
    "500 aa protein instance (k=12). All use Levenshtein distance. A full single-edit neighbourhood of\n"
    "a 1500-char center has ~13k moves (~6 s to score naively), so the budget forces targeted search")

BUDGET_GRACE = 1.25  # an instance is invalid once solve() uses more than budget * grace of CPU


class _OverBudget(BaseException):
    """Raised by the CPU timer inside a candidate; a BaseException so `except Exception` can't swallow it."""


def _on_timer(signum, frame):
    raise _OverBudget


def _budgeted(solve, budget_ms: int, cpu_ms: dict[str, float]):
    """Wrap `solve` with a per-call CPU limit (SIGPROF) and a wall-clock backstop (SIGALRM)."""
    limit = budget_ms * BUDGET_GRACE / 1000

    def wrapped(instance):
        instance.time_budget_ms = budget_ms
        handlers = signal.signal(signal.SIGPROF, _on_timer), signal.signal(signal.SIGALRM, _on_timer)
        over = False
        t0 = time.process_time()
        try:
            try:
                signal.setitimer(signal.ITIMER_PROF, limit)
                signal.setitimer(signal.ITIMER_REAL, 3 * limit + 1)  # sleeping/blocked solvers
                out = solve(instance)
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)
                signal.setitimer(signal.ITIMER_REAL, 0)
        except _OverBudget:
            over = True
        finally:
            signal.signal(signal.SIGPROF, handlers[0])
            signal.signal(signal.SIGALRM, handlers[1])
        used = time.process_time() - t0
        cpu_ms[instance.name] = round(1000 * used, 1)
        if over or used > limit:
            raise TimeoutError(f"over budget: {1000 * used:.0f} ms CPU > {budget_ms} ms x {BUDGET_GRACE}")
        return out

    return wrapped



class MedianStringProblem:
    name = "median_string"
    # research directions handed out round-robin so parallel agents in the same mode diverge
    directions = (
        "speed: faster distance kernels (incremental prefix/suffix DP rows, bit-parallel Myers) so more search fits the budget",
        "starting points: better initial centers (weighted/positional consensus, progressive or star alignment, medoids)",
        "neighbourhood: richer moves (block shifts, pair edits, insert+delete swaps) beyond single edits",
        "acceptance: escape local optima (simulated annealing, tabu, plateau walks, late acceptance)",
        "budget use: anytime design that splits instance.time_budget_ms between restarts/phases adaptively",
        "instance-adaptive: detect metric, alphabet size, noise and indel level and switch strategy per instance",
        "alignment-based consensus: iterative re-alignment of all strings to the center with smarter voting",
        "population: keep several centers and recombine them (crossover of aligned segments, path relinking)",
        "exactness on small inputs: exhaustive or branch-and-bound search where the instance is small enough",
        "robustness: guard against regressions on the instances where the parent is already strong",
    )
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

    def evaluate(self, source: str, split: str, budget_ms: int | None = None) -> EvalResult:
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
        cpu_ms: dict[str, float] = {}
        if budget_ms:
            solve = _budgeted(solve, budget_ms, cpu_ms)
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
                cpu_ms=cpu_ms.get(r.instance_name, 0.0),
                info=(f"k={inst.num_strings} len={lens[0]}-{lens[-1]} |alphabet|={len(inst.alphabet)} "
                      f"metric={inst.metric} mut={inst.metadata.get('mutation_rate')} "
                      f"indel={inst.metadata.get('indel_rate')}"),
            ))
        return EvalResult(split, summary.total_score, summary.total_baseline_score,
                          instances=diags, elapsed=time.perf_counter() - t0)


def _baseline_total(instances) -> int:
    from median_string.metrics import compute_set_median
    return sum(compute_set_median(i.strings, metric=i.metric)[1] for i in instances)


class MedianStringLongProblem(MedianStringProblem):
    """Same contract with MSA-scale objective instances (1500 bp DNA, 500 aa protein)."""

    name = "median_string_long"
    timeouts = {"screen": 30.0, "validate": 120.0, "confirm": 120.0, "holdout": 120.0}  # noqa: RUF012
    _tiers = {"screen": "small", "validate": "long", "confirm": "long_confirm", "holdout": "long_holdout"}  # noqa: RUF012

    def describe(self) -> str:
        return LONG_DESCRIPTION
