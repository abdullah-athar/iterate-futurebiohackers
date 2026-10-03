"""Evaluator and benchmarking harness for Median String / Steiner String solutions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, Union

from .base_solver import BaseSolver, normalize_solver
from .benchmarks import BenchmarkTier, get_benchmark_suite
from .instance import ProblemInstance
from .metrics import compute_set_median, sum_distance
from .solvers import get_solver, list_solvers


@dataclass
class InstanceResult:
    """Evaluation result on a single ProblemInstance."""

    instance_name: str
    candidate: str
    is_valid: bool
    error_message: str
    score: int  # Total Steiner distance (sum of distances) - lower is better
    mean_distance: float
    baseline_score: int  # Set Median score
    improvement_over_baseline: int  # baseline_score - score (positive = better)
    planted_score: int | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Aggregate evaluation summary across all instances in a benchmark suite."""

    solver_name: str
    tier: str
    total_score: int
    total_baseline_score: int
    net_improvement: int
    relative_improvement_pct: float
    win_count: int
    tie_count: int
    loss_count: int
    num_valid: int
    num_instances: int
    total_time_seconds: float
    instance_results: list[InstanceResult] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return self.num_valid == self.num_instances

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "tier": self.tier,
            "total_score": self.total_score,
            "total_baseline_score": self.total_baseline_score,
            "net_improvement": self.net_improvement,
            "relative_improvement_pct": round(self.relative_improvement_pct, 2),
            "win_count": self.win_count,
            "tie_count": self.tie_count,
            "loss_count": self.loss_count,
            "num_valid": self.num_valid,
            "num_instances": self.num_instances,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "instance_results": [r.to_dict() for r in self.instance_results],
        }

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Render a readable summary table in the terminal."""
        print(f"\n{'='*75}")
        print(f"MEDIAN STRING EVALUATION: {self.solver_name} (Tier: {self.tier})")
        print(f"{'='*75}")
        header = f"{'Instance':<26} | {'Score':<6} | {'Baseline':<8} | {'Delta':<7} | {'Time (s)':<8} | {'Valid'}"
        print(header)
        print("-" * len(header))

        for res in self.instance_results:
            delta_str = f"{res.improvement_over_baseline:+d}"
            valid_str = "OK" if res.is_valid else f"ERR: {res.error_message[:15]}"
            print(
                f"{res.instance_name[:26]:<26} | "
                f"{res.score:<6} | "
                f"{res.baseline_score:<8} | "
                f"{delta_str:<7} | "
                f"{res.elapsed_seconds:<8.4f} | "
                f"{valid_str}"
            )

        print("-" * len(header))
        pct_sign = "+" if self.relative_improvement_pct >= 0 else ""
        print(f"Total Score (lower is better): {self.total_score} (Baseline: {self.total_baseline_score})")
        print(f"Net Improvement vs Baseline : {self.net_improvement:+d} ({pct_sign}{self.relative_improvement_pct:.2f}%)")
        print(f"Record (Wins / Ties / Losses): {self.win_count}W / {self.tie_count}T / {self.loss_count}L")
        print(f"Validity                     : {self.num_valid}/{self.num_instances} valid solutions")
        print(f"Total Elapsed Time           : {self.total_time_seconds:.4f}s")
        print(f"{'='*75}\n")


class Evaluator:
    """Benchmark harness to evaluate and compare Median String solvers."""

    def __init__(self, default_tier: BenchmarkTier = "small") -> None:
        self.default_tier = default_tier

    def evaluate_solver(
        self,
        solver: Union[BaseSolver, Callable[[ProblemInstance], str], str],
        benchmark: Union[BenchmarkTier, list[ProblemInstance]] = "small",
        verbose: bool = True,
    ) -> EvaluationSummary:
        """Evaluate a single solver on a benchmark suite.

        Args:
            solver: BaseSolver instance, callable fn(instance) -> str, or registered solver name.
            benchmark: Tier name ("small", "medium", "hard") or list of ProblemInstance objects.
            verbose: If True, prints a summary table to stdout.

        Returns:
            EvaluationSummary dataclass containing detailed performance metrics.
        """
        if isinstance(solver, str):
            solver_obj = get_solver(solver)
            solver_name = solver
        else:
            solver_obj = normalize_solver(solver)
            solver_name = getattr(solver_obj, "name", solver_obj.__class__.__name__)

        if isinstance(benchmark, str):
            instances = get_benchmark_suite(benchmark)
            tier_name = benchmark
        else:
            instances = benchmark
            tier_name = "custom"

        results: list[InstanceResult] = []
        total_time = 0.0

        for inst in instances:
            t0 = time.perf_counter()
            error_msg = ""
            candidate = ""

            try:
                candidate = solver_obj.solve(inst)
            except Exception as e:
                error_msg = f"Crash: {type(e).__name__}: {e}"

            elapsed = time.perf_counter() - t0
            total_time += elapsed

            # Validate solution
            if not error_msg:
                is_valid, validation_err = inst.validate_candidate(candidate)
                if not is_valid:
                    error_msg = validation_err

            # Compute set-median baseline score for reference (respecting constraints)
            _, baseline_score = compute_set_median(
                inst.strings,
                metric=inst.metric,
                target_length=inst.target_length,
                alphabet=inst.alphabet,
            )

            if not error_msg:
                is_valid = True
                score = sum_distance(candidate, inst.strings, metric=inst.metric)
                mean_dist = score / len(inst.strings)
            else:
                is_valid = False
                # Penalty score for invalid candidates: baseline + large penalty
                score = baseline_score + 1000
                mean_dist = float("inf")

            improvement = baseline_score - score

            results.append(
                InstanceResult(
                    instance_name=inst.name,
                    candidate=candidate if is_valid else f"[INVALID: {error_msg}]",
                    is_valid=is_valid,
                    error_message=error_msg,
                    score=score,
                    mean_distance=mean_dist,
                    baseline_score=baseline_score,
                    improvement_over_baseline=improvement,
                    planted_score=inst.known_best_score,
                    elapsed_seconds=elapsed,
                )
            )

        # Aggregate metrics
        total_score = sum(r.score for r in results)
        total_baseline = sum(r.baseline_score for r in results)
        net_improvement = total_baseline - total_score
        rel_improvement = ((total_baseline - total_score) / max(total_baseline, 1)) * 100.0

        win_count = sum(1 for r in results if r.improvement_over_baseline > 0)
        tie_count = sum(1 for r in results if r.improvement_over_baseline == 0)
        loss_count = sum(1 for r in results if r.improvement_over_baseline < 0)
        num_valid = sum(1 for r in results if r.is_valid)

        summary = EvaluationSummary(
            solver_name=solver_name,
            tier=tier_name,
            total_score=total_score,
            total_baseline_score=total_baseline,
            net_improvement=net_improvement,
            relative_improvement_pct=rel_improvement,
            win_count=win_count,
            tie_count=tie_count,
            loss_count=loss_count,
            num_valid=num_valid,
            num_instances=len(instances),
            total_time_seconds=total_time,
            instance_results=results,
        )

        if verbose:
            summary.print_summary()

        return summary

    def compare_solvers(
        self,
        solvers: Sequence[Union[BaseSolver, Callable[[ProblemInstance], str], str]],
        benchmark: Union[BenchmarkTier, list[ProblemInstance]] = "small",
    ) -> list[EvaluationSummary]:
        """Compare multiple solvers on the same benchmark suite."""
        summaries = [self.evaluate_solver(s, benchmark=benchmark, verbose=False) for s in solvers]

        # Sort by total_score ascending (lower is better)
        summaries.sort(key=lambda s: s.total_score)

        print(f"\n{'='*85}")
        print(f"SOLVER LEADERBOARD COMPARISON (Tier: {benchmark if isinstance(benchmark, str) else 'custom'})")
        print(f"{'='*85}")
        header = f"{'Rank':<4} | {'Solver':<24} | {'Score':<7} | {'vs Baseline':<12} | {'Record (W/T/L)':<14} | {'Time (s)':<8} | {'Valid'}"
        print(header)
        print("-" * len(header))

        for rank, s in enumerate(summaries, 1):
            imp_sign = "+" if s.net_improvement >= 0 else ""
            delta_str = f"{imp_sign}{s.net_improvement} ({imp_sign}{s.relative_improvement_pct:.1f}%)"
            record_str = f"{s.win_count}/{s.tie_count}/{s.loss_count}"
            valid_str = f"{s.num_valid}/{s.num_instances}"
            print(
                f"{rank:<4} | "
                f"{s.solver_name[:24]:<24} | "
                f"{s.total_score:<7} | "
                f"{delta_str:<12} | "
                f"{record_str:<14} | "
                f"{s.total_time_seconds:<8.4f} | "
                f"{valid_str}"
            )

        print(f"{'='*85}\n")
        return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Median String / Steiner String Solvers.")
    parser.add_argument(
        "--solver",
        type=str,
        default="set_median",
        help=f"Registered solver to evaluate. Available: {list_solvers()}",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="small",
        choices=["small", "medium", "hard"],
        help="Difficulty tier for the benchmark suite (default: small).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Comma-separated list of solvers to compare (e.g. 'set_median,frequency_consensus,template').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON evaluation results.",
    )
    parser.add_argument(
        "--list-solvers",
        action="store_true",
        help="List all registered solvers and exit.",
    )

    args = parser.parse_args()

    if args.list_solvers:
        print("Registered solvers:", ", ".join(list_solvers()))
        sys.exit(0)

    evaluator = Evaluator(default_tier=args.tier)

    if args.compare:
        solver_names = [s.strip() for s in args.compare.split(",") if s.strip()]
        summaries = evaluator.compare_solvers(solver_names, benchmark=args.tier)
        if args.output:
            data = [s.to_dict() for s in summaries]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved comparison results to {args.output}")
    else:
        summary = evaluator.evaluate_solver(args.solver, benchmark=args.tier, verbose=True)
        if args.output:
            summary.save_json(args.output)
            print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
