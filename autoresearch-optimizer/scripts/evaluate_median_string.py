"""Run evaluation experiments for Median String / Steiner String solvers.

Usage examples:
    # Run evaluation on set median baseline:
    uv run python scripts/evaluate_median_string.py --solver set_median

    # Compare all registered solvers:
    uv run python scripts/evaluate_median_string.py --compare all --tier small

    # Run on medium tier and save JSON report:
    uv run python scripts/evaluate_median_string.py --compare all --tier medium --output artifacts/report.json
"""

import argparse
import sys
from pathlib import Path

# Ensure autoresearch-optimizer root is on sys.path
optimizer_root = Path(__file__).resolve().parent.parent
if str(optimizer_root) not in sys.path:
    sys.path.insert(0, str(optimizer_root))

from median_string import Evaluator, list_solvers


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
        choices=["small", "medium", "hard", "confirm", "holdout"],
        help="Difficulty tier: 'small' (<0.1s), 'medium' (~1s), 'hard' (~5s).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Solvers to compare: comma-separated list or 'all'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save evaluation summary as JSON.",
    )

    args = parser.parse_args()
    evaluator = Evaluator(default_tier=args.tier)

    if args.compare:
        if args.compare.strip().lower() == "all":
            solvers_to_compare = list_solvers()
        else:
            solvers_to_compare = [s.strip() for s in args.compare.split(",") if s.strip()]

        print(f"Comparing solvers: {solvers_to_compare} on benchmark tier '{args.tier}'...")
        summaries = evaluator.compare_solvers(solvers_to_compare, benchmark=args.tier)

        if args.output:
            import json
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in summaries], f, indent=2)
            print(f"Report saved to {out_file}")
    else:
        summary = evaluator.evaluate_solver(args.solver, benchmark=args.tier, verbose=True)
        if args.output:
            summary.save_json(args.output)
            print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
