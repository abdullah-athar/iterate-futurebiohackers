"""CLI: python -m autoresearch --proposer claude --iterations 10 --search-tier medium"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import REGISTRY, build
from .core import SuiteSpec
from .loop import run_loop, save_report
from .proposers import seed_candidate
from .suites import HELDOUT_SEED_OFFSET

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "autoresearch"


def main() -> None:
    p = argparse.ArgumentParser(description="Autoresearch loop for the Median String problem")
    for kind, default in [("proposer", "claude"), ("evaluator", "subprocess"), ("selector", "greedy"),
                          ("memory", "jsonl"), ("budget", "iterations")]:
        p.add_argument(f"--{kind}", default=default, choices=sorted(REGISTRY[kind]))
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--search-tier", default="medium", choices=["small", "medium", "hard"])
    p.add_argument("--timeout", type=float, default=60.0, help="Seconds per candidate evaluation")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--seed", type=int, default=0, help="Random seed for the mock proposer")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    args.run_dir = ARTIFACTS / (args.run_name or time.strftime("%Y%m%d-%H%M%S"))
    print(f"Run directory: {args.run_dir}")

    report = run_loop(
        seed=seed_candidate(),
        proposer=build("proposer", args.proposer, args),
        evaluator=build("evaluator", args.evaluator, args),
        selector=build("selector", args.selector, args),
        memory=build("memory", args.memory, args),
        budget=build("budget", args.budget, args),
        search=SuiteSpec(args.search_tier, 0),
        heldout=SuiteSpec(args.search_tier, HELDOUT_SEED_OFFSET),
    )
    save_report(report, args.run_dir / "summary.json")


if __name__ == "__main__":
    main()
