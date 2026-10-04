"""Measure the evaluator's run-to-run noise on the objective split (local CPU only, no LLM calls).

Solvers are deterministic given their RNG seed, but they stop on a CPU-time deadline, so the work done
(and the score) can vary between two evaluations of the same file. The family-grace epsilon must sit above
this noise so that re-scoring noise never buys a protected family an extra refinement.

Usage: uv run python scripts/measure_eval_noise.py [--problem median_string] [--reps 5] [--out DIR] [FILES...]
(default files: the seed and the time-bounded descriptor fixtures)
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.problem import get_problem  # noqa: E402
from autoresearch.sandbox import get_evaluate  # noqa: E402

DEFAULT = ["autoresearch/seeds/median_string_seed.py", "scripts/descriptor_fixtures/simulated_annealing.py",
           "scripts/descriptor_fixtures/iterated_local_search.py", "scripts/descriptor_fixtures/tabu_search.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--problem", default="median_string")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget-ms", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "calibration")
    args = ap.parse_args()
    problem = get_problem(args.problem)
    evaluate = get_evaluate()
    rows = []
    for f in args.files or DEFAULT:
        src = (ROOT / f).read_text() if not Path(f).is_absolute() else Path(f).read_text()
        scores = []
        for _ in range(args.reps):
            res = evaluate(args.problem, problem, src, problem.objective_split, args.budget_ms)
            scores.append(res.score if res.ok else float("nan"))
        ok = [s for s in scores if s == s]
        row = {"file": f, "scores": scores, "mean": st.mean(ok) if ok else None,
               "sd": st.stdev(ok) if len(ok) > 1 else 0.0, "range": (max(ok) - min(ok)) if ok else None}
        rows.append(row)
        print(f"{f:55s} mean={row['mean']:.1f} sd={row['sd']:.2f} range={row['range']:g}  {scores}")
    worst = max((r["range"] or 0) for r in rows)
    out = {"problem": args.problem, "split": problem.objective_split, "reps": args.reps, "budget_ms": args.budget_ms,
           "rows": rows, "max_range": worst, "suggested_epsilon": max(1.0, 2 * worst)}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"eval_noise-{args.problem}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"max range {worst:g} -> suggested grace epsilon {out['suggested_epsilon']:g} (2 x range); written {path}")


if __name__ == "__main__":
    main()
