"""Compare swarm runs: best objective over time, holdout, cost, and strategy diversity.

Diversity: every evaluated candidate of every run is described with one shared vocabulary
(cache in --out), then per run: distinct core (tier-6) terms and mean pairwise descriptor distance.

Usage: uv run python scripts/compare_runs.py artifacts/runs/ablation-bandit artifacts/runs/ablation-ee
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.descriptors import Embedder, Vocabulary, describe, distance  # noqa: E402
from autoresearch.ledger import RunStore  # noqa: E402
from autoresearch.problem import get_problem  # noqa: E402


def trajectory(entries, marks_min=(5, 10, 15, 20, 25)) -> dict[int, float]:
    t0 = entries[0].timestamp
    best, out, i = float("inf"), {}, 0
    for e in entries:
        if e.scored and e.objective != float("inf") and e.confirmed is not False:
            while i < len(marks_min) and e.timestamp - t0 > marks_min[i] * 60:
                out[marks_min[i]] = best
                i += 1
            best = min(best, e.objective)
    for m in marks_min[i:]:
        out[m] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "descriptors" / "compare")
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    problem = " ".join(get_problem("median_string").describe().strip().splitlines()[:4])
    vocab_path, cache = args.out / "vocab.json", args.out / "descriptors.json"
    vocab = Vocabulary.load(vocab_path)
    emb = Embedder()
    rows = {}
    for run in args.runs:
        store = RunStore(run)
        entries = store.entries()
        seed = entries[0]
        evaluated = [e for e in entries if e.evals and e.id != 0]
        descs = []
        for e in evaluated:
            src = store.read_candidate(e)
            try:
                d = describe(src, vocab, problem, model=args.model, cache_path=cache)
            except RuntimeError:
                continue
            descs.append(d)
            vocab.add(d)
        vocab.save(vocab_path)
        scored = [e for e in entries if e.scored and e.objective != float("inf") and e.confirmed is not False]
        best = min(scored, key=lambda e: (e.objective, e.id))
        hold = json.loads((run / "holdout.json").read_text()) if (run / "holdout.json").exists() else {}
        pair = [distance(a, b, emb) for a, b in combinations(descs, 2)]
        rows[run.name] = {
            "seed": seed.objective, "best": best.objective, "best_id": best.id,
            "gain_pct": round(100 * (seed.objective - best.objective) / seed.objective, 2),
            "holdout_seed": hold.get("seed", {}).get("score"), "holdout_best": hold.get("best", {}).get("score"),
            "generations": max((e.generation or 0 for e in entries), default=0),
            "proposals": len(entries) - 1, "evaluated": len(evaluated),
            "kept": sum(e.status == "kept" for e in entries),
            "new_global_bests": sum(e.improved_global for e in entries if e.id != 0),
            "cost_usd": round(sum(e.usage.get("cost_usd", 0.0) or 0.0 for e in entries), 2),
            "best_over_time": trajectory(entries),
            "distinct_cores": len({d.core for d in descs}),
            "cores": Counter(d.core for d in descs).most_common(),
            "mean_pairwise_distance": round(sum(pair) / len(pair), 3) if pair else None,
            "best_descriptor": None,
        }
        bd = describe(store.read_candidate(best), vocab, problem, model=args.model, cache_path=cache) if best.id else None
        rows[run.name]["best_descriptor"] = str(bd) if bd else "seed"
    for name, r in rows.items():
        print(f"\n== {name} ==")
        for k, v in r.items():
            print(f"  {k:24s} {v}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"\nwritten {args.out / 'comparison.json'}")


if __name__ == "__main__":
    main()
