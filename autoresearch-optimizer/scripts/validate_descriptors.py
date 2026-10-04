"""Validate the descriptor system before using it as a gate.

1. Embedding sanity: 15 hand-labelled term pairs (synonym / related / unrelated), raw vs centred cosine.
2. Self-distance: describe the 10 reference solvers in scripts/descriptor_fixtures twice.
   Pass 1 builds the vocabulary sequentially from the seed; pass 2 re-describes every solver
   (cache bypassed) against the frozen pass-1 vocabulary, as a K-branch drift check would.
   Same-solver distances must sit clearly below between-solver distances; the duplicate threshold
   is placed just above the largest self-distance.

Usage: uv run python scripts/validate_descriptors.py [--model sonnet] [--out artifacts/descriptors]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.descriptors import Embedder, Vocabulary, describe, distance  # noqa: E402

PROBLEM = ("Median string (Steiner string): given a set of strings over a small alphabet, output one string "
           "minimising the total edit distance (Levenshtein or Hamming) to all of them, under a CPU-time budget.")

PAIRS = {
    "synonym": [("simulated annealing", "annealing-based search"), ("genetic algorithm", "evolutionary algorithm"),
                ("hill climbing", "greedy local improvement"), ("set median initialisation", "medoid starting point"),
                ("random restarts", "multi-start")],
    "related": [("simulated annealing", "tabu search"), ("genetic algorithm", "particle swarm optimisation"),
                ("beam search", "branch and bound"), ("substitution moves", "insertion moves"),
                ("geometric cooling", "linear cooling")],
    "unrelated": [("simulated annealing", "column-wise majority vote"), ("tabu list", "dynamic programming"),
                  ("geometric cooling", "one-point crossover"), ("beam search", "stagnation-triggered restart"),
                  ("set median initialisation", "tournament selection")],
}


def embedding_sanity(emb: Embedder) -> dict:
    print("\n== Embedding sanity (cosine: raw -> centred) ==")
    out = {}
    for label, pairs in PAIRS.items():
        rows = []
        for a, b in pairs:
            raw = float((emb.embed([a], centred=False) @ emb.embed([b], centred=False).T)[0, 0])
            cen = float((emb.embed([a]) @ emb.embed([b]).T)[0, 0])
            rows.append({"a": a, "b": b, "raw": round(raw, 3), "centred": round(cen, 3)})
            print(f"  {label:9s} {a:28s} ~ {b:30s} {raw:6.3f} -> {cen:6.3f}")
        out[label] = rows
    for key in ("raw", "centred"):
        means = {lab: np.mean([r[key] for r in out[lab]]) for lab in PAIRS}
        gap = min(r[key] for r in out["synonym"]) - max(r[key] for r in out["unrelated"])
        print(f"  {key:8s} means: " + "  ".join(f"{k}={v:.3f}" for k, v in means.items())
              + f"  | min(synonym) - max(unrelated) = {gap:+.3f}")
        out[f"{key}_means"] = {k: round(float(v), 3) for k, v in means.items()}
        out[f"{key}_syn_unrel_gap"] = round(float(gap), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "descriptors")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    emb = Embedder()
    report = {"model": args.model, "embedding": embedding_sanity(emb)}

    fixtures = sorted((ROOT / "scripts" / "descriptor_fixtures").glob("*.py"))
    names = [f.stem for f in fixtures]
    sources = [f.read_text() for f in fixtures]

    print(f"\n== Pass 1: describing {len(fixtures)} solvers sequentially (vocabulary grows) ==")
    vocab = Vocabulary()
    first, sizes = [], [len(vocab)]
    for name, src in zip(names, sources):
        d = describe(src, vocab, PROBLEM, model=args.model, use_cache=False)
        new = vocab.add(d)
        first.append(d)
        sizes.append(len(vocab))
        print(f"  {name:24s} {d}" + (f"   new: {new}" if new else ""))

    print("\n== Pass 2: re-describing in parallel against the frozen pass-1 vocabulary ==")
    with ThreadPoolExecutor(len(sources)) as pool:
        second = list(pool.map(lambda s: describe(s, vocab, PROBLEM, model=args.model, use_cache=False), sources))
    for name, d in zip(names, second):
        print(f"  {name:24s} {d}")

    n = len(names)
    D = np.array([[distance(first[i], second[j], emb) for j in range(n)] for i in range(n)])
    self_d = np.diag(D)
    # between-solver distances: cross-pass off-diagonal plus within-pass-1 pairs
    between = np.concatenate([D[~np.eye(n, dtype=bool)],
                              [distance(first[i], first[j], emb) for i in range(n) for j in range(i + 1, n)]])

    print("\n== Self-distance test ==")
    for name, s, row in zip(names, self_d, D):
        nearest_other = min(v for j, v in enumerate(row) if names[j] != name)
        print(f"  {name:24s} self={s:.3f}  nearest other={nearest_other:.3f}")
    margin = between.min() - self_d.max()
    overlap = int(np.sum(between <= self_d.max()))
    print(f"  self    : min={self_d.min():.3f} mean={self_d.mean():.3f} max={self_d.max():.3f}")
    print(f"  between : min={between.min():.3f} mean={between.mean():.3f} max={between.max():.3f}")
    print(f"  margin (min between - max self) = {margin:+.3f}; between-pairs at or below max self: {overlap}/{len(between)}")
    threshold = float(self_d.max() + 0.25 * margin) if margin > 0 else None
    if threshold is not None:
        print(f"  PASS: duplicate threshold = {threshold:.3f} (distance; same-algorithm pairs fall below it)")
    else:
        print("  FAIL: separation is poor; descriptors should be display-only and behaviour vectors carry the gate")
    print(f"  vocabulary size: {sizes[0]} seed -> {sizes[-1]} after pass 1 (per solver: {sizes})")

    report |= {
        "solvers": names, "pass1": [d.to_dict() for d in first], "pass2": [d.to_dict() for d in second],
        "distance_matrix": np.round(D, 4).tolist(), "self": np.round(self_d, 4).tolist(),
        "between_min": round(float(between.min()), 4), "between_mean": round(float(between.mean()), 4),
        "self_max": round(float(self_d.max()), 4), "margin": round(float(margin), 4), "overlap": overlap,
        "duplicate_threshold": threshold, "vocab_sizes": sizes,
    }
    path = args.out / f"validation-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1))
    vocab.save(args.out / "validation-vocab.json")
    print(f"\nreport: {path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
