"""Report a scripts/bench.sh benchmark: per problem and arm, mean ± sd over seeds.

Per run: gain over the seed on the search objective, held-out gain (instances the search never saw),
agent cost, describe cost (descriptor arms only; describe_usage.jsonl), wall-clock, evaluated proposals
and proposals the descriptor gate dropped unevaluated. Per arm: the held-out difference to the
reference arm with its standard error (Welch), so a gap smaller than ~2 se is noise.

Usage: uv run python scripts/bench_report.py TAG [--ref initial] [--out artifacts/bench/TAG.md]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch_viz.load import load_run  # noqa: E402
from autoresearch_viz.metrics import summarize  # noqa: E402


def run_row(d: Path) -> dict:
    meta = json.loads((d / "bench.json").read_text())
    run = load_run(d)
    s = summarize(run)
    hold = json.loads((d / "holdout.json").read_text()) if (d / "holdout.json").exists() else {}
    hs, hb = hold.get("seed", {}).get("score"), hold.get("best", {}).get("score")
    usage = d / "describe_usage.jsonl"
    describe = sum(json.loads(l)["cost_usd"] for l in usage.read_text().splitlines()) if usage.exists() else 0.0
    ts = {e["type"]: e["t"] for e in run.events if e.get("type") in ("run_start", "run_end")}
    return {**meta,
            "gain": s.gain_vs_seed_pct,
            "held": 100 * (hs - hb) / hs if hs and hb is not None else math.nan,
            "agent_usd": s.cost_usd, "describe_usd": describe, "usd": s.cost_usd + describe,
            "minutes": (ts["run_end"] - ts["run_start"]) / 60 if len(ts) == 2 else math.nan,
            "evaluated": sum(1 for e in run.proposals if e.evals), "proposals": len(run.proposals),
            "gated": sum(1 for e in run.proposals if e.note.startswith("not selected by max-min"))}


def ms(xs: list[float], digits: int = 2) -> str:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return "n/a"
    return f"{st.mean(xs):.{digits}f} ± {st.stdev(xs):.{digits}f}" if len(xs) > 1 else f"{xs[0]:.{digits}f}"


def welch(a: list[float], b: list[float]) -> str:
    if len(a) < 2 or len(b) < 2:
        return "n/a"
    diff = st.mean(a) - st.mean(b)
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return f"{diff:+.2f} (se {se:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ref", default="initial", help="arm the held-out differences are taken against")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    dirs = sorted(d for d in (ROOT / "artifacts" / "runs").glob(f"{args.tag}-*") if (d / "bench.json").exists())
    if not dirs:
        sys.exit(f"no finished benchmark runs for tag {args.tag!r} (artifacts/runs/{args.tag}-*/bench.json)")
    rows = [run_row(d) for d in dirs]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["problem"], r["arm"])].append(r)

    out = [f"# Benchmark `{args.tag}`", "",
           f"Model {rows[0]['model']}, {rows[0]['agents']} agents x {rows[0]['generations']} generations per run. "
           "Gains are % below the seed's score (higher is better); held-out = instances the search never saw.", ""]
    for problem in sorted({p for p, _ in groups}):
        arms = sorted((a for p, a in groups if p == problem), key=lambda a: (a != args.ref, a))
        ref = [r["held"] for r in groups.get((problem, args.ref), [])]
        out += [f"## {problem}", "",
                f"| arm | commit | n | gain | held-out gain | held-out vs {args.ref} | cost $ (describe $) | minutes "
                "| evaluated / proposals | gate-dropped |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            g = groups[(problem, a)]
            col = lambda k: [r[k] for r in g]  # noqa: E731
            vs = "—" if a == args.ref else welch(col("held"), ref)
            out.append(f"| {a} | {g[0]['sha']} | {len(g)} | {ms(col('gain'))} | {ms(col('held'))} | {vs} "
                       f"| {ms(col('usd'))} ({st.mean(col('describe_usd')):.2f}) | {ms(col('minutes'), 1)} "
                       f"| {st.mean(col('evaluated')):.1f} / {st.mean(col('proposals')):.1f} "
                       f"| {st.mean(col('gated')):.1f} |")
        out.append("")
    out += ["## Runs", "", "| run | gain | held-out | cost $ | describe $ | minutes | evaluated | gate-dropped |",
            "|---|---|---|---|---|---|---|---|"]
    for d, r in zip(dirs, rows):
        out.append(f"| {d.name} | {r['gain']:.2f} | {r['held']:.2f} | {r['usd']:.2f} | {r['describe_usd']:.2f} "
                   f"| {r['minutes']:.1f} | {r['evaluated']}/{r['proposals']} | {r['gated']} |")
    text = "\n".join(out) + "\n"
    print(text)
    path = args.out or ROOT / "artifacts" / "bench" / f"{args.tag}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"written {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
