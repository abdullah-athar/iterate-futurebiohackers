"""CLI: `python -m autoresearch_viz render RUN... -o out.html` / `python -m autoresearch_viz demo`."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .html import render
from .load import load_runs
from .metrics import summarize


def _render(specs: list[str], out: Path, title: str | None, open_browser: bool) -> int:
    runs = load_runs(specs)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs, title) if title else render(runs))
    for s in (summarize(r) for r in runs):
        print(
            f"{s.label:45s} best={s.best_objective:>8.0f}  Δbaseline={s.gain_vs_baseline_pct:+6.1f}%  "
            f"evals={s.evaluated:3d}  dupes={s.counts.get('rejected_duplicate', 0):2d}  tokens={s.tokens:>8,d}"
        )
    print(f"wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autoresearch_viz", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render one HTML dashboard comparing the given run directories")
    r.add_argument("runs", nargs="+", help="run dir (contains ledger.jsonl), a parent dir of runs, or label=path")
    r.add_argument("-o", "--out", type=Path, default=Path("artifacts/viz/dashboard.html"))
    r.add_argument("--title")
    r.add_argument("--open", action="store_true", help="open the result in a browser")
    d = sub.add_parser("demo", help="generate synthetic flavour runs and render them")
    d.add_argument("--out-dir", type=Path, default=Path("artifacts/runs_demo"))
    d.add_argument("-o", "--out", type=Path, default=Path("artifacts/viz/demo_dashboard.html"))
    d.add_argument("--steps", type=int, default=40)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--open", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "demo":
        from .demo import write_demo_runs

        paths = write_demo_runs(a.out_dir, steps=a.steps, seed=a.seed)
        return _render([str(x) for x in paths], a.out, "Autoresearch flavour comparison (synthetic demo)", a.open)
    return _render(a.runs, a.out, a.title, a.open)


if __name__ == "__main__":
    sys.exit(main())
