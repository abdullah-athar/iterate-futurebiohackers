"""Command-line interface.

API mode:    init -> run --llm anthropic --steps 20 -> report --holdout
Agent mode:  init -> (status -> write candidate.py -> submit) x N -> report --holdout
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .ledger import RunStore
from .llm import make_llm
from .loop import LoopConfig, ResearchRun
from .prompts import MODES

DEFAULT_RUN = "artifacts/runs/default"


def _run(args) -> ResearchRun:
    store = RunStore(args.run)
    if not store.exists:
        sys.exit(f"No run at {args.run}. Create one with: python -m autoresearch --run {args.run} init")
    return ResearchRun(store)


def cmd_init(args) -> None:
    store = RunStore(args.run)
    cfg = LoopConfig(problem=args.problem, novelty_threshold=args.novelty_threshold, patience=args.patience)
    seed_src = Path(args.seed).read_text() if args.seed else None
    try:
        run = ResearchRun.create(store, cfg, seed_source=seed_src)
    except FileExistsError as e:
        sys.exit(f"{e}. Use `status` to continue it, or choose another --run directory.")
    e = run.entries()[0]
    print(f"Initialised run at {store.root} (problem={args.problem})")
    print(ResearchRun.describe_entry(e))


def cmd_status(args) -> None:
    run = _run(args)
    ctx = run.context(mode=args.mode, rng=random.Random(args.seed))
    if args.prompt:
        system, user = run.prompt(ctx)
        print(system)
        print(user)
        return
    archive = run.archive()
    print(f"# Run {run.store.root} — problem {run.problem_name}")
    print(f"Global best: #{archive.global_best.id} objective={archive.global_best.objective:g} "
          f"(source: {run.store.root / archive.global_best.source_path})")
    print(f"Front: {[e.id for e in archive.front]}; complementary: "
          f"{ {e.id: won for e, won in archive.complementary()} }")
    print(f"\nSuggested mode: {ctx.mode} — {MODES[ctx.mode]}")
    if ctx.extra.get("plateau"):
        print(f"PLATEAU: {ctx.extra['plateau']} proposals without global improvement.")
    print(f"Parents: {[p.id for p in ctx.parents]} -> " + ", ".join(str(run.store.root / p.source_path) for p in ctx.parents))
    if ctx.extra.get("falsified"):
        print("\nFalsified hypotheses (do not re-propose):")
        for i, v, h in ctx.extra["falsified"]:
            print(f"  #{i} [{v}] {h}")
    print("\n## Diagnostics\n" + ctx.diagnostics)
    print("\n## Ledger\n" + ctx.digest)
    print(f"\nNext: write a solver file, then `python -m autoresearch --run {run.store.root} submit --file <file> "
          f"--hypothesis \"...\" --mode {ctx.mode} --parent {' '.join(str(p.id) for p in ctx.parents)}`")


def cmd_submit(args) -> None:
    run = _run(args)
    source = Path(args.file).read_text()
    parents = args.parent if args.parent is not None else [run.archive().global_best.id]
    e = run.submit(source, args.hypothesis, args.mode, parents, proposer=args.proposer,
                   prompt_tokens=args.prompt_tokens, completion_tokens=args.completion_tokens)
    print(ResearchRun.describe_entry(e))
    from .problem import EvalResult
    from .prompts import format_diagnostics
    for d in e.evals.values():
        print(format_diagnostics(EvalResult.from_dict(d), run.archive(), label=f"#{e.id} "))


def cmd_run(args) -> None:
    run = _run(args)
    llm = make_llm(args.llm)
    print(f"Running {args.steps} step(s) with {llm.name} on {run.store.root}")
    run.run(llm, args.steps, seed=args.seed)
    print("\n" + "=" * 80)
    from .report import render
    print(render(run))


def cmd_report(args) -> None:
    run = _run(args)
    from .report import render
    text = render(run, holdout=args.holdout, holdout_front=args.holdout_front)
    out = Path(args.output) if args.output else run.store.root / "report.md"
    out.write_text(text)
    print(text)
    print(f"[report written to {out}]")


def cmd_best(args) -> None:
    run = _run(args)
    best = run.archive().global_best
    src = run.store.read_candidate(best)
    if args.output:
        Path(args.output).write_text(src)
        print(f"Wrote best candidate #{best.id} (objective {best.objective:g}) to {args.output}")
    else:
        print(src)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="autoresearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=DEFAULT_RUN, help=f"run directory (default {DEFAULT_RUN})")
    # accept --run after the subcommand too, e.g. `autoresearch status --run artifacts/runs/x`
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))

    s = sub.add_parser("init", help="create a run and evaluate the seed solver")
    s.add_argument("--problem", default="median_string")
    s.add_argument("--seed", help="path to a custom seed solver file")
    s.add_argument("--novelty-threshold", type=float, default=0.95)
    s.add_argument("--patience", type=int, default=4)
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("status", help="show archive, diagnostics, suggested mode (agent mode)")
    s.add_argument("--mode", choices=list(MODES))
    s.add_argument("--seed", type=int, default=None, help="RNG seed for parent selection")
    s.add_argument("--prompt", action="store_true", help="print the exact proposer prompt instead")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("submit", help="novelty-gate, evaluate and record a candidate (agent mode)")
    s.add_argument("--file", required=True)
    s.add_argument("--hypothesis", required=True)
    s.add_argument("--mode", default="tune", choices=list(MODES))
    s.add_argument("--parent", type=int, nargs="*", default=None)
    s.add_argument("--proposer", default="agent")
    s.add_argument("--prompt-tokens", type=int, default=0)
    s.add_argument("--completion-tokens", type=int, default=0)
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("run", help="API mode: let an LLM drive N steps")
    s.add_argument("--llm", default="anthropic", help="anthropic|gemini|openai|mock[:model]")
    s.add_argument("--steps", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="render report.md (optionally with held-out evaluation)")
    s.add_argument("--holdout", action="store_true")
    s.add_argument("--holdout-front", action="store_true", help="also evaluate every front member on holdout")
    s.add_argument("--output")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("best", help="print or export the best candidate's source")
    s.add_argument("--output")
    s.set_defaults(fn=cmd_best)

    args = p.parse_args(argv)
    args.fn(args)
