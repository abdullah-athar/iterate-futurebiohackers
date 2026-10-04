"""Command-line interface for agent mode.

init -> (status -> write candidate.py -> submit) x N -> report --holdout
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

from .ledger import RunStore
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
    cfg = LoopConfig(problem=args.problem, novelty_threshold=args.novelty_threshold, patience=args.patience,
                     modes=args.modes, exploit=args.exploit)
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


def cmd_try(args) -> None:
    """Evaluate a solver file on one split and print diagnostics; never touches the ledger."""
    from .guard import check_imports
    from .problem import get_problem
    from .prompts import format_diagnostics
    from .sandbox import get_evaluate

    store = RunStore(args.run)
    cfg = store.config() if store.exists else {}
    name = cfg.get("problem", "median_string")
    budget = args.budget_ms or cfg.get("time_budget_ms", LoopConfig.time_budget_ms)
    problem = get_problem(name)
    source = Path(args.file).read_text()
    violations = check_imports(source, problem.allowed_imports)
    if violations:
        sys.exit("disallowed in candidate: " + ", ".join(violations))
    res = get_evaluate()(name, problem, source, args.split, budget)
    print(format_diagnostics(res, label=f"try {args.file}: "))
    print(f"(budget {budget} ms CPU per instance; cpu_ms per instance: "
          + ", ".join(f"{i.name}={i.cpu_ms:g}" for i in res.instances) + ")")


def cmd_swarm(args) -> None:
    from . import swarm

    if sys.platform == "darwin":
        # keep the Mac awake: sleep pauses the monotonic clock behind the agents' turn deadline
        import subprocess
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    if args.eval == "modal":
        os.environ["AUTORESEARCH_EVAL"] = "modal"
        if not args.no_deploy:
            from .modal_eval import deploy
            deploy()
    store = RunStore(args.run)
    if not store.exists:
        cfg = LoopConfig(problem=args.problem, time_budget_ms=args.budget_ms, modes=args.modes, exploit=args.exploit,
                         descriptors=not args.no_descriptors, describe_model=args.describe_model,
                         **{f: getattr(args, f) for f in DIVERSITY_FLAGS if getattr(args, f, None) is not None})
        from .scheduler import validate_config
        errs = validate_config(cfg)
        if errs:
            sys.exit("invalid configuration: " + "; ".join(errs))
        run = ResearchRun.create(store, cfg)
        print(f"Initialised run at {store.root}\n" + ResearchRun.describe_entry(run.entries()[0]))
    run = ResearchRun(store)
    emit = swarm.Events(run, time.time())
    propose = swarm.claude_proposer(args.model, args.turn_s, args.eval, args.max_budget_usd, emit)
    budget_ms = run.config.time_budget_ms
    evaluate = (swarm.modal_evaluator(run.problem_name, budget_ms) if args.eval == "modal"
                else swarm.local_evaluator(run.problem_name, budget_ms))
    from .budget import RunBudget
    budget = RunBudget(store, max_usd=args.max_run_usd, max_calls=args.max_run_calls, max_tokens=args.max_run_tokens,
                       unknown_session_usd=args.max_budget_usd)
    swarm.run_swarm(run, args.agents, args.turn_s, args.budget_min * 60, propose, evaluate, emit,
                    seed=args.seed, max_generations=args.generations, eval_name=args.eval,
                    budget=budget if budget.capped else None, per_agent_usd_cap=args.max_budget_usd)
    from .report import render
    text = render(run)
    (store.root / "report.md").write_text(text)
    print(f"\n[report written to {store.root / 'report.md'}]")


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


def _ablation_args(s) -> None:
    """Controls for the simple-loop vs full-framework comparison (apply to a new run only)."""
    s.add_argument("--modes", nargs="+", choices=list(MODES),
                   help="restrict prompt modes, e.g. `--modes tune` for a plain incumbent-only loop (default: all)")
    s.add_argument("--exploit", type=float, default=LoopConfig.exploit,
                   help=f"probability the parent is the global best rather than a front member (default {LoopConfig.exploit}; "
                        "1.0 = no per-instance archive)")


DIVERSITY_FLAGS = (
    "descriptor_backend", "distance_metric", "distance_policy", "family_diagnostics", "diverse_parent_selection",
    "semantic_retry_before_codegen", "entropy_controller", "family_grace", "grace_min_evaluations",
    "grace_max_evaluations", "grace_patience", "grace_epsilon", "max_protected_families", "max_proposal_regenerations",
    "top_k", "allocation_window", "stagnation_generations", "explore_boost_fraction", "max_override_fraction",
    "archive_threshold", "batch_threshold", "drift_threshold_max", "bandit_counts_all_attempts", "tune_exact_only",
    "grace_max_attempts",
)


def _diversity_args(s) -> None:
    """Diversity extension (autoresearch/scheduler.py), new run only; everything off = the reference."""
    g = s.add_argument_group("diversity (all off by default; see README)")
    g.add_argument("--descriptor-backend", dest="descriptor_backend", choices=["canonical", "existing"])
    g.add_argument("--distance-metric", dest="distance_metric", choices=["weighted_jaccard", "minilm"])
    g.add_argument("--distance-policy", dest="distance_policy", choices=["off", "observe", "soft", "strict"])
    g.add_argument("--family-diagnostics", dest="family_diagnostics", action="store_true", default=None,
                   help="describe candidates for family statistics with every policy off (cost logged as diagnostic)")
    g.add_argument("--diverse-parents", dest="diverse_parent_selection", action="store_true", default=None)
    g.add_argument("--plan-first", dest="semantic_retry_before_codegen", action="store_true", default=None,
                   help="structured plan + distance check + reservation before each agent session (one extra call each)")
    g.add_argument("--entropy-controller", dest="entropy_controller", action="store_true", default=None)
    g.add_argument("--family-grace", dest="family_grace", action="store_true", default=None)
    g.add_argument("--grace-min", dest="grace_min_evaluations", type=int)
    g.add_argument("--grace-max", dest="grace_max_evaluations", type=int)
    g.add_argument("--grace-patience", dest="grace_patience", type=int)
    g.add_argument("--grace-epsilon", dest="grace_epsilon", type=float)
    g.add_argument("--max-protected", dest="max_protected_families", type=int)
    g.add_argument("--max-regenerations", dest="max_proposal_regenerations", type=int)
    g.add_argument("--top-k", dest="top_k", type=int)
    g.add_argument("--allocation-window", dest="allocation_window", type=int)
    g.add_argument("--stagnation-generations", dest="stagnation_generations", type=int)
    g.add_argument("--explore-boost", dest="explore_boost_fraction", type=float)
    g.add_argument("--max-override", dest="max_override_fraction", type=float)
    g.add_argument("--archive-threshold", dest="archive_threshold", type=float,
                   help="calibrated min distance of a new_family proposal to family representatives")
    g.add_argument("--batch-threshold", dest="batch_threshold", type=float,
                   help="calibrated min distance between proposals of one batch")
    g.add_argument("--drift-threshold-max", dest="drift_threshold_max", type=float)
    g.add_argument("--tune-exact-only", dest="tune_exact_only", action="store_true", default=None,
                   help="tune children (grace tries included) are refused only as exact copies, not as near-duplicates")
    g.add_argument("--grace-max-attempts", dest="grace_max_attempts", type=int)
    g.add_argument("--legacy-bandit-counting", dest="bandit_counts_all_attempts", action="store_false", default=None,
                   help="do not count duplicates/guard rejections/empty sessions as bandit tries (old behaviour)")


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
    _ablation_args(s)
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("status", help="show archive, diagnostics, suggested mode (agent mode)")
    s.add_argument("--mode", choices=list(MODES))
    s.add_argument("--seed", type=int, default=None, help="RNG seed for parent selection")
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

    s = sub.add_parser("try", help="evaluate a solver file on one split without recording it")
    s.add_argument("--file", required=True)
    s.add_argument("--split", default="validate", choices=["screen", "validate"])
    s.add_argument("--budget-ms", type=int, help="CPU budget per instance (default: the run's, else 1000)")
    s.set_defaults(fn=cmd_try)

    s = sub.add_parser("swarm", help="run N local Claude Code agents per generation; evaluate on Modal")
    s.add_argument("--agents", type=int, default=32)
    s.add_argument("--budget-min", type=float, default=20, help="wall-clock budget for the whole run")
    s.add_argument("--turn-s", type=int, default=180, help="wall-clock limit per agent session")
    s.add_argument("--generations", type=int, help="stop after this many generations")
    s.add_argument("--model", default="sonnet", help="Claude Code --model (alias or full id)")
    s.add_argument("--eval", choices=["modal", "local"], default="modal")
    s.add_argument("--no-deploy", action="store_true", help="skip `modal deploy` of the eval app")
    s.add_argument("--max-budget-usd", type=float, help="per-session spend cap passed to claude")
    s.add_argument("--problem", default="median_string", help="problem for a new run")
    _ablation_args(s)
    _diversity_args(s)
    s.add_argument("--budget-ms", type=int, default=1000, help="CPU ms per instance for a new run")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-descriptors", action="store_true",
                   help="new run without the exploration-exploitation layer (descriptor contract for tune, "
                        "max-min novelty selection for new_family, research landscape)")
    s.add_argument("--max-run-usd", type=float, help="run-level spend cap (agents + describe/plan calls); "
                   "a generation starts only if its estimated cost still fits")
    s.add_argument("--max-run-calls", type=int, help="run-level cap on provider calls (sessions + auxiliary)")
    s.add_argument("--max-run-tokens", type=int, help="run-level token cap (stops before the next generation)")
    s.add_argument("--describe-model", default="sonnet",
                   help="model that writes the descriptors, independent of --model so the descriptors (and the "
                        "distance thresholds tuned on them) stay comparable across runs")
    s.set_defaults(fn=cmd_swarm)

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
