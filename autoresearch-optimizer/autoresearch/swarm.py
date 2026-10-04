"""Parallel research: N coding agents per generation propose solvers, evaluation fans out
(Modal or local), and this process is the only writer of the ledger and events.

Per generation:  allocate modes/parents (UCB bandit) -> N agents write candidate.py +
hypothesis.txt in their own workspace -> guard + novelty gate (in arrival order) ->
evaluate the survivors in parallel -> record in order -> next generation, until the wall-clock
budget cannot fit another generation. Then the best and the seed are scored on holdout.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import run_spend
from .loop import ResearchRun, evaluate_candidate
from .prompts import MODES, build_user_prompt

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Assignment:
    worker: int
    mode: str
    parent_ids: list[int]
    direction: str = ""
    notes: str = ""  # extra STATUS.md section (exploration-exploitation layer: descriptor contract, research landscape)
    # diversity scheduler decisions copied into the entry's usage: ucb_mode, override (reason), target_family,
    # grace_family, focus, parent_reason ... (empty when the scheduler is off)
    meta: dict = field(default_factory=dict)


def allocate(run: ResearchRun, n: int, rng: random.Random) -> list[Assignment]:
    """Spread n agents over the eligible modes: one each first, the rest in proportion to UCB score.

    An untried mode is weighted like the best tried one and capped at an even share, so a mode with
    no evidence yet cannot take most of a generation."""
    entries = run.entries()
    archive = run.archive(entries)
    scores = run.mode_scores(entries, archive)
    finite = [v for v in scores.values() if v != float("inf")]
    weights = {m: (max(finite, default=1.0) if v == float("inf") else v) for m, v in scores.items()}
    cap = {m: -(-n // len(scores)) if v == float("inf") and finite else n for m, v in scores.items()}
    counts = Counter({m: 1 for m in list(scores)[:n]})
    while sum(counts.values()) < n:
        open_modes = [m for m in scores if counts[m] < cap[m]] or list(scores)
        # largest remaining deficit against the proportional target gets the next agent
        total = sum(weights[m] for m in open_modes)
        m = max(open_modes, key=lambda m: n * weights[m] / total - counts[m])
        counts[m] += 1
    modes = [m for m in MODES if m in counts for _ in range(counts[m])]
    directions = list(getattr(run.problem, "directions", ()) or [""])
    rng.shuffle(directions)
    return [Assignment(i, m, [p.id for p in run.pick_parents(m, archive, rng)], directions[i % len(directions)])
            for i, m in enumerate(modes)]


AGENT_MD = """# Autoresearch worker {worker} — generation {gen}, mode `{mode}`

You are one of {n} agents proposing solvers in parallel this generation
({mix}). Your job: ONE hypothesis, ONE complete solver file. A harness evaluates it after you stop.

Your research direction: **{direction}**
Other agents cover the other directions, so stay on yours and make it work within your mode.
Ideas already in the ledger (see STATUS.md) count as taken: build on them or go elsewhere.

Time limit: {turn_s} s of wall clock; you are stopped at the deadline. Have a complete
`candidate.py` and `hypothesis.txt` written by ~{half} s, then refine only if time remains.

1. Read STATUS.md: the problem, your mode, the parent solver(s) (also saved here as parent_<id>.py;
   candidate.py starts as a copy of the first parent), per-instance diagnostics, the research
   ledger and the falsified hypotheses.
2. Write one line to hypothesis.txt: what you change, why, and on which instances you expect a lower score.
3. Edit candidate.py: a complete file defining `solve(instance) -> str`; stdlib + median_string.metrics
   only; deterministic; each solve() gets instance.time_budget_ms of CPU (over 1.25x = invalid instance).
4. Self-test with `./try` (validate split on remote CPUs, ~15 s) or `./try candidate.py screen`.
   It prints per-instance score vs baseline/best_known, cpu_ms and errors. Fix crashes and overruns.
5. Stop when candidate.py and hypothesis.txt are final. Only candidate.py is submitted.
   Never revert candidate.py to the parent: if your self-test shows no gain, leave your attempted
   change in place. The harness then records the hypothesis as falsified, so other agents don't repeat it.
"""

PROMPT = "Read AGENT.md and STATUS.md in the current directory and follow them. Work only in this directory."


def write_workspace(run: ResearchRun, a: Assignment, gen: int, n: int, mix: str, turn_s: int, ws: Path,
                    eval_backend: str) -> None:
    ctx = run.context(mode=a.mode, parent_ids=a.parent_ids)
    (ws / "STATUS.md").write_text(build_user_prompt(run.problem.describe(), ctx, run.config.descriptors)
                                  + (f"\n\n{a.notes}" if a.notes else ""))
    (ws / "AGENT.md").write_text(AGENT_MD.format(worker=a.worker, gen=gen, mode=a.mode, n=n, mix=mix,
                                                 turn_s=turn_s, half=turn_s // 2,
                                                 direction=a.direction or "your choice"))
    for p, src in zip(ctx.parents, ctx.parent_sources):
        (ws / f"parent_{p.id}.py").write_text(src)
    (ws / "candidate.py").write_text(ctx.parent_sources[0])
    try_sh = ws / "try"
    try_sh.write_text(f"#!/bin/sh\nexport PYTHONPATH={ROOT} AUTORESEARCH_EVAL={eval_backend}\n"
                      f'exec {sys.executable} -m autoresearch --run {run.store.root.resolve()} try '
                      f'--file "${{1:-candidate.py}}" --split "${{2:-validate}}"\n')
    try_sh.chmod(0o755)


def run_claude(ws: Path, turn_s: int, model: str, max_budget_usd: float | None = None) -> dict:
    """Run one headless Claude Code session in `ws`; kill its process group at the deadline."""
    cmd = ["claude", "-p", PROMPT, "--output-format", "json", "--model", model,
           "--permission-mode", "acceptEdits", "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep"]
    if max_budget_usd:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    env = {k: v for k, v in os.environ.items() if not k.endswith("_SEED") or not k.startswith("AUTORESEARCH_")}
    t0 = time.time()
    with open(ws / "agent.json", "w") as out, open(ws / "agent.err", "w") as err:
        proc = subprocess.Popen(cmd, cwd=ws, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env,
                                start_new_session=True)
        try:
            proc.wait(timeout=turn_s)
            outcome = "ok" if proc.returncode == 0 else "agent_error"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            outcome = "timeout"
    # cost_usd stays None (unknown) unless the session reported it: a killed session may still have spent
    usage = {"model": model, "seconds": round(time.time() - t0, 1), "outcome": outcome, "cost_usd": None}
    try:
        res = json.loads((ws / "agent.json").read_text())
        u = res.get("usage", {})
        usage.update(cost_usd=res.get("total_cost_usd"), num_turns=res.get("num_turns"),
                     prompt_tokens=u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                     + u.get("cache_creation_input_tokens", 0),
                     completion_tokens=u.get("output_tokens", 0))
        if res.get("is_error"):
            usage["outcome"] = res.get("subtype") or "agent_error"
    except (json.JSONDecodeError, OSError):
        pass
    return usage


def claude_proposer(model: str, turn_s: int, eval_backend: str, max_budget_usd: float | None, emit):
    """propose_many backed by local headless Claude Code sessions (one thread each)."""

    def propose_many(run: ResearchRun, gen: int, assignments: list[Assignment]) -> list[dict]:
        mix = ", ".join(f"{c} on {m}" for m, c in Counter(a.mode for a in assignments).items())
        agents_dir = run.store.root / "agents"
        agents_dir.mkdir(exist_ok=True)

        def one(a: Assignment) -> dict:
            ws = Path(tempfile.mkdtemp(prefix=f"autoresearch-g{gen:02d}-w{a.worker:02d}-"))
            write_workspace(run, a, gen, len(assignments), mix, turn_s, ws, eval_backend)
            parent_src = (ws / "candidate.py").read_text()
            usage = run_claude(ws, turn_s, model, max_budget_usd)
            source = (ws / "candidate.py").read_text()
            hyp_path = ws / "hypothesis.txt"
            hypothesis = hyp_path.read_text().strip().splitlines()[0] if hyp_path.exists() and hyp_path.read_text().strip() else ""
            if source == parent_src or not hypothesis:
                source, usage["outcome"] = "", usage["outcome"] if usage["outcome"] != "ok" else "no_file"
            tag = f"g{gen:02d}-w{a.worker:02d}"
            for name in ("agent.json", "agent.err"):
                if (ws / name).exists():
                    shutil.copy(ws / name, agents_dir / f"{tag}-{name}")
            shutil.rmtree(ws, ignore_errors=True)
            return {"assignment": a, "source": source, "hypothesis": hypothesis, "usage": usage}

        results = []
        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            futures = [pool.submit(one, a) for a in assignments]
            for f in as_completed(futures):
                r = f.result()
                u = r["usage"]
                emit("agent_done", gen=gen, worker=r["assignment"].worker, mode=r["assignment"].mode,
                     outcome=u["outcome"], seconds=u["seconds"], cost_usd=u.get("cost_usd", 0.0),
                     hypothesis=r["hypothesis"][:160])
                results.append(r)
        return results

    return propose_many


def modal_evaluator(problem_name: str, budget_ms: int):
    from .modal_eval import remote_cascade_map

    def evaluate_many(sources: list[str]) -> list:
        out: list = [None] * len(sources)
        for i, r in remote_cascade_map(problem_name, sources, budget_ms):
            out[i] = r
        return out

    return evaluate_many


def local_evaluator(problem_name: str, budget_ms: int, workers: int = 4):
    def evaluate_many(sources: list[str]) -> list:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda s: evaluate_candidate(problem_name, s, budget_ms), sources))

    return evaluate_many


class Events:
    """Append-only events.jsonl next to the ledger, plus one status line per event on stdout."""

    def __init__(self, run: ResearchRun, t0: float, log=print) -> None:
        self.path = run.store.root / "events.jsonl"
        self.t0, self.log = t0, log

    def __call__(self, kind: str, **data) -> None:
        rec = {"t": time.time(), "type": kind, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.log(f"[{int(rec['t'] - self.t0) // 60:02d}:{int(rec['t'] - self.t0) % 60:02d}] {format_event(rec)}")


def format_event(r: dict) -> str:
    k = r["type"]
    if k == "run_start":
        return f"run start: {r['agents']} agents/generation, budget {r['budget_s'] // 60} min, turn {r['turn_s']} s, eval={r['eval']}"
    if k == "gen_start":
        return f"gen {r['gen']} start: {r['mix']}; best #{r['best_id']}={r['best']:g}"
    if k == "agent_done":
        return f"gen {r['gen']} w{r['worker']:02d} [{r['mode']}] {r['outcome']} {r['seconds']:.0f}s ${r['cost_usd'] or 0:.2f} — {r['hypothesis'][:90]}"
    if k == "eval_start":
        return f"gen {r['gen']} evaluating {r['n']} candidate(s) ({r['dup']} duplicate, {r['guard']} guard, {r['empty']} empty skipped)"
    if k == "entry":
        star = " ** NEW BEST **" if r.get("new_best") else ""
        obj = "-" if r["objective"] is None else r["objective"]
        return f"gen {r['gen']} #{r['id']} w{r['worker']:02d} [{r['mode']}] {r['status']}/{r['verdict']} {obj}{star}"
    if k == "gen_end":
        return (f"gen {r['gen']} end: best #{r['best_id']}={r['best']:g}; {r['evaluated']} evaluated, {r['kept']} kept; "
                f"{r['seconds']:.0f}s; cost so far ${r['cost_usd']:.2f}")
    if k == "descriptor_gate":
        return (f"gen {r['gen']} descriptors: tune {r['tune_kept']} kept, {r['drifted']} drifted to new_family; "
                f"new_family pool {r['pool']} -> {r['selected']}/{r['slots']} selected (min dist {r['min_dists']})")
    if k == "schedule":
        ov = "; ".join(f"w{o['worker']} {o['from']}->{o['to']} ({o['reason']})" for o in r["overrides"] if "worker" in o)
        ctl = r.get("controller") or {}
        return (f"gen {r['gen']} schedule: ucb {dict(Counter(r['ucb_modes']))}; overrides: {ov or 'none'}"
                + (f"; controller {'ON' if ctl.get('active') else 'off'} ({ctl.get('why')})" if ctl else ""))
    if k == "distance_gate":
        return (f"gen {r['gen']} distance [{r['policy']}, {r['metric']}]: {r['decisions']}; described {r['described']}, "
                f"undescribed {r['undescribed']}")
    if k == "families":
        top, rec = r["top"], r["recent"]
        return (f"gen {r['gen']} families: {r['families']} known, new {r['new_families'] or '-'}; top-{top['n']} "
                f"H={top['H']:.2f} ({top['families']} fam); recent H={rec['H']:.2f} ({rec['families']} fam)")
    if k == "mode_drift":
        return f"gen {r['gen']} w{r['worker']:02d} [{r['mode']}] drifted {r['from_family']} -> {r['to_family']} (d={r['distance']})"
    if k == "plan":
        return f"gen {r['gen']} w{r['worker']:02d} plan {r['status']} after {r['attempts']} attempt(s): {r.get('summary', '')[:80]}"
    if k == "plan_divergence":
        return f"gen {r['gen']} w{r['worker']:02d} code differs from its plan (d={r['distance']}): plan-only {r['only_plan']}, code-only {r['only_code']}"
    if k == "budget_stop":
        return f"budget stop before gen {r['gen']}: {r['reason']}"
    if k == "holdout":
        return f"holdout: seed {r['seed']:g} -> best #{r['best_id']} {r['best']:g} (baseline {r['baseline']:g})"
    if k == "run_end":
        return f"run end: {r['generations']} generation(s), {r['proposals']} proposals, best #{r['best_id']}={r['best']:g}"
    return json.dumps(r)


def run_swarm(run: ResearchRun, agents: int, turn_s: int, budget_s: float, propose_many, evaluate_many, emit,
              seed: int = 0, max_generations: int | None = None, eval_name: str = "", budget=None,
              per_agent_usd_cap: float | None = None) -> None:
    """`budget` (RunBudget) adds run-level caps on cost/calls/tokens; a generation starts only if its
    estimated spend (last generation's mean session cost, else `per_agent_usd_cap`) still fits."""
    rng = random.Random(seed)
    t0 = time.time()
    entries = run.entries()
    gen = max((e.generation or 0 for e in entries), default=0) + 1
    eval_s, generations, proposals = 60.0, 0, 0
    layer = None
    if run.config.descriptors:
        from .exploration_exploitation import ExplorationExploitation
        layer = ExplorationExploitation(run)
        layer.describer.budget = budget
    div = None
    from .scheduler import DiversityLayer, needs_families
    if needs_families(run.config):
        div = DiversityLayer(run, budget)
    describe_s = 60.0 if (layer or div) else 0.0
    emit("run_start", agents=agents, budget_s=int(budget_s), turn_s=turn_s,
         eval=eval_name + (", descriptors" if layer else "") + (", diversity" if div else ""),
         caps={} if budget is None else {"usd": budget.max_usd, "calls": budget.max_calls, "tokens": budget.max_tokens})
    stop = ""
    while (time.time() - t0 + turn_s + describe_s + eval_s + 30 <= budget_s
           and (max_generations is None or generations < max_generations)):
        if budget is not None:
            ok, why = budget.reserve_generation(agents, _per_agent_estimate(run, per_agent_usd_cap))
            if not ok:
                stop = why
                emit("budget_stop", gen=gen, reason=why)
                break
        tg = time.time()
        assignments = allocate(run, agents, rng)
        if layer:
            assignments = layer.prepare(assignments, rng)
        if div:
            assignments = div.plan(assignments, gen, rng, emit)
        best = run.archive().global_best
        emit("gen_start", gen=gen, mix=", ".join(f"{m} {c}" for m, c in Counter(a.mode for a in assignments).items()),
             best_id=best.id, best=best.objective, assignments=[{**asdict(a), "notes": ""} for a in assignments])
        results = propose_many(run, gen, assignments)
        results.sort(key=lambda r: r["assignment"].worker)
        next_id = run.store.next_id()
        arrivals: list[tuple[int, str]] = []
        for i, r in enumerate(results):  # every result becomes the entry next_id + i (empty sessions too)
            if not r["source"]:
                continue
            # with descriptors, tune children only lose to exact copies (refinement is the point)
            exact = layer is not None and r["assignment"].mode == "tune"
            r["pre"] = run.precheck(r["source"], extra_prior=arrivals, threshold=1.0 if exact else None)
            arrivals.append((next_id + i, r["source"]))
        if layer or div:
            td = time.time()
            if layer:
                layer.gate(results, gen, emit)
            if div:
                div.after_precheck(results, gen, emit)
            describe_s = max(30.0, time.time() - td)
        emit("eval_start", gen=gen, n=len(passed(results)), dup=sum(1 for r in results if r.get("pre") and r["pre"][1].is_duplicate),
             guard=sum(1 for r in results if r.get("pre") and r["pre"][0]), empty=sum(1 for r in results if not r["source"]))
        eval_s, n = evaluate_and_record(run, gen, results, evaluate_many, emit)
        if budget is not None:
            budget.release_generation()
        proposals += n
        if div:
            div.after_generation(gen, emit)
        emit_gen_end(run, gen, tg, emit, **({"vocab": layer.describer.end_generation()} if layer else {}))
        gen += 1
        generations += 1
    write_holdout(run, emit)
    best = run.archive().global_best
    emit("run_end", generations=generations, proposals=proposals, best_id=best.id, best=best.objective,
         **({"stopped_by": stop} if stop else {}))


def _per_agent_estimate(run: ResearchRun, cap: float | None) -> float | None:
    """Mean reported cost of the last generation's sessions; before any, the per-session cap (or None)."""
    entries = run.entries()
    last = max((e.generation or 0 for e in entries), default=0)
    costs = [e.usage["cost_usd"] for e in entries
             if e.generation == last and last and e.usage.get("cost_usd") is not None]
    if costs:
        return sum(costs) / len(costs)
    return cap


def passed(results: list[dict]) -> list[dict]:
    """Results with a source that passed the import guard and the novelty gate."""
    return [r for r in results if r["source"] and not r["pre"][0] and not r["pre"][1].is_duplicate]


def evaluate_and_record(run: ResearchRun, gen: int, results: list[dict], evaluate_many, emit) -> tuple[float, int]:
    """Evaluate the results that passed the gates, then record every result with a source in list order.
    Returns (evaluation wall-clock seconds, number recorded)."""
    todo = passed(results)
    te = time.time()
    for r, ev in zip(todo, evaluate_many([r["source"] for r in todo]) if todo else []):
        r["evals"] = ev if isinstance(ev, dict) else {}
        if not isinstance(ev, dict):
            r["usage"]["eval_error"] = repr(ev)[:300]
    eval_s = max(30.0, time.time() - te)
    n = 0
    for r in results:
        a, u = r["assignment"], r["usage"]
        for k, v in a.meta.items():
            u.setdefault(k, v)
        common = dict(generation=gen, usage=u, elapsed=u.get("seconds", 0.0),
                      prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0))
        if not r["source"]:  # the session cost money and was a try of its mode: keep it in the ledger
            e = run.record_no_output(a.mode, a.parent_ids, f"claude-code:{u.get('model', '?')}",
                                     r.get("hypothesis", ""), **common)
        else:
            e = run.record(r["source"], r["hypothesis"], a.mode, a.parent_ids, f"claude-code:{u.get('model', '?')}",
                           r.get("evals", {}), r["pre"], note=r.get("note", ""), **common)
        emit("entry", gen=gen, id=e.id, worker=a.worker, mode=a.mode, status=e.status, verdict=e.verdict,
             objective=None if not e.scored else e.objective, new_best=e.improved_global)
        n += 1
    return eval_s, n


def emit_gen_end(run: ResearchRun, gen: int, tg: float, emit, **extra) -> None:
    entries = run.entries()
    best = run.archive(entries).global_best
    gen_entries = [e for e in entries if e.generation == gen]
    spend = run_spend(run.store)
    # cost_usd = agent sessions (as before); total_usd adds describe/plan calls; unknown = unreported costs
    emit("gen_end", gen=gen, best_id=best.id, best=best.objective, seconds=round(time.time() - tg, 1),
         evaluated=sum(1 for e in gen_entries if e.evals), kept=sum(1 for e in gen_entries if e.status == "kept"),
         cost_usd=round(spend.by_kind.get("agent", 0.0), 4), total_usd=round(spend.usd, 4),
         spend_by_kind={k: round(v, 4) for k, v in spend.by_kind.items()}, unknown_cost=spend.unknown_cost,
         calls=spend.calls, tokens=spend.tokens, **extra)


def write_holdout(run: ResearchRun, emit) -> None:
    """Score the seed and the global best on the never-searched holdout split -> holdout.json."""
    entries = run.entries()
    seed, best = entries[0], run.archive(entries).global_best
    res = {label: (e.id, run.evaluate_holdout(e)) for label, e in (("seed", seed), ("best", best))}
    (run.store.root / "holdout.json").write_text(json.dumps(
        {label: {"id": i, **r.to_dict()} for label, (i, r) in res.items()}, indent=1))
    emit("holdout", seed=res["seed"][1].score, best=res["best"][1].score, best_id=best.id,
         baseline=res["best"][1].baseline)
