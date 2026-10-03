"""Parallel research: N coding agents per generation propose solvers, evaluation fans out
(Modal or local), and this process is the only writer of the ledger and events.

Per generation:  allocate modes/parents (UCB bandit) -> N agents write candidate.py +
hypothesis.txt in their own workspace -> guard + novelty gate (in arrival order) ->
evaluate the survivors in parallel -> record in order -> next generation, until the wall-clock
budget cannot fit another generation. Then the best and the seed are scored on holdout.

With --hypothesis-first, each agent's idea is first stated by one cheap tool-less call, a
hypothesis gate drops repeated ideas, and only the survivors get a coding session (see
hypothesis_gate.py).
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
from dataclasses import asdict, dataclass
from pathlib import Path

from .hypothesis_gate import ask_claude
from .loop import ResearchRun, evaluate_candidate
from .prompts import MODES, build_user_prompt

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Assignment:
    worker: int
    mode: str
    parent_ids: list[int]
    direction: str = ""


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

HYP_PROMPT = """You are worker {worker} of {n} agents proposing median-string solvers in parallel this generation
({mix}). Your mode is `{mode}` and your research direction is: {direction}.
Other agents cover the other directions, so stay on yours. Ideas already in the research ledger
below count as taken: build on them or go elsewhere.

Reply with ONE line and nothing else: the change you would make to the parent solver, why, and on
which instances you expect a lower score. Name the technique precisely (e.g. "late-acceptance walk
over insert/delete moves", not "improve the search"). A gate compares the lines of all agents; only
distinct ideas get implemented.

{status}"""
HYP_SYSTEM = "You are a research agent. Propose one algorithmic change. Reply with one line only."

PROMPT = "Read AGENT.md and STATUS.md in the current directory and follow them. Work only in this directory."
CODE_PROMPT = ("Read AGENT.md and STATUS.md in the current directory and follow them. Work only in this directory. "
               "Your hypothesis is already in hypothesis.txt and passed a gate that rejects ideas already taken: "
               "skip step 2 and implement exactly that idea.")


def write_workspace(run: ResearchRun, a: Assignment, gen: int, n: int, mix: str, turn_s: int, ws: Path,
                    eval_backend: str) -> None:
    ctx = run.context(mode=a.mode, parent_ids=a.parent_ids)
    (ws / "STATUS.md").write_text(build_user_prompt(run.problem.describe(), ctx))
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


def run_claude(ws: Path, turn_s: int, model: str, max_budget_usd: float | None = None, prompt: str = PROMPT,
               effort: str | None = None) -> dict:
    """Run one headless Claude Code session in `ws`; kill its process group at the deadline."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model,
           "--permission-mode", "acceptEdits", "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep"]
    if max_budget_usd:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    if effort:
        cmd += ["--effort", effort]
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
    usage = {"model": model, "seconds": round(time.time() - t0, 1), "outcome": outcome}
    try:
        res = json.loads((ws / "agent.json").read_text())
        u = res.get("usage", {})
        usage.update(cost_usd=res.get("total_cost_usd", 0.0), num_turns=res.get("num_turns"),
                     prompt_tokens=u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                     + u.get("cache_creation_input_tokens", 0),
                     completion_tokens=u.get("output_tokens", 0))
        if res.get("is_error"):
            usage["outcome"] = res.get("subtype") or "agent_error"
    except (json.JSONDecodeError, OSError):
        pass
    return usage


def _read_hypothesis(ws: Path) -> str:
    p = ws / "hypothesis.txt"
    text = p.read_text().strip() if p.exists() else ""
    return text.splitlines()[0] if text else ""


def _merge_usage(hyp: dict, code: dict | None = None) -> dict:
    """One ledger usage for a two-phase agent: totals (hypothesis call + gate share + coding session),
    with the hypothesis call and the gate share also kept separately."""
    code = code or {}
    out = {**hyp, **code}
    for k in ("prompt_tokens", "completion_tokens", "seconds", "num_turns"):
        out[k] = (hyp.get(k) or 0) + (code.get(k) or 0)
    out["cost_usd"] = (hyp.get("cost_usd") or 0) + (hyp.get("gate_cost_usd") or 0) + (code.get("cost_usd") or 0)
    out.update({f"hyp_{k}": hyp.get(k) or 0 for k in ("cost_usd", "prompt_tokens", "completion_tokens", "seconds")})
    out["gate_cost_usd"] = hyp.get("gate_cost_usd") or 0
    return out


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("-• ").strip("`*\"' ")
        if line:
            return line
    return ""


def claude_proposer(model: str, turn_s: int, eval_backend: str, max_budget_usd: float | None, emit,
                    hypothesis_first: bool = False, hyp_turn_s: int = 60, gate=None, effort: str | None = None):
    """propose_many backed by local headless Claude Code sessions (one thread each).

    With `hypothesis_first`, each agent's idea is first stated by one tool-less call over its
    STATUS.md (a few k tokens instead of a coding session); `gate(new, prior)` drops ideas that
    repeat an earlier one, and only the survivors get a coding session, told to implement it."""

    def propose_many(run: ResearchRun, gen: int, assignments: list[Assignment]) -> list[dict]:
        mix = ", ".join(f"{c} on {m}" for m, c in Counter(a.mode for a in assignments).items())
        agents_dir = run.store.root / "agents"
        agents_dir.mkdir(exist_ok=True)
        n = len(assignments)

        def finish(a: Assignment, ws: Path, usage: dict, hypothesis: str, parent_src: str, **extra) -> dict:
            source = (ws / "candidate.py").read_text() if "gate" not in extra else ""
            if "gate" not in extra and (source == parent_src or not hypothesis):
                source, usage["outcome"] = "", usage["outcome"] if usage["outcome"] != "ok" else "no_file"
            tag = f"g{gen:02d}-w{a.worker:02d}"
            for name in ("agent.json", "agent.err"):
                if (ws / name).exists():
                    shutil.copy(ws / name, agents_dir / f"{tag}-{name}")
            shutil.rmtree(ws, ignore_errors=True)
            emit("agent_done", gen=gen, worker=a.worker, mode=a.mode, outcome=usage["outcome"],
                 seconds=usage["seconds"], cost_usd=usage.get("cost_usd", 0.0), hypothesis=hypothesis[:160])
            return {"assignment": a, "source": source, "hypothesis": hypothesis, "usage": usage, **extra}

        def workspace(a: Assignment) -> tuple[Path, str]:
            ws = Path(tempfile.mkdtemp(prefix=f"autoresearch-g{gen:02d}-w{a.worker:02d}-"))
            write_workspace(run, a, gen, n, mix, turn_s, ws, eval_backend)
            return ws, (ws / "candidate.py").read_text()

        def one(a: Assignment) -> dict:
            ws, parent_src = workspace(a)
            usage = run_claude(ws, turn_s, model, max_budget_usd, effort=effort)
            return finish(a, ws, usage, _read_hypothesis(ws), parent_src)

        def hypothesis_only(a: Assignment) -> dict:
            ws, parent_src = workspace(a)
            prompt = HYP_PROMPT.format(worker=a.worker, n=n, mix=mix, mode=a.mode,
                                       direction=a.direction or "your choice", status=(ws / "STATUS.md").read_text())
            try:
                text, usage = ask_claude(prompt, HYP_SYSTEM, model, hyp_turn_s, effort)
                hypothesis, usage["outcome"] = _first_line(text), "ok"
            except (subprocess.SubprocessError, OSError, ValueError) as e:
                hypothesis, usage = "", {"model": model, "seconds": float(hyp_turn_s), "outcome": f"hyp_error: {e}"[:200]}
            if hypothesis:
                (ws / "hypothesis.txt").write_text(hypothesis + "\n")
            return {"assignment": a, "ws": ws, "parent_src": parent_src, "usage": usage, "hypothesis": hypothesis}

        def implement(h: dict) -> dict:
            code = run_claude(h["ws"], turn_s, model, max_budget_usd, prompt=CODE_PROMPT, effort=effort)
            return finish(h["assignment"], h["ws"], _merge_usage(h["usage"], code),
                          _read_hypothesis(h["ws"]) or h["hypothesis"], h["parent_src"])

        def parallel(fn, items: list) -> list[dict]:
            if not items:
                return []
            with ThreadPoolExecutor(max_workers=len(items)) as pool:
                return [f.result() for f in as_completed([pool.submit(fn, x) for x in items])]

        if not hypothesis_first:
            return parallel(one, assignments)

        hyps = sorted(parallel(hypothesis_only, assignments), key=lambda h: h["assignment"].worker)
        stated = [h for h in hyps if h["hypothesis"]]
        prior = [(f"#{e.id}", e.hypothesis) for e in run.entries()[-40:] if e.mode != "seed" and e.hypothesis]
        verdicts, gate_usage = gate([(f"w{h['assignment'].worker:02d}", h["hypothesis"]) for h in stated], prior)
        share = (gate_usage.get("cost_usd", 0.0) or 0.0) / max(len(stated), 1)
        for h, v in zip(stated, verdicts):
            h["verdict"] = v
            h["usage"]["gate_cost_usd"] = share
        rejected = [h for h in stated if not h["verdict"].keep]
        emit("hypothesis_gate", gen=gen, stated=len(stated), kept=len(stated) - len(rejected), rejected=len(rejected),
             judge=gate_usage.get("judge", "lexical"), fallback=gate_usage.get("fallback", ""),
             cost_usd=round(gate_usage.get("cost_usd", 0.0) or 0.0, 4),
             decisions=[{"worker": h["assignment"].worker, **h["verdict"].to_dict()} for h in stated])
        results = [finish(h["assignment"], h["ws"], {**h["usage"], "outcome": "no_hypothesis"}, "", h["parent_src"])
                   for h in hyps if not h["hypothesis"]]
        for h in rejected:
            results.append(finish(h["assignment"], h["ws"], {**_merge_usage(h["usage"]), "outcome": "rejected_hypothesis"},
                                  h["hypothesis"], h["parent_src"], gate=h["verdict"].to_dict()))
        return results + parallel(implement, [h for h in stated if h["verdict"].keep])

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
    if k == "hypothesis_gate":
        fb = f", fallback {r['fallback']}" if r.get("fallback") else ""
        return (f"gen {r['gen']} hypothesis gate: {r['kept']} kept, {r['rejected']} same idea as an earlier one "
                f"({r['judge']}{fb}, ${r['cost_usd']:.3f})")
    if k == "eval_start":
        return (f"gen {r['gen']} evaluating {r['n']} candidate(s) ({r['dup']} duplicate, {r['guard']} guard, "
                f"{r.get('hyp', 0)} hypothesis-gate, {r['empty']} empty skipped)")
    if k == "entry":
        star = " ** NEW BEST **" if r.get("new_best") else ""
        obj = "-" if r["objective"] is None else r["objective"]
        return f"gen {r['gen']} #{r['id']} w{r['worker']:02d} [{r['mode']}] {r['status']}/{r['verdict']} {obj}{star}"
    if k == "gen_end":
        return (f"gen {r['gen']} end: best #{r['best_id']}={r['best']:g}; {r['evaluated']} evaluated, {r['kept']} kept; "
                f"{r['seconds']:.0f}s; cost so far ${r['cost_usd']:.2f}")
    if k == "holdout":
        return f"holdout: seed {r['seed']:g} -> best #{r['best_id']} {r['best']:g} (baseline {r['baseline']:g})"
    if k == "run_end":
        return f"run end: {r['generations']} generation(s), {r['proposals']} proposals, best #{r['best_id']}={r['best']:g}"
    return json.dumps(r)


def run_swarm(run: ResearchRun, agents: int, turn_s: int, budget_s: float, propose_many, evaluate_many, emit,
              seed: int = 0, max_generations: int | None = None, eval_name: str = "") -> None:
    rng = random.Random(seed)
    t0 = time.time()
    entries = run.entries()
    gen = max((e.generation or 0 for e in entries), default=0) + 1
    eval_s, generations, proposals = 60.0, 0, 0
    emit("run_start", agents=agents, budget_s=int(budget_s), turn_s=turn_s, eval=eval_name)
    while time.time() - t0 + turn_s + eval_s + 30 <= budget_s and (max_generations is None or generations < max_generations):
        tg = time.time()
        assignments = allocate(run, agents, rng)
        best = run.archive().global_best
        emit("gen_start", gen=gen, mix=", ".join(f"{m} {c}" for m, c in Counter(a.mode for a in assignments).items()),
             best_id=best.id, best=best.objective, assignments=[asdict(a) for a in assignments])
        results = propose_many(run, gen, assignments)
        results.sort(key=lambda r: r["assignment"].worker)
        next_id = run.store.next_id()
        arrivals: list[tuple[int, str]] = []
        for r in (r for r in results if r["source"]):
            r["pre"] = run.precheck(r["source"], extra_prior=arrivals)
            arrivals.append((next_id + len(arrivals), r["source"]))
        todo = [r for r in results if r["source"] and not r["pre"][0] and not r["pre"][1].is_duplicate]
        gated = [r for r in results if r.get("gate")]
        emit("eval_start", gen=gen, n=len(todo), dup=sum(1 for r in results if r.get("pre") and r["pre"][1].is_duplicate),
             guard=sum(1 for r in results if r.get("pre") and r["pre"][0]), hyp=len(gated),
             empty=sum(1 for r in results if not r["source"] and not r.get("gate")))
        te = time.time()
        for r, ev in zip(todo, evaluate_many([r["source"] for r in todo])):
            r["evals"] = ev if isinstance(ev, dict) else {}
            if not isinstance(ev, dict):
                r["usage"]["eval_error"] = repr(ev)[:300]
        eval_s = max(30.0, time.time() - te)
        for r in (r for r in results if r["source"]):
            a, u = r["assignment"], r["usage"]
            e = run.record(r["source"], r["hypothesis"], a.mode, a.parent_ids, f"claude-code:{u.get('model', '?')}",
                           r.get("evals", {}), r["pre"], generation=gen, usage=u, elapsed=u.get("seconds", 0.0),
                           prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0))
            emit("entry", gen=gen, id=e.id, worker=a.worker, mode=a.mode, status=e.status, verdict=e.verdict,
                 objective=None if not e.scored else e.objective, new_best=e.improved_global)
            proposals += 1
        for r in gated:
            a, u, g = r["assignment"], r["usage"], r["gate"]
            e = run.record_rejected_hypothesis(r["hypothesis"], a.mode, a.parent_ids, f"claude-code:{u.get('model', '?')}",
                                               g["similar_to"], g["reason"], generation=gen, usage=u,
                                               elapsed=u.get("seconds", 0.0), prompt_tokens=u.get("prompt_tokens", 0),
                                               completion_tokens=u.get("completion_tokens", 0))
            emit("entry", gen=gen, id=e.id, worker=a.worker, mode=a.mode, status=e.status, verdict=e.verdict,
                 objective=None, new_best=False)
            proposals += 1
        entries = run.entries()
        best = run.archive(entries).global_best
        gen_entries = [e for e in entries if e.generation == gen]
        emit("gen_end", gen=gen, best_id=best.id, best=best.objective, seconds=round(time.time() - tg, 1),
             evaluated=sum(1 for e in gen_entries if e.evals), kept=sum(1 for e in gen_entries if e.status == "kept"),
             cost_usd=round(sum(e.usage.get("cost_usd", 0.0) or 0.0 for e in entries), 2))
        gen += 1
        generations += 1
    write_holdout(run, emit)
    best = run.archive().global_best
    emit("run_end", generations=generations, proposals=proposals, best_id=best.id, best=best.objective)


def write_holdout(run: ResearchRun, emit) -> None:
    """Score the seed and the global best on the never-searched holdout split -> holdout.json."""
    entries = run.entries()
    seed, best = entries[0], run.archive(entries).global_best
    res = {label: (e.id, run.evaluate_holdout(e)) for label, e in (("seed", seed), ("best", best))}
    (run.store.root / "holdout.json").write_text(json.dumps(
        {label: {"id": i, **r.to_dict()} for label, (i, r) in res.items()}, indent=1))
    emit("holdout", seed=res["seed"][1].score, best=res["best"][1].score, best_id=best.id,
         baseline=res["best"][1].baseline)
