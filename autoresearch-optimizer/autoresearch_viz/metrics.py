"""Per-run series and flavour scoreboard derived from a loaded Run."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .load import STATUS_KEPT, STATUS_ORDER, STATUS_REJECTED_DUPLICATE, Entry, Run


@dataclass
class Point:
    entry_id: int
    evals: int  # cumulative evaluations (duplicates skipped by the novelty gate are not counted)
    tokens: int  # cumulative agent tokens (duplicates still cost their proposal tokens)
    seconds: float  # cumulative wall-clock
    best: float  # best objective so far
    improved: bool  # this entry set a new global best
    cost: float = 0.0  # cumulative agent cost in USD (Claude Code's estimate)


def best_so_far(run: Run) -> list[Point]:
    pts: list[Point] = []
    best = math.inf
    evals = tokens = 0
    seconds = cost = 0.0
    t0 = None
    stamps = [e.timestamp for e in run.entries if e.timestamp is not None]
    use_stamps = len(stamps) == len(run.entries) and stamps == sorted(stamps) and len(stamps) > 1
    for e in run.entries:
        if e.was_evaluated and not e.is_seed:
            evals += 1
        tokens += e.tokens
        cost += e.cost
        if use_stamps:
            t0 = e.timestamp if t0 is None else t0
            seconds = max(seconds, e.timestamp - t0)
        else:
            seconds += e.elapsed
        improved = e.scored and e.objective < best and e.confirmed is not False
        if improved:
            best = e.objective
        if math.isfinite(best):
            pts.append(Point(e.id, evals, tokens, seconds, best, improved and not e.is_seed, cost))
    return pts


def model_switches(run: Run) -> list[tuple[Point, str, str]]:
    """(point just before the switch, old proposer, new proposer) each time a run's proposer changes,
    e.g. a `--model sonnet,opus` schedule handing generation 2 to Opus."""
    pts = {p.entry_id: p for p in best_so_far(run)}
    out, prev = [], None
    for e in run.entries:
        if e.is_seed:
            continue
        if prev is not None and e.proposer != prev.proposer and prev.id in pts:
            out.append((pts[prev.id], prev.proposer, e.proposer))
        prev = e
    return out


@dataclass
class ModeStat:
    tried: int = 0
    global_wins: int = 0
    instance_wins: int = 0


@dataclass
class Summary:
    label: str
    problem: str
    objective_split: str
    proposals: int
    evaluated: int
    counts: dict[str, int]
    seed_objective: float
    baseline: float
    best_objective: float
    best_known: float | None
    best_entry_id: int | None
    gain_vs_baseline_pct: float
    gain_vs_seed_pct: float
    holdout_pct: float | None
    holdout_score: float | None
    holdout_baseline: float | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    evals_to_first_gain: int | None
    tokens_per_pct: float | None
    modes: dict[str, ModeStat] = field(default_factory=dict)
    instance_wins_only: int = 0
    cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def gap_closed_pct(self) -> float | None:
        """Share of the baseline -> best-known gap that was closed."""
        if self.best_known is None or not math.isfinite(self.baseline):
            return None
        gap = self.baseline - self.best_known
        return 100.0 * (self.baseline - self.best_objective) / gap if gap > 0 else None


def _pct(baseline: float, score: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(score) or baseline == 0:
        return math.nan
    return 100.0 * (baseline - score) / abs(baseline)


def summarize(run: Run) -> Summary:
    seed, best = run.seed, run.best
    seed_obj = seed.objective if seed and seed.scored else math.nan
    seed_eval = run.objective_eval(seed) if seed else None
    best_eval = run.objective_eval(best) if best else None
    baseline = (best_eval or seed_eval).baseline if (best_eval or seed_eval) else math.nan
    best_obj = best.objective if best else math.nan
    ref_eval = best_eval or seed_eval
    best_known = None
    if ref_eval and ref_eval.instances and all(i.best_known is not None for i in ref_eval.instances):
        best_known = sum(i.best_known for i in ref_eval.instances)

    counts = {s: 0 for s in STATUS_ORDER}
    for e in run.proposals:
        counts[e.status] = counts.get(e.status, 0) + 1
    modes: dict[str, ModeStat] = {}
    inst_only = 0
    for e in run.proposals:
        if not e.was_evaluated:
            continue
        m = modes.setdefault(e.mode or "?", ModeStat())
        m.tried += 1
        if e.improved_global:
            m.global_wins += 1
        elif e.improved_instances:
            m.instance_wins += 1
            inst_only += 1

    pts = best_so_far(run)
    first_gain = next((p.evals for p in pts if p.improved), None)
    prompt = sum(e.prompt_tokens for e in run.entries)
    completion = sum(e.completion_tokens for e in run.entries)
    seconds = pts[-1].seconds if pts else sum(e.elapsed for e in run.entries)
    gain_base = _pct(baseline, best_obj)
    gain_seed = _pct(seed_obj, best_obj)
    tokens_per_pct = (prompt + completion) / gain_seed if gain_seed and gain_seed > 0 else None

    hold = best.evals.get("holdout") if best else None
    return Summary(
        label=run.label,
        problem=run.problem,
        objective_split=run.objective_split,
        proposals=len(run.proposals),
        evaluated=sum(1 for e in run.proposals if e.was_evaluated),
        counts=counts,
        seed_objective=seed_obj,
        baseline=baseline,
        best_objective=best_obj,
        best_known=best_known,
        best_entry_id=best.id if best else None,
        gain_vs_baseline_pct=gain_base,
        gain_vs_seed_pct=gain_seed,
        holdout_pct=hold.improvement_pct if hold and hold.ok else None,
        holdout_score=hold.score if hold and hold.ok else None,
        holdout_baseline=hold.baseline if hold and hold.ok else None,
        prompt_tokens=prompt,
        completion_tokens=completion,
        seconds=seconds,
        evals_to_first_gain=first_gain,
        tokens_per_pct=tokens_per_pct,
        modes=modes,
        instance_wins_only=inst_only,
        cost_usd=sum(e.cost for e in run.entries),
    )


@dataclass
class InstanceRow:
    name: str
    baseline: float
    best_known: float | None
    seed: float | None
    best_by_run: dict[str, tuple[float, int]]  # label -> (best score on this instance, entry id)


def per_instance(runs: list[Run]) -> list[InstanceRow]:
    rows: dict[str, InstanceRow] = {}
    for run in runs:
        for e in run.entries:
            ev = run.objective_eval(e)
            if not ev or not ev.ok:
                continue
            for inst in ev.instances:
                if not inst.valid or not math.isfinite(inst.score):
                    continue
                row = rows.setdefault(inst.name, InstanceRow(inst.name, inst.baseline, inst.best_known, None, {}))
                if e.is_seed and row.seed is None:
                    row.seed = inst.score
                cur = row.best_by_run.get(run.label)
                if cur is None or inst.score < cur[0]:
                    row.best_by_run[run.label] = (inst.score, e.id)
    return list(rows.values())


def best_delta_text(run: Run, e: Entry, best_before: float) -> str:
    if not e.scored:
        return "—"
    if not math.isfinite(best_before):
        return "seed"
    d = e.objective - best_before
    return f"{d:+.0f}" if d else "±0"
