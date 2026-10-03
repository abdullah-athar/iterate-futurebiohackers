"""Render a research run (ledger + archive + optional held-out evaluation) as Markdown."""

from __future__ import annotations

from .archive import _inst_score
from .ledger import (
    STATUS_REJECTED_DUPLICATE,
    STATUS_REJECTED_GUARD,
    STATUS_SEED,
    VERDICT_UNCONFIRMED,
    Entry,
)
from .loop import ResearchRun
from .problem import EvalResult


def _obj(e: Entry) -> str:
    return "-" if e.objective is None else ("fail" if e.objective == float("inf") else f"{e.objective:g}")


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation between two score lists (no scipy dependency)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if vx == 0 or vy == 0 else cov / (vx * vy)


def render(run: ResearchRun, holdout: bool = False, holdout_front: bool = False) -> str:
    entries = run.entries()
    archive = run.archive(entries)
    seed = next((e for e in entries if e.status == STATUS_SEED), None)
    best = archive.global_best
    proposals = [e for e in entries if e.status != STATUS_SEED]
    evaluated = [e for e in proposals if e.status not in (STATUS_REJECTED_DUPLICATE, STATUS_REJECTED_GUARD)]
    dups = sum(e.status == STATUS_REJECTED_DUPLICATE for e in proposals)
    guarded = sum(e.status == STATUS_REJECTED_GUARD for e in proposals)
    tokens = sum(e.tokens for e in proposals)
    proposers = sorted({e.proposer for e in proposals}) or ["-"]

    out = [f"# Autoresearch report — {run.problem_name}", "",
           f"- run dir: `{run.store.root}`", f"- proposers: {', '.join(proposers)}",
           (f"- proposals: {len(proposals)} ({len(evaluated)} evaluated, {dups} rejected by novelty gate, "
            f"{guarded} rejected by import guard — neither evaluated)"),
           f"- LLM tokens: {tokens:,} (prompt {sum(e.prompt_tokens for e in proposals):,} / completion {sum(e.completion_tokens for e in proposals):,})"]
    if seed and best and seed.scored:
        gain = seed.objective - best.objective
        out.append(f"- objective ({run.problem.objective_split} split, lower is better): seed {seed.objective:g} → best {best.objective:g} "
                   f"(**{100 * gain / max(seed.objective, 1):.1f}% better**, baseline {best.eval_result(run.problem.objective_split).baseline:g})")
        if tokens:
            out.append(f"- research efficiency: {gain / (tokens / 1000):.2f} objective points per 1k tokens; "
                       f"{gain / max(len(evaluated), 1):.2f} per evaluated proposal")
    verdicts: dict[str, int] = {}
    for e in proposals:
        verdicts[e.verdict or "-"] = verdicts.get(e.verdict or "-", 0) + 1
    out.append("- hypothesis verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))
    claimed = [e for e in proposals if e.confirmed is not None]
    confirm_split = run.problem.confirm_split
    if claimed and confirm_split:
        unconfirmed = [e for e in claimed if e.confirmed is False]
        out.append(f"- confirmation re-tests: {len(claimed)} claimed new bests re-run on fresh '{confirm_split}' instances, "
                   f"{len(claimed) - len(unconfirmed)} confirmed, {len(unconfirmed)} rejected as noise/overfit"
                   + (f" (#{', #'.join(str(e.id) for e in unconfirmed)})" if unconfirmed else ""))
    splits = [s for s in run.problem.splits if s not in (confirm_split, "holdout")]
    if len(splits) >= 2:
        both = [e for e in entries if all(e.evals.get(s) and e.eval_result(s).ok for s in splits[:2])]
        rho = spearman([e.eval_result(splits[0]).score for e in both], [e.eval_result(splits[1]).score for e in both])
        if rho is not None:
            out.append(f"- proxy fidelity: Spearman ρ={rho:.2f} between '{splits[0]}' and '{splits[1]}' scores "
                       f"(n={len(both)}; how well the cheap screen predicted the objective split)")
    out += ["", "## Trajectory", "",
            "| # | parent | mode | status | verdict | objective | Δ vs best so far | tokens | hypothesis |", "|---|---|---|---|---|---|---|---|---|"]
    best_so_far = float("inf")
    for e in entries:
        delta = ""
        if e.scored:
            if best_so_far != float("inf"):
                delta = f"{best_so_far - e.objective:+g}"
            if e.verdict != VERDICT_UNCONFIRMED:
                best_so_far = min(best_so_far, e.objective)
        hyp = e.hypothesis.replace("|", "/")[:110]
        out.append(f"| {e.id} | {','.join(map(str, e.parent_ids)) or '-'} | {e.mode} | {e.status} | {e.verdict or '-'} | {_obj(e)} | {delta} | {e.tokens} | {hyp} |")
    out += ["", "## Pareto front (best on at least one instance)", "",
            "| # | objective | wins on instances | hypothesis |", "|---|---|---|---|"]
    for e in archive.front:
        won = archive.instances_won_by(e)
        out.append(f"| {e.id} | {_obj(e)} | {', '.join(won) or '(global best only)'} | {e.hypothesis.replace('|', '/')[:90]} |")
    out += ["", "## Per-instance best vs baseline / best_known", "",
            "| instance | baseline | best_known | front best | by # | seed |", "|---|---|---|---|---|---|"]
    for name, e in sorted(archive.per_instance_best.items()):
        bk = archive.per_instance_best_known.get(name)
        seed_score = _inst_score(seed, name) if seed and seed.scored else float("nan")
        out.append(f"| {name} | {archive.per_instance_baseline[name]:g} | {'-' if bk is None else f'{bk:g}'} | "
                   f"{_inst_score(e, name):g} | {e.id} | {seed_score:g} |")
    modes: dict[str, list[int]] = {}
    for e in evaluated:
        m = modes.setdefault(e.mode, [0, 0, 0])
        m[0] += 1
        m[1] += int(e.improved_global)
        m[2] += int(bool(e.improved_instances) and not e.improved_global)
    out += ["", "## Prompt-mode statistics (bandit arms)", "", "| mode | tried | new global best | new instance best only |", "|---|---|---|---|"]
    for m, (n, g, i) in sorted(modes.items()):
        out.append(f"| {m} | {n} | {g} | {i} |")
    if holdout and best:
        out += ["", "## Held-out evaluation (fresh seeds never seen during search)", ""]
        targets = [("seed", seed)] if seed else []
        targets.append(("best", best))
        if holdout_front:
            targets += [(f"front #{e.id}", e) for e in archive.front if e.id != best.id]
        results: list[tuple[str, Entry, EvalResult]] = [(label, e, run.evaluate_holdout(e)) for label, e in targets]
        out += ["| candidate | # | total | baseline | vs baseline | W/T/L vs baseline | time |", "|---|---|---|---|---|---|---|"]
        for label, e, r in results:
            w = sum(i.score < i.baseline for i in r.instances)
            t = sum(i.score == i.baseline for i in r.instances)
            l = sum(i.score > i.baseline for i in r.instances)
            tot = "fail" if r.error else f"{r.score:g}"
            out.append(f"| {label} | {e.id} | {tot} | {r.baseline:g} | {r.improvement_pct:+.1f}% | {w}/{t}/{l} | {r.elapsed:.1f}s |")
        if results:
            names = [i.name for i in results[-1][2].instances]
            out += ["", "| instance | " + " | ".join(l for l, _, _ in results) + " | baseline | best_known |",
                    "|---|" + "---|" * (len(results) + 2)]
            for n in names:
                row = []
                for _, _, r in results:
                    s = next((i.score for i in r.instances if i.name == n), None)
                    row.append("-" if s is None else f"{s:g}")
                ref = next((i for i in results[-1][2].instances if i.name == n), None)
                out.append(f"| {n} | " + " | ".join(row) + f" | {ref.baseline:g} | {'-' if ref.best_known is None else f'{ref.best_known:g}'} |")
    return "\n".join(out) + "\n"
