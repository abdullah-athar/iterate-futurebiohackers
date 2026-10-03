"""Context shown to the proposing agent.

The proposer (a coding agent) must *reflect* before editing: it is shown per-instance
diagnostics of the parent(s) and a digest of what has been tried, and must write a one-line
hypothesis plus a complete solver file. Prompt *modes* steer the kind of change requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .archive import Archive
from .ledger import Entry
from .problem import EvalResult

MODES: dict[str, str] = {
    "tune": "Keep the parent's algorithm. Make one focused change: a better neighbourhood, "
            "a smarter start, a restart schedule, or a parameter choice. Justify it from the diagnostics.",
    "fix_losers": "Target the instances where the parent is furthest from best_known or loses to the "
                  "baseline. Diagnose *why* the algorithm fails on them (e.g. indels, high noise, large "
                  "alphabet, Hamming metric) and change the algorithm to fix that case without regressing others.",
    "new_family": "Propose a genuinely different algorithmic approach from everything in the ledger "
                  "(e.g. alignment-based consensus, DP over a center sequence, beam search over edit operations, "
                  "simulated annealing, iterative refinement against the current center). Do not tune the parent.",
    "merge": "You are given two solvers that are each best on different instances. Combine their "
             "strengths into one solver (e.g. run both and keep the better, or use one's start with the "
             "other's local search). The result must beat both on their respective instances.",
}

@dataclass
class Context:
    mode: str
    parents: list[Entry]
    parent_sources: list[str]
    diagnostics: str
    digest: str
    rejection_note: str = ""
    extra: dict = field(default_factory=dict)


def format_diagnostics(res: EvalResult, archive: Archive | None = None, label: str = "") -> str:
    lines = [(f"{label}split={res.split} total={res.score:g} baseline={res.baseline:g} "
              f"({res.improvement_pct:+.1f}% vs baseline) elapsed={res.elapsed:.1f}s")]
    if res.error:
        lines.append(f"  ERROR: {res.error}")
    hdr = f"  {'instance':<28}{'score':>7}{'base':>7}{'best_known':>11}{'front_best':>11}{'time':>7}  info"
    lines.append(hdr)
    for i in res.instances:
        fb = ""
        if archive and i.name in archive.per_instance_best:
            e = archive.per_instance_best[i.name]
            fb = f"{_inst_score(e, i.name):g}(#{e.id})"
        flag = "" if i.valid else f"  INVALID: {i.error}"
        bk = "" if i.best_known is None else f"{i.best_known:g}"
        worse = " <- loses to baseline" if i.valid and i.score > i.baseline else ""
        lines.append(f"  {i.name:<28}{i.score:>7g}{i.baseline:>7g}{bk:>11}{fb:>11}{i.elapsed:>6.2f}s  {i.info}{worse}{flag}")
    return "\n".join(lines)


def _inst_score(entry: Entry, name: str) -> float:
    from .archive import _inst_score as f
    return f(entry, name)


def format_digest(entries: list[Entry], limit: int = 12) -> str:
    rows = []
    for e in entries[-limit:]:
        obj = "-" if e.objective is None else ("fail" if e.objective == float("inf") else f"{e.objective:g}")
        extra = ""
        if e.status == "rejected_duplicate":
            extra = f" (dup of #{e.novelty.get('nearest_id')}, sim={e.novelty.get('max_similarity')})"
        elif e.improved_instances:
            extra = f" (new best on: {', '.join(e.improved_instances)})"
        elif e.note:
            extra = f" ({e.note[:80]})"
        rows.append(f"  #{e.id:<3} parent={e.parent_ids} mode={e.mode:<11} {e.status:<19} verdict={e.verdict or '-':<12} objective={obj:<7} {e.hypothesis[:90]}{extra}")
    return "\n".join(rows) if rows else "  (empty)"


def build_user_prompt(problem_description: str, ctx: Context) -> str:
    parts = [problem_description.strip(), "", f"## Mode: {ctx.mode}", MODES[ctx.mode], ""]
    for e, src in zip(ctx.parents, ctx.parent_sources):
        parts += [f"## Parent #{e.id} ({e.status}, objective={e.objective:g}) — hypothesis: {e.hypothesis}",
                  "```python", src.strip(), "```", ""]
    parts += ["## Diagnostics (per instance; 'front_best' = best score any candidate achieved on that instance)",
              ctx.diagnostics, "", "## Research ledger (most recent last)", ctx.digest, ""]
    if ctx.extra.get("falsified"):
        parts += ["## Falsified hypotheses (evidence says these do not help — do not re-propose, build on why they failed)"]
        parts += [f"- #{i} [{v}] {h}" for i, v, h in ctx.extra["falsified"]] + [""]
    if ctx.extra.get("plateau"):
        parts += [f"## Note: no global improvement for {ctx.extra['plateau']} proposals — change direction.", ""]
    if ctx.rejection_note:
        parts += ["## Rejected", ctx.rejection_note, ""]
    parts.append("Now reflect on the evidence, write the complete solver to candidate.py and a one-line "
                 "hypothesis (what you change, why, and on which instances you expect a lower score) to hypothesis.txt.")
    return "\n".join(parts)
