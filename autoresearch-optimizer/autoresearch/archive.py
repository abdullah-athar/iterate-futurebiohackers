"""Pareto-per-instance archive derived from the ledger.

The archive keeps the global best *and* every candidate that is best on at least one
instance of the objective split. Members that beat the global best somewhere are
"complementary" and are offered to the proposer for merging.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .ledger import Entry


@dataclass
class Archive:
    objective_split: str
    global_best: Entry | None = None
    per_instance_best: dict[str, Entry] = field(default_factory=dict)
    per_instance_baseline: dict[str, float] = field(default_factory=dict)
    per_instance_best_known: dict[str, float | None] = field(default_factory=dict)

    @classmethod
    def build(cls, entries: list[Entry], objective_split: str) -> Archive:
        arc = cls(objective_split=objective_split)
        scored = [e for e in entries if e.scored and e.evals.get(objective_split)]
        if not scored:
            return arc
        confirmed = [e for e in scored if e.confirmed is not False] or scored
        arc.global_best = min(confirmed, key=lambda e: (e.objective, e.id))
        for e in scored:
            res = e.eval_result(objective_split)
            for inst in res.instances:
                if not inst.valid:
                    continue
                arc.per_instance_baseline[inst.name] = inst.baseline
                arc.per_instance_best_known[inst.name] = inst.best_known
                cur = arc.per_instance_best.get(inst.name)
                if cur is None or _inst_key(e, inst.name) < _inst_key(cur, inst.name):
                    arc.per_instance_best[inst.name] = e
        return arc

    @property
    def front(self) -> list[Entry]:
        seen: dict[int, Entry] = {}
        if self.global_best:
            seen[self.global_best.id] = self.global_best
        for e in self.per_instance_best.values():
            seen.setdefault(e.id, e)
        return sorted(seen.values(), key=lambda e: e.objective)

    def instances_won_by(self, entry: Entry) -> list[str]:
        return sorted(n for n, e in self.per_instance_best.items() if e.id == entry.id)

    def complementary(self) -> list[tuple[Entry, list[str]]]:
        """Front members (other than the global best) with the instances they win."""
        if not self.global_best:
            return []
        out = []
        for e in self.front:
            if e.id == self.global_best.id:
                continue
            won = self.instances_won_by(e)
            if won:
                out.append((e, won))
        return out

    def select_parent(self, rng: random.Random, exploit: float = 0.7) -> Entry:
        """Power-law-ish parent sampling: mostly the leader, sometimes a front member."""
        front = self.front
        if len(front) <= 1 or rng.random() < exploit:
            return self.global_best
        return rng.choice([e for e in front if e.id != self.global_best.id])

    def improvement_of(self, entry: Entry) -> tuple[bool, list[str]]:
        """Would `entry` (not yet in the archive) improve the global best / any instance?"""
        res = entry.eval_result(self.objective_split)
        if res is None or not entry.scored:
            return False, []
        improved_global = self.global_best is None or entry.objective < self.global_best.objective
        improved = []
        for inst in res.instances:
            if not inst.valid:
                continue
            cur = self.per_instance_best.get(inst.name)
            if cur is None or inst.score < _inst_score(cur, inst.name):
                improved.append(inst.name)
        return improved_global, improved


def _inst_score(entry: Entry, name: str) -> float:
    for inst in entry.eval_result(entry_objective_split(entry)).instances:
        if inst.name == name:
            return inst.score
    return float("inf")


def _inst_key(entry: Entry, name: str) -> tuple[float, float, int]:
    return (_inst_score(entry, name), entry.objective, entry.id)


def entry_objective_split(entry: Entry) -> str:
    # evaluations are keyed by split; the objective split is the last non-holdout one present
    for split in ("validate", "screen"):
        if split in entry.evals:
            return split
    return next(iter(entry.evals))
