"""Schema-tolerant loading of autoresearch run directories.

A run directory contains `ledger.jsonl` (one JSON object per proposal) and optionally
`config.json`, `events.jsonl` (swarm progress) and `holdout.json` (final held-out scores). Every field is read with a default so that the dashboard keeps working when
the loop adds or renames fields.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUS_SEED = "seed"
STATUS_KEPT = "kept"
STATUS_EVALUATED = "evaluated"
STATUS_REJECTED_DUPLICATE = "rejected_duplicate"
STATUS_REJECTED_SCREEN = "rejected_screen"
STATUS_FAILED = "failed"
STATUS_REJECTED_GUARD = "rejected_guard"

STATUS_ORDER = [STATUS_KEPT, STATUS_EVALUATED, STATUS_REJECTED_SCREEN, STATUS_REJECTED_DUPLICATE,
                STATUS_REJECTED_GUARD, STATUS_FAILED]
STATUS_LABELS = {
    STATUS_SEED: "seed",
    STATUS_KEPT: "kept (improved)",
    STATUS_EVALUATED: "evaluated, no gain",
    STATUS_REJECTED_SCREEN: "rejected at screen",
    STATUS_REJECTED_DUPLICATE: "duplicate (not evaluated)",
    STATUS_FAILED: "crashed / over budget",
    STATUS_REJECTED_GUARD: "disallowed import (not evaluated)",
}


def _num(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    if isinstance(v, str):
        if v.lower() in ("inf", "infinity", "+inf"):
            return math.inf
        try:
            return float(v)
        except ValueError:
            return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class InstanceDiag:
    name: str
    score: float
    baseline: float
    best_known: float | None = None
    valid: bool = True
    error: str = ""
    elapsed: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstanceDiag:
        return cls(
            name=str(d.get("name", d.get("instance_name", "?"))),
            score=_num(d.get("score"), math.inf),
            baseline=_num(d.get("baseline", d.get("baseline_score")), math.nan),
            best_known=_num(d.get("best_known", d.get("planted_score"))),
            valid=bool(d.get("valid", d.get("is_valid", True))),
            error=str(d.get("error", d.get("error_message", "")) or ""),
            elapsed=_num(d.get("elapsed", d.get("elapsed_seconds")), 0.0),
        )


@dataclass
class Eval:
    split: str
    score: float
    baseline: float
    instances: list[InstanceDiag] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and math.isfinite(self.score)

    @property
    def improvement_pct(self) -> float:
        if not self.ok or not self.baseline:
            return math.nan
        return 100.0 * (self.baseline - self.score) / max(self.baseline, 1.0)

    @classmethod
    def from_dict(cls, split: str, d: dict[str, Any]) -> Eval:
        insts = [InstanceDiag.from_dict(i) for i in d.get("instances", d.get("instance_results", []))]
        score = _num(d.get("score", d.get("total_score")))
        baseline = _num(d.get("baseline", d.get("total_baseline_score")))
        if score is None:
            score = sum(i.score for i in insts) if insts else math.inf
        if baseline is None:
            baseline = sum(i.baseline for i in insts) if insts else math.nan
        return cls(
            split=str(d.get("split", split)),
            score=score,
            baseline=baseline,
            instances=insts,
            elapsed=_num(d.get("elapsed", d.get("total_time_seconds")), 0.0),
            error=str(d.get("error", "") or ""),
        )


@dataclass
class Entry:
    id: int
    parent_ids: list[int]
    mode: str
    hypothesis: str
    status: str
    proposer: str = "agent"
    objective: float | None = None
    improved_global: bool = False
    improved_instances: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    timestamp: float | None = None
    evals: dict[str, Eval] = field(default_factory=dict)
    novelty: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    source_path: str = ""
    verdict: str = ""
    confirmed: bool | None = None
    generation: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return float(self.usage.get("cost_usd") or 0.0)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def scored(self) -> bool:
        return self.objective is not None and math.isfinite(self.objective)

    @property
    def is_seed(self) -> bool:
        return self.status == STATUS_SEED

    @property
    def was_evaluated(self) -> bool:
        return self.status not in (STATUS_REJECTED_DUPLICATE, STATUS_REJECTED_GUARD)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entry:
        evals = {}
        for split, ev in (d.get("evals") or {}).items():
            if isinstance(ev, dict):
                evals[split] = Eval.from_dict(split, ev)
        pids = d.get("parent_ids", d.get("parents", []))
        if isinstance(pids, int):
            pids = [pids]
        novelty = dict(d.get("novelty") or {})
        novelty.setdefault("duplicate_of", novelty.get("nearest_id"))
        novelty.setdefault("similarity", novelty.get("max_similarity"))
        return cls(
            id=int(d.get("id", 0)),
            parent_ids=[int(p) for p in (pids or [])],
            mode=str(d.get("mode", "") or ""),
            hypothesis=str(d.get("hypothesis", "") or ""),
            status=str(d.get("status", STATUS_EVALUATED)),
            proposer=str(d.get("proposer", "agent") or "agent"),
            objective=_num(d.get("objective")),
            improved_global=bool(d.get("improved_global", False)),
            improved_instances=list(d.get("improved_instances") or []),
            prompt_tokens=int(d.get("prompt_tokens", 0) or 0),
            completion_tokens=int(d.get("completion_tokens", 0) or 0),
            elapsed=_num(d.get("elapsed"), 0.0),
            timestamp=_num(d.get("timestamp")),
            evals=evals,
            novelty=novelty,
            note=str(d.get("note", "") or ""),
            source_path=str(d.get("source_path", "") or ""),
            verdict=str(d.get("verdict", "") or ""),
            confirmed=d.get("confirmed"),
            generation=d.get("generation"),
            usage=dict(d.get("usage") or {}),
        )


@dataclass
class Run:
    label: str
    path: Path
    config: dict[str, Any]
    entries: list[Entry]
    objective_split: str
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def problem(self) -> str:
        return str(self.config.get("problem", "?"))

    @property
    def seed(self) -> Entry | None:
        return next((e for e in self.entries if e.is_seed), None)

    @property
    def proposals(self) -> list[Entry]:
        return [e for e in self.entries if not e.is_seed]

    def objective_eval(self, e: Entry) -> Eval | None:
        return e.evals.get(self.objective_split)

    @property
    def best(self) -> Entry | None:
        # same rule as autoresearch.archive: a gain that failed its fresh-instance re-test never counts
        scored = [e for e in self.entries if e.scored and e.confirmed is not False]
        return min(scored, key=lambda e: (e.objective, e.id)) if scored else None

    def describe(self) -> str:
        """One-line description of the flavour from config (skipping bookkeeping keys)."""
        skip = {"label", "problem", "synthetic", "description"}
        if self.config.get("description"):
            return str(self.config["description"])
        parts = [f"{k}={v}" for k, v in self.config.items() if k not in skip and not isinstance(v, (dict, list))]
        return ", ".join(parts)


def _infer_objective_split(entries: list[Entry], config: dict[str, Any]) -> str:
    if config.get("objective_split"):
        return str(config["objective_split"])
    seen: list[str] = []
    for e in entries:
        for s in e.evals:
            if s not in seen:
                seen.append(s)
    for e in entries:
        if e.scored:
            for s, ev in e.evals.items():
                if ev.score == e.objective and s != "holdout":
                    return s
    for pref in ("validate", "objective", "medium"):
        if pref in seen:
            return pref
    non_holdout = [s for s in seen if s != "holdout"]
    return non_holdout[-1] if non_holdout else (seen[0] if seen else "validate")


def load_run(path: str | Path, label: str | None = None) -> Run:
    root = Path(path)
    ledger = root / "ledger.jsonl"
    if not ledger.exists():
        raise FileNotFoundError(f"no ledger.jsonl in {root}")
    config: dict[str, Any] = {}
    cfg = root / "config.json"
    if cfg.exists():
        try:
            config = json.loads(cfg.read_text())
        except json.JSONDecodeError:
            config = {}
    entries: list[Entry] = []
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(Entry.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    entries.sort(key=lambda e: e.id)
    events = []
    if (root / "events.jsonl").exists():
        for line in (root / "events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if (root / "holdout.json").exists():
        try:
            by_id = {e.id: e for e in entries}
            for rec in json.loads((root / "holdout.json").read_text()).values():
                if rec.get("id") in by_id:
                    by_id[rec["id"]].evals["holdout"] = Eval.from_dict("holdout", rec)
        except (json.JSONDecodeError, AttributeError):
            pass
    return Run(
        label=label or str(config.get("label") or root.name),
        path=root,
        config=config,
        entries=entries,
        objective_split=_infer_objective_split(entries, config),
        events=events,
    )


def load_runs(specs: list[str]) -> list[Run]:
    """Accept `path` or `label=path`; a path without a ledger is scanned for child run dirs."""
    runs: list[Run] = []
    for spec in specs:
        label, _, p = spec.rpartition("=") if "=" in spec and not Path(spec).exists() else ("", "", spec)
        root = Path(p)
        if (root / "ledger.jsonl").exists():
            runs.append(load_run(root, label or None))
            continue
        children = sorted(c for c in root.glob("*") if (c / "ledger.jsonl").exists()) if root.is_dir() else []
        if not children:
            raise FileNotFoundError(f"{root}: not a run directory and contains no runs")
        runs.extend(load_run(c) for c in children)
    return runs
