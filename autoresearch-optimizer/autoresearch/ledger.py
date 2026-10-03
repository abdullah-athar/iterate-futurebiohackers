"""Append-only experiment ledger (JSONL) plus the on-disk layout of a research run."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .problem import EvalResult

STATUS_SEED = "seed"
STATUS_KEPT = "kept"                  # improved global best or entered the per-instance front
STATUS_EVALUATED = "evaluated"        # valid, fully evaluated, no improvement
STATUS_REJECTED_DUPLICATE = "rejected_duplicate"   # novelty gate; not evaluated
STATUS_REJECTED_SCREEN = "rejected_screen"         # failed/invalid/worse-than-baseline on screen
STATUS_FAILED = "failed"              # crash or timeout on the objective split
STATUS_REJECTED_GUARD = "rejected_guard"           # disallowed imports/calls; not evaluated

VERDICT_SUPPORTED = "supported"            # new global best, confirmed on fresh instances
VERDICT_PARTIAL = "partial"                # better on some instances, not globally
VERDICT_FALSIFIED = "falsified"            # evaluated, no gain
VERDICT_UNCONFIRMED = "unconfirmed"        # won on validate but not on the fresh confirm set (noise/overfit)
VERDICT_INCONCLUSIVE = "inconclusive"      # crashed/timed out/invalid: hypothesis never actually tested
VERDICT_UNTESTED = "untested"              # rejected before evaluation (duplicate or guard)


@dataclass
class Entry:
    id: int
    parent_ids: list[int]
    mode: str
    hypothesis: str
    status: str
    proposer: str = "agent"
    source_path: str = ""
    novelty: dict[str, Any] = field(default_factory=dict)
    evals: dict[str, dict[str, Any]] = field(default_factory=dict)
    objective: float | None = None
    improved_global: bool = False
    improved_instances: list[str] = field(default_factory=list)
    confirmed: bool | None = None      # result of the fresh-instance re-test of a claimed global best
    verdict: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    note: str = ""
    timestamp: float = field(default_factory=time.time)
    generation: int | None = None      # swarm generation (None in single-agent mode)
    usage: dict[str, Any] = field(default_factory=dict)   # agent cost_usd, num_turns, seconds, model, outcome

    def eval_result(self, split: str) -> EvalResult | None:
        d = self.evals.get(split)
        return EvalResult.from_dict(d) if d else None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def scored(self) -> bool:
        return self.objective is not None and self.objective != float("inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entry:
        return cls(**d)


class RunStore:
    """Directory layout: config.json, ledger.jsonl, candidates/NNNN.py, report.md."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.ledger_path = self.root / "ledger.jsonl"
        self.candidates_dir = self.root / "candidates"
        self.config_path = self.root / "config.json"

    @property
    def exists(self) -> bool:
        return self.ledger_path.exists()

    def create(self, config: dict[str, Any]) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2))
        self.ledger_path.touch()

    def config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text())

    def entries(self) -> list[Entry]:
        if not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("objective") == "inf":
                    d["objective"] = float("inf")
                out.append(Entry.from_dict(d))
        return out

    def append(self, entry: Entry) -> None:
        d = entry.to_dict()
        if d["objective"] == float("inf"):
            d["objective"] = "inf"
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(d) + "\n")

    def next_id(self) -> int:
        entries = self.entries()
        return (max(e.id for e in entries) + 1) if entries else 0

    def write_candidate(self, cid: int, source: str) -> str:
        path = self.candidates_dir / f"{cid:04d}.py"
        path.write_text(source)
        return str(path.relative_to(self.root))

    def read_candidate(self, entry: Entry) -> str:
        return (self.root / entry.source_path).read_text()
