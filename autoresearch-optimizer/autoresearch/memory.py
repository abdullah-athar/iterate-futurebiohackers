"""Memory: the research history, persisted to disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .core import Record


class JsonlLedger:
    """Appends every record to ledger.jsonl and keeps champion.py up to date."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "ledger.jsonl"
        self.records: list[Record] = []

    def record(self, rec: Record) -> None:
        self.records.append(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        if rec.accepted:
            (self.run_dir / "champion.py").write_text(rec.candidate.code)

    def champion(self) -> Record | None:
        return next((r for r in reversed(self.records) if r.accepted), None)

    def summary(self, k: int = 10) -> str:
        lines = []
        for r in self.records[-k:]:
            status = "ACCEPTED" if r.accepted else ("rejected" if r.result.valid else "FAILED")
            line = f"- iter {r.iteration} [{status}] score={r.result.score}: {r.candidate.rationale[:300]}"
            if r.result.error:
                line += f"\n  error: {r.result.error[-500:]}"
            lines.append(line)
        return "\n".join(lines) or "(none yet)"
