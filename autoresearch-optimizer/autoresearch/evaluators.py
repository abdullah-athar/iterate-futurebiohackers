"""Candidate evaluators: turn a Candidate into a scored Result."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .core import Candidate, Result, SuiteSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SubprocessEvaluator:
    """Runs each candidate in a fresh Python process with a wall-clock timeout.

    This isolates the loop from candidate code that hangs, crashes, or leaks state.
    """

    def __init__(self, workdir: Path, timeout_s: float = 60.0) -> None:
        self.workdir = Path(workdir) / "candidates"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    def evaluate(self, candidate: Candidate, suite: SuiteSpec) -> Result:
        path = self.workdir / f"{candidate.id}.py"
        path.write_text(candidate.code)
        cmd = [sys.executable, "-m", "autoresearch.run_candidate", str(path), suite.tier, str(suite.seed_offset)]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return self._failure(candidate, suite, f"Timeout after {self.timeout_s:.0f}s", time.perf_counter() - t0)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            return self._failure(candidate, suite, _tail(proc.stderr), elapsed)

        summary = json.loads(proc.stdout.strip().splitlines()[-1])
        errors = [r["error_message"] for r in summary["instance_results"] if r["error_message"]]
        return Result(
            candidate_id=candidate.id,
            suite=str(suite),
            score=summary["total_score"],
            valid=summary["num_valid"] == summary["num_instances"],
            error="; ".join(errors[:3]),
            baseline_score=summary["total_baseline_score"],
            elapsed=elapsed,
            per_instance=[
                {k: r[k] for k in ("instance_name", "score", "baseline_score", "planted_score", "elapsed_seconds")}
                for r in summary["instance_results"]
            ],
        )

    def _failure(self, candidate: Candidate, suite: SuiteSpec, error: str, elapsed: float) -> Result:
        return Result(candidate_id=candidate.id, suite=str(suite), score=float("inf"), valid=False, error=error, elapsed=elapsed)


def _tail(text: str, n: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-n:])
