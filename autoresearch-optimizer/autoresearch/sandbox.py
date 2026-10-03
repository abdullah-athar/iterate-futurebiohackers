"""Trusted side of candidate evaluation.

Two process boundaries: `evaluate_in_subprocess` runs one whole split evaluation in a fresh
interpreter with a wall-clock limit (the containment layer); inside it, `run_candidate` runs the
candidate's code in yet another process (`candidate_runner`) that only ever sees the public view
of each instance and only returns strings. Scoring happens here, against our own copy of the data.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from median_string.evaluator import SolveOutcome
from median_string.instance import ProblemInstance

from .problem import EvalResult, Problem

ROOT = Path(__file__).resolve().parent.parent


def get_evaluate():
    """Evaluation backend: local subprocess, or Modal when AUTORESEARCH_EVAL=modal."""
    if os.environ.get("AUTORESEARCH_EVAL") == "modal":
        from .modal_eval import remote_evaluate
        return remote_evaluate
    return evaluate_in_subprocess


@dataclass
class CandidateRun:
    outcomes: list[SolveOutcome] = field(default_factory=list)  # one per instance, in order
    cpu_ms: dict[str, float] = field(default_factory=dict)
    error: str = ""  # whole-run failure: load error, crash, timeout or unreadable response


def run_candidate(source: str, instances: list[ProblemInstance], budget_ms: int | None,
                  timeout: float) -> CandidateRun:
    """Run `source` on the public view of each instance in a `candidate_runner` process and return
    its raw outputs for scoring here. The child gets no `AUTORESEARCH_*` variables (hidden seeds)."""
    views = [i.public_view() for i in instances]
    request = {"source": source, "budget_ms": budget_ms,
               "instances": [{"name": v.name, "strings": v.strings, "alphabet": v.alphabet,
                              "target_length": v.target_length, "metric": v.metric} for v in views]}
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTORESEARCH_")}
    with tempfile.TemporaryDirectory(prefix="autoresearch-candidate-") as tmp:
        req, resp = Path(tmp) / "request.json", Path(tmp) / "response.json"
        req.write_text(json.dumps(request))
        cmd = [sys.executable, "-m", "autoresearch.candidate_runner", str(req), str(resp)]
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return CandidateRun(error=f"Timeout: candidate process exceeded {timeout:.0f}s")
        if proc.returncode != 0 or not resp.exists():
            tail = " | ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-5:])
            return CandidateRun(error=f"Candidate process crashed (exit {proc.returncode}): {tail}")
        try:
            data = json.loads(resp.read_text())
        except json.JSONDecodeError:
            return CandidateRun(error="Candidate process produced no readable response")
    if not isinstance(data, dict):
        return CandidateRun(error="Candidate process produced no readable response")
    if data.get("load_error"):
        return CandidateRun(error=f"Failed to load solver: {data['load_error']}")
    by_name = {r.get("name"): r for r in data.get("instances", []) if isinstance(r, dict)}
    run = CandidateRun()
    for inst in instances:
        r = by_name.get(inst.name)
        if r is None:
            run.outcomes.append(SolveOutcome(error="Crash: candidate process returned no result for this instance"))
            continue
        cand = r.get("candidate")
        error = str(r.get("error") or "")
        if not error and not isinstance(cand, str):
            error = "Candidate must be a string."
        run.outcomes.append(SolveOutcome(candidate=cand if isinstance(cand, str) else "", error=error,
                                         elapsed_seconds=float(r.get("elapsed") or 0.0)))
        run.cpu_ms[inst.name] = float(r.get("cpu_ms") or 0.0)
    return run


def evaluator_hash() -> str:
    """Short hash of the evaluator's own source (benchmark, scoring, runner). Stored in each run's
    config so results produced by different evaluator versions cannot be mixed up silently."""
    files = sorted((ROOT / "median_string").glob("*.py")) + sorted((ROOT / "autoresearch" / "problems").glob("*.py"))
    files += [Path(__file__), ROOT / "autoresearch" / "candidate_runner.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def evaluate_in_subprocess(problem_name: str, problem: Problem, source: str, split: str,
                           budget_ms: int | None = None) -> EvalResult:
    """Run `problem.evaluate(source, split, budget_ms)` in a fresh interpreter; kill it on timeout."""
    timeout = problem.timeouts.get(split, 120.0)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(source)
        src_path = f.name
    cmd = [sys.executable, "-m", "autoresearch.worker", problem_name, split, src_path]
    if budget_ms is not None:
        cmd.append(str(budget_ms))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return _failed(problem, split, f"Timeout: evaluation exceeded {timeout:.0f}s on split '{split}'", time.perf_counter() - t0)
    finally:
        Path(src_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return _failed(problem, split, "Worker crashed: " + " | ".join(tail), time.perf_counter() - t0)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return _failed(problem, split, "Worker produced no JSON result", time.perf_counter() - t0)
    return EvalResult.from_dict(payload)


def _failed(problem: Problem, split: str, msg: str, elapsed: float = 0.0) -> EvalResult:
    baseline = problem.evaluate("def solve(instance):\n    return instance.strings[0]\n", split).baseline
    return EvalResult(split=split, score=float("inf"), baseline=baseline, elapsed=round(elapsed, 2), error=msg)
