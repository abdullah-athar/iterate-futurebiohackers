"""Load candidate solver source and evaluate it in an isolated subprocess with a timeout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

from .problem import EvalResult, Problem

ROOT = Path(__file__).resolve().parent.parent


def load_solve_function(source: str):
    module = types.ModuleType("candidate_solver")
    module.__file__ = "<candidate>"
    exec(compile(source, "<candidate>", "exec"), module.__dict__)  # noqa: S102 - candidates run in a subprocess
    solve = module.__dict__.get("solve")
    if not callable(solve):
        raise TypeError("candidate must define a top-level function `solve(instance) -> str`")
    return solve


def evaluate_in_subprocess(problem_name: str, problem: Problem, source: str, split: str) -> EvalResult:
    """Run `problem.evaluate(source, split)` in a fresh interpreter; kill it on timeout."""
    timeout = problem.timeouts.get(split, 120.0)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(source)
        src_path = f.name
    cmd = [sys.executable, "-m", "autoresearch.worker", problem_name, split, src_path]
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
