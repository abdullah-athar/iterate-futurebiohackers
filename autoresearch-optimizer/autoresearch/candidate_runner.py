"""Untrusted side of an evaluation: run candidate code in a process that never scores anything.

The trusted side (`sandbox.run_candidate`) starts this module with a request file and a response
file. The request holds the candidate source, the CPU budget and the *public view* of each
instance (name, strings, alphabet, target_length, metric) and nothing else: no planted answer,
reference score or generator seed, and the `AUTORESEARCH_*` environment variables are stripped.
The response is one record per instance: the returned string (or an error), wall time and CPU
time. Validation, distances and baselines are computed by the parent from its own copy of the
inputs, so nothing this process returns, prints or mutates can change a score.
"""

from __future__ import annotations

import json
import signal
import sys
import time
import types
from pathlib import Path

from median_string.instance import ProblemInstance

BUDGET_GRACE = 1.25  # an instance is invalid once solve() uses more than budget * grace of CPU


class _OverBudget(BaseException):
    """Raised by the CPU timer inside a candidate; a BaseException so `except Exception` can't swallow it."""


def _on_timer(signum, frame):
    raise _OverBudget


def load_solve_function(source: str):
    module = types.ModuleType("candidate_solver")
    module.__file__ = "<candidate>"
    exec(compile(source, "<candidate>", "exec"), module.__dict__)  # noqa: S102 - this process exists to run untrusted code
    solve = module.__dict__.get("solve")
    if not callable(solve):
        raise TypeError("candidate must define a top-level function `solve(instance) -> str`")
    return solve


def solve_one(solve, instance: ProblemInstance, budget_ms: int | None) -> dict:
    """Run solve(instance) under a per-call CPU limit (SIGPROF) and a wall-clock backstop (SIGALRM)."""
    rec = {"name": instance.name, "candidate": None, "error": "", "elapsed": 0.0, "cpu_ms": 0.0}
    limit = budget_ms * BUDGET_GRACE / 1000 if budget_ms else None
    handlers = signal.signal(signal.SIGPROF, _on_timer), signal.signal(signal.SIGALRM, _on_timer)
    over, out = False, None
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        try:
            if limit:
                signal.setitimer(signal.ITIMER_PROF, limit)
                signal.setitimer(signal.ITIMER_REAL, 3 * limit + 1)  # sleeping/blocked solvers
            out = solve(instance)
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
            signal.setitimer(signal.ITIMER_REAL, 0)
    except _OverBudget:
        over = True
    except BaseException as e:  # noqa: BLE001 - a crash (or exit()) is an invalid result for this instance only
        rec["error"] = f"Crash: {type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGPROF, handlers[0])
        signal.signal(signal.SIGALRM, handlers[1])
    used = time.process_time() - c0
    rec["elapsed"] = time.perf_counter() - t0
    rec["cpu_ms"] = round(1000 * used, 1)
    if over or (limit and used > limit):
        rec["error"] = f"Crash: TimeoutError: over budget: {1000 * used:.0f} ms CPU > {budget_ms} ms x {BUDGET_GRACE}"
    elif not rec["error"]:
        if isinstance(out, str):
            rec["candidate"] = out
        else:
            rec["error"] = f"Candidate must be a string, got {type(out).__name__}."
    return rec


def main() -> None:
    req_path, resp_path = sys.argv[1:3]
    req = json.loads(Path(req_path).read_text())
    budget_ms = req.get("budget_ms")
    resp: dict = {"instances": []}
    try:
        solve = load_solve_function(req["source"])
    except BaseException as e:  # noqa: BLE001 - a load failure fails the whole split
        resp["load_error"] = f"{type(e).__name__}: {e}"
    else:
        for d in req["instances"]:
            inst = ProblemInstance(name=d["name"], strings=list(d["strings"]), alphabet=d["alphabet"],
                                   target_length=d.get("target_length"), metric=d.get("metric", "levenshtein"),
                                   time_budget_ms=budget_ms)
            resp["instances"].append(solve_one(solve, inst, budget_ms))
    Path(resp_path).write_text(json.dumps(resp))


if __name__ == "__main__":
    main()
