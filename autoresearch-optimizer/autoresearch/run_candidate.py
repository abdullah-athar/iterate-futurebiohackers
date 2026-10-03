"""Evaluate one candidate solver file and print the result as JSON (run in a subprocess).

Usage: python -m autoresearch.run_candidate <solver.py> <tier> <seed_offset>
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys

from median_string.base_solver import BaseSolver
from median_string.evaluator import Evaluator

from .core import SuiteSpec
from .suites import build_suite


def load_solver(path: str) -> BaseSolver:
    spec = importlib.util.spec_from_file_location("candidate_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseSolver)
        and obj is not BaseSolver
        and not inspect.isabstract(obj)
        and obj.__module__ == module.__name__
    ]
    if not classes:
        raise ValueError("No BaseSolver subclass defined in candidate module")
    return classes[-1]()


def main() -> None:
    path, tier, offset = sys.argv[1], sys.argv[2], int(sys.argv[3])
    solver = load_solver(path)
    summary = Evaluator().evaluate_solver(solver, benchmark=build_suite(SuiteSpec(tier, offset)), verbose=False)
    print(json.dumps(summary.to_dict()))


if __name__ == "__main__":
    main()
