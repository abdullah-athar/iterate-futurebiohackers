"""Subprocess entry point: evaluate one candidate on one split and print JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .problem import get_problem


def main() -> None:
    problem_name, split, src_path = sys.argv[1:4]
    budget_ms = int(sys.argv[4]) if len(sys.argv) > 4 else None
    problem = get_problem(problem_name)
    result = problem.evaluate(Path(src_path).read_text(), split, budget_ms)
    sys.stdout.write("\n" + json.dumps(result.to_dict()) + "\n")


if __name__ == "__main__":
    main()
