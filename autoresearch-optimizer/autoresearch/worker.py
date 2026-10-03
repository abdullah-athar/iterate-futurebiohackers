"""Subprocess entry point: evaluate one candidate on one split and print JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .problem import get_problem


def main() -> None:
    problem_name, split, src_path = sys.argv[1:4]
    problem = get_problem(problem_name)
    result = problem.evaluate(Path(src_path).read_text(), split)
    sys.stdout.write("\n" + json.dumps(result.to_dict()) + "\n")


if __name__ == "__main__":
    main()
