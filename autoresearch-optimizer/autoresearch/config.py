"""Component registry: map CLI names to factories so components can be mixed and matched.

Add a new implementation by writing a class with the matching interface (see core.py) and
registering a factory here. Each factory receives the parsed CLI args.
"""

from __future__ import annotations

from .evaluators import SubprocessEvaluator
from .loop import IterationBudget
from .memory import JsonlLedger
from .proposers import ClaudeProposer, MockProposer
from .selection import GreedySelector

REGISTRY = {
    "proposer": {
        "claude": lambda a: ClaudeProposer(model=a.model, effort=a.effort, timeout_s=a.timeout),
        "mock": lambda a: MockProposer(seed=a.seed),
    },
    "evaluator": {
        "subprocess": lambda a: SubprocessEvaluator(a.run_dir, timeout_s=a.timeout),
    },
    "selector": {
        "greedy": lambda a: GreedySelector(),
    },
    "memory": {
        "jsonl": lambda a: JsonlLedger(a.run_dir),
    },
    "budget": {
        "iterations": lambda a: IterationBudget(a.iterations),
    },
}


def build(kind: str, name: str, args):
    try:
        return REGISTRY[kind][name](args)
    except KeyError:
        raise SystemExit(f"Unknown {kind} {name!r}; choose from {sorted(REGISTRY[kind])}")
