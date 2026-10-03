"""Evaluate candidates on Modal CPUs while the proposing agents stay on the local machine.

Deploy (the swarm does this itself at start):  uv run modal deploy -m autoresearch.modal_eval
Then set AUTORESEARCH_EVAL=modal so `submit`, `try` and holdout evaluation run remotely.
Confirm/holdout seeds come from the `autoresearch-heldout` secret (AUTORESEARCH_CONFIRM_SEED,
AUTORESEARCH_HOLDOUT_SEED), which only the evaluators see.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "autoresearch-eval"
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rapidfuzz==3.14.6")
    .env({"PYTHONPATH": "/root/ar"})
    .add_local_dir(ROOT, "/root/ar", ignore=["**/.venv", "**/__pycache__", "artifacts", "notebooks", "data", "**/.DS_Store"])
)
app = modal.App(APP_NAME, image=image)
heldout = [modal.Secret.from_name("autoresearch-heldout")]


@app.function(cpu=1.0, memory=1024, timeout=600, max_containers=96, secrets=heldout)
def evaluate(problem_name: str, source: str, split: str, budget_ms: int | None) -> dict:
    """One split in a fresh subprocess (same containment as local evaluation)."""
    from autoresearch.problem import get_problem
    from autoresearch.sandbox import evaluate_in_subprocess

    return evaluate_in_subprocess(problem_name, get_problem(problem_name), source, split, budget_ms).to_dict()


@app.function(cpu=1.0, memory=1024, timeout=900, max_containers=96, secrets=heldout)
def evaluate_cascade(problem_name: str, source: str, budget_ms: int | None) -> dict:
    """screen -> validate -> confirm for one candidate (see loop.evaluate_candidate)."""
    from autoresearch.loop import evaluate_candidate
    from autoresearch.sandbox import evaluate_in_subprocess

    return evaluate_candidate(problem_name, source, budget_ms, evaluate_in_subprocess)


def remote_evaluate(problem_name, problem, source, split, budget_ms):
    """Drop-in for sandbox.evaluate_in_subprocess that runs on the deployed app."""
    from .problem import EvalResult

    fn = modal.Function.from_name(APP_NAME, "evaluate")
    return EvalResult.from_dict(fn.remote(problem_name, source, split, budget_ms))


def remote_cascade_map(problem_name: str, sources: list[str], budget_ms: int | None):
    """Evaluate many candidates in parallel containers; yields (index, evals | Exception)."""
    fn = modal.Function.from_name(APP_NAME, "evaluate_cascade")
    args = [(problem_name, s, budget_ms) for s in sources]
    yield from enumerate(fn.starmap(args, order_outputs=True, return_exceptions=True))


def deploy() -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "modal", "deploy", "-m", "autoresearch.modal_eval"], cwd=ROOT, check=True)
