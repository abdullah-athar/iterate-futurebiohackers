# program.md — instructions for a coding agent driving the autoresearch loop

You are the proposer in an automated algorithm-research loop. The loop (not you) owns the
evaluation, the ledger, the novelty gate and the archive. Your job each turn: read the evidence,
form one hypothesis, write one solver file, submit it. Work from `autoresearch-optimizer/`.

## Setup (once)

```sh
uv sync --frozen
uv run python -m autoresearch --run artifacts/runs/<name> init --problem median_string
```

## The loop (repeat until told to stop or ~20 submissions)

1. `uv run python -m autoresearch --run artifacts/runs/<name> status`
   - Shows the global best, the Pareto front, the **suggested mode** (tune / fix_losers /
     new_family / merge), the parent file(s) to start from, a per-instance diagnostics table
     (score vs set-median baseline vs best_known planted score vs best on the front) and the
     ledger of everything tried.
2. Read the parent source and the diagnostics. Decide **one** change and write it as a
   falsifiable hypothesis ("X because Y, expect lower score on Z").
   - Respect the mode. `new_family` means a different algorithm, not a tweak.
   - Do not resubmit an idea the ledger shows was tried; near-duplicates are rejected unevaluated.
3. Write the complete solver to a scratch file, e.g. `artifacts/runs/<name>/scratch/candidate.py`:
   - must define `def solve(instance) -> str`; stdlib + `median_string.metrics` only; deterministic.
   - time budget: whole `screen` split < 20 s, whole `validate` split < 90 s.
4. `uv run python -m autoresearch --run artifacts/runs/<name> submit --file <file> --hypothesis "..." --mode <mode> --parent <ids> --proposer <your-name>`
   - Output: `kept` (new global best or new best on some instance), `evaluated` (valid, no gain),
     `rejected_duplicate`, `rejected_guard` (disallowed import/call), `rejected_screen`
     (invalid/crash/worse than baseline on screen), `failed`.
   - A candidate that beats the global best on `validate` is **re-tested on a fresh `confirm`
     set** before it is accepted; if it scores *worse* than the incumbent there the gain is treated
     as noise/overfit and the candidate is not promoted (verdict `unconfirmed`). Every submission gets a verdict:
     `supported` / `partial` / `falsified` / `unconfirmed` / `inconclusive` / `untested`.
     `status` lists the falsified hypotheses — read them, do not re-propose them, build on *why*
     they failed.
5. Go to 1. The loop's bandit will shift the suggested mode after plateaus; follow it.

## Finish

```sh
uv run python -m autoresearch --run artifacts/runs/<name> report --holdout
uv run python -m autoresearch --run artifacts/runs/<name> best --output median_string/solvers/discovered.py
```

The report includes the trajectory, the front, per-instance bests, prompt-mode statistics and a
held-out evaluation on fresh instances the search never saw. Summarise it for the human.
