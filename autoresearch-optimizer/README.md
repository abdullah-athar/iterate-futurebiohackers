# Autoresearch optimizer

Shared workspace for building an autoresearch optimizer: propose algorithm changes, evaluate them against a reproducible baseline, and keep improvements with an experiment record.

This idea fits **Track 1: AI Automated Discovery of Algorithms — Build Your Own Autoresearch Framework**. The benchmark, objective, and search strategy are team decisions; claim a workstream in [TASKS.md](TASKS.md). This folder is an initial scaffold; the optimizer and benchmark are still to be implemented. The team must choose one official track for its submission.

## Get started

From the repository root, with `uv` and `just` installed:

```sh
just autoresearch-setup
just autoresearch-smoke
```

This workspace uses Python 3.11 and its own `.venv/`, `pyproject.toml`, and committed `uv.lock`. To add dependencies or run experiments, work inside this folder:

```sh
cd autoresearch-optimizer
uv sync --frozen
uv add <package>
uv run python scripts/<experiment>.py
```

Commit both `pyproject.toml` and `uv.lock` when changing dependencies.

## Layout

```text
autoresearch-optimizer/
  README.md         # setup, scope, and collaboration
  TASKS.md          # workstream ownership and next steps
  pyproject.toml    # workspace dependencies
  uv.lock           # reproducible dependency resolution
  autoresearch/     # research loop: propose -> novelty gate -> cascade eval -> archive -> ledger
  autoresearch_viz/ # dashboard that visualises runs and compares flavours (see below)
  scripts/          # runnable experiments and optimizer code
  notebooks/        # exploration
  data/             # local inputs, ignored by Git
  artifacts/        # local results, ignored by Git
```

## Visualise runs and compare flavours

`autoresearch_viz` renders one self-contained HTML dashboard (no network access needed, so it
works in the demo video) from one or more run directories (`artifacts/runs/<name>/ledger.jsonl`).
Each run is a "flavour" of the research loop: a different proposer, prompt policy, archive or
novelty setting on the same problem.

```bash
# compare every run under artifacts/runs
uv run python -m autoresearch_viz render artifacts/runs -o artifacts/viz/dashboard.html --open

# pick runs and give them display names
uv run python -m autoresearch_viz render "claude=artifacts/runs/claude_a" "gemini=artifacts/runs/gemini_a"

# synthetic runs (clearly labelled) to iterate on the dashboard before real runs exist
uv run python -m autoresearch_viz demo --open
```

The dashboard shows: best objective vs evaluations / LLM tokens / wall-clock (with baseline and
planted-optimum lines and the hypothesis behind every improvement), a flavour scoreboard (gain on
the objective split, held-out gain, evaluations, tokens, tokens per 1% gained, duplicates skipped
by the novelty gate), per-instance bars, the outcome mix of proposals, and the full research
trajectory of each run as a lab notebook. Loading is schema-tolerant: missing ledger fields fall
back to sensible defaults.

## Work together

1. Claim an unowned task in [TASKS.md](TASKS.md), and agree on the benchmark and evaluation contract before implementing the search loop.
2. Start a feature branch from the latest `main`, for example `autoresearch/baseline` or `autoresearch/search-loop`.
3. Keep runnable code in `scripts/` and exploration in `notebooks/`. Load relevant shared science skills from the repository's `.agents/skills/` when useful.
4. For every experiment, record the objective, baseline, candidate, seed, data split, compute budget, elapsed time, score, and source commit. Save raw outputs in `artifacts/`; include a concise result summary and reproduction command in the PR.
5. Open a focused PR and get a teammate's approval before merging. Keep credentials and participant-only access codes in local environment variables or an ignored `.env` file.

## Hackathon deliverables

From the organizer information shared with the team:

- Build during the event and credit reused libraries, APIs, and pretrained models.
- Prepare a two-minute project demo, the GitHub repository link, and a short description.
- Submit code and demos by **Sunday, 4 October 2026 at 14:45 BST**.
- Plan the first-round presentation: 1:30 pitch, 1:30 live demo, and 2:00 Q&A.
- Show novelty, performance, interpretability, and ease of use for the autoresearch challenge.

Event information: [Iterate AI x Science Hackathon](https://iterate.inc/london-ai-science).
