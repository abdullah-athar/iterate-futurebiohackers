# Autoresearch optimizer

Shared workspace for building an autoresearch optimizer: propose algorithm changes, evaluate them against a reproducible baseline, and keep improvements with an experiment record.

This idea fits **Track 1: AI Automated Discovery of Algorithms — Build Your Own Autoresearch Framework**. The benchmark, objective, and search strategy are team decisions; claim a workstream in [TASKS.md](TASKS.md). The team must choose one official track for its submission.

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
  scripts/          # runnable experiments and optimizer code
  notebooks/        # exploration
  data/             # local inputs, ignored by Git
  artifacts/        # local results, ignored by Git
```

## Autoresearch loop

`autoresearch/` runs a simple hill-climbing loop. Each iteration, Claude rewrites the current champion solver. The new solver is scored in a subprocess with a timeout, and it is kept only if it lowers the total distance on the search suite. At the end, the champion is scored on a held-out suite: the same tier regenerated with seeds offset by 10,000, which the loop never sees.

```sh
just autoresearch-run --proposer mock --iterations 3 --search-tier small     # offline dry run
just autoresearch-run --proposer claude --iterations 10 --search-tier medium # needs ANTHROPIC_API_KEY
just autoresearch-test
```

Each run writes `ledger.jsonl`, `champion.py`, `summary.json`, and every candidate file to `artifacts/autoresearch/<run>/`.

Every component sits behind a small interface in `autoresearch/core.py` and is chosen by name in `autoresearch/config.py`:

| Component | Interface | Implementations |
| --- | --- | --- |
| Proposer | `propose(ctx) -> Candidate` | `claude`, `mock` |
| Evaluator | `evaluate(candidate, suite) -> Result` | `subprocess` |
| Selector | `accept(memory, result)`, `parents(memory)` | `greedy` |
| Memory | `record`, `champion`, `summary` | `jsonl` |
| Budget | `exhausted(memory)` | `iterations` |

To add a variant, such as a population selector or a reflective memory, write a class with the same methods, register it in `REGISTRY`, and select it with `--selector <name>`. Only `autoresearch/loop.py` knows how the components fit together.

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
