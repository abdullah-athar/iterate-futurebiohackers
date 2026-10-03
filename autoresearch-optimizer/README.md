# Autoresearch optimizer

Shared workspace for building an autoresearch optimizer: propose algorithm changes, evaluate them against a reproducible baseline, and keep improvements with an experiment record.

This is our entry for **Track 1: AI Automated Discovery of Algorithms — Build Your Own Autoresearch Framework**. `median_string/` is the benchmark + evaluator (Median/Steiner string, lower total edit distance is better); `autoresearch/` is the research loop that drives it. Claim a workstream in [TASKS.md](TASKS.md).

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

## The autoresearch loop (`autoresearch/`)

One loop, two ways to drive it, one ledger:

```sh
cd autoresearch-optimizer && uv sync --frozen

# API mode: an LLM proposes candidates (ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)
uv run python -m autoresearch --run artifacts/runs/demo init --problem median_string
uv run python -m autoresearch --run artifacts/runs/demo run --llm anthropic --steps 20   # or gemini | openai[:model] | mock
uv run python -m autoresearch --run artifacts/runs/demo report --holdout

# Agent mode: Claude Code / Antigravity / Devin / you are the proposer
#   -> tell the agent: "read autoresearch/program.md and start a research run"
uv run python -m autoresearch --run artifacts/runs/demo status          # evidence + suggested mode + parent file
uv run python -m autoresearch --run artifacts/runs/demo submit --file cand.py --hypothesis "..." --mode fix_losers --parent 2
uv run python -m autoresearch --run artifacts/runs/demo best --output median_string/solvers/discovered.py
```

Every turn: **novelty gate → import guard → cascade evaluation (screen → validate) → confirmation re-test → archive → ledger**.
A candidate is a single file defining `solve(instance) -> str`; the loop owns scoring, so the proposer can only *propose*.
`artifacts/runs/<name>/` holds `ledger.jsonl` (hypothesis, parents, mode, status, verdict, per-instance scores, tokens), `candidates/NNNN.py` and `report.md`.

### Ideas we tried to move the needle on

Each maps to a known hard problem in LLM autoresearch; the point of the hackathon entry is that these are cheap and interpretable.

| Hard problem | What the loop does | Where |
| --- | --- | --- |
| Scalar scores are a poor gradient (GEPA) | The proposer sees a **per-instance diagnostics table**: score vs set-median baseline vs planted `best_known` vs best on the front, runtime, metric/noise metadata; it must write a falsifiable hypothesis before code. | `prompts.py`, `status` |
| Good solvers get thrown away because they lose on aggregate | **Pareto-per-instance archive**: anything best on *some* instance is kept and offered for a `merge` with the leader. | `archive.py` |
| Paying tokens + compute for re-proposed ideas (Shinka) | **Rejection sampling / novelty gate**: AST-normalise (strip docstrings, α-rename locals), reject ≥0.95 similar candidates *before* evaluation; the proposer is told why and re-asked. | `novelty.py` |
| Is the improvement real, or selection on noise? | **Confirmation re-test**: a claimed new global best is re-run on a fresh-seed `confirm` set and only promoted if it also wins there (verdict `unconfirmed` otherwise). Final numbers come from a `holdout` set the search never saw. | `loop.py::_confirm`, `benchmarks.py` |
| Do cheap experiments predict expensive ones? | **Cascade** (small `screen` → medium `validate`, early-reject below baseline) and the report prints the **proxy fidelity** (Spearman ρ between screen and validate scores). | `loop.py::_cascade`, `report.py` |
| Research taste: tune vs. investigate vs. abandon | **UCB bandit over prompt modes** (`tune`, `fix_losers`, `new_family`, `merge`) with a **plateau detector** that bans `tune` after N flat proposals. | `loop.py::choose_mode` |
| Hypotheses that never get revised | Every submission gets a **verdict** (`supported` / `partial` / `falsified` / `unconfirmed` / `inconclusive` / `untested`); falsified ones are shown back to the proposer as "do not re-propose, build on why they failed". | `ledger.py`, `prompts.py` |
| Memory decay over long runs | The append-only **ledger** is the memory; `status` replays it, runs are resumable, and any agent can pick up another agent's run. | `ledger.py` |
| Reward hacking / untrustworthy evidence | Candidates return a string; the harness recomputes the score in a **separate subprocess with timeouts**; a static **import guard** blocks `os`/`subprocess`/benchmark-generator imports before evaluation; the ledger is written only by the loop. | `sandbox.py`, `guard.py` |
| Research efficiency | Prompt/completion **tokens are recorded per proposal**; the report prints objective points per 1k tokens and per evaluated proposal. | `report.py` |

Adding another problem = one class implementing `autoresearch/problem.py::Problem` (`describe`, `seed_source`, `evaluate(source, split)`), registered in `PROBLEMS`.

Run the tests with `just autoresearch-test` (or `uv run python scripts/test_autoresearch.py`).

## Layout

```text
autoresearch-optimizer/
  README.md         # setup, scope, and collaboration
  TASKS.md          # workstream ownership and next steps
  pyproject.toml    # workspace dependencies
  uv.lock           # reproducible dependency resolution
  autoresearch/     # the research loop: problem adapter, novelty gate, archive, LLM backends, CLI, program.md
  median_string/    # benchmark, metrics, baseline solvers, evaluator
  scripts/          # runnable experiments and tests
  notebooks/        # exploration
  data/             # local inputs, ignored by Git
  artifacts/        # local results, ignored by Git
```

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
