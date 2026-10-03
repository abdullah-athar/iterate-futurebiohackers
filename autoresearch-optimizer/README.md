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

The proposer is always a coding agent (Claude Code here; Codex or any agent that can edit files
works the same way). The loop owns evaluation and the ledger, so an agent can only *propose*.

### How a run works

```mermaid
flowchart LR
    A["1. Decide what to ask for<br/>small tweak, fix weak spots,<br/>new approach, or combine two"]
    B["2. Agents write ideas<br/>many Claude Code agents<br/>on your laptop, in parallel"]
    C["3. Drop repeats<br/>ideas we've already tried<br/>are skipped"]
    D["4. Test in the cloud<br/>each idea runs on Modal,<br/>1 second per test case"]
    E["5. Keep real wins<br/>a new best must also win<br/>on fresh test cases"]
    F["6. Final check<br/>score the best idea on<br/>test cases nobody saw"]
    A --> B --> C --> D --> E
    E -->|time left| A
    E -->|time up| F
```

1. **Decide what to ask for.** The loop tracks which kinds of request have produced
   improvements so far and asks for more of those.
2. **Agents write ideas.** Each agent reads what has been tried, what worked and what failed.
   It then writes one solver and a one-line hypothesis explaining why it should be better.
3. **Drop repeats.** A solver that is basically a copy of an earlier one is thrown out
   without being run.
4. **Test in the cloud.** Every solver runs on Modal, with a fixed 1-second compute limit per
   test case, so ideas compete on quality *within* the same budget.
5. **Keep real wins.** A solver counts as the new best only if its win also holds on fresh
   test cases, so lucky results are filtered out. Every result is logged, and the next round
   sees everything.
6. **Final check.** When time runs out, the best solver is scored on a hidden set of test
   cases that the search never used.

You can watch all of this live in the dashboard (`just autoresearch-viz serve ...`).

### Budgets

| Budget | Default | Where to change |
| --- | --- | --- |
| CPU per `solve()` call, per instance | **1000 ms**; over 1250 ms the instance is invalid | `swarm --budget-ms` (new run) / `config.json` |
| One agent session | **180 s** wall clock, then the agent is stopped and its last files are used | `swarm --turn-s` |
| Whole run | **20 min** wall clock; no new generation starts if one cannot finish | `swarm --budget-min` |
| Agents per generation | **32** | `swarm --agents` |
| Evaluation containers | up to 96 in parallel, 1 CPU each | `autoresearch/modal_eval.py` |

Solvers see their budget as `instance.time_budget_ms` and should return their best answer
before it runs out. The evaluator enforces it with a CPU timer (`SIGPROF`) that solver code
cannot catch with `except Exception`.

### Commands

```sh
# one-time: Modal login (uv run modal setup) and the hidden held-out seeds
uv run modal secret create autoresearch-heldout AUTORESEARCH_CONFIRM_SEED=<int> AUTORESEARCH_HOLDOUT_SEED=<int>

# from the repo root
just autoresearch-swarm-smoke                                   # 2 agents, 1 generation, ~1-2 min
just autoresearch-swarm --run artifacts/runs/swarm-1            # 32 agents/generation, 20 min
just autoresearch-swarm --run artifacts/runs/swarm-1 --agents 8 --budget-min 10 --model opus
just autoresearch-viz serve artifacts/runs/swarm-1 --open      # live dashboard (run in a second terminal)

# single-agent mode (tell the agent: "read autoresearch/program.md and start a research run")
just autoresearch --run artifacts/runs/demo init
just autoresearch --run artifacts/runs/demo status
just autoresearch --run artifacts/runs/demo try --file cand.py
just autoresearch --run artifacts/runs/demo submit --file cand.py --hypothesis "..." --mode fix_losers --parent 2
just autoresearch --run artifacts/runs/demo report --holdout
```

Set `AUTORESEARCH_EVAL=modal` to send single-agent `submit`/`try` evaluations to Modal too
(`uv run modal deploy -m autoresearch.modal_eval` first; the swarm deploys it itself).

A run directory `artifacts/runs/<name>/` holds:
- `ledger.jsonl`: hypothesis, parents, mode, status, verdict, per-instance scores, CPU ms, agent tokens and cost;
- `candidates/NNNN.py`;
- `events.jsonl`: swarm progress;
- `holdout.json`;
- `agents/`: each agent's Claude Code JSON output;
- `report.md`.

### What is hard about autoresearch, and what we did about it

Plain-English version. Each item is a known weakness of LLM-driven research loops (FunSearch,
AlphaEvolve, OpenEvolve, ShinkaEvolve, GEPA, AI-Scientist-style agents), followed by how this
loop handles it.

1. **A single score hides what went wrong.**
   - *The problem:* most loops tell the model "your solver scored 518", with no sign of which
     inputs it fails on or why.
   - *What we do:* every agent gets a per-instance table: its score against the simple baseline,
     against the planted answer, and against the best any candidate has reached on that input,
     plus CPU time and the input's properties (noise level, indels, alphabet). It must state a
     falsifiable hypothesis ("X because Y, expect lower score on Z") before it writes code.
2. **Models keep re-proposing the same idea.**
   - *The problem:* LLMs drift back to the obvious change, and every repeat costs tokens and compute.
   - *What we do:* a novelty gate compares each new solver with every earlier one, after stripping
     comments and renaming variables. Near-copies are rejected *before* any evaluation, including
     copies of what another agent proposed in the same generation. Each agent is also told what
     the other agents are working on, and that it should skip the most obvious next step.
3. **"Improvements" that are really luck.**
   - *The problem:* try hundreds of variants and some will win on the test set by chance.
   - *What we do:* a claimed new best must also hold up on a second set of fresh instances
     (*confirm*) before it counts; otherwise it is marked `unconfirmed`. Final numbers come from a
     *holdout* set the search never sees. On Modal, the seeds that generate those sets live in a
     secret only the evaluators can read, so an agent cannot regenerate them and hard-code answers.
4. **Good ideas get thrown away because they lose on average.**
   - *The problem:* keeping only the single best solver discards one that is excellent on some inputs.
   - *What we do:* a Pareto archive keeps every solver that is best on at least one instance. The
     `merge` mode hands an agent two such solvers and asks it to combine them.
5. **The loop doesn't know when to stop tweaking.**
   - *The problem:* without guidance, agents make small edits to the leader long after that has
     stopped paying off.
   - *What we do:* a bandit tracks which kind of request (`tune`, `fix_losers`, `new_family`,
     `merge`) has been producing gains and gives those modes more agents. After a run of proposals
     with no gain, `tune` is switched off so agents have to try something different.
6. **Unlimited compute makes results meaningless.**
   - *The problem:* with no time limit, brute force wins and the benchmark saturates. Our first
     solver already matched the planted answer everywhere.
   - *What we do:* every `solve()` call gets a hard 1000 ms CPU budget per instance. Agents must
     find algorithms that are both good and fast, which separates ideas far better.
7. **Agents can game their own evaluation.**
   - *The problem:* an agent that grades itself, or can read the test generator, can fake progress.
   - *What we do:*
     - agents only return source code;
     - the harness re-scores it in a separate process or a separate Modal container;
     - an import guard blocks `os`, `subprocess`, the benchmark generator, `exec` and `open`;
     - only the orchestrator writes the ledger.
8. **Serial loops are slow.**
   - *The problem:* one agent at a time gives roughly one hypothesis a minute.
   - *What we do:* N agents work in parallel each generation, and every evaluation runs in its own
     Modal container. Generations stay in sync so each one builds on all previous results.
9. **Long runs forget and are hard to watch.**
   - *What we do:* the append-only ledger is the memory. Falsified hypotheses are fed back as
     "don't repeat this; build on why it failed". `events.jsonl` and the live dashboard show what
     every agent is doing while the run is in progress.

**Not solved yet:**
- Agents given the same evidence still tend to converge on similar ideas.
- Each generation waits for its slowest agent (capped by the turn limit).
- Modal CPUs are about 1.5x slower and noisier than a laptop, so a solver right at its budget can
  flip between valid and invalid.
- Many parallel sessions on a Claude subscription can hit usage limits.
- The reported dollar cost is Claude Code's own estimate.

Adding another problem means writing one class that implements `autoresearch/problem.py::Problem`
(`describe`, `seed_source`, `evaluate(source, split, budget_ms)`) and registering it in `PROBLEMS`.

Run the tests with `just autoresearch-test`.

## Layout

```text
autoresearch-optimizer/
  README.md         # setup, scope, and collaboration
  TASKS.md          # workstream ownership and next steps
  pyproject.toml    # workspace dependencies
  uv.lock           # reproducible dependency resolution
  autoresearch/     # research loop: propose -> novelty gate -> cascade eval -> archive -> ledger
  autoresearch_viz/ # live / static dashboard for runs (see below)
  median_string/    # benchmark, metrics, baseline solvers, evaluator
  scripts/          # runnable experiments and tests
  notebooks/        # exploration
  data/             # local inputs, ignored by Git
  artifacts/        # local results, ignored by Git
```

## Visualise runs (`autoresearch_viz/`)

A self-contained HTML dashboard: no network access is needed, so it also works offline in a demo.
It shows:
- a live "Now" panel: generation, agents back, elapsed time, cost, latest events;
- best objective against evaluations, agent tokens and wall clock, with the hypothesis behind
  every improvement;
- a scoreboard per run, including held-out gain;
- per-instance bars;
- the outcome mix of all proposals;
- the trajectory, with each proposal's verdict.

```sh
just autoresearch-viz serve artifacts/runs/swarm-1 --open          # live: refreshes every 3 s during a run
just autoresearch-viz render artifacts/runs -o artifacts/viz/all.html   # static snapshot comparing all runs
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
