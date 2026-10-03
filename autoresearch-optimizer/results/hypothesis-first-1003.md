# Hypothesis-first swarm: cost per agent (2026-10-03)

Three 1-generation smoke runs, 2 Sonnet agents each (`--turn-s 120 --eval local --max-budget-usd 1`).
Costs are Claude Code's `total_cost_usd` per agent. One run per row, so scores are noisy: these
runs check the mechanics and the cost per phase, not solver quality.

| Run | Hypothesis step | Gate (Haiku) | Coding session | Total per agent | Best (validate) | Holdout best (seed 1024, baseline 1147) |
|---|---|---|---|---|---|---|
| `smoke-tcava-1003-185121`, no gate | — | — | $0.11 | **$0.11** | 512 | 935 |
| `hypfirst-tcava-1003-201223`, hypothesis as an agent session | $0.08 (97k input tokens) | $0.005 | $0.18 (resumed session) | $0.26 | 515 | 940 |
| `hypfirst2-tcava-1003-201626`, hypothesis as one tool-less call | **$0.015 (2.6k input tokens)** | $0.005 | $0.11 | **$0.13** | 524 | 1018 |

What changed between the two hypothesis-first runs: the hypothesis was first written by a full
Claude Code session (whose built-in instructions and tool turns make up most of its ~100k input
tokens), then by one call with no tools, a one-line system prompt and STATUS.md inline.

Break-even of the final version: every distinct idea costs about $0.02 more than without the gate,
and every repeated idea saves its coding session (about $0.11). The gate pays for itself once
roughly 1 idea in 6 is a repeat (0.11 = 0.02 + (1 - d) × 0.11 gives d ≈ 0.18). In these runs the
2 agents had different research directions and no idea was repeated; the saving needs runs with
more agents than directions (the default is 32 agents and 10 directions) to be measured.

Judge check (one Haiku call, 3.8k input tokens, $0.011): of 4 sample hypotheses, it rejected a
rewording of a ledger idea (multi-start from top-4 set medians) and a second Myers bit-parallel
kernel, and kept the 2 distinct ideas.

Reproduce: `uv run python -m autoresearch --run artifacts/runs/<name> swarm --agents 2 --turn-s 120 --generations 1 --eval local --max-budget-usd 1 --hypothesis-first`
