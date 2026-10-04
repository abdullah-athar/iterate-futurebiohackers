# Rerun: Haiku vs Sonnet vs Opus, 4 agents × 2 generations (mid objective, 2026-10-03)

Same settings as [models4-haiku-sonnet-opus-1003.md](models4-haiku-sonnet-opus-1003.md), run a second time
to check that the ranking holds: `swarm --agents 4 --generations 2 --eval local --max-budget-usd 1 --seed 0
--model <m>`, one model after the other. Dashboard:
[models4-rerun-dashboard-1003.html](models4-rerun-dashboard-1003.html) (the dashed vertical lines on the
tokens / wall-clock / cost tabs mark where each run really stopped).

| | Haiku (run 1 → run 2) | Sonnet (run 1 → run 2) | Opus (run 1 → run 2) |
|---|---|---|---|
| Best on validate (seed 22,816; planted reference 18,787) | 19,794 → 20,415 | 19,280 → 19,172 | **18,525 → 18,718** |
| Gain vs seed | 13.2% → 10.5% | 15.5% → 16.0% | **18.8% → 18.0%** |
| Holdout best (seed 22,575) | 19,786 → 20,393 | 19,200 → 18,920 | **18,441 → 18,628** |
| Agent tokens | 3.22 M → 4.05 M | **0.61 M → 0.67 M** | 0.92 M → 0.96 M |
| Agent cost (Claude Code estimate) | $0.85 → $1.17 | **$0.54 → $0.54** | $1.32 → $1.34 |
| Objective points gained per dollar | 3,553 → 2,050 | **6,599 → 6,716** | 3,259 → 3,049 |
| Wall clock (run start to end, holdout included) | 6:27 → 5:27 | **1:17 → 1:30** | 2:36 → 2:23 |

- The ranking is the same in both runs: Opus finds the best solver (again below the planted reference,
  on validate and on holdout), Sonnet is the cheapest, fastest and most efficient per dollar, Haiku is last
  on quality and is not cheaper than Sonnet.
- Sonnet and Opus are stable to within ~1% in score and cost. Haiku varies the most (−3 points of gain,
  +38% cost), consistent with its long sessions near the 180 s limit.
- One duplicate proposal each for Haiku and Sonnet in run 2 was stopped by the novelty gate before
  evaluation (7 evaluations instead of 8).
