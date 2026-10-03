# Haiku vs Sonnet vs Opus, 4 agents × 2 generations (mid objective, 2026-10-03)

Settings: `swarm --agents 4 --generations 2 --eval local --max-budget-usd 1 --seed 0 --model <m>`, run one
after the other on the default `median_string` objective (`mid` tier: four DNA instances of 280–688 bp +
one 320 aa protein) with the trusted evaluator from #20. One run per model, so differences of a few
percent are within noise. Dashboard: [models4-haiku-sonnet-opus-1003.html](models4-haiku-sonnet-opus-1003.html)
("Research progress" card, "vs tokens" tab for objective against tokens). Quality vs efficiency view:
[models4-comparison-1003.html](models4-comparison-1003.html), made with `scripts/compare_models.py`.

| | Haiku | Sonnet | Opus |
|---|---|---|---|
| Best on validate (seed 22,816; planted reference 18,787) | 19,794 | 19,280 | **18,525** |
| Gain vs seed | 13.2% | 15.5% | **18.8%** |
| Gain vs strongest classical solver (`template`, 23,085) | 14.3% | 16.5% | **19.8%** |
| Holdout best (seed 22,575; set median 23,986) | 19,786 | 19,200 | **18,441** |
| Agent tokens (8 sessions) | 3.22 M | **0.61 M** | 0.92 M |
| Agent cost (Claude Code estimate) | $0.85 | **$0.54** | $1.32 |
| Objective points gained per 1k tokens | 0.94 | **5.84** | 4.66 |
| Objective points gained per dollar | 3,553 | **6,599** | 3,259 |
| Wall clock (run start to end, holdout included) | 6 min 27 s | **1 min 17 s** | 2 min 36 s |
| Agent sessions stopped at the 180 s limit | 2 of 8 | 0 | 0 |

- Opus is the only model that beats the planted reference (18,525 vs 18,787 on validate), and it keeps the
  lead on the holdout instances.
- Sonnet is the most efficient: the most improvement per token and per dollar, and the fastest run.
- Haiku is the cheapest per token but used 5× Sonnet's tokens and hit the session time limit twice, so it
  was neither the cheapest run nor the best.
