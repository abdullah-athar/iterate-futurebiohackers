# Haiku vs Sonnet, 8 agents, generation 1 (mid objective, 2026-10-03)

Settings: `swarm --agents 8 --generations 2 --eval local --max-budget-usd 1 --seed 0 --model <m>`, default
`median_string` objective (`mid` tier: four DNA instances of 280–688 bp + one 320 aa protein), trusted
evaluator from #20. Dashboard: [models8-haiku-sonnet-1003.html](models8-haiku-sonnet-1003.html).

Only generation 1 is usable. During generation 2 every agent failed before writing code: Sonnet's with
"API Error 400: You have reached your specified API usage limits", Haiku's with "Not logged in" (the
signed-in account changed mid-run). The Opus run hit the usage limit on every call and produced no
proposal, so it is left out. One run per model.

| | Haiku | Sonnet |
|---|---|---|
| Best on validate (seed 22,816; planted reference 18,787) | 21,545 | **19,211** |
| Gain vs seed | 5.6% | **15.8%** |
| Gain vs strongest classical solver (`template`, 23,085) | 6.7% | **16.8%** |
| Holdout best (seed 22,575; set median 23,986) | 21,401 | **19,219** |
| Agent tokens (generation 1, 8 sessions) | 3.76 M | 0.97 M |
| Agent cost (Claude Code estimate) | $1.51 | **$0.68** |
| Generation wall clock | 140 s | 68 s |

Haiku is cheaper per token but used about 4× more tokens than Sonnet, so its generation cost more,
took twice as long and ended 12% further from the planted reference.
