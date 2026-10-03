# CIFAR-100 speedrun leaderboard

- Generated 2026-10-03 17:04 from `artifacts/speedrun_runs/registry.jsonl`: 4 variant rows, 1 control rows.
- k = none (k.json missing or invalid; dtime_adj uses each row's k_used where present)
- GPU used: 84.9/1263 min (ledger `artifacts/speedrun_runs/gpu_ledger.jsonl`: 36 containers, 24 guard misses, 7.3 min lost).

Score: `dtime_adj = dtime - dacc_pp / k` (seconds; dtime = variant minus control mean prepare+train time, dacc in accuracy percentage points, both from paired trials); lower is better. dtime_adj is recomputed from the current k for every row with paired data; rows without paired data sit at the bottom sorted by dtime. `!` marks nonfinite > 0 or a verdict that is not qualified/complete; `*` marks a dtime_adj computed from the row's own k_used (no k.json).

## Variants (lower dtime_adj is better)

| rank | label | round | params | hypothesis | n | mean acc % | acc std | dacc pp +- SE | dtime s | dtime_adj s | mean time s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ! e7.5 | calib | `{"epochs":7.5}` | exchange-rate calibration: 1.0 fewer epochs; k = dacc/dtime | 8/8 | 74.75 | 0.12 | -0.38 +- 0.11 | -0.83 |  | 6.52 | 17 warm |  | A100 SXM4 @ 400 W | BELOW 75% |
| 2 | ! e8.0 | calib | `{"epochs":8.0}` | exchange-rate calibration: 0.5 fewer epochs; k = dacc/dtime | 8/8 | 74.97 | 0.31 | -0.16 +- 0.13 | -0.41 |  | 6.95 | 18 warm |  | A100 SXM4 @ 400 W | BELOW 75% |
| 3 | ref-sxm500-n40 | reference | `{}` | reference: current main recipe, cold build, official target | 40/40 | 75.19 | 0.25 |  |  |  | 6.94 | 95 cold |  | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 4 | ref-sxm400-n40 | reference | `{}` | reference: current main recipe, cold build, official target | 40/40 | 75.20 | 0.28 |  |  |  | 7.41 | 133 cold |  | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |

## Controls (power limits seen: A100 SXM4 @ 400 W x1)

| job | label | round | GPU @ power | n | mean acc % | acc std | mean time s | time std | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epochs | control | calib | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.35 | 0.01 | 133 warm | QUALIFIED (>= 75%) |

## GPU hit rates (from the ledger)

| GPU @ power | containers | share | guard misses | GPU min |
| --- | --- | --- | --- | --- |
| A100 SXM4 @ ? | 23 | 64% | 22 | 15.7 |
| A100 PCIe @ ? | 6 | 17% | 0 | 39.9 |
| A100 PCIe @ 300 W | 2 | 6% | 2 | 0.5 |
| A100 SXM4 @ 400 W | 2 | 6% | 0 | 14.7 |
| ? @ ? | 1 | 3% | 0 | 3.4 |
| A100 PCIe / SXM4 @ ? | 1 | 3% | 0 | 4.0 |
| A100 SXM4 @ 500 W | 1 | 3% | 0 | 6.7 |

Guard misses: 24/36 containers (67%), 7.3 GPU min lost; ledger total 84.9 min.

Pareto plot: `pareto.svg` (x = mean prepare+train s, y = mean accuracy %).
