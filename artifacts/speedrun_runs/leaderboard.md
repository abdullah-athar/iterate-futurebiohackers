# CIFAR-100 speedrun leaderboard

- Generated 2026-10-03 17:42 from `artifacts/speedrun_runs/registry.jsonl`: 48 variant rows, 19 control rows.
- k = 0.445 pp/s from `artifacts/speedrun_runs/k.json` (calib/epochs: control 8.5 vs 8.0 (dacc -0.16 pp, dtime -0.41 s) vs 7.5 (dacc -0.38 pp, dtime -0.83 s), n=8 paired, A100-SXM4-80GB 400 W, least squares through the origin, 2026-10-03T17:35:00)
- GPU used: 294.0/1263 min (ledger `artifacts/speedrun_runs/gpu_ledger.jsonl`: 63 containers, 30 guard misses, 7.8 min lost).

Score: `dtime_adj = dtime - dacc_pp / k` (seconds; dtime = variant minus control mean prepare+train time, dacc in accuracy percentage points, both from paired trials); lower is better. dtime_adj is recomputed from the current k for every row with paired data; rows without paired data sit at the bottom sorted by dtime. `!` marks nonfinite > 0 or a verdict that is not qualified/complete; `*` marks a dtime_adj computed from the row's own k_used (no k.json).

## Variants (lower dtime_adj is better)

| rank | label | round | params | hypothesis | n | mean acc % | acc std | dacc pp +- SE | dtime s | dtime_adj s | mean time s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | w128-256-768 | r1 | `{"widths":[128,256,768]}` | aggressive group 2 cut compensated by group 3 | 8/8 | 75.23 | 0.25 | +0.03 +- 0.15 | -1.16 | -1.24 | 6.18 | 126 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 2 | w128-320-768 | r1 | `{"widths":[128,320,768]}` | move capacity from group 2 (40% of time) to the cheap 3x3... | 8/8 | 75.56 | 0.29 | +0.43 +- 0.15 | -0.01 | -0.99 | 6.91 | 73 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 3 | bias32 | r1 | `{"bias_scaler":32.0}` | BN biases learn too fast at 64x | 8/8 | 75.44 | 0.24 | +0.32 +- 0.14 | +0.01 | -0.70 | 7.48 | 22 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 4 | d3-2-3 | r1 | `{"depths":[3,2,3]}` | drop the residual pair in group 2 (40% of time) | 8/8 | 75.01 | 0.22 | -0.11 +- 0.14 | -0.87 | -0.61 | 6.53 | 40 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 5 | ls0.25 | r1 | `{"label_smoothing":0.25}` | slightly less label smoothing | 8/8 | 75.33 | 0.18 | +0.21 +- 0.09 | +0.00 | -0.47 | 7.44 | 23 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 6 | bn0.7 | r1 | `{"bn_momentum":0.7}` | even faster BN statistics for a short schedule | 8/8 | 75.29 | 0.23 | +0.16 +- 0.10 | -0.03 | -0.40 | 7.24 | 35 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 7 | lr10.8 | r1 | `{"lr":10.8}` | lr x1.2: faster progress per step in a short schedule | 8/8 | 75.30 | 0.26 | +0.17 +- 0.14 | +0.01 | -0.37 | 7.33 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 8 | ls0.35 | r1 | `{"label_smoothing":0.35}` | slightly more label smoothing (100 classes tolerate stron... | 8/8 | 75.27 | 0.28 | +0.14 +- 0.14 | +0.01 | -0.32 | 7.45 | 20 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 9 | bs768 | r1 | `{"batch_size":768}` | smaller batch = more steps per epoch; the sum loss keeps... | 8/8 | 75.55 | 0.35 | +0.28 +- 0.17 | +0.34 | -0.29 | 7.77 | 118 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 10 | bn0.5 | r1 | `{"bn_momentum":0.5}` | slower BN statistics (momentum 0.6 is already very fast) | 8/8 | 75.25 | 0.18 | +0.12 +- 0.09 | +0.00 | -0.27 | 7.27 | 31 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 11 | scale0.1389 | r1 | `{"scaling_factor":0.1388888888888889}` | logit scale x1.25: sharper logits | 8/8 | 75.23 | 0.31 | +0.10 +- 0.16 | -0.04 | -0.26 | 7.30 | 35 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 12 | jitter0.3 | r1 | `{"jitter":0.3}` | strong photometric jitter; may hurt at 8.5 epochs (under-... | 8/8 | 75.38 | 0.19 | +0.11 +- 0.14 | -0.01 | -0.26 | 7.33 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 13 | mom0.8 | r1 | `{"momentum":0.8}` | lower momentum (lr is decoupled, so this changes the effe... | 8/8 | 75.24 | 0.23 | +0.11 +- 0.11 | +0.02 | -0.24 | 7.33 | 15 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 14 | ! d2-3-3 | r1 | `{"depths":[2,3,3]}` | drop the residual pair in group 1 (largest spatial map):... | 8/8 | 74.96 | 0.24 | -0.17 +- 0.08 | -0.59 | -0.21 | 6.80 | 38 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 15 | cutout4 | r1 | `{"cutout":4}` | small cutout as extra regularisation (airbench96 uses cut... | 8/8 | 75.20 | 0.16 | +0.08 +- 0.10 | +0.00 | -0.18 | 6.96 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 16 | jitter0.2 | r1 | `{"jitter":0.2}` | moderate photometric jitter; same mechanism, stronger | 8/8 | 75.34 | 0.23 | +0.08 +- 0.12 | +0.00 | -0.17 | 7.34 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 17 | whiten4 | r1 | `{"whiten_bias_epochs":4}` | train the whitening bias longer | 8/8 | 75.21 | 0.22 | +0.08 +- 0.05 | +0.05 | -0.13 | 7.32 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 18 | translate1 | r1 | `{"translate":1}` | less translation = less regularisation, closer fit in few... | 8/8 | 75.26 | 0.18 | +0.04 +- 0.13 | -0.03 | -0.12 | 7.37 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 19 | jitter0.1 | r1 | `{"jitter":0.1}` | mild per-image brightness/contrast jitter regularises; ac... | 8/8 | 75.31 | 0.32 | +0.04 +- 0.12 | -0.02 | -0.11 | 7.32 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 20 | lr7.2 | r1 | `{"lr":7.2}` | lr x0.8: the 8.5-epoch schedule may be over-aggressive | 8/8 | 75.17 | 0.25 | +0.04 +- 0.12 | +0.02 | -0.08 | 7.33 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 21 | wd0.0168 | r1 | `{"weight_decay":0.0168}` | wd x1.4: stronger regularisation; with lookahead EMA may... | 8/8 | 75.33 | 0.15 | +0.03 +- 0.12 | +0.01 | -0.06 | 7.30 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 22 | warmup0.15 | r1 | `{"warmup":0.15}` | shorter warmup leaves more steps at high lr | 8/8 | 75.15 | 0.26 | +0.03 +- 0.15 | +0.01 | -0.05 | 7.45 | 22 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 23 | ! e8.0 | calib | `{"epochs":8.0}` | exchange-rate calibration: 0.5 fewer epochs; k = dacc/dtime | 8/8 | 74.97 | 0.31 | -0.16 +- 0.13 | -0.41 | -0.05 | 6.95 | 18 warm |  | A100 SXM4 @ 400 W | BELOW 75% |
| 24 | final0.15 | r1 | `{"final_lr":0.15}` | decay less: the lookahead EMA already averages the noise | 8/8 | 75.13 | 0.14 | +0.00 +- 0.12 | -0.01 | -0.01 | 6.99 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 25 | w96-384-576 | r1 | `{"widths":[96,384,576]}` | group 1 is 36% of forward time at 31x31; narrowing it sav... | 8/8 | 75.05 | 0.20 | -0.26 +- 0.10 | -0.58 | +0.00 | 6.81 | 53 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 26 | ! e7.5 | calib | `{"epochs":7.5}` | exchange-rate calibration: 1.0 fewer epochs; k = dacc/dtime | 8/8 | 74.75 | 0.12 | -0.38 +- 0.11 | -0.83 | +0.01 | 6.52 | 17 warm |  | A100 SXM4 @ 400 W | BELOW 75% |
| 27 | translate3 | r1 | `{"translate":3}` | more translation = more regularisation; accuracy gain buy... | 8/8 | 75.22 | 0.23 | +0.01 +- 0.12 | +0.03 | +0.02 | 7.43 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 28 | w128-384-512 | r1 | `{"widths":[128,384,512]}` | narrower head group: small time saving, tests whether gro... | 8/8 | 75.01 | 0.23 | -0.19 +- 0.14 | -0.39 | +0.02 | 6.94 | 110 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 29 | no-cudagraphs | r1 | `{"compile":"max-autotune-no-cudagraphs"}` | cheaper build; measures what CUDA graphs are worth per step | 8/8 | 75.13 | 0.22 | +0.00 +- 0.00 | +0.05 | +0.05 | 7.39 | 44 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 30 | ! w96-320-640 | r1 | `{"widths":[96,320,640]}` | narrow groups 1 and 2, widen group 3: time saving with pa... | 8/8 | 74.75 | 0.17 | -0.37 +- 0.08 | -0.76 | +0.08 | 6.16 | 77 warm | 0 | A100 SXM4 @ 500 W | BELOW 75% |
| 31 | ! silu | r1 | `{"activation":"silu"}` | SiLU is cheaper than erf-GELU and often equal in accuracy... | 8/8 | 75.00 | 0.38 | -0.13 +- 0.18 | -0.21 | +0.09 | 7.24 | 69 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 32 | final0.03 | r1 | `{"final_lr":0.03}` | decay further at the end for a sharper final convergence | 8/8 | 75.07 | 0.34 | -0.05 +- 0.09 | -0.01 | +0.11 | 6.99 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 33 | ! silu-gather | r1 | `{"activation":"silu","crop_gather":true}` | cheap-swaps bundle: SiLU plus the sync-free vectorised crop | 8/8 | 74.95 | 0.25 | -0.18 +- 0.13 | -0.23 | +0.16 | 7.11 | 37 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 34 | warmup0.3 | r1 | `{"warmup":0.3}` | longer warmup stabilises the fp16 early phase | 8/8 | 75.05 | 0.36 | -0.08 +- 0.18 | -0.01 | +0.17 | 6.98 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 35 | bs1536 | r1 | `{"batch_size":1536}` | larger batch = fewer steps and less Python overhead per e... | 8/8 | 75.11 | 0.25 | -0.16 +- 0.09 | -0.18 | +0.17 | 7.25 | 115 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 36 | scale0.0889 | r1 | `{"scaling_factor":0.08888888888888889}` | logit scale x0.8: softer logits with label smoothing 0.3 | 8/8 | 75.02 | 0.20 | -0.10 +- 0.07 | -0.02 | +0.21 | 7.32 | 36 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 37 | bias128 | r1 | `{"bias_scaler":128.0}` | BN biases can learn faster still (airbench uses 64x on CI... | 8/8 | 75.03 | 0.36 | -0.09 +- 0.19 | +0.01 | +0.22 | 7.47 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 38 | mom0.9 | r1 | `{"momentum":0.9}` | higher momentum smooths the noisy few-epoch trajectory | 8/8 | 75.19 | 0.22 | -0.11 +- 0.16 | +0.01 | +0.24 | 7.30 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 39 | whiten2 | r1 | `{"whiten_bias_epochs":2}` | freeze the whitening bias earlier: the bias-grad graph ru... | 8/8 | 75.15 | 0.18 | -0.15 +- 0.12 | -0.03 | +0.30 | 7.29 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 40 | ema3 | r1 | `{"ema_every":3}` | more frequent lookahead averaging | 8/8 | 75.16 | 0.22 | -0.14 +- 0.14 | +0.03 | +0.35 | 7.35 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 41 | ema10 | r1 | `{"ema_every":10}` | less frequent lookahead averaging: fewer EMA kernels, may... | 8/8 | 75.11 | 0.14 | -0.18 +- 0.14 | +0.01 | +0.42 | 7.33 | 15 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 42 | ! w64-384-576 | r1 | `{"widths":[64,384,576]}` | halve group 1; big time saving, accuracy risk | 8/8 | 74.37 | 0.22 | -0.94 +- 0.11 | -1.58 | +0.54 | 5.82 | 55 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 43 | ! cutout8 | r1 | `{"cutout":8}` | larger cutout; likely too strong for 8.5 epochs | 8/8 | 74.70 | 0.20 | -0.42 +- 0.11 | +0.01 | +0.96 | 6.96 | 10 warm | 0 | A100 SXM4 @ 500 W | BELOW 75% |
| 44 | ! ema-off | r1 | `{"ema_every":0}` | no lookahead EMA: saves the EMA kernels (0.33 ms x 82) at... | 8/8 | 74.65 | 0.45 | -0.48 +- 0.19 | +0.01 | +1.09 | 7.47 | 21 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 45 | ! d3-3-2 | r1 | `{"depths":[3,3,2]}` | drop the residual pair in group 3 (cheapest group): small... | 8/8 | 74.19 | 0.13 | -0.94 +- 0.08 | -0.33 | +1.78 | 7.11 | 57 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 46 | ! wd0.0084 | r1 | `{"weight_decay":0.0084}` | wd x0.7: less regularisation for a short schedule | 8/8 | 74.50 | 0.20 | -0.80 +- 0.14 | +0.02 | +1.82 | 7.31 | 21 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 47 | ref-sxm500-n40 | reference | `{}` | reference: current main recipe, cold build, official target | 40/40 | 75.19 | 0.25 |  |  |  | 6.94 | 95 cold |  | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 48 | ref-sxm400-n40 | reference | `{}` | reference: current main recipe, cold build, official target | 40/40 | 75.20 | 0.28 |  |  |  | 7.41 | 133 cold |  | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |

## Controls (power limits seen: A100 SXM4 @ 400 W x16, A100 SXM4 @ 500 W x3)

| job | label | round | GPU @ power | n | mean acc % | acc std | mean time s | time std | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epochs | control | calib | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.35 | 0.01 | 133 warm | QUALIFIED (>= 75%) |
| aug-translate | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.22 | 0.21 | 7.40 | 0.04 | 18 warm | QUALIFIED (>= 75%) |
| aug-cutout | control | r1 | A100 SXM4 @ 500 W | 8/8 | 75.13 | 0.22 | 6.95 | 0.02 | 12 warm | QUALIFIED (>= 75%) |
| arch-depths-a | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.39 | 0.02 | 17 warm | QUALIFIED (>= 75%) |
| arch-widths-a | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.31 | 0.25 | 7.40 | 0.14 | 17 warm | QUALIFIED (>= 75%) |
| aug-jitter | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.26 | 0.23 | 7.34 | 0.02 | 19 warm | QUALIFIED (>= 75%) |
| opt-lr-mom | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.31 | 0.02 | 18 warm | QUALIFIED (>= 75%) |
| opt-mom-wd | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.30 | 0.32 | 7.30 | 0.02 | 24 warm | QUALIFIED (>= 75%) |
| arch-widths-b | control | r1 | A100 SXM4 @ 500 W | 8/8 | 75.13 | 0.22 | 6.92 | 0.02 | 12 warm | QUALIFIED (>= 75%) |
| arch-depths-b | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.44 | 0.03 | 25 warm | QUALIFIED (>= 75%) |
| arch-widths-c | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.20 | 0.36 | 7.34 | 0.01 | 24 warm | QUALIFIED (>= 75%) |
| opt-ls-warmup | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.44 | 0.03 | 23 warm | QUALIFIED (>= 75%) |
| opt-warmup-final | control | r1 | A100 SXM4 @ 500 W | 8/8 | 75.13 | 0.22 | 7.00 | 0.02 | 11 warm | QUALIFIED (>= 75%) |
| opt-scale | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.34 | 0.02 | 17 warm | QUALIFIED (>= 75%) |
| opt-ema-whiten | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.30 | 0.32 | 7.32 | 0.02 | 17 warm | QUALIFIED (>= 75%) |
| opt-whiten-bn | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.27 | 0.02 | 17 warm | QUALIFIED (>= 75%) |
| opt-bias-ema | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.47 | 0.02 | 25 warm | QUALIFIED (>= 75%) |
| sys-compile | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.13 | 0.22 | 7.34 | 0.03 | 27 warm | QUALIFIED (>= 75%) |
| opt-batch | control | r1 | A100 SXM4 @ 400 W | 8/8 | 75.27 | 0.20 | 7.43 | 0.02 | 18 warm | QUALIFIED (>= 75%) |

## GPU hit rates (from the ledger)

| GPU @ power | containers | share | guard misses | GPU min |
| --- | --- | --- | --- | --- |
| A100 SXM4 @ ? | 23 | 37% | 22 | 15.7 |
| A100 SXM4 @ 400 W | 17 | 27% | 0 | 161.8 |
| A100 PCIe @ 300 W | 9 | 14% | 8 | 2.0 |
| A100 PCIe @ ? | 6 | 10% | 0 | 39.9 |
| A100 SXM4 @ 500 W | 4 | 6% | 0 | 33.4 |
| A100 (mixed) @ ? | 2 | 3% | 0 | 33.8 |
| ? @ ? | 1 | 2% | 0 | 3.4 |
| A100 PCIe / SXM4 @ ? | 1 | 2% | 0 | 4.0 |

Guard misses: 30/63 containers (48%), 7.8 GPU min lost; ledger total 294.0 min.

Pareto plot: `pareto.svg` (x = mean prepare+train s, y = mean accuracy %).
