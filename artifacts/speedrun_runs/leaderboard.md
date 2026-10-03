# CIFAR-100 speedrun leaderboard

- Generated 2026-10-03 19:35 from `artifacts/speedrun_runs/registry.jsonl`: 124 variant rows, 46 control rows.
- k = 0.7 pp/s from `artifacts/speedrun_runs/k.json` (r6/k2 on the promoted recipe (8.75 ep, 24 px first quarter, stack), SXM 400 W, n=8 paired, control 75.37 pct / 5.67 s: e8.25 -0.23 pp / -0.37 s, e9.25 +0.17 pp / +0.24 s; least squares 0.65 pp/s, rounded to 0.7. Earlier bases: 128/384/576 (rounds calib, r1) 0.445; 96/256/768 at 8.5 ep (rounds r3, r4, r5) 1.0., 2026-10-03T19:30:00)
- GPU used: 748.9/1263 min (ledger `artifacts/speedrun_runs/gpu_ledger.jsonl`: 142 containers, 76 guard misses, 19.3 min lost).

Score: `dtime_adj = dtime - dacc_pp / k` (seconds; dtime = variant minus control mean prepare+train time, dacc in accuracy percentage points, both from paired trials); lower is better. dtime_adj is recomputed from the current k for every row with paired data; rows without paired data sit at the bottom sorted by dtime. `!` marks nonfinite > 0 or a verdict that is not qualified/complete; `*` marks a dtime_adj computed from the row's own k_used (no k.json).

## Variants (lower dtime_adj is better)

| rank | label | round | params | hypothesis | n | mean acc % | acc std | dacc pp +- SE | dtime s | dtime_adj s | mean time s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | w128-256-768 | r1 | `{"widths":[128,256,768]}` | aggressive group 2 cut compensated by group 3 | 8/8 | 75.23 | 0.25 | +0.03 +- 0.15 | -1.16 | -1.24 | 6.18 | 126 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 2 | w128-320-768 | r1 | `{"widths":[128,320,768]}` | move capacity from group 2 (40% of time) to the cheap 3x3... | 8/8 | 75.56 | 0.29 | +0.43 +- 0.15 | -0.01 | -0.99 | 6.91 | 73 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 3 | S7-res24s0.25-e8.75 | r5 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":24,"resolution_switch":0.25,"epochs":8.75}` | lead candidate trimmed by a quarter epoch (~5.6 s) in cas... | 16/16 | 75.34 | 0.25 | +0.43 +- 0.07 | -0.38 | -0.81 | 5.70 | 26 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 4 | S7-res24s0.25-e9.0 | r5 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":24,"resolution_switch":0.25,"epochs":9.0}` | r4: +0.42 +- 0.13 pp, -0.19 s (75.40% / 5.77 s); n=16 con... | 16/16 | 75.39 | 0.18 | +0.48 +- 0.08 | -0.22 | -0.71 | 5.86 | 27 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 5 | bias32 | r1 | `{"bias_scaler":32.0}` | BN biases learn too fast at 64x | 8/8 | 75.44 | 0.24 | +0.32 +- 0.14 | +0.01 | -0.70 | 7.48 | 22 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 6 | S7-res24s0.25-e9.25 | r5 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":24,"resolution_switch":0.25,"epochs":9.25}` | lead candidate with a quarter epoch more (~5.93 s) as the... | 16/16 | 75.49 | 0.24 | +0.58 +- 0.09 | -0.06 | -0.64 | 6.02 | 25 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 7 | S6-res24s0.5-e9.5 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"train_resolution":24,"resolution_switch":0.5,"epochs":9.5}` | r3 stack6-res24-s0.5 (74.56% / 5.00 s) with one epoch mor... | 8/8 | 75.05 | 0.27 | +0.07 +- 0.11 | -0.57 | -0.64 | 5.39 | 32 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 8 | ! stack6-res24-s0.5 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"train_resolution":24,"resolution_switch":0.5}` | six gainers plus half the steps at 24 px | 8/8 | 74.56 | 0.29 | -0.43 +- 0.12 | -1.06 | -0.63 | 5.00 | 58 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 9 | d3-2-3 | r1 | `{"depths":[3,2,3]}` | drop the residual pair in group 2 (40% of time) | 8/8 | 75.01 | 0.22 | -0.11 +- 0.14 | -0.87 | -0.61 | 6.53 | 40 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 10 | S7-res24s0.25-e9.0 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":24,"resolution_switch":0.25,"epochs":9.0}` | 24 px for the first quarter (-0.49 s, -0.35 pp alone) pai... | 8/8 | 75.40 | 0.21 | +0.42 +- 0.13 | -0.19 | -0.61 | 5.77 | 95 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 11 | S7-res28-e8.75 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.5,"epochs":8.75}` | same with a quarter epoch more: ~75.5% at ~5.85 s | 8/8 | 75.17 | 0.14 | +0.17 +- 0.09 | -0.36 | -0.53 | 5.94 | 25 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 12 | S7-res28-e9.0 | r5 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.5,"epochs":9.0}` | r4: +0.33 +- 0.11 pp, -0.18 s (75.33%); n=16 confirmation | 16/16 | 75.34 | 0.21 | +0.43 +- 0.08 | -0.09 | -0.52 | 5.92 | 33 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 13 | S7-res28-e9.0 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.5,"epochs":9.0}` | same at 9.0 epochs: ~75.6% at ~6.0 s (time limit check) | 8/8 | 75.33 | 0.18 | +0.33 +- 0.11 | -0.18 | -0.51 | 6.12 | 24 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 14 | ! S7-res28-e8.5 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.5,"epochs":8.5}` | stack4 + jitter + scale x1.25 (momentum 0.8 dropped: -0.1... | 8/8 | 74.95 | 0.24 | -0.04 +- 0.16 | -0.53 | -0.48 | 5.77 | 76 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 15 | S7-res28-e8.75 | r5 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.5,"epochs":8.75}` | r4: +0.17 +- 0.09 pp, -0.36 s (75.17%); n=16 confirmation | 16/16 | 75.17 | 0.21 | +0.26 +- 0.08 | -0.22 | -0.48 | 5.80 | 25 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 16 | ls0.25 | r1 | `{"label_smoothing":0.25}` | slightly less label smoothing | 8/8 | 75.33 | 0.18 | +0.21 +- 0.09 | +0.00 | -0.47 | 7.44 | 23 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 17 | stack6-res28-s0.5 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"train_resolution":28,"resolution_switch":0.5}` | six gainers plus half the steps at 28 px | 8/8 | 75.03 | 0.20 | +0.04 +- 0.11 | -0.41 | -0.45 | 5.66 | 51 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 18 | ! res24-s0.5-stack4 | r3 | `{"train_resolution":24,"resolution_switch":0.5,"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7}` | resizing paid for by the four gainers | 8/8 | 74.34 | 0.32 | -0.64 +- 0.14 | -1.09 | -0.44 | 4.98 | 95 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 19 | S6-res28-e9.0 | r5 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"train_resolution":28,"resolution_switch":0.5,"epochs":9.0}` | r4: +0.22 +- 0.18 pp, -0.11 s (75.22% / 5.90 s); n=16 con... | 16/16 | 75.21 | 0.27 | +0.30 +- 0.11 | -0.09 | -0.40 | 5.92 | 25 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 20 | bn0.7 | r1 | `{"bn_momentum":0.7}` | even faster BN statistics for a short schedule | 8/8 | 75.29 | 0.23 | +0.16 +- 0.10 | -0.03 | -0.40 | 7.24 | 35 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 21 | lr10.8 | r1 | `{"lr":10.8}` | lr x1.2: faster progress per step in a short schedule | 8/8 | 75.30 | 0.26 | +0.17 +- 0.14 | +0.01 | -0.37 | 7.33 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 22 | ! stack6-e8 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"epochs":8.0}` | six gainers with 0.5 epoch trimmed (k): aiming at ~75.35%... | 8/8 | 75.00 | 0.27 | +0.02 +- 0.13 | -0.33 | -0.34 | 5.68 | 17 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 23 | S6-res28-e9.0 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"train_resolution":28,"resolution_switch":0.5,"epochs":9.0}` | r3 stack6-res28-s0.5 (75.03% / 5.66 s) with 0.5 epoch mor... | 8/8 | 75.22 | 0.35 | +0.22 +- 0.18 | -0.11 | -0.34 | 5.90 | 44 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 24 | ls0.35 | r1 | `{"label_smoothing":0.35}` | slightly more label smoothing (100 classes tolerate stron... | 8/8 | 75.27 | 0.28 | +0.14 +- 0.14 | +0.01 | -0.32 | 7.45 | 20 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 25 | bs768 | r1 | `{"batch_size":768}` | smaller batch = more steps per epoch; the sum loss keeps... | 8/8 | 75.55 | 0.35 | +0.28 +- 0.17 | +0.34 | -0.29 | 7.77 | 118 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 26 | S7-e8.5 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"epochs":8.5}` | S7 without resizing: additivity of scale x1.25 on top of... | 8/8 | 75.27 | 0.31 | +0.28 +- 0.14 | -0.00 | -0.28 | 6.01 | 37 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 27 | ! stack6-e7.5 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"epochs":7.5}` | six gainers with 1 epoch trimmed | 8/8 | 74.61 | 0.28 | -0.38 +- 0.17 | -0.66 | -0.28 | 5.40 | 32 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 28 | bn0.5 | r1 | `{"bn_momentum":0.5}` | slower BN statistics (momentum 0.6 is already very fast) | 8/8 | 75.25 | 0.18 | +0.12 +- 0.09 | +0.00 | -0.27 | 7.27 | 31 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 29 | scale0.1389 | r1 | `{"scaling_factor":0.1388888888888889}` | logit scale x1.25: sharper logits | 8/8 | 75.23 | 0.31 | +0.10 +- 0.16 | -0.04 | -0.26 | 7.30 | 35 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 30 | jitter0.3 | r1 | `{"jitter":0.3}` | strong photometric jitter; may hurt at 8.5 epochs (under-... | 8/8 | 75.38 | 0.19 | +0.11 +- 0.14 | -0.01 | -0.26 | 7.33 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 31 | ! S8-res28-e8.75 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.5,"epochs":8.75}` | stack6 + scale x1.25 with 28 px, 8.75 epochs | 8/8 | 74.99 | 0.14 | +0.00 +- 0.10 | -0.25 | -0.25 | 5.77 | 50 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 32 | mom0.8 | r1 | `{"momentum":0.8}` | lower momentum (lr is decoupled, so this changes the effe... | 8/8 | 75.24 | 0.23 | +0.11 +- 0.11 | +0.02 | -0.24 | 7.33 | 15 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 33 | stack4 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7}` | four strongest gainers (old net sum +0.86 pp if additive) | 8/8 | 75.22 | 0.25 | +0.24 +- 0.07 | +0.01 | -0.23 | 6.02 | 36 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 34 | ! d2-3-3 | r1 | `{"depths":[2,3,3]}` | drop the residual pair in group 1 (largest spatial map):... | 8/8 | 74.96 | 0.24 | -0.17 +- 0.08 | -0.59 | -0.21 | 6.80 | 38 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 35 | stack6 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"momentum":0.8}` | six gainers | 8/8 | 75.30 | 0.21 | +0.32 +- 0.13 | +0.11 | -0.21 | 6.12 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 36 | ! res24-s0.5 | r3 | `{"train_resolution":24,"resolution_switch":0.5}` | 24 px for the first half (his 28 px/9 epochs attempt miss... | 8/8 | 74.04 | 0.17 | -0.94 +- 0.12 | -1.13 | -0.19 | 4.90 | 32 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 37 | cutout4 | r1 | `{"cutout":4}` | small cutout as extra regularisation (airbench96 uses cut... | 8/8 | 75.20 | 0.16 | +0.08 +- 0.10 | +0.00 | -0.18 | 6.96 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 38 | jitter0.2 | r1 | `{"jitter":0.2}` | moderate photometric jitter; same mechanism, stronger | 8/8 | 75.34 | 0.23 | +0.08 +- 0.12 | +0.00 | -0.17 | 7.34 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 39 | scale0.1389 | r3 | `{"scaling_factor":0.1388888888888889}` | old net: +0.10 pp at equal time (sharper logits) | 8/8 | 75.23 | 0.29 | +0.24 +- 0.16 | +0.07 | -0.17 | 6.25 | 36 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 40 | ! w64-256-768 | r6 | `{"widths":[64,256,768]}` | priority 5: group 1 at 64 channels (largest maps); on the... | 8/8 | 74.90 | 0.22 | -0.40 +- 0.13 | -0.71 | -0.15 | 4.99 | 92 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 41 | crop-triton | r6 | `{"crop_mode":"triton"}` | PR #5's Triton crop/flip kernel instead of the 25-mask cr... | 8/8 | 75.36 | 0.26 | +0.07 +- 0.08 | -0.04 | -0.14 | 5.71 | 80 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 42 | ! res24-s0.25 | r3 | `{"train_resolution":24,"resolution_switch":0.25}` | 24 px for the first quarter of the steps; low-res graph c... | 8/8 | 74.63 | 0.24 | -0.35 +- 0.14 | -0.49 | -0.14 | 5.54 | 77 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 43 | stack2 | r3 | `{"bias_scaler":32.0,"label_smoothing":0.25}` | the two strongest gainers together: additivity test | 8/8 | 75.21 | 0.18 | +0.22 +- 0.06 | +0.09 | -0.13 | 6.27 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 44 | whiten4 | r1 | `{"whiten_bias_epochs":4}` | train the whitening bias longer | 8/8 | 75.21 | 0.22 | +0.08 +- 0.05 | +0.05 | -0.13 | 7.32 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 45 | bias32 | r3 | `{"bias_scaler":32.0}` | old net: +0.32 pp at equal time; BN biases learn too fast... | 8/8 | 75.09 | 0.27 | +0.11 +- 0.14 | -0.02 | -0.13 | 6.04 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 46 | d2-3-3 | r6 | `{"depths":[2,3,3]}` | depth 2 in group 1 (drop its residual pair) | 8/8 | 75.05 | 0.15 | -0.24 +- 0.11 | -0.47 | -0.12 | 5.23 | 68 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 47 | translate1 | r1 | `{"translate":1}` | less translation = less regularisation, closer fit in few... | 8/8 | 75.26 | 0.18 | +0.04 +- 0.13 | -0.03 | -0.12 | 7.37 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 48 | ! res28-s0.5 | r3 | `{"train_resolution":28,"resolution_switch":0.5}` | 28 px for the first half | 8/8 | 74.70 | 0.26 | -0.28 +- 0.12 | -0.39 | -0.11 | 5.67 | 68 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 49 | jitter0.1 | r1 | `{"jitter":0.1}` | mild per-image brightness/contrast jitter regularises; ac... | 8/8 | 75.31 | 0.32 | +0.04 +- 0.12 | -0.02 | -0.11 | 7.32 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 50 | lr7.2 | r1 | `{"lr":7.2}` | lr x0.8: the 8.5-epoch schedule may be over-aggressive | 8/8 | 75.17 | 0.25 | +0.04 +- 0.12 | +0.02 | -0.08 | 7.33 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 51 | s20x20-24x40-28x60-e9.25 | r7 | `{"res_schedule":[[20,0.2],[24,0.4],[28,0.6]],"epochs":9.25}` | three-stage ramp with half an epoch bought back | 8/8 | 75.20 | 0.38 | -0.09 +- 0.16 | -0.20 | -0.07 | 5.51 | 76 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 52 | ! S7-res28s0.75-e9.0 | r4 | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":28,"resolution_switch":0.75,"epochs":9.0}` | 28 px for three quarters (-0.62 s, -0.73 pp alone) at 9.0... | 8/8 | 74.74 | 0.27 | -0.24 +- 0.10 | -0.31 | -0.07 | 5.65 | 73 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 53 | wd0.0168 | r1 | `{"weight_decay":0.0168}` | wd x1.4: stronger regularisation; with lookahead EMA may... | 8/8 | 75.33 | 0.15 | +0.03 +- 0.12 | +0.01 | -0.06 | 7.30 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 54 | w96-256-640 | r6 | `{"widths":[96,256,640]}` | group 3 at 640 channels | 8/8 | 75.10 | 0.20 | -0.19 +- 0.06 | -0.33 | -0.06 | 5.30 | 113 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 55 | ls0.25 | r3 | `{"label_smoothing":0.25}` | old net: +0.21 pp at equal time | 8/8 | 75.00 | 0.28 | +0.02 +- 0.08 | -0.04 | -0.06 | 6.01 | 17 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 56 | lr10.8 | r3 | `{"lr":10.8}` | old net: +0.17 pp at equal time | 8/8 | 75.02 | 0.15 | +0.04 +- 0.11 | -0.02 | -0.06 | 6.04 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 57 | s20x25-28x50-e9.25 | r7 | `{"res_schedule":[[20,0.25],[28,0.5]],"epochs":9.25}` | two-stage 20/28 ramp with half an epoch bought back | 8/8 | 75.29 | 0.15 | -0.01 +- 0.06 | -0.07 | -0.06 | 5.58 | 68 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 58 | warmup0.15 | r1 | `{"warmup":0.15}` | shorter warmup leaves more steps at high lr | 8/8 | 75.15 | 0.26 | +0.03 +- 0.15 | +0.01 | -0.05 | 7.45 | 22 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 59 | ! e8.0 | calib | `{"epochs":8.0}` | exchange-rate calibration: 0.5 fewer epochs; k = dacc/dtime | 8/8 | 74.97 | 0.31 | -0.16 +- 0.13 | -0.41 | -0.05 | 6.95 | 18 warm |  | A100 SXM4 @ 400 W | BELOW 75% |
| 60 | ! s20x25-24x50 | r7 | `{"res_schedule":[[20,0.25],[24,0.5]]}` | 20 then 24 px over the first half | 8/8 | 74.83 | 0.25 | -0.46 +- 0.11 | -0.71 | -0.05 | 4.97 | 44 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 61 | bn0.7 | r3 | `{"bn_momentum":0.7}` | old net: +0.16 pp at equal time | 8/8 | 75.01 | 0.30 | +0.03 +- 0.14 | -0.01 | -0.04 | 6.08 | 45 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 62 | e8.25 | r6 | `{"epochs":8.25}` | k on the promoted recipe: 0.5 epoch less | 8/8 | 75.13 | 0.27 | -0.23 +- 0.14 | -0.37 | -0.03 | 5.30 | 27 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 63 | jitter0.3 | r3 | `{"jitter":0.3}` | old net: +0.11 pp at equal time; own RNG keeps pairing | 8/8 | 75.03 | 0.27 | +0.05 +- 0.13 | +0.02 | -0.03 | 6.11 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 64 | s24x25-28x50 | r7 | `{"res_schedule":[[24,0.25],[28,0.5]]}` | the promoted 24 px first quarter plus a 28 px second quarter | 8/8 | 75.14 | 0.22 | -0.15 +- 0.09 | -0.24 | -0.02 | 5.54 | 57 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 65 | e9.0 | r3 | `{"epochs":9.0}` | k upward: 0.5 more epochs | 8/8 | 75.34 | 0.15 | +0.36 +- 0.11 | +0.35 | -0.01 | 6.35 | 23 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 66 | final0.15 | r1 | `{"final_lr":0.15}` | decay less: the lookahead EMA already averages the noise | 8/8 | 75.13 | 0.14 | +0.00 +- 0.12 | -0.01 | -0.01 | 6.99 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 67 | e9.25 | r6 | `{"epochs":9.25}` | k on the promoted recipe: 0.5 epoch more | 8/8 | 75.54 | 0.25 | +0.17 +- 0.12 | +0.24 | -0.00 | 5.91 | 27 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 68 | w96-384-576 | r1 | `{"widths":[96,384,576]}` | group 1 is 36% of forward time at 31x31; narrowing it sav... | 8/8 | 75.05 | 0.20 | -0.26 +- 0.10 | -0.58 | +0.00 | 6.81 | 53 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 69 | ! e7.5 | calib | `{"epochs":7.5}` | exchange-rate calibration: 1.0 fewer epochs; k = dacc/dtime | 8/8 | 74.75 | 0.12 | -0.38 +- 0.11 | -0.83 | +0.01 | 6.52 | 17 warm |  | A100 SXM4 @ 400 W | BELOW 75% |
| 70 | translate3 | r1 | `{"translate":3}` | more translation = more regularisation; accuracy gain buy... | 8/8 | 75.22 | 0.23 | +0.01 +- 0.12 | +0.03 | +0.02 | 7.43 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 71 | w128-384-512 | r1 | `{"widths":[128,384,512]}` | narrower head group: small time saving, tests whether gro... | 8/8 | 75.01 | 0.23 | -0.19 +- 0.14 | -0.39 | +0.02 | 6.94 | 110 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 72 | ! w64-d233-triton-e9.5 | r7 | `{"widths":[64,256,768],"depths":[2,3,3],"crop_mode":"triton","epochs":9.5}` | group-1 cuts plus the Triton crop, with 0.75 epoch bought... | 8/8 | 74.91 | 0.18 | -0.38 +- 0.11 | -0.52 | +0.03 | 5.30 | 89 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 73 | ! e8.0 | r3 | `{"epochs":8.0}` | k on Abdullah's 96/256/768 network: 0.5 fewer epochs | 8/8 | 74.63 | 0.27 | -0.35 +- 0.13 | -0.31 | +0.04 | 5.69 | 20 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 74 | no-cudagraphs | r1 | `{"compile":"max-autotune-no-cudagraphs"}` | cheaper build; measures what CUDA graphs are worth per step | 8/8 | 75.13 | 0.22 | +0.00 +- 0.00 | +0.05 | +0.05 | 7.39 | 44 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 75 | scale1.5 | r6 | `{"scaling_factor":0.16666666666666666}` | priority 6: logit scale 1.5/9 (1.25/9 gave +0.24 pp over... | 8/8 | 75.32 | 0.24 | +0.03 +- 0.13 | +0.11 | +0.06 | 5.85 | 116 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 76 | d3-2-3 | r6 | `{"depths":[3,2,3]}` | depth 2 in group 2 (on the old net: -0.87 s for -0.11 pp) | 8/8 | 75.04 | 0.19 | -0.25 +- 0.08 | -0.29 | +0.07 | 5.34 | 76 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 77 | ! w96-320-640 | r1 | `{"widths":[96,320,640]}` | narrow groups 1 and 2, widen group 3: time saving with pa... | 8/8 | 74.75 | 0.17 | -0.37 +- 0.08 | -0.76 | +0.08 | 6.16 | 77 warm | 0 | A100 SXM4 @ 500 W | BELOW 75% |
| 78 | w80-256-768 | r6 | `{"widths":[80,256,768]}` | group 1 at 80 channels | 8/8 | 75.10 | 0.22 | -0.19 +- 0.12 | -0.18 | +0.09 | 5.52 | 87 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 79 | ! silu | r1 | `{"activation":"silu"}` | SiLU is cheaper than erf-GELU and often equal in accuracy... | 8/8 | 75.00 | 0.38 | -0.13 +- 0.18 | -0.21 | +0.09 | 7.24 | 69 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 80 | ! res28-s0.75 | r3 | `{"train_resolution":28,"resolution_switch":0.75}` | 28 px for three quarters | 8/8 | 74.25 | 0.15 | -0.73 +- 0.09 | -0.62 | +0.11 | 5.44 | 38 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 81 | final0.03 | r1 | `{"final_lr":0.03}` | decay further at the end for a sharper final convergence | 8/8 | 75.07 | 0.34 | -0.05 +- 0.09 | -0.01 | +0.11 | 6.99 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 82 | ! s20x20-24x40-28x60 | r7 | `{"res_schedule":[[20,0.2],[24,0.4],[28,0.6]]}` | three-stage ramp 20->24->28->32 over 60% of the steps | 8/8 | 74.81 | 0.20 | -0.48 +- 0.10 | -0.57 | +0.12 | 5.10 | 65 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 83 | switch0.35 | r6 | `{"resolution_switch":0.35}` | 24 px for 35% of the steps instead of 25%: more saving, a... | 8/8 | 75.05 | 0.14 | -0.32 +- 0.12 | -0.31 | +0.14 | 5.36 | 24 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 84 | jitter0.45 | r6 | `{"jitter":0.45}` | stronger photometric jitter | 8/8 | 75.22 | 0.29 | -0.07 +- 0.14 | +0.06 | +0.16 | 5.80 | 32 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 85 | ! mom0.8 | r3 | `{"momentum":0.8}` | old net: +0.11 pp at equal time | 8/8 | 74.83 | 0.33 | -0.15 +- 0.11 | +0.01 | +0.16 | 6.10 | 20 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 86 | ! silu-gather | r1 | `{"activation":"silu","crop_gather":true}` | cheap-swaps bundle: SiLU plus the sync-free vectorised crop | 8/8 | 74.95 | 0.25 | -0.18 +- 0.13 | -0.23 | +0.16 | 7.11 | 37 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 87 | warmup0.3 | r1 | `{"warmup":0.3}` | longer warmup stabilises the fp16 early phase | 8/8 | 75.05 | 0.36 | -0.08 +- 0.18 | -0.01 | +0.17 | 6.98 | 10 warm | 0 | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 88 | bs1536 | r1 | `{"batch_size":1536}` | larger batch = fewer steps and less Python overhead per e... | 8/8 | 75.11 | 0.25 | -0.16 +- 0.09 | -0.18 | +0.17 | 7.25 | 115 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 89 | w96-256-896 | r6 | `{"widths":[96,256,896]}` | group 3 at 896 channels: accuracy to trade for epochs | 8/8 | 75.63 | 0.24 | +0.34 +- 0.12 | +0.67 | +0.18 | 6.30 | 110 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 90 | ! filt6-0.75 | r7 | `{"filter_start":6,"filter_keep":0.75}` | priority 3 (airbench96_faster spirit): from epoch 6 skip... | 8/8 | 74.91 | 0.29 | -0.47 +- 0.10 | -0.46 | +0.20 | 5.21 | 25 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 91 | scale0.0889 | r1 | `{"scaling_factor":0.08888888888888889}` | logit scale x0.8: softer logits with label smoothing 0.3 | 8/8 | 75.02 | 0.20 | -0.10 +- 0.07 | -0.02 | +0.21 | 7.32 | 36 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 92 | bias128 | r1 | `{"bias_scaler":128.0}` | BN biases can learn faster still (airbench uses 64x on CI... | 8/8 | 75.03 | 0.36 | -0.09 +- 0.19 | +0.01 | +0.22 | 7.47 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 93 | ! s20x20-24x40-28x60-e9.5 | r7 | `{"res_schedule":[[20,0.2],[24,0.4],[28,0.6]],"epochs":9.5}` | three-stage ramp with 0.75 epoch bought back | 8/8 | 74.99 | 0.14 | -0.30 +- 0.11 | -0.19 | +0.24 | 5.52 | 41 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 94 | mom0.9 | r1 | `{"momentum":0.9}` | higher momentum smooths the noisy few-epoch trajectory | 8/8 | 75.19 | 0.22 | -0.11 +- 0.16 | +0.01 | +0.24 | 7.30 | 21 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 95 | ! res24-s0.75 | r3 | `{"train_resolution":24,"resolution_switch":0.75}` | 24 px for three quarters: largest saving, largest accurac... | 8/8 | 73.18 | 0.20 | -1.80 +- 0.11 | -1.54 | +0.26 | 4.49 | 32 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 96 | ! s20x25-28x50 | r7 | `{"res_schedule":[[20,0.25],[28,0.5]]}` | priority 2: 20 px for the first quarter (instead of 24),... | 8/8 | 74.84 | 0.20 | -0.45 +- 0.10 | -0.35 | +0.30 | 5.32 | 69 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 97 | whiten2 | r1 | `{"whiten_bias_epochs":2}` | freeze the whitening bias earlier: the bias-grad graph ru... | 8/8 | 75.15 | 0.18 | -0.15 +- 0.12 | -0.03 | +0.30 | 7.29 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 98 | ! cutout4 | r3 | `{"cutout":4}` | old net: +0.08 pp at equal time | 8/8 | 74.73 | 0.25 | -0.26 +- 0.12 | +0.05 | +0.31 | 6.23 | 16 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 99 | ! s20x35 | r7 | `{"res_schedule":[[20,0.35]]}` | 20 px alone for 35% of the steps | 8/8 | 74.82 | 0.14 | -0.47 +- 0.10 | -0.36 | +0.31 | 5.32 | 74 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 100 | ! w64-d233 | r7 | `{"widths":[64,256,768],"depths":[2,3,3]}` | priority 5 stack: the two favourable group-1 cuts togethe... | 8/8 | 74.56 | 0.29 | -0.73 +- 0.15 | -0.71 | +0.33 | 5.11 | 142 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 101 | bs1536-e9.25 | r6 | `{"batch_size":1536,"epochs":9.25}` | batch 1536 with the time saving spent on 0.5 epoch | 8/8 | 75.12 | 0.13 | -0.18 +- 0.11 | +0.10 | +0.35 | 5.77 | 24 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 102 | ema3 | r1 | `{"ema_every":3}` | more frequent lookahead averaging | 8/8 | 75.16 | 0.22 | -0.14 +- 0.14 | +0.03 | +0.35 | 7.35 | 16 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 103 | ! bs1536 | r6 | `{"batch_size":1536}` | priority 4: fewer steps, better GEMM tiling; sum loss kee... | 8/8 | 74.86 | 0.30 | -0.43 +- 0.17 | -0.21 | +0.41 | 5.46 | 154 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 104 | e9.5 | r3 | `{"epochs":9.5}` | his selected recipe (75.25% / 6.86 s at n=40) in the same... | 8/8 | 75.25 | 0.29 | +0.27 +- 0.14 | +0.68 | +0.41 | 6.68 | 22 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 105 | ema10 | r1 | `{"ema_every":10}` | less frequent lookahead averaging: fewer EMA kernels, may... | 8/8 | 75.11 | 0.14 | -0.18 +- 0.14 | +0.01 | +0.42 | 7.33 | 15 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 106 | ! filt5-0.75 | r7 | `{"filter_start":5,"filter_keep":0.75}` | filter from epoch 5 | 8/8 | 74.66 | 0.18 | -0.72 +- 0.10 | -0.59 | +0.44 | 5.09 | 23 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 107 | ! w64-triton-s20x20-24x40-28x60-e9.5 | r7 | `{"widths":[64,256,768],"crop_mode":"triton","res_schedule":[[20,0.2],[24,0.4],[28,0.6]],"epochs":9.5}` | group 1 at 64 plus the three-stage ramp and the Triton cr... | 8/8 | 74.74 | 0.22 | -0.55 +- 0.09 | -0.26 | +0.53 | 5.56 | 236 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 108 | ! w64-384-576 | r1 | `{"widths":[64,384,576]}` | halve group 1; big time saving, accuracy risk | 8/8 | 74.37 | 0.22 | -0.94 +- 0.11 | -1.58 | +0.54 | 5.82 | 55 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 109 | ! s16x15-20x30-24x45-28x60 | r7 | `{"res_schedule":[[16,0.15],[20,0.3],[24,0.45],[28,0.6]]}` | four-stage ramp from 16 px (exploratory; the cold build m... | 8/8 | 74.38 | 0.27 | -0.91 +- 0.14 | -0.49 | +0.81 | 5.09 | 171 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 110 | ! filt6-0.5 | r7 | `{"filter_start":6,"filter_keep":0.5}` | from epoch 6 train on the hardest half only | 8/8 | 74.12 | 0.16 | -1.26 +- 0.07 | -0.86 | +0.93 | 4.82 | 23 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 111 | ! cutout8 | r1 | `{"cutout":8}` | larger cutout; likely too strong for 8.5 epochs | 8/8 | 74.70 | 0.20 | -0.42 +- 0.11 | +0.01 | +0.96 | 6.96 | 10 warm | 0 | A100 SXM4 @ 500 W | BELOW 75% |
| 112 | ! ema-off | r1 | `{"ema_every":0}` | no lookahead EMA: saves the EMA kernels (0.33 ms x 82) at... | 8/8 | 74.65 | 0.45 | -0.48 +- 0.19 | +0.01 | +1.09 | 7.47 | 21 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 113 | ! bs2048 | r6 | `{"batch_size":2048}` | batch 2048 (NaN watch) | 8/8 | 74.32 | 0.25 | -0.97 +- 0.12 | -0.26 | +1.13 | 5.41 | 149 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 114 | ! hard0.75-online | r6 | `{"hard_fraction":0.75,"proxy_mode":"online"}` | same selection scored online by the proxy every batch | 8/8 | 73.91 | 0.24 | -1.38 +- 0.11 | -0.65 | +1.33 | 5.00 | 62 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 115 | ! hard0.75-offline | r6 | `{"hard_fraction":0.75}` | priority 3: PR #5's proxy hard-example selection (offline... | 8/8 | 73.75 | 0.17 | -1.55 +- 0.11 | -0.56 | +1.64 | 5.09 | 405 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 116 | ! d3-3-2 | r1 | `{"depths":[3,3,2]}` | drop the residual pair in group 3 (cheapest group): small... | 8/8 | 74.19 | 0.13 | -0.94 +- 0.08 | -0.33 | +1.78 | 7.11 | 57 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 117 | ! wd0.0084 | r1 | `{"weight_decay":0.0084}` | wd x0.7: less regularisation for a short schedule | 8/8 | 74.50 | 0.20 | -0.80 +- 0.14 | +0.02 | +1.82 | 7.31 | 21 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 118 | ! hard0.5-offline | r6 | `{"hard_fraction":0.5}` | keep the hardest half of each batch | 8/8 | 67.81 | 0.53 | -7.49 +- 0.21 | -1.41 | +9.28 | 4.24 | 176 warm | 0 | A100 SXM4 @ 400 W | BELOW 75% |
| 119 | final2-defaults-400W | final | `{"count_nonfinite":true}` |  | 40/40 | 75.30 | 0.25 |  |  |  | 5.72 | 187 cold | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 120 | final-S7-res24q-e8.75-400W | final | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":24,"resolution_switch":0.25,"epochs":8.75,"count_nonfinite":true}` |  | 40/40 | 75.34 | 0.28 |  |  |  | 5.76 | 187 cold | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 121 | final-S7-res24q-e9.0-400W | final | `{"bias_scaler":32.0,"label_smoothing":0.25,"lr":10.8,"bn_momentum":0.7,"jitter":0.3,"scaling_factor":0.1388888888888889,"train_resolution":24,"resolution_switch":0.25,"epochs":9.0,"count_nonfinite":true}` |  | 40/40 | 75.40 | 0.27 |  |  |  | 5.87 | 157 cold | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 122 | defaults-torchlogs | r6 | `{}` | priority 1: do the 24->32 resolution switch or the first... | 3/3 | 75.51 | 0.21 |  |  |  | 5.99 | 44 warm | 0 | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |
| 123 | ref-sxm500-n40 | reference | `{}` | reference: current main recipe, cold build, official target | 40/40 | 75.19 | 0.25 |  |  |  | 6.94 | 95 cold |  | A100 SXM4 @ 500 W | QUALIFIED (>= 75%) |
| 124 | ref-sxm400-n40 | reference | `{}` | reference: current main recipe, cold build, official target | 40/40 | 75.20 | 0.28 |  |  |  | 7.41 | 133 cold |  | A100 SXM4 @ 400 W | QUALIFIED (>= 75%) |

## Controls (power limits seen: A100 SXM4 @ 400 W x43, A100 SXM4 @ 500 W x3)

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
| calib2 | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 6.00 | 0.01 | 143 warm | BELOW 75% |
| acc-a | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 6.06 | 0.14 | 18 warm | BELOW 75% |
| acc-b | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 6.09 | 0.02 | 23 warm | BELOW 75% |
| stack | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 6.00 | 0.02 | 18 warm | BELOW 75% |
| acc-c | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.99 | 0.23 | 6.18 | 0.02 | 18 warm | BELOW 75% |
| stack-trim | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.99 | 0.23 | 6.06 | 0.01 | 18 warm | BELOW 75% |
| res-24 | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 6.03 | 0.02 | 30 warm | BELOW 75% |
| res-28 | control | r3 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 6.07 | 0.03 | 24 warm | BELOW 75% |
| s7-res28 | control | r4 | A100 SXM4 @ 400 W | 8/8 | 75.00 | 0.23 | 6.30 | 0.12 | 16 warm | BELOW 75% |
| s6-s8 | control | r4 | A100 SXM4 @ 400 W | 8/8 | 74.99 | 0.23 | 6.02 | 0.02 | 16 warm | BELOW 75% |
| s7-res24 | control | r4 | A100 SXM4 @ 400 W | 8/8 | 74.98 | 0.22 | 5.96 | 0.01 | 21 warm | BELOW 75% |
| confirm16 | control | r5 | A100 SXM4 @ 400 W | 16/16 | 74.91 | 0.25 | 6.01 | 0.01 | 18 warm | BELOW 75% |
| confirm16b | control | r5 | A100 SXM4 @ 400 W | 16/16 | 74.91 | 0.25 | 6.08 | 0.01 | 17 warm | BELOW 75% |
| k2 | control | r6 | A100 SXM4 @ 400 W | 8/8 | 75.37 | 0.25 | 5.67 | 0.13 | 25 warm | QUALIFIED (>= 75%) |
| arch-a | control | r6 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.70 | 0.11 | 24 warm | QUALIFIED (>= 75%) |
| arch-b | control | r6 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.63 | 0.04 | 24 warm | QUALIFIED (>= 75%) |
| fine-sys | control | r6 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.75 | 0.02 | 33 warm | QUALIFIED (>= 75%) |
| batch | control | r6 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.67 | 0.01 | 25 warm | QUALIFIED (>= 75%) |
| select | control | r6 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.65 | 0.01 | 25 warm | QUALIFIED (>= 75%) |
| staged-epochs-2 | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.30 | 0.27 | 5.65 | 0.02 | 24 warm | QUALIFIED (>= 75%) |
| staged-a-2 | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.78 | 0.08 | 35 warm | QUALIFIED (>= 75%) |
| filter | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.38 | 0.23 | 5.68 | 0.01 | 24 warm | QUALIFIED (>= 75%) |
| staged-epochs-1 | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.71 | 0.01 | 24 warm | QUALIFIED (>= 75%) |
| staged-a-1 | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.67 | 0.02 | 25 warm | QUALIFIED (>= 75%) |
| staged-b-1 | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.30 | 0.27 | 5.69 | 0.02 | 32 warm | QUALIFIED (>= 75%) |
| staged-b-2 | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.58 | 0.01 | 34 warm | QUALIFIED (>= 75%) |
| stack-arch | control | r7 | A100 SXM4 @ 400 W | 8/8 | 75.29 | 0.24 | 5.82 | 0.03 | 40 warm | QUALIFIED (>= 75%) |

## GPU hit rates (from the ledger)

| GPU @ power | containers | share | guard misses | GPU min |
| --- | --- | --- | --- | --- |
| A100 SXM4 @ 400 W | 69 | 49% | 19 | 609.9 |
| A100 PCIe @ 300 W | 29 | 20% | 28 | 7.0 |
| A100 SXM4 @ ? | 23 | 16% | 22 | 15.7 |
| A100 SXM4 @ 500 W | 11 | 8% | 7 | 35.1 |
| A100 PCIe @ ? | 6 | 4% | 0 | 39.9 |
| A100 (mixed) @ ? | 2 | 1% | 0 | 33.8 |
| ? @ ? | 1 | 1% | 0 | 3.4 |
| A100 PCIe / SXM4 @ ? | 1 | 1% | 0 | 4.0 |

Guard misses: 76/142 containers (54%), 19.3 GPU min lost; ledger total 748.9 min.

Pareto plot: `pareto.svg` (x = mean prepare+train s, y = mean accuracy %).
