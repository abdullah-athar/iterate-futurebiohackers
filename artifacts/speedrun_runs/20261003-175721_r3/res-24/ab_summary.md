# r3/res-24

2026-10-03 18:05, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B899WTVE1CP1YQEPGRMVR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 406.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B899WTVE1CP1YQEPGRMVR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 6.03 | control | control | 29.75 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res24-s0.25 | `{"resolution_switch": 0.25, "train_resolution": 24}` | 8/8 | 74.63 | 0.24 | -0.35 +- 0.14 | 5.54 | -0.49 | +0.29 | 76.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res24-s0.5 | `{"resolution_switch": 0.5, "train_resolution": 24}` | 8/8 | 74.04 | 0.17 | -0.94 +- 0.12 | 4.90 | -1.13 | +0.98 | 31.95 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res24-s0.75 | `{"resolution_switch": 0.75, "train_resolution": 24}` | 8/8 | 73.18 | 0.20 | -1.80 +- 0.11 | 4.49 | -1.54 | +2.50 | 32.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
