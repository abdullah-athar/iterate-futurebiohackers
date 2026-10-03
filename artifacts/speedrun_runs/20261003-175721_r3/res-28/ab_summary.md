# r3/res-28

2026-10-03 18:06, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B899WCHQH43F05WDFYQNR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 477.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B899WCHQH43F05WDFYQNR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 6.07 | control | control | 24.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res28-s0.5 | `{"resolution_switch": 0.5, "train_resolution": 28}` | 8/8 | 74.70 | 0.26 | -0.28 +- 0.12 | 5.67 | -0.39 | +0.23 | 68.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res28-s0.75 | `{"resolution_switch": 0.75, "train_resolution": 28}` | 8/8 | 74.25 | 0.15 | -0.73 +- 0.09 | 5.44 | -0.62 | +1.02 | 38.22 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res24-s0.5-stack4 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.5, "train_resolution": 24}` | 8/8 | 74.34 | 0.32 | -0.64 +- 0.14 | 4.98 | -1.09 | +0.36 | 95.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
