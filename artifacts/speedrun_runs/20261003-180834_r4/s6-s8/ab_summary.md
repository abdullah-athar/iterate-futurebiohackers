# r4/s6-s8

2026-10-03 18:15, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41BTSTB5JFX23PGZFHZYVKR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 392.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41BTSTB5JFX23PGZFHZYVKR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S6-res28-e9.0 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8, "resolution_switch": 0.5, "train_resolution": 28}` | 8/8 | 75.22 | 0.35 | +0.22 +- 0.18 | 5.90 | -0.11 | -0.34 | 43.51 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.99 | 0.23 | control | 6.02 | control | control | 16.34 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S7-e8.5 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 8.5, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "scaling_factor": 0.1388888888888889}` | 8/8 | 75.27 | 0.31 | +0.28 +- 0.14 | 6.01 | -0.00 | -0.28 | 37.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S8-res28-e8.75 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 8.75, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8, "resolution_switch": 0.5, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 8/8 | 74.99 | 0.14 | +0.00 +- 0.10 | 5.77 | -0.25 | -0.25 | 50.34 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
