# r4/s7-res28

2026-10-03 18:15, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41BTSTB7J80636P79QBP5RR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 391.9 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41BTSTB7J80636P79QBP5RR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S7-res28-e8.5 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 8.5, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.5, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 8/8 | 74.95 | 0.24 | -0.04 +- 0.16 | 5.77 | -0.53 | -0.48 | 75.87 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.00 | 0.23 | control | 6.30 | control | control | 16.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S7-res28-e8.75 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 8.75, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.5, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 8/8 | 75.17 | 0.14 | +0.17 +- 0.09 | 5.94 | -0.36 | -0.53 | 24.75 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S7-res28-e9.0 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.5, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 8/8 | 75.33 | 0.18 | +0.33 +- 0.11 | 6.12 | -0.18 | -0.51 | 24.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
