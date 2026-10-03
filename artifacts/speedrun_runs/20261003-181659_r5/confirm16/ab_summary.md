# r5/confirm16

2026-10-03 18:26, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41C9RJGP5X2SDDTAGWM4PHR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 540.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41C9RJGP5X2SDDTAGWM4PHR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S7-res28-e9.0 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.5, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 16/16 | 75.34 | 0.21 | +0.43 +- 0.08 | 5.92 | -0.09 | -0.52 | 33.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 74.91 | 0.25 | control | 6.01 | control | control | 17.80 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S7-res28-e8.75 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 8.75, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.5, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 16/16 | 75.17 | 0.21 | +0.26 +- 0.08 | 5.80 | -0.22 | -0.48 | 24.76 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S6-res28-e9.0 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8, "resolution_switch": 0.5, "train_resolution": 28}` | 16/16 | 75.21 | 0.27 | +0.30 +- 0.11 | 5.92 | -0.09 | -0.40 | 25.23 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
