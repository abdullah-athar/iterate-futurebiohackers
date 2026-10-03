# r4/s7-res24

2026-10-03 18:17, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41BWS913Q857Y8273HRYKTR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 474.7 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41BWFBGGMC6DH9Q98PV1K3R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41BWS913Q857Y8273HRYKTR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S7-res24s0.25-e9.0 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.25, "scaling_factor": 0.1388888888888889, "train_resolution": 24}` | 8/8 | 75.40 | 0.21 | +0.42 +- 0.13 | 5.77 | -0.19 | -0.61 | 95.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 5.96 | control | control | 21.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S6-res24s0.5-e9.5 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.5, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8, "resolution_switch": 0.5, "train_resolution": 24}` | 8/8 | 75.05 | 0.27 | +0.07 +- 0.11 | 5.39 | -0.57 | -0.64 | 31.62 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S7-res28s0.75-e9.0 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 9.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "resolution_switch": 0.75, "scaling_factor": 0.1388888888888889, "train_resolution": 28}` | 8/8 | 74.74 | 0.27 | -0.24 +- 0.10 | 5.65 | -0.31 | -0.07 | 73.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
