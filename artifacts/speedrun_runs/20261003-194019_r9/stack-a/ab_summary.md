# r9/stack-a

2026-10-03 19:47, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41H5RW53DQ9YP5Q2CQMZ3ZR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 336.9 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41H5RW53DQ9YP5Q2CQMZ3ZR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-e8.75 | `{"bias_scaler": 16.0, "lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 75.01 | 0.23 | +0.17 +- 0.07 | 4.88 | +0.07 | -0.17 | 27.71 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.83 | 0.13 | control | 4.81 | control | control | 24.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S-e9.0 | `{"bias_scaler": 16.0, "epochs": 9.0, "lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 75.17 | 0.31 | +0.34 +- 0.08 | 4.94 | +0.13 | -0.35 | 24.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S-e9.25 | `{"bias_scaler": 16.0, "epochs": 9.25, "lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 75.19 | 0.23 | +0.35 +- 0.10 | 5.12 | +0.31 | -0.19 | 23.10 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
