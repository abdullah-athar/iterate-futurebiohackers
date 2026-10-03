# r26/k26-a

2026-10-04 00:35, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M421H5RBT39DCQXRV77WNV3R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 464.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M421H5RBT39DCQXRV77WNV3R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr10.5 | `{"lr": 10.5}` | 8/8 | 75.06 | 0.23 | +0.01 +- 0.11 | 4.52 | +0.01 | -0.00 | 48.07 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.05 | 0.20 | control | 4.51 | control | control | 60.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| lr12.5 | `{"lr": 12.5}` | 8/8 | 75.13 | 0.31 | +0.08 +- 0.14 | 4.50 | -0.01 | -0.09 | 48.05 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| wd0.012 | `{"weight_decay": 0.012}` | 8/8 | 74.92 | 0.32 | -0.13 +- 0.11 | 4.50 | -0.01 | +0.12 | 46.38 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
