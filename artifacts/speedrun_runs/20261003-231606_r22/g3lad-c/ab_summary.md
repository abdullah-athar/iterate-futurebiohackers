# r22/g3lad-c

2026-10-03 23:22, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41XDERQP2KX8M1GTR5R6GTR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 371.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41XDERQP2KX8M1GTR5R6GTR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner640-e9.75 | `{"epochs": 9.75, "g3_pair": "inner640"}` | 8/8 | 75.09 | 0.26 | -0.12 +- 0.09 | 4.64 | +0.02 | +0.13 | 58.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.62 | control | control | 49.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner640-e10.0 | `{"epochs": 10.0, "g3_pair": "inner640"}` | 8/8 | 75.32 | 0.18 | +0.11 +- 0.12 | 4.77 | +0.14 | +0.03 | 48.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
