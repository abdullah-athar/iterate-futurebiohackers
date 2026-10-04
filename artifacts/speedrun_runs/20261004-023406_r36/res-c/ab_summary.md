# r36/res-c

2026-10-04 02:41, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M428SQK8NAPJ2M1YRXS83M2R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 376.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M428SQK8NAPJ2M1YRXS83M2R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r28x0.6-e10.0 | `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.25], [28, 0.6]]}` | 8/8 | 75.05 | 0.20 | -0.10 +- 0.09 | 4.08 | -0.10 | -0.00 | 113.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.15 | 0.24 | control | 4.18 | control | control | 36.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r28x0.6-e10.25 | `{"epochs": 10.25, "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.25], [28, 0.6]]}` | 8/8 | 75.19 | 0.18 | +0.04 +- 0.07 | 4.19 | +0.01 | -0.04 | 35.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
