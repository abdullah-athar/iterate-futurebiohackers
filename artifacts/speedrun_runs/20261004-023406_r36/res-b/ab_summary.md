# r36/res-b

2026-10-04 02:40, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M428R0448ZQ6DES99MZM2EMR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 396.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M428R0448ZQ6DES99MZM2EMR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24x0.35-e10.0 | `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.35], [28, 0.5]]}` | 8/8 | 75.30 | 0.19 | +0.15 +- 0.09 | 4.03 | -0.12 | -0.27 | 137.39 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.15 | 0.24 | control | 4.15 | control | control | 34.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24x0.35-e10.25 | `{"epochs": 10.25, "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.35], [28, 0.5]]}` | 8/8 | 75.17 | 0.15 | +0.03 +- 0.07 | 4.14 | -0.01 | -0.03 | 35.16 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
