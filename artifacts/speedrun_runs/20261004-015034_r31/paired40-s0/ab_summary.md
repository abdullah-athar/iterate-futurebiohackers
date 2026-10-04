# r31/paired40-s0

2026-10-04 01:59, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M42689J0S0MZG7722FP5NXPR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 40 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 513.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M42689J0S0MZG7722FP5NXPR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack-e10.0-s0 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 40/40 | 75.13 | 0.24 | -0.07 +- 0.06 | 4.34 | -0.12 | -0.05 | 39.20 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.20 | 0.27 | control | 4.46 | control | control | 35.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
