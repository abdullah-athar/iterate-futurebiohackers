# r31/paired40-s40

2026-10-04 01:47, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M425JG8YK8TA953H01X6SY6R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 40 trial(s) per run, seeds from 40, warm compile cache, k = 1.0, container wall 517.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M425JG8YK8TA953H01X6SY6R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack-e10.0-s40 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 40/40 | 75.20 | 0.25 | -0.02 +- 0.06 | 4.33 | -0.14 | -0.11 | 39.33 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.22 | 0.25 | control | 4.46 | control | control | 35.90 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
