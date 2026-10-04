# r34/paired40-s0-pool

2026-10-04 02:18, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4279QVR841P222V7Y7BDB2R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 40 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 600.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4279QVR841P222V7Y7BDB2R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avgsum-e10.0-s0 | `{"global_pool": "fullpool_avgsum"}` | 40/40 | 75.24 | 0.25 | +0.03 +- 0.06 | 4.33 | -0.13 | -0.17 | 125.95 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.20 | 0.27 | control | 4.47 | control | control | 35.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
