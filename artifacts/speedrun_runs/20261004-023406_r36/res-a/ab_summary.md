# r36/res-a

2026-10-04 02:40, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M428R044WJ6F1C7TD5X9FWTR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 369.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M428R044WJ6F1C7TD5X9FWTR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e9.75 | `{"epochs": 9.75, "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 75.15 | 0.22 | +0.02 +- 0.11 | 4.11 | -0.06 | -0.08 | 104.84 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.17 | control | control | 34.78 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| e10.25 | `{"epochs": 10.25, "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 75.25 | 0.27 | +0.12 +- 0.13 | 4.30 | +0.12 | +0.01 | 36.10 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
