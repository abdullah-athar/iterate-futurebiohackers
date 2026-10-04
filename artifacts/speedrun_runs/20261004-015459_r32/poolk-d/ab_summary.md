# r32/poolk-d

2026-10-04 02:05, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426GCDWGVH00Y99BBVD1B2R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 606.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426GCDWGVH00Y99BBVD1B2R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fullpool-avgsum-n16 | `{"global_pool": "fullpool_avgsum"}` | 16/16 | 75.31 | 0.26 | +0.11 +- 0.09 | 4.27 | -0.12 | -0.23 | 135.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.20 | 0.27 | control | 4.40 | control | control | 33.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g2-192-fullpool-avgsum-e10.0 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 16/16 | 75.14 | 0.32 | -0.06 +- 0.11 | 4.16 | -0.24 | -0.18 | 130.59 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
