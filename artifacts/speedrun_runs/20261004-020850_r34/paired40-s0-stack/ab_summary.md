# r34/paired40-s0-stack

2026-10-04 02:20, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4279R4XWAXENJHQGCBGWB0R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 40 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 683.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4279R4XWAXENJHQGCBGWB0R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-192-avgsum-e10.0-s0 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 40/40 | 75.16 | 0.23 | -0.05 +- 0.06 | 4.23 | -0.26 | -0.22 | 164.07 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.20 | 0.27 | control | 4.50 | control | control | 47.60 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
