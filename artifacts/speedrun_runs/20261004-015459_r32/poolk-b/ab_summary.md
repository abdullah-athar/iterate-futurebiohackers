# r32/poolk-b

2026-10-04 02:07, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426GCNDFYMM4VHE207XB40R, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 711.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426GCNDFYMM4VHE207XB40R CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-192-fullpool-e10.0 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "fullpool"}` | 16/16 | 75.06 | 0.25 | -0.14 +- 0.08 | 4.20 | -0.29 | -0.15 | 167.29 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.20 | 0.27 | control | 4.49 | control | control | 44.55 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g2-192-fullpool-sum-e10.0 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "fullpool_sum"}` | 16/16 | 75.13 | 0.26 | -0.07 +- 0.11 | 4.26 | -0.23 | -0.16 | 171.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
