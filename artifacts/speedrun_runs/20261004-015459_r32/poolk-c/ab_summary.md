# r32/poolk-c

2026-10-04 02:05, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426GC5538DG795Y31M2SAQR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 630.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426GC5538DG795Y31M2SAQR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-192-fullpool-sum-e9.75 | `{"epochs": 9.75, "g2_pair": "inner192", "global_pool": "fullpool_sum"}` | 16/16 | 75.02 | 0.23 | -0.18 +- 0.10 | 4.15 | -0.31 | -0.13 | 146.91 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.20 | 0.27 | control | 4.47 | control | control | 36.90 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g2-192-fullpool-e9.75 | `{"epochs": 9.75, "g2_pair": "inner192", "global_pool": "fullpool"}` | 16/16 | 74.94 | 0.21 | -0.26 +- 0.08 | 4.10 | -0.36 | -0.10 | 146.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
