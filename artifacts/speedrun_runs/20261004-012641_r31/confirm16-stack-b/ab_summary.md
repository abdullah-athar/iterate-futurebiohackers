# r31/confirm16-stack-b

2026-10-04 01:33, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M424WJKGV2BKSQ7Z6NVBHE0R, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 376.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M424WJKGV2BKSQ7Z6NVBHE0R CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack-e9.75-n16 | `{"epochs": 9.75, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 16/16 | 74.96 | 0.31 | -0.19 +- 0.09 | 4.28 | -0.22 | -0.03 | 52.39 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 16/16 | 75.15 | 0.24 | control | 4.49 | control | control | 47.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
