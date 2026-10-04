# r37/g1-b

2026-10-04 02:48, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4292DG27ZPANJG7WQA50VPR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 484.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4292DG27ZPANJG7WQA50VPR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g1-40-e10.0 | `{"g1_pair": "inner40", "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 75.00 | 0.23 | -0.14 +- 0.12 | 4.14 | -0.04 | +0.10 | 155.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.15 | 0.24 | control | 4.18 | control | control | 105.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g1-40-e10.25 | `{"epochs": 10.25, "g1_pair": "inner40", "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 75.19 | 0.13 | +0.04 +- 0.11 | 4.24 | +0.06 | +0.02 | 34.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
