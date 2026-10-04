# r35/width-b

2026-10-04 02:44, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M428PKQC6FNYN43HWGRXN97R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 648.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M428PKQC6FNYN43HWGRXN97R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w896-e9.0 | `{"epochs": 9.0, "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "widths": [64, 256, 896]}` | 8/8 | 74.71 | 0.14 | -0.44 +- 0.10 | 4.18 | -0.05 | +0.39 | 169.99 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.15 | 0.24 | control | 4.23 | control | control | 105.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w288-e10.0 | `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "widths": [64, 288, 768]}` | 8/8 | 75.42 | 0.32 | +0.27 +- 0.13 | 4.76 | +0.54 | +0.27 | 172.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
