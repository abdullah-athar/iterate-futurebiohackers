# r31/confirm16-stack-a

2026-10-04 01:34, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M424WJQNYC1G6C67KX5MPBER, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 451.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M424WJQNYC1G6C67KX5MPBER CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack-e10.0-n16 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 16/16 | 75.16 | 0.24 | +0.04 +- 0.08 | 4.34 | -0.13 | -0.17 | 38.67 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.12 | 0.22 | control | 4.47 | control | control | 65.39 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| stack-e9.875-n16 | `{"epochs": 9.875, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 16/16 | 75.06 | 0.28 | -0.06 +- 0.11 | 4.31 | -0.16 | -0.10 | 36.73 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
