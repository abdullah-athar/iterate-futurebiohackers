# r22/g3lad-a

2026-10-03 23:21, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41XDEGEVY05TSJ64FWEFS8R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 309.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41XDEGEVY05TSJ64FWEFS8R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner512-e9.75 | `{"epochs": 9.75, "g3_pair": "inner512"}` | 8/8 | 74.92 | 0.26 | -0.28 +- 0.11 | 4.34 | -0.19 | +0.09 | 42.97 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.54 | control | control | 40.40 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner512-e10.0 | `{"epochs": 10.0, "g3_pair": "inner512"}` | 8/8 | 75.17 | 0.24 | -0.04 +- 0.08 | 4.43 | -0.10 | -0.06 | 41.07 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
