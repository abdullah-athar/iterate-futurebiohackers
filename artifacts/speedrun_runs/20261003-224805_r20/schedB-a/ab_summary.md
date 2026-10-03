# r20/schedB-a

2026-10-03 22:54, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41VVX4A3XW3P6KNQQ35T19R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 307.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41VT52BD4YBKVES7N2JX79R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41VVX4A3XW3P6KNQQ35T19R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-24q30-28h | `{"epochs": 9.5, "resolution_schedule": [[24, 0.3], [28, 0.5]]}` | 8/8 | 75.18 | 0.25 | -0.02 +- 0.08 | 4.42 | -0.09 | -0.07 | 36.65 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.51 | control | control | 35.58 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| B-24q-28q75 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.25], [28, 0.75]]}` | 8/8 | 74.78 | 0.24 | -0.42 +- 0.06 | 4.49 | -0.02 | +0.40 | 34.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
