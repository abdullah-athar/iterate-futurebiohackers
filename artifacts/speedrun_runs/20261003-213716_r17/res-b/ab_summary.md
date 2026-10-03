# r17/res-b

2026-10-03 21:42, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41QRNTTD16GMY5DK0GY3ZCR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 318.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41QRGE1HYXVS05276P1X4HR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41QRNTTD16GMY5DK0GY3ZCR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24h | `{"resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.70 | 0.42 | -0.34 +- 0.11 | 3.89 | -0.73 | -0.39 | 39.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.04 | 0.24 | control | 4.62 | control | control | 37.07 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24q-28h-e9.25 | `{"epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.23 | 0.25 | +0.19 +- 0.12 | 4.42 | -0.19 | -0.38 | 50.53 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
