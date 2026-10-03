# r18/h24-b

2026-10-03 21:54, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RD40M5TG752ZNA7ABMCAR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 328.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RD40M5TG752ZNA7ABMCAR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24h-e10.25 | `{"epochs": 10.25, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.17 | 0.29 | +0.01 +- 0.13 | 4.41 | -0.21 | -0.22 | 40.10 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.17 | 0.21 | control | 4.62 | control | control | 39.83 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24h-28q75-e9.75 | `{"epochs": 9.75, "resolution_schedule": [[24, 0.5], [28, 0.75]]}` | 8/8 | 74.63 | 0.15 | -0.53 +- 0.09 | 3.97 | -0.65 | -0.12 | 52.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
