# r18/h24-c

2026-10-03 21:53, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RDAJ2MM59Y7RHEN07Q5NR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 274.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41RD4E271M07E76W1C17GCR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RDAJ2MM59Y7RHEN07Q5NR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r20q25-24h-e9.75 | `{"epochs": 9.75, "resolution_schedule": [[20, 0.25], [24, 0.5]]}` | 8/8 | 74.62 | 0.20 | -0.42 +- 0.11 | 4.11 | -0.53 | -0.12 | 36.10 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.04 | 0.24 | control | 4.64 | control | control | 26.80 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24h-e9.5 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.13 | 0.21 | +0.10 +- 0.09 | 4.12 | -0.52 | -0.61 | 25.05 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
