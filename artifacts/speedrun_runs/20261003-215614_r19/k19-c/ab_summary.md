# r19/k19-c

2026-10-03 22:02, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RV7CGXAKATPW6DN5PKD0R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 378.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RV7CGXAKATPW6DN5PKD0R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ls0.2 | `{"epochs": 9.5, "label_smoothing": 0.2, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.05 | 0.31 | -0.09 +- 0.11 | 4.15 | -0.00 | +0.08 | 79.29 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.16 | control | control | 25.99 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| translate1 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "translate": 1}` | 8/8 | 74.91 | 0.22 | -0.22 +- 0.10 | 4.16 | +0.00 | +0.22 | 25.22 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| translate3 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "translate": 3}` | 8/8 | 74.56 | 0.16 | -0.57 +- 0.10 | 4.16 | +0.00 | +0.57 | 25.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
