# r3/calib2

2026-10-03 17:56, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41AN5GH0V8TR9FH8XT9NY1R, CLOUD_PROVIDER_GCP/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 464.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41AN5GH0V8TR9FH8XT9NY1R CLOUD_PROVIDER_GCP/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 6.00 | control | control | 143.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| e8.0 | `{"epochs": 8.0}` | 8/8 | 74.63 | 0.27 | -0.35 +- 0.13 | 5.69 | -0.31 | +0.48 | 20.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| e9.0 | `{"epochs": 9.0}` | 8/8 | 75.34 | 0.15 | +0.36 +- 0.11 | 6.35 | +0.35 | -0.47 | 22.80 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| e9.5 | `{"epochs": 9.5}` | 8/8 | 75.25 | 0.29 | +0.27 +- 0.14 | 6.68 | +0.68 | +0.08 | 22.22 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
