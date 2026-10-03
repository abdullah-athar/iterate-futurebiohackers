# calib/epochs

2026-10-03 17:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M417RJXA4WTX1YE6N1KH9S4R, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = None, container wall 399.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M417RJXA4WTX1YE6N1KH9S4R CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.35 | control | control | 132.88 (warm) | None | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| e8.0 | `{"epochs": 8.0}` | 8/8 | 74.97 | 0.31 | -0.16 +- 0.13 | 6.95 | -0.41 | n/a | 17.85 (warm) | None | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| e7.5 | `{"epochs": 7.5}` | 8/8 | 74.75 | 0.12 | -0.38 +- 0.11 | 6.52 | -0.83 | n/a | 17.28 (warm) | None | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
