# r3/acc-a

2026-10-03 18:02, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B5SWPK4PX9JEAD0PJ47AR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 314.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B5SWPK4PX9JEAD0PJ47AR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 6.06 | control | control | 17.96 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| bias32 | `{"bias_scaler": 32.0}` | 8/8 | 75.09 | 0.27 | +0.11 +- 0.14 | 6.04 | -0.02 | -0.27 | 17.28 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ls0.25 | `{"label_smoothing": 0.25}` | 8/8 | 75.00 | 0.28 | +0.02 +- 0.08 | 6.01 | -0.04 | -0.09 | 16.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| lr10.8 | `{"lr": 10.8}` | 8/8 | 75.02 | 0.15 | +0.04 +- 0.11 | 6.04 | -0.02 | -0.11 | 16.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
