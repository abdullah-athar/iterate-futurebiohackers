# r7/staged-epochs-1

2026-10-03 19:28, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G1YAEXH959MVDCBGCCBZR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 333.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G1YAEXH959MVDCBGCCBZR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s20x20-24x40-28x60-e9.25 | `{"epochs": 9.25, "res_schedule": [[20, 0.2], [24, 0.4], [28, 0.6]]}` | 8/8 | 75.20 | 0.38 | -0.09 +- 0.16 | 5.51 | -0.20 | -0.07 | 75.83 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.71 | control | control | 24.43 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| s20x20-24x40-28x60-e9.5 | `{"epochs": 9.5, "res_schedule": [[20, 0.2], [24, 0.4], [28, 0.6]]}` | 8/8 | 74.99 | 0.14 | -0.30 +- 0.11 | 5.52 | -0.19 | +0.24 | 40.78 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
