# r19/k19-b

2026-10-03 22:01, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RV70MSNTMM2820W65WJMR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 322.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RV70MSNTMM2820W65WJMR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final0.15 | `{"epochs": 9.5, "final_lr": 0.15, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.89 | 0.30 | +0.03 +- 0.09 | 4.11 | -0.01 | -0.03 | 27.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 74.86 | 0.22 | control | 4.12 | control | control | 25.36 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| ema4 | `{"ema_every": 4, "epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.79 | 0.29 | -0.07 +- 0.08 | 4.11 | -0.00 | +0.07 | 25.69 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| ema8 | `{"ema_every": 8, "epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.87 | 0.30 | +0.01 +- 0.07 | 4.12 | +0.00 | -0.00 | 26.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
