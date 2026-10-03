# r1/opt-mom-wd

2026-10-03 17:24, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418QJX56VHRWMG78CBF11WR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 559.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418QJX56VHRWMG78CBF11WR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.30 | 0.32 | control | 7.30 | control | control | 24.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| mom0.9 | `{"momentum": 0.9}` | 8/8 | 75.19 | 0.22 | -0.11 +- 0.16 | 7.30 | +0.01 | +0.24 | 20.69 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| wd0.0084 | `{"weight_decay": 0.0084}` | 8/8 | 74.50 | 0.20 | -0.80 +- 0.14 | 7.31 | +0.02 | +1.82 | 20.53 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| wd0.0168 | `{"weight_decay": 0.0168}` | 8/8 | 75.33 | 0.15 | +0.03 +- 0.12 | 7.30 | +0.01 | -0.06 | 20.78 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
