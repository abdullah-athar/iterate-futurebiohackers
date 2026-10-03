# r1/opt-bias-ema

2026-10-03 17:34, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41974VS9471D4C1AGNKWQKR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 647.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41974VS9471D4C1AGNKWQKR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.47 | control | control | 24.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bias32 | `{"bias_scaler": 32.0}` | 8/8 | 75.44 | 0.24 | +0.32 +- 0.14 | 7.48 | +0.01 | -0.70 | 21.96 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bias128 | `{"bias_scaler": 128.0}` | 8/8 | 75.03 | 0.36 | -0.09 +- 0.19 | 7.47 | +0.01 | +0.22 | 21.24 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ema-off | `{"ema_every": 0}` | 8/8 | 74.65 | 0.45 | -0.48 +- 0.19 | 7.47 | +0.01 | +1.09 | 21.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
