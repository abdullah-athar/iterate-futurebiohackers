# r8/knobs-d

2026-10-03 19:38, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41GJH9C04N5DH8QTRHEZY8R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 382.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41GJH9C04N5DH8QTRHEZY8R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mom0.9 | `{"momentum": 0.9, "widths": [64, 256, 768]}` | 8/8 | 74.59 | 0.12 | -0.31 +- 0.09 | 4.84 | -0.01 | +0.43 | 91.75 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 74.90 | 0.22 | control | 4.85 | control | control | 23.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| ema4 | `{"ema_every": 4, "widths": [64, 256, 768]}` | 8/8 | 74.86 | 0.19 | -0.04 +- 0.12 | 4.86 | +0.01 | +0.06 | 23.70 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| e9.25 | `{"epochs": 9.25, "widths": [64, 256, 768]}` | 8/8 | 75.18 | 0.25 | +0.28 +- 0.07 | 5.10 | +0.25 | -0.15 | 22.98 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
