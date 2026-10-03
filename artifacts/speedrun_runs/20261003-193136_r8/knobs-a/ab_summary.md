# r8/knobs-a

2026-10-03 19:38, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41GJC2WYFPEFTXAVXV6TYZR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 385.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41GJC2WYFPEFTXAVXV6TYZR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bias16 | `{"bias_scaler": 16.0, "widths": [64, 256, 768]}` | 8/8 | 74.97 | 0.21 | +0.14 +- 0.07 | 4.84 | -0.05 | -0.25 | 89.98 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 74.83 | 0.13 | control | 4.89 | control | control | 25.59 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| ls0.2 | `{"label_smoothing": 0.2, "widths": [64, 256, 768]}` | 8/8 | 74.83 | 0.32 | -0.00 +- 0.13 | 4.84 | -0.05 | -0.05 | 25.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| lr12 | `{"lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 75.00 | 0.22 | +0.16 +- 0.11 | 4.83 | -0.06 | -0.29 | 24.41 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
