# r6/arch-b

2026-10-03 19:11, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EX86RNH5QGT54QMECAS3R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 558.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EX86RNH5QGT54QMECAS3R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d3-2-3 | `{"depths": [3, 2, 3]}` | 8/8 | 75.04 | 0.19 | -0.25 +- 0.08 | 5.34 | -0.29 | -0.04 | 76.34 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.63 | control | control | 24.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w96-256-640 | `{"widths": [96, 256, 640]}` | 8/8 | 75.10 | 0.20 | -0.19 +- 0.06 | 5.30 | -0.33 | -0.14 | 113.23 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w96-256-896 | `{"widths": [96, 256, 896]}` | 8/8 | 75.63 | 0.24 | +0.34 +- 0.12 | 6.30 | +0.67 | +0.33 | 109.80 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
