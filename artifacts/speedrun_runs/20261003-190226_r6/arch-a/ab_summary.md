# r6/arch-a

2026-10-03 19:11, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EX86G469MY1EME0AD0RNR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 499.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EX86G469MY1EME0AD0RNR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w64-256-768 | `{"widths": [64, 256, 768]}` | 8/8 | 74.90 | 0.22 | -0.40 +- 0.13 | 4.99 | -0.71 | -0.32 | 92.25 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.70 | control | control | 24.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w80-256-768 | `{"widths": [80, 256, 768]}` | 8/8 | 75.10 | 0.22 | -0.19 +- 0.12 | 5.52 | -0.18 | +0.01 | 86.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d2-3-3 | `{"depths": [2, 3, 3]}` | 8/8 | 75.05 | 0.15 | -0.24 +- 0.11 | 5.23 | -0.47 | -0.22 | 68.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
