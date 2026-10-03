# r1/arch-widths-b

2026-10-03 17:24, NVIDIA A100-SXM4-80GB @ 500.00 W (task ta-01M418QHJ4F1MHZMSVMH7E59DR, CLOUD_PROVIDER_UNSPECIFIED/eu), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 565.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M418QHJ4F1MHZMSVMH7E59DR CLOUD_PROVIDER_UNSPECIFIED/eu]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 6.92 | control | control | 11.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| w128-320-768 | `{"widths": [128, 320, 768]}` | 8/8 | 75.56 | 0.29 | +0.43 +- 0.15 | 6.91 | -0.01 | -0.99 | 73.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| w96-320-640 | `{"widths": [96, 320, 640]}` | 8/8 | 74.75 | 0.17 | -0.37 +- 0.08 | 6.16 | -0.76 | +0.08 | 77.40 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | BELOW 75% |
