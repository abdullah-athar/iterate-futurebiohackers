# r8/knobs-c

2026-10-03 19:39, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41GJH9CMC7KBCJME8CYD0WR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 446.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41GJH9CMC7KBCJME8CYD0WR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bn0.8 | `{"bn_momentum": 0.8, "widths": [64, 256, 768]}` | 8/8 | 74.81 | 0.21 | -0.12 +- 0.08 | 4.83 | -0.06 | +0.11 | 90.20 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 74.93 | 0.22 | control | 4.89 | control | control | 49.12 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| scale1.4 | `{"scaling_factor": 0.15555555555555556, "widths": [64, 256, 768]}` | 8/8 | 74.80 | 0.20 | -0.13 +- 0.11 | 4.83 | -0.06 | +0.14 | 64.39 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| jitter0.2 | `{"jitter": 0.2, "widths": [64, 256, 768]}` | 8/8 | 74.93 | 0.23 | -0.00 +- 0.09 | 4.84 | -0.05 | -0.05 | 24.51 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
