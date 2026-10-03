# r15/knobs-3

2026-10-03 20:51, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41MPJJ6TKKED62MN3NK393R, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 452.9 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41MPJJ6TKKED62MN3NK393R CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bn0.6 | `{"bn_momentum": 0.6}` | 8/8 | 75.32 | 0.16 | -0.21 +- 0.10 | 5.02 | -0.14 | +0.15 | 85.40 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.17 | control | control | 36.94 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| whiten-eps4x | `{"whiten_eps": 0.002}` | 8/8 | 75.44 | 0.32 | -0.09 +- 0.16 | 5.13 | -0.04 | +0.08 | 33.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| head-lr0.5 | `{"head_lr": 0.5}` | 8/8 | 74.96 | 0.29 | -0.57 +- 0.09 | 5.09 | -0.08 | +0.73 | 32.49 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
