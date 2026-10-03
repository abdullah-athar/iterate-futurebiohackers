# r15/s2conv

2026-10-03 20:51, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41MPJHXYQSJFARQSQNPWEAR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 475.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41MPJHXYQSJFARQSQNPWEAR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1-s2 | `{"conv1_stride": 2}` | 8/8 | 74.16 | 0.26 | -1.17 +- 0.12 | 4.74 | -0.36 | +1.31 | 107.07 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.33 | 0.19 | control | 5.10 | control | control | 35.41 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| head-lr2 | `{"head_lr": 2.0}` | 8/8 | 75.30 | 0.30 | -0.03 +- 0.10 | 5.28 | +0.18 | +0.23 | 37.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| switch0.35 | `{"resolution_switch": 0.35}` | 8/8 | 75.11 | 0.22 | -0.22 +- 0.12 | 4.84 | -0.25 | +0.05 | 37.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
