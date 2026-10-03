# r1/opt-whiten-bn

2026-10-03 17:33, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41976CMS6W6J87TDZCDVC0R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 638.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41976CMS6W6J87TDZCDVC0R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.27 | control | control | 17.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| whiten4 | `{"whiten_bias_epochs": 4}` | 8/8 | 75.21 | 0.22 | +0.08 +- 0.05 | 7.32 | +0.05 | -0.13 | 16.65 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bn0.5 | `{"bn_momentum": 0.5}` | 8/8 | 75.25 | 0.18 | +0.12 +- 0.09 | 7.27 | +0.00 | -0.27 | 30.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bn0.7 | `{"bn_momentum": 0.7}` | 8/8 | 75.29 | 0.23 | +0.16 +- 0.10 | 7.24 | -0.03 | -0.40 | 35.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
