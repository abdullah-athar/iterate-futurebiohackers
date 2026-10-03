# r16/port-b

2026-10-03 21:30, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41PQTWVDJ82B2Y6ZQK0BQQR, CLOUD_PROVIDER_AZURE/eu-south), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 635.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41PQQ6FDQMKJKG2200ZNFHR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41PQTWVDJ82B2Y6ZQK0BQQR CLOUD_PROVIDER_AZURE/eu-south]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wd0.0238 | `{"weight_decay": 0.0238}` | 8/8 | 75.02 | 0.18 | -0.07 +- 0.04 | 4.60 | +0.00 | +0.07 | 189.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.09 | 0.19 | control | 4.60 | control | control | 35.43 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bn0.7 | `{"bn_momentum": 0.7}` | 8/8 | 75.13 | 0.13 | +0.04 +- 0.08 | 4.61 | +0.00 | -0.03 | 114.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.3 | `{"color_jitter": [0.3, 0.3]}` | 8/8 | 75.16 | 0.28 | +0.07 +- 0.10 | 4.60 | +0.00 | -0.07 | 47.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
