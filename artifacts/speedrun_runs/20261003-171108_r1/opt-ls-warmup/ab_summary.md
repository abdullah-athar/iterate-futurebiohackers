# r1/opt-ls-warmup

2026-10-03 17:28, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4193TR0WB24TBVNTPB4PZ0R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 398.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4193TR0WB24TBVNTPB4PZ0R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.44 | control | control | 22.60 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ls0.25 | `{"label_smoothing": 0.25}` | 8/8 | 75.33 | 0.18 | +0.21 +- 0.09 | 7.44 | +0.00 | -0.47 | 22.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ls0.35 | `{"label_smoothing": 0.35}` | 8/8 | 75.27 | 0.28 | +0.14 +- 0.14 | 7.45 | +0.01 | -0.32 | 20.48 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| warmup0.15 | `{"warmup": 0.15}` | 8/8 | 75.15 | 0.26 | +0.03 +- 0.15 | 7.45 | +0.01 | -0.05 | 22.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
