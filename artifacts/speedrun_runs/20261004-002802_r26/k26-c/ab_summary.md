# r26/k26-c

2026-10-04 00:39, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M421H6831GQCQ5VYACCR10DR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 682.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M421H6831GQCQ5VYACCR10DR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ls0.2 | `{"label_smoothing": 0.2}` | 8/8 | 75.05 | 0.19 | -0.02 +- 0.12 | 4.46 | -0.02 | +0.00 | 156.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.48 | control | control | 48.45 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ls0.3 | `{"label_smoothing": 0.3}` | 8/8 | 75.11 | 0.22 | +0.03 +- 0.11 | 4.47 | -0.01 | -0.05 | 151.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| warmup0.15 | `{"warmup": 0.15}` | 8/8 | 75.09 | 0.26 | +0.01 +- 0.16 | 4.47 | -0.01 | -0.02 | 48.64 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
