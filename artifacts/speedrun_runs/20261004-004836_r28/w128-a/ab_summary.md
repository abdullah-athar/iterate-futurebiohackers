# r28/w128-a

2026-10-04 00:58, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M422PV3CFBEAQPZC8WP9686R, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 556.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M422PV3CFBEAQPZC8WP9686R CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w128-d233-e8.25 | `{"depths": [2, 3, 3], "epochs": 8.25, "widths": [128, 256, 768]}` | 8/8 | 74.84 | 0.24 | -0.19 +- 0.16 | 4.26 | -0.10 | +0.10 | 220.25 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.04 | 0.27 | control | 4.36 | control | control | 57.97 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w128-d233-e9.0 | `{"depths": [2, 3, 3], "epochs": 9.0, "widths": [128, 256, 768]}` | 8/8 | 75.17 | 0.23 | +0.13 +- 0.15 | 4.62 | +0.26 | +0.13 | 60.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
