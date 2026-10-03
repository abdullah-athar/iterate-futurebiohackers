# r28/w128-b

2026-10-04 00:58, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M422PV40CY0BCSY2PVSASHYR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 557.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M422PV40CY0BCSY2PVSASHYR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w128-d233-e9.5 | `{"depths": [2, 3, 3], "epochs": 9.5, "widths": [128, 256, 768]}` | 8/8 | 75.29 | 0.26 | +0.22 +- 0.09 | 4.93 | +0.54 | +0.32 | 217.42 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.39 | control | control | 56.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w128-d233-e10.0 | `{"depths": [2, 3, 3], "epochs": 10.0, "widths": [128, 256, 768]}` | 8/8 | 75.69 | 0.17 | +0.62 +- 0.10 | 5.16 | +0.77 | +0.15 | 57.18 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
