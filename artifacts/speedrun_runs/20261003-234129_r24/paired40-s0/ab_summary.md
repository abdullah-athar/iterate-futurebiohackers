# r24/paired40-s0

2026-10-03 23:50, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41YVXY0QQQS9ETA3R4W9KXR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 40 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 512.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41YVXY0QQQS9ETA3R4W9KXR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner512-e10.0-s0 | `{"epochs": 10.0, "g3_pair": "inner512"}` | 40/40 | 75.20 | 0.27 | -0.01 +- 0.06 | 4.51 | -0.09 | -0.08 | 35.09 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.21 | 0.23 | control | 4.60 | control | control | 34.52 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
