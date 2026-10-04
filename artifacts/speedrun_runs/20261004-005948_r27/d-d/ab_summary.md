# r27/d-d

2026-10-04 01:09, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M423CXVT9EYN5GGCT576XDVR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 494.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M423CXVT9EYN5GGCT576XDVR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-inner192 | `{"g2_pair": "inner192"}` | 8/8 | 75.10 | 0.24 | +0.07 +- 0.15 | 4.33 | -0.13 | -0.20 | 190.02 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.03 | 0.26 | control | 4.46 | control | control | 42.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g2-inner192-e10.25 | `{"epochs": 10.25, "g2_pair": "inner192"}` | 8/8 | 75.04 | 0.24 | +0.01 +- 0.13 | 4.44 | -0.02 | -0.03 | 45.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
