# r17/res-d

2026-10-03 21:44, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41QRFVQQB6F8E8KABMZJ3GR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 416.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41QRFVQQB6F8E8KABMZJ3GR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r20-24-28 | `{"resolution_schedule": [[20, 0.15], [24, 0.35], [28, 0.55]]}` | 8/8 | 74.83 | 0.27 | -0.20 +- 0.13 | 4.05 | -0.66 | -0.46 | 141.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.04 | 0.24 | control | 4.71 | control | control | 32.52 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24q25-28q60 | `{"resolution_schedule": [[24, 0.25], [28, 0.6]]}` | 8/8 | 74.98 | 0.17 | -0.06 +- 0.09 | 4.33 | -0.38 | -0.32 | 42.97 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
