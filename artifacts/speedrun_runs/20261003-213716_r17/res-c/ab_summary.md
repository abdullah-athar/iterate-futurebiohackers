# r17/res-c

2026-10-03 21:43, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41QRGE12SP85X3KJVBYQJBR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 369.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41QRGE12SP85X3KJVBYQJBR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24q-28h-e9.5 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.28 | 0.25 | +0.18 +- 0.11 | 4.61 | -0.12 | -0.30 | 34.73 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.10 | 0.15 | control | 4.73 | control | control | 26.76 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24q-28h-bs512 | `{"batch_size": 512, "epochs": 8.0, "lr": 16.0, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 74.76 | 0.38 | -0.34 +- 0.10 | 4.21 | -0.52 | -0.18 | 121.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
