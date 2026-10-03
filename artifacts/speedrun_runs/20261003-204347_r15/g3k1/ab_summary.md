# r15/g3k1

2026-10-03 20:54, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41MPJJ6CVK6A93FD9K3M55R, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 631.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41MPJJ6CVK6A93FD9K3M55R CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k331 | `{"conv_kernels": [3, 3, 1]}` | 8/8 | 73.75 | 0.26 | -1.78 +- 0.15 | 4.25 | -0.76 | +1.78 | 121.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.01 | control | control | 39.86 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| k311 | `{"conv_kernels": [3, 1, 1]}` | 8/8 | 72.31 | 0.32 | -3.22 +- 0.13 | 3.87 | -1.14 | +3.45 | 134.71 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| res24-28 | `{"res_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.31 | 0.26 | -0.22 +- 0.13 | 5.05 | +0.04 | +0.36 | 88.09 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
