# r19/k19-g

2026-10-03 22:01, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RV7NM63BAHQDMC8JWQVZR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 331.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RV7NM63BAHQDMC8JWQVZR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aten-only | `{"autotune_backends": "ATEN", "epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.91 | 0.26 | -0.23 +- 0.09 | 4.14 | -0.00 | +0.23 | 140.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.14 | control | control | 39.12 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
