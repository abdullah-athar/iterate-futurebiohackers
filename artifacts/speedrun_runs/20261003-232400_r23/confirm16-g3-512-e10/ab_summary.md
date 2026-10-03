# r23/confirm16-g3-512-e10

2026-10-03 23:29, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41XVY3ZMHMSEYEXKY9Q4APR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 334.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41XVY3ZMHMSEYEXKY9Q4APR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner512-e10.0-n16 | `{"epochs": 10.0, "g3_pair": "inner512"}` | 16/16 | 75.20 | 0.27 | +0.02 +- 0.10 | 4.46 | -0.11 | -0.13 | 51.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.18 | 0.25 | control | 4.56 | control | control | 49.22 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
