# r25/sys-tune-a

2026-10-04 00:16, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4200B2TMMVVDKKJCW735ZCR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 925.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4200B2TMMVVDKKJCW735ZCR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cdt | `{"inductor_tuning": ["coordinate_descent_tuning"]}` | 8/8 | 75.11 | 0.27 | +0.16 +- 0.10 | 4.38 | -0.09 | -0.25 | 404.22 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.95 | 0.11 | control | 4.47 | control | control | 33.28 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| cdt-aggfusion | `{"inductor_tuning": ["coordinate_descent_tuning", "aggressive_fusion"]}` | 8/8 | 75.04 | 0.24 | +0.08 +- 0.08 | 4.38 | -0.09 | -0.17 | 152.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| aggfusion | `{"inductor_tuning": ["aggressive_fusion"]}` | 8/8 | 75.07 | 0.28 | +0.12 +- 0.08 | 4.46 | -0.01 | -0.13 | 106.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
