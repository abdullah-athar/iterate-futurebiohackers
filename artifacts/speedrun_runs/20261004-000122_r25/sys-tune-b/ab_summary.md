# r25/sys-tune-b

2026-10-04 00:08, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4200B2TAENVZ7P05F9SEW0R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 424.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4200B2TAENVZ7P05F9SEW0R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aten-only | `{"autotune_backends": "ATEN"}` | 8/8 | 75.10 | 0.24 | +0.03 +- 0.16 | 4.45 | -0.00 | -0.03 | 104.39 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.45 | control | control | 33.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| triton-only | `{"autotune_backends": "TRITON"}` | 8/8 | 75.09 | 0.19 | +0.02 +- 0.14 | 4.45 | -0.00 | -0.02 | 97.64 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
