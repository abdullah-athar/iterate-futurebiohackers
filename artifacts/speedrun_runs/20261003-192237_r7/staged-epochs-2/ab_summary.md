# r7/staged-epochs-2

2026-10-03 19:27, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G3V3RRZHE9PFTW4SKEGQR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 233.9 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41G1YQ7CDEB92E02T0JB3BR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G3V3RRZHE9PFTW4SKEGQR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s20x25-28x50-e9.25 | `{"epochs": 9.25, "res_schedule": [[20, 0.25], [28, 0.5]]}` | 8/8 | 75.29 | 0.15 | -0.01 +- 0.06 | 5.58 | -0.07 | -0.06 | 67.65 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.30 | 0.27 | control | 5.65 | control | control | 24.28 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
