# r20/method-b

2026-10-03 22:41, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41V3EWM3PBR9VM30X2P16AR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 358.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41V3EWM3PBR9VM30X2P16AR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-skipg0 | `{"epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]], "skip_residual_groups": [0], "skip_residual_until": 0.25}` | 0/8 | n/a | n/a | control | n/a | control | control | 88.28 (warm) | None | NVIDIA A100-SXM4-80GB @ 400.00 W | INCOMPLETE (TypeError: expected Tensor as element 1 in argument 2, but got NoneType) |
| control | `{}` | 8/8 | 75.08 | 0.11 | control | 4.35 | control | control | 56.64 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| A-skipg1 | `{"epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]], "skip_residual_groups": [1], "skip_residual_until": 0.25}` | 0/8 | n/a | n/a | control | n/a | control | control | 84.39 (warm) | None | NVIDIA A100-SXM4-80GB @ 400.00 W | INCOMPLETE (TypeError: expected Tensor as element 4 in argument 2, but got NoneType) |
