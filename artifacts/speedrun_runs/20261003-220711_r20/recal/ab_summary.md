# r20/recal

2026-10-03 22:13, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41SF951ZRH4G0T3YQW0R88R, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 356.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41SF951ZRH4G0T3YQW0R88R CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-bnrecal2 | `{"bn_recal_batches": 2, "epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.05 | 0.21 | -0.03 +- 0.17 | 4.40 | +0.01 | +0.04 | 57.72 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.09 | 0.37 | control | 4.39 | control | control | 49.64 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| A-bnrecal5 | `{"bn_recal_batches": 5, "epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.09 | 0.17 | +0.01 +- 0.17 | 4.42 | +0.03 | +0.03 | 48.78 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
