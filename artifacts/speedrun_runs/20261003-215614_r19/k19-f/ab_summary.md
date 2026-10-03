# r19/k19-f

2026-10-03 22:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RV7DK5T71D3XS9G361C7R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 510.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RV7DK5T71D3XS9G361C7R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bsched512 | `{"batch_schedule": [[512, 0.5]], "epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.04 | 0.18 | -0.10 +- 0.08 | 4.33 | +0.18 | +0.28 | 205.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.14 | control | control | 35.42 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bsched512-lr14 | `{"batch_schedule": [[512, 0.5]], "epochs": 9.5, "lr": 14.0, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.02 | 0.16 | -0.11 +- 0.08 | 4.33 | +0.19 | +0.30 | 60.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
