# r19/k19-d

2026-10-03 22:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RV70MT13DTNKHKA9H9NFR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 479.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RV70MT13DTNKHKA9H9NFR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wd0.012 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "weight_decay": 0.012}` | 8/8 | 74.81 | 0.26 | -0.33 +- 0.13 | 4.20 | +0.00 | +0.33 | 36.68 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.19 | control | control | 42.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| lr10.3 | `{"epochs": 9.5, "lr": 10.3, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.80 | 0.17 | -0.33 +- 0.08 | 4.20 | +0.00 | +0.33 | 34.65 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| scale1.5 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "scaling_factor": 0.16666666666666666}` | 8/8 | 75.12 | 0.24 | -0.01 +- 0.08 | 4.19 | -0.00 | +0.01 | 115.84 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
