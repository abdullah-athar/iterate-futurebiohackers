# r19/k19-e

2026-10-03 22:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RV7DK2REQ1JQQYJZV4KKR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 466.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RV7DK2REQ1JQQYJZV4KKR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whiten5 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "whiten_bias_epochs": 5}` | 8/8 | 74.84 | 0.29 | -0.30 +- 0.11 | 4.37 | +0.20 | +0.50 | 42.45 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.17 | control | control | 38.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.1 | `{"color_jitter": [0.1, 0.1], "epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.87 | 0.26 | -0.27 +- 0.09 | 4.21 | +0.04 | +0.31 | 38.42 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| cutout4 | `{"cutout": 4, "epochs": 9.5, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.70 | 0.24 | -0.43 +- 0.09 | 4.17 | +0.01 | +0.44 | 91.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
