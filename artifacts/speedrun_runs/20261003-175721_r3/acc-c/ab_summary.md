# r3/acc-c

2026-10-03 18:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B8EJ7J5HQMFR2FXYJ6KJR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 341.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41B85P7KF1GGH147N0VJPVR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B8EJ7J5HQMFR2FXYJ6KJR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.99 | 0.23 | control | 6.18 | control | control | 18.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| cutout4 | `{"cutout": 4}` | 8/8 | 74.73 | 0.25 | -0.26 +- 0.12 | 6.23 | +0.05 | +0.64 | 16.20 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| scale0.1389 | `{"scaling_factor": 0.1388888888888889}` | 8/8 | 75.23 | 0.29 | +0.24 +- 0.16 | 6.25 | +0.07 | -0.46 | 36.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| stack2 | `{"bias_scaler": 32.0, "label_smoothing": 0.25}` | 8/8 | 75.21 | 0.18 | +0.22 +- 0.06 | 6.27 | +0.09 | -0.41 | 16.40 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
